/**
 * Copyright (c) 2026 Dan McLeran
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Iris species classifier built on TinyMind.
//
// Dataset: https://archive.ics.uci.edu/dataset/53/iris
//
// A 4-input -> 8-hidden -> 3-output MLP in Q8.8 fixed-point, tanh hidden +
// sigmoid output. The three sigmoid outputs are a one-hot encoding of the
// species (setosa, versicolor, virginica); the predicted class is argmax.
//
// The program loads `iris.data` (the CSV distributed by UCI) from the working
// directory. `make run` cd's into ./output, so the Makefile copies the data
// there. If the file is missing the program exits with a message -- the file
// is tiny (~4 KB) and ships with the example.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"
#include "constants.hpp"
#include "nnproperties.hpp"

// ---------------------------------------------------------------------------
// Q-format neural network definition
// ---------------------------------------------------------------------------

static constexpr size_t NUMBER_OF_FIXED_BITS = 16;
static constexpr size_t NUMBER_OF_FRACTIONAL_BITS = 16;
typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
typedef typename ValueType::FullWidthValueType FullWidthValueType;

static constexpr size_t NUMBER_OF_INPUTS = 4;
static constexpr size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static constexpr size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 8;
static constexpr size_t NUMBER_OF_OUTPUTS = 3;

struct RandomNumberGenerator
{
    static ValueType generateRandomWeight()
    {
        // Symmetric small init in [-0.5, 0.5] (raw Q-format units).
        const FullWidthValueType half =
            tinymind::Constants<ValueType>::one().getValue() / 2;
        const FullWidthValueType weight =
            (static_cast<FullWidthValueType>(rand()) % (2 * half + 1)) - half;
        return ValueType(weight);
    }
};

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    tinymind::ReluActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>,
    NUMBER_OF_OUTPUTS> TransferFunctionsType;

typedef tinymind::MultilayerPerceptron<
    ValueType,
    NUMBER_OF_INPUTS,
    NUMBER_OF_HIDDEN_LAYERS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
    TransferFunctionsType> NeuralNetworkType;

static NeuralNetworkType gNet;

// ---------------------------------------------------------------------------
// Sample + float<->Q helpers
// ---------------------------------------------------------------------------

struct Sample
{
    double feat[4]; // sepal len/width, petal len/width (cm)
    int    label;   // 0=setosa, 1=versicolor, 2=virginica
};

static inline ValueType toQ(double v)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    ValueType q;
    q.setValue(static_cast<FullWidthValueType>(std::lround(v * scale)));
    return q;
}

static inline double fromQ(const ValueType& q)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    return static_cast<double>(q.getValue()) / scale;
}

// Per-feature z-score statistics fit on the training split.
struct FeatureStats
{
    double mean[4];
    double stdev[4];
};

static void fitStats(const std::vector<Sample>& train, FeatureStats& st)
{
    for (size_t f = 0; f < 4; ++f) { st.mean[f] = 0.0; st.stdev[f] = 0.0; }
    for (const auto& s : train)
        for (size_t f = 0; f < 4; ++f) st.mean[f] += s.feat[f];
    const double n = static_cast<double>(train.size());
    for (size_t f = 0; f < 4; ++f) st.mean[f] /= n;
    for (const auto& s : train)
        for (size_t f = 0; f < 4; ++f)
        {
            const double d = s.feat[f] - st.mean[f];
            st.stdev[f] += d * d;
        }
    for (size_t f = 0; f < 4; ++f)
    {
        st.stdev[f] = std::sqrt(st.stdev[f] / n);
        if (st.stdev[f] < 1e-6) st.stdev[f] = 1.0;
    }
}

static FeatureStats gStats;

// z-score, then /3 so typical values sit inside Q16.16's stable range.
static inline void toInput(const Sample& s, ValueType* in)
{
    for (size_t f = 0; f < 4; ++f)
    {
        const double z = (s.feat[f] - gStats.mean[f]) / gStats.stdev[f] / 3.0;
        in[f] = toQ(z);
    }
}

// ---------------------------------------------------------------------------
// CSV loader for iris.data (UCI format: 4 floats + species name per line)
// ---------------------------------------------------------------------------

static bool loadCsv(const std::string& path, std::vector<Sample>& out)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ','))
        {
            cells.push_back(cell);
        }

        if (cells.size() < 5)
        {
            continue;
        }

        Sample s;
        for (size_t f = 0; f < 4; ++f)
        {
            s.feat[f] = std::stod(cells[f]);
        }
        const std::string& name = cells[4];
        if (name.find("setosa") != std::string::npos)          s.label = 0;
        else if (name.find("versicolor") != std::string::npos) s.label = 1;
        else if (name.find("virginica") != std::string::npos)  s.label = 2;
        else continue;
        out.push_back(s);
    }

    return !out.empty();
}

static const char* const CLASS_NAMES[3] = { "setosa", "versicolor", "virginica" };

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::srand(7U);

    std::vector<Sample> data;
    if (!loadCsv("iris.data", data))
    {
        std::cerr << "Could not open iris.data in the working directory.\n"
                     "Run via `make run` (it copies the bundled file into ./output)."
                  << std::endl;
        return 1;
    }
    std::cout << "Loaded " << data.size() << " rows from iris.data." << std::endl;

    // Deterministic shuffle + 80/20 split.
    std::srand(7U);
    for (size_t i = data.size(); i > 1; --i)
    {
        const size_t j = static_cast<size_t>(rand()) % i;
        std::swap(data[i - 1], data[j]);
    }
    const size_t nTrain = (data.size() * 4) / 5;
    std::vector<Sample> train(data.begin(), data.begin() + nTrain);
    std::vector<Sample> test(data.begin() + nTrain, data.end());
    std::cout << "Train: " << train.size() << "  Test: " << test.size() << std::endl;

    fitStats(train, gStats);

    // ---- Training ----------------------------------------------------------
    const unsigned iterations  = 30000U;
    const unsigned reportEvery =  1000U;
    ValueType input[NUMBER_OF_INPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType learned[NUMBER_OF_OUTPUTS];
    double errSum = 0.0;

    std::ofstream lossCsv("iris_loss.csv");
    lossCsv << "iter,avg_err" << std::endl;

    for (unsigned it = 0; it < iterations; ++it)
    {
        const Sample& s = train[static_cast<size_t>(rand()) % train.size()];
        toInput(s, input);
        for (size_t k = 0; k < NUMBER_OF_OUTPUTS; ++k)
        {
            target[k] = toQ(s.label == static_cast<int>(k) ? 1.0 : 0.0);
        }

        gNet.feedForward(input);
        const ValueType err = gNet.calculateError(target);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(err))
        {
            gNet.trainNetwork(target);
        }
        errSum += std::fabs(fromQ(err));

        if ((it + 1) % reportEvery == 0)
        {
            std::cout << "iter " << std::setw(6) << (it + 1)
                      << "   avg|err| = " << std::fixed << std::setprecision(4)
                      << (errSum / reportEvery) << std::endl;
            lossCsv << (it + 1) << "," << (errSum / reportEvery) << std::endl;
            errSum = 0.0;
        }
    }
    lossCsv.close();

    // ---- Evaluation --------------------------------------------------------
    int confusion[3][3] = { {0,0,0}, {0,0,0}, {0,0,0} };
    std::ofstream scatter("iris_scatter.csv");
    scatter << "petal_len,petal_width,true_class,pred_class" << std::endl;

    size_t correct = 0;
    for (const auto& s : test)
    {
        toInput(s, input);
        gNet.feedForward(input);
        gNet.getLearnedValues(learned);
        int pred = 0;
        double best = fromQ(learned[0]);
        for (size_t k = 1; k < NUMBER_OF_OUTPUTS; ++k)
        {
            const double v = fromQ(learned[k]);
            if (v > best) { best = v; pred = static_cast<int>(k); }
        }
        ++confusion[s.label][pred];
        if (pred == s.label) ++correct;
        scatter << s.feat[2] << "," << s.feat[3] << ","
                << s.label << "," << pred << std::endl;
    }
    scatter.close();

    {
        std::ofstream cm("iris_confusion.csv");
        cm << "actual,pred_setosa,pred_versicolor,pred_virginica" << std::endl;
        for (int r = 0; r < 3; ++r)
        {
            cm << CLASS_NAMES[r];
            for (int c = 0; c < 3; ++c) cm << "," << confusion[r][c];
            cm << std::endl;
        }
    }

    const double acc = static_cast<double>(correct) / static_cast<double>(test.size());
    std::cout << "\nTest accuracy: " << std::fixed << std::setprecision(4)
              << acc << "  (" << correct << "/" << test.size() << ")" << std::endl;

    return 0;
}
