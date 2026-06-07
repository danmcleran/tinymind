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

// Handwritten-digit classifier built on TinyMind.
//
// Dataset: UCI Optical Recognition of Handwritten Digits
//          https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
//
// A 64-input -> 32-hidden -> 10-output MLP in Q16.16 fixed-point, ReLU hidden +
// sigmoid output. Each input is one pixel (intensity 0..16) of an 8x8 bitmap,
// flattened row-major. The ten sigmoid outputs are a one-hot encoding of the
// digit (0..9); the predicted class is argmax.
//
// The program loads `optdigits.tra` (train) and `optdigits.tes` (test) from the
// working directory. `make run` cd's into ./output, so the Makefile copies the
// data there. If a file is missing the program exits with a message.

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

static constexpr size_t NUMBER_OF_PIXELS = 64;
static constexpr size_t NUMBER_OF_INPUTS = NUMBER_OF_PIXELS;
static constexpr size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static constexpr size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 32;
static constexpr size_t NUMBER_OF_OUTPUTS = 10;

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
    double feat[NUMBER_OF_PIXELS]; // pixel intensities 0..16 (8x8 row-major)
    int    pixel[NUMBER_OF_PIXELS]; // raw int pixels, for the image-grid CSV
    int    label;                  // digit 0..9
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
    double mean[NUMBER_OF_PIXELS];
    double stdev[NUMBER_OF_PIXELS];
};

static void fitStats(const std::vector<Sample>& train, FeatureStats& st)
{
    for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f) { st.mean[f] = 0.0; st.stdev[f] = 0.0; }
    for (const auto& s : train)
        for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f) st.mean[f] += s.feat[f];
    const double n = static_cast<double>(train.size());
    for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f) st.mean[f] /= n;
    for (const auto& s : train)
        for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f)
        {
            const double d = s.feat[f] - st.mean[f];
            st.stdev[f] += d * d;
        }
    for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f)
    {
        st.stdev[f] = std::sqrt(st.stdev[f] / n);
        // Many border pixels are constant zero -> stdev ~0. Guard so those
        // features map to a constant 0 input instead of dividing by ~0.
        if (st.stdev[f] < 1e-6) st.stdev[f] = 1.0;
    }
}

static FeatureStats gStats;

// z-score, then /3 so typical values sit inside Q16.16's stable range.
static inline void toInput(const Sample& s, ValueType* in)
{
    for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f)
    {
        const double z = (s.feat[f] - gStats.mean[f]) / gStats.stdev[f] / 3.0;
        in[f] = toQ(z);
    }
}

// ---------------------------------------------------------------------------
// CSV loader for optdigits.tra / optdigits.tes
// (UCI format: 64 int pixels + class label per line)
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

        if (cells.size() < NUMBER_OF_PIXELS + 1)
        {
            continue;
        }

        Sample s;
        for (size_t f = 0; f < NUMBER_OF_PIXELS; ++f)
        {
            s.pixel[f] = std::stoi(cells[f]);
            s.feat[f]  = static_cast<double>(s.pixel[f]);
        }
        s.label = std::stoi(cells[NUMBER_OF_PIXELS]);
        if (s.label < 0 || s.label > 9)
        {
            continue;
        }
        out.push_back(s);
    }

    return !out.empty();
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::srand(7U);

    std::vector<Sample> train;
    std::vector<Sample> test;
    if (!loadCsv("optdigits.tra", train))
    {
        std::cerr << "Could not open optdigits.tra in the working directory.\n"
                     "Run via `make run` (it copies the bundled file into ./output)."
                  << std::endl;
        return 1;
    }
    if (!loadCsv("optdigits.tes", test))
    {
        std::cerr << "Could not open optdigits.tes in the working directory.\n"
                     "Run via `make run` (it copies the bundled file into ./output)."
                  << std::endl;
        return 1;
    }
    std::cout << "Loaded " << train.size() << " train rows (optdigits.tra), "
              << test.size() << " test rows (optdigits.tes)." << std::endl;

    fitStats(train, gStats);

    // ---- Training ----------------------------------------------------------
    const unsigned iterations  = 60000U;
    const unsigned reportEvery =  2000U;
    ValueType input[NUMBER_OF_INPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType learned[NUMBER_OF_OUTPUTS];
    double errSum = 0.0;

    std::ofstream lossCsv("digits_loss.csv");
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
    int confusion[NUMBER_OF_OUTPUTS][NUMBER_OF_OUTPUTS];
    for (size_t r = 0; r < NUMBER_OF_OUTPUTS; ++r)
        for (size_t c = 0; c < NUMBER_OF_OUTPUTS; ++c)
            confusion[r][c] = 0;

    std::vector<int> predOf(test.size(), 0);
    size_t correct = 0;
    for (size_t i = 0; i < test.size(); ++i)
    {
        const Sample& s = test[i];
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
        predOf[i] = pred;
        ++confusion[s.label][pred];
        if (pred == s.label) ++correct;
    }

    {
        std::ofstream cm("digits_confusion.csv");
        cm << "actual";
        for (size_t c = 0; c < NUMBER_OF_OUTPUTS; ++c) cm << ",pred_" << c;
        cm << std::endl;
        for (size_t r = 0; r < NUMBER_OF_OUTPUTS; ++r)
        {
            cm << r;
            for (size_t c = 0; c < NUMBER_OF_OUTPUTS; ++c) cm << "," << confusion[r][c];
            cm << std::endl;
        }
    }

    // ---- Image-grid sample (up to 40 test images, mix correct + misclassified) ----
    {
        const size_t MAX_SAMPLES = 40;
        std::vector<size_t> chosen;

        // Mix: reserve up to half the grid for misclassified cases (the hard
        // examples worth eyeballing), fill the rest with correctly classified
        // images so the grid shows both successes and failures.
        const size_t MAX_MISS = MAX_SAMPLES / 2;
        size_t nMiss = 0;
        for (size_t i = 0; i < test.size() && nMiss < MAX_MISS; ++i)
            if (predOf[i] != test[i].label) { chosen.push_back(i); ++nMiss; }
        for (size_t i = 0; i < test.size() && chosen.size() < MAX_SAMPLES; ++i)
            if (predOf[i] == test[i].label) chosen.push_back(i);

        std::ofstream samp("digits_samples.csv");
        for (size_t p = 0; p < NUMBER_OF_PIXELS; ++p) samp << "pixel" << p << ",";
        samp << "true,pred" << std::endl;
        for (size_t idx : chosen)
        {
            const Sample& s = test[idx];
            for (size_t p = 0; p < NUMBER_OF_PIXELS; ++p) samp << s.pixel[p] << ",";
            samp << s.label << "," << predOf[idx] << std::endl;
        }
    }

    const double acc = static_cast<double>(correct) / static_cast<double>(test.size());
    std::cout << "\nTest accuracy: " << std::fixed << std::setprecision(4)
              << acc << "  (" << correct << "/" << test.size() << ")" << std::endl;

    return 0;
}
