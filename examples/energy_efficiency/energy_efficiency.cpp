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

// UCI Energy Efficiency regression built on TinyMind.
//
// Dataset: https://archive.ics.uci.edu/dataset/242/energy+efficiency
//
// An 8-input -> 16-hidden -> 2-output MLP in Q16.16 fixed-point, ReLU hidden +
// LINEAR output (this is regression, not classification). The eight building
// design features (X1..X8) predict the two energy loads: Y1 = heating load,
// Y2 = cooling load.
//
// Inputs are z-score normalized (training-set statistics) then scaled by 1/3 so
// typical values sit inside Q16.16's stable range. Because the output activation
// is linear, the targets are standardized to z-score as well during training;
// predictions are de-standardized back to real load units before reporting.
//
// The program loads `ENB2012_data.csv` (the CSV distributed by UCI) from the
// working directory. `make run` cd's into ./output, so the Makefile copies the
// data there. If the file is missing the program exits with a message.

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

static constexpr size_t NUMBER_OF_INPUTS = 8;
static constexpr size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static constexpr size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 16;
static constexpr size_t NUMBER_OF_OUTPUTS = 2;

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

// Regression: ReLU hidden, LINEAR output. The transfer-function policy needs
// NumberOfOutputNeurons (5th template arg) set to 2 or it static_asserts.
//
// A linear output head produces an unbounded gradient. In Q16.16 that gradient
// saturates the fixed-point multiply, which makes the effective step size
// independent of the learning rate and the hidden weights explode. Swapping the
// default no-op gradient clipper (9th template arg) for GradientClipByValue
// (clamp to [-1, 1]) keeps each step bounded so the regression trains stably.
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    tinymind::ReluActivationPolicy<ValueType>,
    tinymind::LinearActivationPolicy<ValueType>,
    NUMBER_OF_OUTPUTS,
    tinymind::DefaultNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, NUMBER_OF_OUTPUTS>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType> > TransferFunctionsType;

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
    double feat[8]; // X1..X8 building design features
    double load[2]; // Y1 = heating load, Y2 = cooling load
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
    double mean[8];
    double stdev[8];
};

// Per-target z-score statistics fit on the training split.
struct TargetStats
{
    double mean[2];
    double stdev[2];
};

static void fitFeatureStats(const std::vector<Sample>& train, FeatureStats& st)
{
    for (size_t f = 0; f < 8; ++f) { st.mean[f] = 0.0; st.stdev[f] = 0.0; }
    for (const auto& s : train)
        for (size_t f = 0; f < 8; ++f) st.mean[f] += s.feat[f];
    const double n = static_cast<double>(train.size());
    for (size_t f = 0; f < 8; ++f) st.mean[f] /= n;
    for (const auto& s : train)
        for (size_t f = 0; f < 8; ++f)
        {
            const double d = s.feat[f] - st.mean[f];
            st.stdev[f] += d * d;
        }
    for (size_t f = 0; f < 8; ++f)
    {
        st.stdev[f] = std::sqrt(st.stdev[f] / n);
        if (st.stdev[f] < 1e-6) st.stdev[f] = 1.0;
    }
}

static void fitTargetStats(const std::vector<Sample>& train, TargetStats& st)
{
    for (size_t t = 0; t < 2; ++t) { st.mean[t] = 0.0; st.stdev[t] = 0.0; }
    for (const auto& s : train)
        for (size_t t = 0; t < 2; ++t) st.mean[t] += s.load[t];
    const double n = static_cast<double>(train.size());
    for (size_t t = 0; t < 2; ++t) st.mean[t] /= n;
    for (const auto& s : train)
        for (size_t t = 0; t < 2; ++t)
        {
            const double d = s.load[t] - st.mean[t];
            st.stdev[t] += d * d;
        }
    for (size_t t = 0; t < 2; ++t)
    {
        st.stdev[t] = std::sqrt(st.stdev[t] / n);
        if (st.stdev[t] < 1e-6) st.stdev[t] = 1.0;
    }
}

static FeatureStats gFeatStats;
static TargetStats  gTargetStats;

// z-score, then /3 so typical values sit inside Q16.16's stable range.
static inline void toInput(const Sample& s, ValueType* in)
{
    for (size_t f = 0; f < 8; ++f)
    {
        const double z = (s.feat[f] - gFeatStats.mean[f]) / gFeatStats.stdev[f] / 3.0;
        in[f] = toQ(z);
    }
}

// Targets are standardized to z-score and then scaled by 1/TARGET_SCALE so the
// linear-output gradients stay small enough to avoid Q16.16 saturation (a large
// standardized error otherwise overflows the fixed-point gradient term).
static constexpr double TARGET_SCALE = 6.0;

static inline void toTarget(const Sample& s, ValueType* tgt)
{
    for (size_t t = 0; t < 2; ++t)
    {
        const double z = (s.load[t] - gTargetStats.mean[t]) / gTargetStats.stdev[t] / TARGET_SCALE;
        tgt[t] = toQ(z);
    }
}

// De-standardize a network output back to real load units.
static inline double fromTarget(double z, size_t t)
{
    return z * TARGET_SCALE * gTargetStats.stdev[t] + gTargetStats.mean[t];
}

// ---------------------------------------------------------------------------
// CSV loader for ENB2012_data.csv (UCI format: X1..X8,Y1,Y2 + 2 empty cols)
// ---------------------------------------------------------------------------

static bool loadCsv(const std::string& path, std::vector<Sample>& out)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    std::getline(in, line); // skip header row

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

        // Need X1..X8 (0..7) plus Y1,Y2 (8,9); trailing empty cols ignored.
        if (cells.size() < 10 || cells[0].empty())
        {
            continue;
        }

        Sample s;
        for (size_t f = 0; f < 8; ++f)
        {
            s.feat[f] = std::stod(cells[f]);
        }
        s.load[0] = std::stod(cells[8]); // Y1 heating load
        s.load[1] = std::stod(cells[9]); // Y2 cooling load
        out.push_back(s);
    }

    return !out.empty();
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::srand(7U);

    std::vector<Sample> data;
    if (!loadCsv("ENB2012_data.csv", data))
    {
        std::cerr << "Could not open ENB2012_data.csv in the working directory.\n"
                     "Run via `make run` (it copies the bundled file into ./output)."
                  << std::endl;
        return 1;
    }
    std::cout << "Loaded " << data.size() << " rows from ENB2012_data.csv." << std::endl;

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

    fitFeatureStats(train, gFeatStats);
    fitTargetStats(train, gTargetStats);

    // With gradient clipping in place the step size is well behaved; a modest
    // learning rate with momentum and acceleration disabled gives a clean,
    // stable regression fit. (The default nonzero momentum + acceleration terms
    // accumulate independently of the learning rate and slowly destabilize a
    // linear-output regressor.)
    gNet.setLearningRate(toQ(0.02));
    gNet.setMomentumRate(toQ(0.0));
    gNet.setAccelerationRate(toQ(0.0));

    // ---- Training ----------------------------------------------------------
    const unsigned iterations  = 40000U;
    const unsigned reportEvery =  2000U;
    ValueType input[NUMBER_OF_INPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType learned[NUMBER_OF_OUTPUTS];
    double errSum = 0.0;

    std::ofstream lossCsv("energy_loss.csv");
    lossCsv << "iter,avg_err" << std::endl;

    for (unsigned it = 0; it < iterations; ++it)
    {
        const Sample& s = train[static_cast<size_t>(rand()) % train.size()];
        toInput(s, input);
        toTarget(s, target);

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

    // ---- Evaluation (real load units) --------------------------------------
    std::ofstream pred("energy_pred.csv");
    pred << "sample,y1_true,y1_pred,y2_true,y2_pred" << std::endl;

    // Accumulators for MAE and R^2 per target.
    double sumAbs[2]   = { 0.0, 0.0 };
    double sumSqRes[2] = { 0.0, 0.0 };
    double sumTrue[2]  = { 0.0, 0.0 };
    double sumSqTrue[2] = { 0.0, 0.0 };

    for (size_t i = 0; i < test.size(); ++i)
    {
        const Sample& s = test[i];
        toInput(s, input);
        gNet.feedForward(input);
        gNet.getLearnedValues(learned);

        double yPred[2];
        for (size_t t = 0; t < 2; ++t)
        {
            yPred[t] = fromTarget(fromQ(learned[t]), t);
            const double r = yPred[t] - s.load[t];
            sumAbs[t]    += std::fabs(r);
            sumSqRes[t]  += r * r;
            sumTrue[t]   += s.load[t];
            sumSqTrue[t] += s.load[t] * s.load[t];
        }

        pred << i << ","
             << s.load[0] << "," << yPred[0] << ","
             << s.load[1] << "," << yPred[1] << std::endl;
    }
    pred.close();

    const double n = static_cast<double>(test.size());
    double mae[2], r2[2];
    for (size_t t = 0; t < 2; ++t)
    {
        mae[t] = sumAbs[t] / n;
        const double meanTrue = sumTrue[t] / n;
        const double ssTot = sumSqTrue[t] - n * meanTrue * meanTrue;
        r2[t] = (ssTot > 1e-9) ? (1.0 - sumSqRes[t] / ssTot) : 0.0;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nHeating load (Y1):  MAE=" << mae[0] << "  R2=" << r2[0] << std::endl;
    std::cout << "Cooling load (Y2):  MAE=" << mae[1] << "  R2=" << r2[1] << std::endl;

    return 0;
}
