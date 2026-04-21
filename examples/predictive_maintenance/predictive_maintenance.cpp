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

// AI4I 2020 Predictive Maintenance binary classifier built on TinyMind.
//
// Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
//
// The program tries to load `ai4i2020.csv` from the working directory (the CSV
// distributed by UCI). If the file is not present it falls back to a synthetic
// generator that follows the documented AI4I 2020 generative/failure rules so
// the example still trains end-to-end without a download.
//
// The network is a 7-input -> 8-hidden -> 1-output MLP in Q16.16 fixed-point,
// ReLU hidden + sigmoid output. The six process features plus a two-dim
// one-hot for product variant (L, M; H = [0,0]) drive a binary classifier for
// the `Machine failure` column. Because failures are only ~3.4% of the data,
// training draws 50/50 balanced mini-samples from positive and negative pools.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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

static constexpr size_t NUMBER_OF_INPUTS = 7;
static constexpr size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static constexpr size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 8;
static constexpr size_t NUMBER_OF_OUTPUTS = 1;

struct RandomNumberGenerator
{
    static ValueType generateRandomWeight()
    {
        const FullWidthValueType one = tinymind::Constants<ValueType>::one().getValue();
        const FullWidthValueType negOne = tinymind::Constants<ValueType>::negativeOne().getValue();
        const FullWidthValueType weight =
            (std::rand() % (one + one - negOne)) + negOne;

        return ValueType(weight);
    }
};

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    tinymind::ReluActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>> TransferFunctionsType;

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
    double airTempK;
    double procTempK;
    double rpm;
    double torqueNm;
    double toolWearMin;
    int    variant; // 0=H, 1=M, 2=L
    int    label;   // 0/1
};

static inline ValueType toQ(double v)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    ValueType q;
    q.setValue(static_cast<FullWidthValueType>(v * scale));
    return q;
}

static inline double fromQ(const ValueType& q)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    return static_cast<double>(q.getValue()) / scale;
}

// ---------------------------------------------------------------------------
// CSV loader for ai4i2020.csv (as distributed by UCI)
// ---------------------------------------------------------------------------
//
// Header: UDI,Product ID,Type,Air temperature [K],Process temperature [K],
//         Rotational speed [rpm],Torque [Nm],Tool wear [min],
//         Machine failure,TWF,HDF,PWF,OSF,RNF

static bool loadCsv(const std::string& path, std::vector<Sample>& out)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    std::getline(in, line); // skip header

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

        if (cells.size() < 9)
        {
            continue;
        }

        Sample s;
        const std::string& type = cells[2];
        s.variant = (type == "L") ? 2 : (type == "M") ? 1 : 0;
        s.airTempK    = std::stod(cells[3]);
        s.procTempK   = std::stod(cells[4]);
        s.rpm         = std::stod(cells[5]);
        s.torqueNm    = std::stod(cells[6]);
        s.toolWearMin = std::stod(cells[7]);
        s.label       = std::stoi(cells[8]);
        out.push_back(s);
    }

    return !out.empty();
}

// ---------------------------------------------------------------------------
// Synthetic fallback: follows the documented AI4I 2020 generative and
// failure-labelling rules. Used when ai4i2020.csv is not present.
// ---------------------------------------------------------------------------

static void synthesizeDataset(std::vector<Sample>& out, std::size_t n, std::mt19937& rng)
{
    std::normal_distribution<double> airNoise(0.0, 2.0);
    std::normal_distribution<double> procNoise(0.0, 1.0);
    std::normal_distribution<double> torqueDist(40.0, 10.0);
    std::normal_distribution<double> rpmDist(1538.0, 180.0);
    std::uniform_real_distribution<double> wearDist(0.0, 253.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    out.clear();
    out.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        Sample s{};
        const double u = unit(rng);
        s.variant = (u < 0.5) ? 2 : (u < 0.8) ? 1 : 0; // 50% L, 30% M, 20% H

        s.airTempK  = 300.0 + airNoise(rng);
        s.procTempK = s.airTempK + 10.0 + procNoise(rng);

        double torque = torqueDist(rng);
        if (torque < 3.8)  torque = 3.8;
        if (torque > 76.6) torque = 76.6;
        s.torqueNm = torque;

        double rpm = rpmDist(rng);
        if (rpm < 1168.0) rpm = 1168.0;
        if (rpm > 2886.0) rpm = 2886.0;
        s.rpm = rpm;

        s.toolWearMin = wearDist(rng);

        // Failure-mode rules.
        const bool twf  = (s.toolWearMin >= 200.0 && s.toolWearMin <= 240.0 && unit(rng) < 0.10);
        const bool hdf  = (std::fabs(s.airTempK - s.procTempK) < 8.6 && s.rpm < 1380.0);
        const double P  = s.torqueNm * (s.rpm * 2.0 * M_PI / 60.0);
        const bool pwf  = (P < 3500.0 || P > 9000.0);
        const double osfThresh = (s.variant == 2 ? 11000.0 : s.variant == 1 ? 12000.0 : 13000.0);
        const bool osf  = (s.toolWearMin * s.torqueNm > osfThresh);
        const bool rnf  = (unit(rng) < 0.001);

        s.label = (twf || hdf || pwf || osf || rnf) ? 1 : 0;
        out.push_back(s);
    }
}

// ---------------------------------------------------------------------------
// Standardize numeric features (z-score) using training-set statistics.
// ---------------------------------------------------------------------------

struct FeatureStats
{
    double mean[5];
    double stdev[5];
};

static void fitStats(const std::vector<Sample>& train, FeatureStats& st)
{
    for (size_t f = 0; f < 5; ++f) { st.mean[f] = 0.0; st.stdev[f] = 0.0; }
    for (const auto& s : train)
    {
        st.mean[0] += s.airTempK;
        st.mean[1] += s.procTempK;
        st.mean[2] += s.rpm;
        st.mean[3] += s.torqueNm;
        st.mean[4] += s.toolWearMin;
    }
    const double n = static_cast<double>(train.size());
    for (size_t f = 0; f < 5; ++f) st.mean[f] /= n;

    for (const auto& s : train)
    {
        const double v[5] = { s.airTempK, s.procTempK, s.rpm, s.torqueNm, s.toolWearMin };
        for (size_t f = 0; f < 5; ++f)
        {
            const double d = v[f] - st.mean[f];
            st.stdev[f] += d * d;
        }
    }
    for (size_t f = 0; f < 5; ++f)
    {
        st.stdev[f] = std::sqrt(st.stdev[f] / n);
        if (st.stdev[f] < 1e-6) st.stdev[f] = 1.0;
    }
}

static void toInput(const Sample& s, const FeatureStats& st, ValueType* in)
{
    const double v[5] = { s.airTempK, s.procTempK, s.rpm, s.torqueNm, s.toolWearMin };
    for (size_t f = 0; f < 5; ++f)
    {
        // Scale z-score by 1/3 so typical values sit within Q16.16's sweet spot.
        const double z = (v[f] - st.mean[f]) / st.stdev[f] / 3.0;
        in[f] = toQ(z);
    }
    in[5] = toQ(s.variant == 2 ? 1.0 : 0.0); // L
    in[6] = toQ(s.variant == 1 ? 1.0 : 0.0); // M
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::srand(7U);
    std::mt19937 rng(7U);

    std::vector<Sample> data;
    if (loadCsv("ai4i2020.csv", data))
    {
        std::cout << "Loaded " << data.size() << " rows from ai4i2020.csv." << std::endl;
    }
    else
    {
        std::cout << "ai4i2020.csv not found; synthesizing 10000 rows using the "
                     "documented AI4I 2020 generative rules." << std::endl;
        synthesizeDataset(data, 10000, rng);
    }

    std::shuffle(data.begin(), data.end(), rng);
    const size_t nTrain = (data.size() * 4) / 5;
    std::vector<Sample> train(data.begin(), data.begin() + nTrain);
    std::vector<Sample> test(data.begin() + nTrain, data.end());

    FeatureStats st;
    fitStats(train, st);

    // Split training samples by class for balanced sampling.
    std::vector<size_t> pos, neg;
    for (size_t i = 0; i < train.size(); ++i)
    {
        (train[i].label ? pos : neg).push_back(i);
    }

    std::cout << "Train: " << train.size() << " (pos=" << pos.size()
              << ", neg=" << neg.size() << ")  Test: " << test.size() << std::endl;

    if (pos.empty())
    {
        std::cerr << "No positive samples in training set." << std::endl;
        return 1;
    }

    // Training loop with 50/50 balanced sampling.
    const unsigned iterations   = 40000U;
    const unsigned reportEvery  =  2000U;
    ValueType input[NUMBER_OF_INPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType learned[NUMBER_OF_OUTPUTS];
    double errSum = 0.0;

    std::uniform_int_distribution<size_t> posPick(0, pos.size() - 1);
    std::uniform_int_distribution<size_t> negPick(0, neg.size() - 1);
    std::bernoulli_distribution coin(0.5);

    for (unsigned it = 0; it < iterations; ++it)
    {
        const Sample& s = coin(rng) ? train[pos[posPick(rng)]] : train[neg[negPick(rng)]];
        toInput(s, st, input);
        target[0] = toQ(s.label ? 1.0 : 0.0);

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
                      << "   avg|err| = "
                      << std::fixed << std::setprecision(4)
                      << (errSum / reportEvery) << std::endl;
            errSum = 0.0;
        }
    }

    // Evaluate on held-out test set.
    size_t tp = 0, fp = 0, tn = 0, fn = 0;
    for (const auto& s : test)
    {
        toInput(s, st, input);
        gNet.feedForward(input);
        gNet.getLearnedValues(learned);
        const bool predFail = fromQ(learned[0]) >= 0.5;
        const bool realFail = s.label != 0;
        if (predFail && realFail)      ++tp;
        else if (predFail && !realFail) ++fp;
        else if (!predFail && realFail) ++fn;
        else                            ++tn;
    }

    const double acc   = static_cast<double>(tp + tn) / static_cast<double>(test.size());
    const double prec  = (tp + fp) ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
    const double rec   = (tp + fn) ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
    const double f1    = (prec + rec) > 0.0 ? (2.0 * prec * rec) / (prec + rec) : 0.0;

    std::cout << "\nConfusion matrix (rows=actual, cols=predicted):" << std::endl;
    std::cout << "               pred no-fail   pred fail" << std::endl;
    std::cout << "  actual no-fail " << std::setw(8) << tn << std::setw(14) << fp << std::endl;
    std::cout << "  actual fail    " << std::setw(8) << fn << std::setw(14) << tp << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\naccuracy=" << acc
              << "  precision=" << prec
              << "  recall="    << rec
              << "  F1="        << f1 << std::endl;

    return 0;
}
