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

// Gas-sensor-array drift classifier built on TinyMind.
//
// Inspired by the UCI "Gas Sensor Array Drift Dataset":
//   https://archive.ics.uci.edu/dataset/224
//
// The real dataset is large (13910 measurements, 16 chemo-resistive sensors x 8
// features = 128 dimensions, recorded over 36 months in 10 batches). Rather than
// bundle a multi-megabyte multi-file download, this example follows the
// `examples/predictive_maintenance` precedent and SYNTHESIZES a
// physically-motivated dataset with documented rules (see synthesizeDataset()).
//
// The headline phenomenon is SENSOR DRIFT: metal-oxide gas sensors age, so a
// model trained on early measurements slowly loses accuracy on later ones. We
// reproduce that here by generating 10 sequential "batches" (months) and
// applying a progressively growing per-sensor multiplicative gain and additive
// offset. We train ONLY on batch 1 (the freshest, cleanest data) and then
// evaluate accuracy on every batch 1..10 -- the resulting accuracy-vs-batch
// curve is the drift curve, and it slopes down.
//
// Network: 128-input -> 32-hidden -> 6-output MLP in Q16.16 fixed-point,
// ReLU hidden + sigmoid output. The six sigmoid outputs are a one-hot encoding
// of the gas class; the predicted class is argmax.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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
// Q8.8 collapses on a 128-input net -- use Q16.16.
typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
typedef typename ValueType::FullWidthValueType FullWidthValueType;

static constexpr size_t NUMBER_OF_INPUTS = 128; // 16 sensors x 8 features
static constexpr size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static constexpr size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 32;
static constexpr size_t NUMBER_OF_OUTPUTS = 6; // 6 gas classes

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

// FixedPointTransferFunctions takes NumberOfOutputs as its 5th template arg
// (after the two activation policies); it MUST match NUMBER_OF_OUTPUTS or a
// static_assert fires ("TransferFunctionPolicy NumberOfOutputNeurons is
// incorrect").
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
    double feat[NUMBER_OF_INPUTS]; // 128 sensor-feature responses
    int    label;                  // 0..5 gas class
    int    batch;                  // 1..10 month/batch index
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

// Per-feature z-score statistics fit on the batch-1 training split.
struct FeatureStats
{
    double mean[NUMBER_OF_INPUTS];
    double stdev[NUMBER_OF_INPUTS];
};

static void fitStats(const std::vector<Sample>& train, FeatureStats& st)
{
    for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f) { st.mean[f] = 0.0; st.stdev[f] = 0.0; }
    for (const auto& s : train)
        for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f) st.mean[f] += s.feat[f];
    const double n = static_cast<double>(train.size());
    for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f) st.mean[f] /= n;
    for (const auto& s : train)
        for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f)
        {
            const double d = s.feat[f] - st.mean[f];
            st.stdev[f] += d * d;
        }
    for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f)
    {
        st.stdev[f] = std::sqrt(st.stdev[f] / n);
        if (st.stdev[f] < 1e-6) st.stdev[f] = 1.0;
    }
}

static FeatureStats gStats;

// z-score using BATCH-1 TRAIN statistics, then /3 so typical values sit inside
// Q16.16's stable range. The SAME batch-1 statistics are applied to every later
// batch -- that is the point: drift pushes later batches off the trained
// distribution, and the normalization does not compensate.
static inline void toInput(const Sample& s, ValueType* in)
{
    for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f)
    {
        const double z = (s.feat[f] - gStats.mean[f]) / gStats.stdev[f] / 3.0;
        in[f] = toQ(z);
    }
}

// ---------------------------------------------------------------------------
// Synthetic dataset generator (physically-motivated, documented rules).
// ---------------------------------------------------------------------------
//
// 6 gas classes (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene).
// Each class has a distinct 128-dim mean response vector built so the classes
// are cleanly separable in batch 1. Per-sample Gaussian noise gives spread.
//
// DRIFT: each of the 10 batches applies a progressively growing per-sensor-
// feature multiplicative gain and additive offset. The drift magnitude scales
// with (batch - 1), so batch 1 is drift-free and batch 10 is the most degraded.
// A model trained on batch 1 sees later batches as shifted/scaled and its
// accuracy falls -- the drift curve.

static const char* const CLASS_NAMES[NUMBER_OF_OUTPUTS] = {
    "Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"
};

static constexpr int    NUM_CLASSES         = static_cast<int>(NUMBER_OF_OUTPUTS);
static constexpr int    NUM_BATCHES         = 10;
static constexpr int    SAMPLES_PER_CLASS   = 150;
static constexpr double SAMPLE_NOISE_STDEV  = 0.18; // per-sample Gaussian spread
static constexpr double DRIFT_GAIN_PER_BATCH   = 0.16; // multiplicative drift slope
static constexpr double DRIFT_OFFSET_PER_BATCH = 0.13; // additive drift slope

// rand()-based standard-normal sample (Box-Muller); deterministic under srand.
static double randUniform()
{
    return (static_cast<double>(rand()) + 0.5) /
           (static_cast<double>(RAND_MAX) + 1.0);
}

static double randNormal()
{
    double u1 = randUniform();
    double u2 = randUniform();
    if (u1 < 1e-12) u1 = 1e-12;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// Distinct per-class mean response vectors. Each class lights up a different
// subset of the 16 sensors with a class-specific profile across the 8 features.
static void buildClassMeans(double means[NUM_CLASSES][NUMBER_OF_INPUTS])
{
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        for (int sensor = 0; sensor < 16; ++sensor)
        {
            // Each class has a characteristic activation per sensor.
            const double phase = 0.6 * c + 0.37 * sensor;
            const double sensorGain = 1.0 + 0.9 * std::sin(phase);
            for (int feat = 0; feat < 8; ++feat)
            {
                const int idx = sensor * 8 + feat;
                // Feature profile (e.g. steady-state vs transient slopes).
                const double featProfile =
                    std::cos(0.5 * feat + 0.31 * c) + 0.4 * std::sin(0.8 * feat);
                means[c][idx] = sensorGain * (0.8 + 0.5 * featProfile);
            }
        }
    }
}

static void synthesizeDataset(std::vector<Sample>& out)
{
    double means[NUM_CLASSES][NUMBER_OF_INPUTS];
    buildClassMeans(means);

    // Per-sensor-feature drift direction (fixed pattern, scaled per batch).
    double driftGainDir[NUMBER_OF_INPUTS];
    double driftOffDir[NUMBER_OF_INPUTS];
    for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f)
    {
        driftGainDir[f] = std::sin(0.21 * static_cast<double>(f) + 1.3);
        driftOffDir[f]  = std::cos(0.17 * static_cast<double>(f) + 0.7);
    }

    out.clear();
    out.reserve(static_cast<size_t>(NUM_BATCHES) * NUM_CLASSES * SAMPLES_PER_CLASS);

    for (int b = 1; b <= NUM_BATCHES; ++b)
    {
        const double driftStep = static_cast<double>(b - 1); // 0 at batch 1
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            for (int n = 0; n < SAMPLES_PER_CLASS; ++n)
            {
                Sample s{};
                s.label = c;
                s.batch = b;
                for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f)
                {
                    const double gain =
                        1.0 + DRIFT_GAIN_PER_BATCH * driftStep * driftGainDir[f];
                    const double offset =
                        DRIFT_OFFSET_PER_BATCH * driftStep * driftOffDir[f];
                    const double clean = means[c][f];
                    const double noisy = clean + SAMPLE_NOISE_STDEV * randNormal();
                    s.feat[f] = gain * noisy + offset;
                }
                out.push_back(s);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::srand(7U);

    // ---- Synthesize the drift dataset --------------------------------------
    std::vector<Sample> data;
    synthesizeDataset(data);
    std::cout << "Synthesized " << data.size() << " samples across "
              << NUM_BATCHES << " batches (" << NUM_CLASSES << " classes, "
              << SAMPLES_PER_CLASS << " samples/class/batch)." << std::endl;

    // ---- Split batch 1 into 80/20 train/val --------------------------------
    std::vector<Sample> batch1;
    for (const auto& s : data)
        if (s.batch == 1) batch1.push_back(s);

    // Deterministic shuffle of batch 1.
    for (size_t i = batch1.size(); i > 1; --i)
    {
        const size_t j = static_cast<size_t>(rand()) % i;
        std::swap(batch1[i - 1], batch1[j]);
    }
    const size_t nTrain = (batch1.size() * 4) / 5;
    std::vector<Sample> train(batch1.begin(), batch1.begin() + nTrain);
    std::vector<Sample> val(batch1.begin() + nTrain, batch1.end());
    std::cout << "Batch 1 -> train: " << train.size()
              << "  val: " << val.size() << std::endl;

    // Fit normalization on the batch-1 training split only.
    fitStats(train, gStats);

    // ---- Training (batch 1 only) -------------------------------------------
    const unsigned iterations  = 50000U;
    const unsigned reportEvery =  2000U;
    ValueType input[NUMBER_OF_INPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType learned[NUMBER_OF_OUTPUTS];
    double errSum = 0.0;

    std::ofstream lossCsv("gas_loss.csv");
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

    // ---- Helper: predicted class via argmax --------------------------------
    auto predictClass = [&](const Sample& s) -> int {
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
        return pred;
    };

    // ---- Batch-1 validation confusion matrix -------------------------------
    int confusion[NUM_CLASSES][NUM_CLASSES];
    for (int r = 0; r < NUM_CLASSES; ++r)
        for (int c = 0; c < NUM_CLASSES; ++c) confusion[r][c] = 0;

    size_t valCorrect = 0;
    for (const auto& s : val)
    {
        const int pred = predictClass(s);
        ++confusion[s.label][pred];
        if (pred == s.label) ++valCorrect;
    }
    const double batch1ValAcc =
        static_cast<double>(valCorrect) / static_cast<double>(val.size());

    {
        std::ofstream cm("gas_confusion.csv");
        cm << "actual";
        for (int c = 0; c < NUM_CLASSES; ++c) cm << ",pred_" << c;
        cm << std::endl;
        for (int r = 0; r < NUM_CLASSES; ++r)
        {
            cm << CLASS_NAMES[r];
            for (int c = 0; c < NUM_CLASSES; ++c) cm << "," << confusion[r][c];
            cm << std::endl;
        }
    }

    // ---- Drift curve: accuracy on each batch 1..10 -------------------------
    std::ofstream driftCsv("gas_drift.csv");
    driftCsv << "batch,accuracy" << std::endl;

    std::cout << "\nDrift curve (model trained on batch 1 only):" << std::endl;
    double batch10Acc = 0.0;
    for (int b = 1; b <= NUM_BATCHES; ++b)
    {
        size_t correct = 0;
        size_t total = 0;
        for (const auto& s : data)
        {
            if (s.batch != b) continue;
            // For batch 1, score on the held-out val split only so the curve's
            // first point is an honest in-distribution number; later batches are
            // entirely unseen, so score all of them.
            if (b == 1)
            {
                continue; // handled below via the val split
            }
            ++total;
            if (predictClass(s) == s.label) ++correct;
        }

        double acc;
        if (b == 1)
        {
            acc = batch1ValAcc;
        }
        else
        {
            acc = static_cast<double>(correct) / static_cast<double>(total);
        }
        if (b == NUM_BATCHES) batch10Acc = acc;

        driftCsv << b << "," << std::fixed << std::setprecision(4) << acc << std::endl;
        std::cout << "  batch " << std::setw(2) << b
                  << "   accuracy = " << std::fixed << std::setprecision(4)
                  << acc << std::endl;
    }
    driftCsv.close();

    std::cout << "\nBatch 1 (val) accuracy: " << std::fixed << std::setprecision(4)
              << batch1ValAcc << std::endl;
    std::cout << "Batch 10 accuracy:      " << std::fixed << std::setprecision(4)
              << batch10Acc << "   (drift degradation)" << std::endl;

    return 0;
}
