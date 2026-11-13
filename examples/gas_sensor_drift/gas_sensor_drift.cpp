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

// Gas-sensor-array drift classifier built on tinymind.
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
#include <random>
#include <string>
#include <vector>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"
#include "constants.hpp"
#include "nnproperties.hpp"
#include "densehead.hpp"
#include "prototypehead.hpp"
#include "include/tinymind_rounding.hpp"

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
            static_cast<FullWidthValueType>((rand() % (2 * half + 1)) - half);
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

// ---------------------------------------------------------------------------
// Adapting to drift on the device
// ---------------------------------------------------------------------------
//
// The trained MLP is split in two: a frozen backbone (the 128 -> 32 ReLU
// layer) and a head (the 32 -> 6 layer). On each drifted batch a few labelled
// samples per class are drawn (the "shots"); the rest of the batch is the
// test set. Seven ways to use them are compared, each starting from the
// shipped model:
//
//   none       the shipped model, unchanged
//   renorm     z-score statistics recomputed from the shots, labels unused
//   renorm_all z-score statistics from the whole batch, labels unused (a
//              reference: unlabelled data is cheap on a device)
//   prototype  tinymind::PrototypeHead on the backbone's embedding: one mean
//              embedding per class from the shots, nearest mean wins
//   head       tinymind::DenseHead loaded with the shipped head and trained
//              on the shots with softmax cross-entropy (head-only fine-tuning)
//   bias       only the backbone's 32 biases trained, the head frozen; the
//              gradient comes from DenseHead::backward()
//   bias_head  the biases and the head trained together
//
// Each (batch, shots) cell is averaged over 10 random draws of the shots.

static constexpr size_t EMBEDDING = NUMBER_OF_NEURONS_PER_HIDDEN_LAYER;
typedef tinymind::DenseHead<ValueType, EMBEDDING, NUMBER_OF_OUTPUTS> HeadType;
typedef tinymind::PrototypeHead<ValueType, EMBEDDING, NUMBER_OF_OUTPUTS> PrototypeType;

enum Method
{
    METHOD_NONE = 0,
    METHOD_RENORM,
    METHOD_RENORM_ALL,
    METHOD_PROTOTYPE,
    METHOD_HEAD,
    METHOD_BIAS,
    METHOD_BIAS_HEAD,
    NUM_METHODS
};

static const char* const METHOD_NAMES[NUM_METHODS] = {
    "none", "renorm", "renorm_all", "prototype", "head", "bias", "bias_head"
};

static const int SHOTS[] = {1, 2, 5, 10, 20};
static constexpr size_t NUM_SHOTS = sizeof(SHOTS) / sizeof(SHOTS[0]);
static constexpr int DRAWS = 10;
static constexpr unsigned ADAPT_EPOCHS = 30U;
static constexpr double ADAPT_LEARNING_RATE = 0.05;

typedef std::vector<ValueType> Vec;

// The frozen backbone: the MLP's input-to-hidden weights and biases, read
// out of the trained network.
struct Backbone
{
    ValueType weights[EMBEDDING][NUMBER_OF_INPUTS];
    ValueType biases[EMBEDDING];

    void load(NeuralNetworkType& net)
    {
        for (size_t k = 0; k < EMBEDDING; ++k)
        {
            for (size_t i = 0; i < NUMBER_OF_INPUTS; ++i)
            {
                weights[k][i] = net.getInputLayerWeightForNeuronAndConnection(i, k);
            }
            biases[k] = net.getInputLayerBiasNeuronWeightForConnection(k);
        }
    }

    // W x without the bias, cached per sample so that a bias change costs
    // 32 additions instead of a 128 x 32 product.
    void preActivation(const ValueType* input, ValueType* pre) const
    {
        for (size_t k = 0; k < EMBEDDING; ++k)
        {
            ValueType sum(0);
            for (size_t i = 0; i < NUMBER_OF_INPUTS; ++i)
            {
                sum += weights[k][i] * input[i];
            }
            pre[k] = sum;
        }
    }
};

static inline void embed(const ValueType* pre, const ValueType* bias, ValueType* h)
{
    const ValueType zero(0);
    for (size_t k = 0; k < EMBEDDING; ++k)
    {
        const ValueType z = pre[k] + bias[k];
        h[k] = (z > zero) ? z : zero;
    }
}

// Normalization statistics from a set of samples (labels unused).
static void statsFrom(const std::vector<Sample>& data, const std::vector<size_t>& idx, FeatureStats& st)
{
    std::vector<Sample> subset;
    subset.reserve(idx.size());
    for (size_t j : idx) subset.push_back(data[j]);
    fitStats(subset, st);
}

static inline void toInputWith(const Sample& s, const FeatureStats& st, ValueType* in)
{
    for (size_t f = 0; f < NUMBER_OF_INPUTS; ++f)
    {
        in[f] = toQ((s.feat[f] - st.mean[f]) / st.stdev[f] / 3.0);
    }
}

// Accuracy of `head` (or `proto` when not null) on the samples whose cached
// pre-activations are `pre`, with backbone biases `bias`.
static double scoreHead(const HeadType& head, const PrototypeType* proto,
                        const std::vector<Vec>& pre, const std::vector<int>& labels,
                        const ValueType* bias)
{
    ValueType h[EMBEDDING];
    size_t correct = 0;
    for (size_t j = 0; j < pre.size(); ++j)
    {
        embed(pre[j].data(), bias, h);
        const size_t p = (proto != nullptr) ? proto->predict(h) : head.predict(h);
        correct += (static_cast<int>(p) == labels[j]) ? 1U : 0U;
    }
    return static_cast<double>(correct) / static_cast<double>(pre.size());
}

// Trains the backbone biases (when trainBias) and the head (when trainHead)
// on the shots for ADAPT_EPOCHS epochs of per-sample SGD.
static void adapt(HeadType& head, ValueType* bias, const std::vector<Vec>& shotPre,
                  const std::vector<int>& shotLabels, const bool trainBias, const bool trainHead,
                  std::mt19937& rng)
{
    const ValueType rate = toQ(ADAPT_LEARNING_RATE);
    const ValueType zero(0);
    ValueType h[EMBEDDING];
    ValueType logits[NUMBER_OF_OUTPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType deltas[NUMBER_OF_OUTPUTS];
    ValueType hDeltas[EMBEDDING];
    std::vector<size_t> order(shotPre.size());
    for (size_t j = 0; j < order.size(); ++j) order[j] = j;

    for (unsigned epoch = 0; epoch < ADAPT_EPOCHS; ++epoch)
    {
        for (size_t j = order.size(); j > 1; --j)
        {
            std::swap(order[j - 1], order[rng() % j]);
        }
        for (size_t j : order)
        {
            const ValueType* pre = shotPre[j].data();
            embed(pre, bias, h);
            if (trainHead)
            {
                head.trainClass(h, static_cast<size_t>(shotLabels[j]), rate, trainBias ? hDeltas : nullptr);
            }
            else
            {
                for (size_t o = 0; o < NUMBER_OF_OUTPUTS; ++o)
                {
                    target[o] = toQ(shotLabels[j] == static_cast<int>(o) ? 1.0 : 0.0);
                }
                head.forward(h, logits);
                head.outputDeltas(logits, target, deltas);
                head.backward(deltas, hDeltas);
            }
            if (trainBias)
            {
                // dL/db through the ReLU: the hidden delta where the unit is on.
                for (size_t k = 0; k < EMBEDDING; ++k)
                {
                    if ((pre[k] + bias[k]) > zero)
                    {
                        bias[k] += tinymind::multiplyRoundNearest(rate, hDeltas[k]);
                    }
                }
            }
        }
    }
}

static void runAdaptation(NeuralNetworkType& net, const std::vector<Sample>& data,
                          const std::vector<Sample>& val)
{
    Backbone backbone;
    backbone.load(net);

    ValueType shippedWeights[HeadType::NumberOfWeights];
    ValueType shippedBiases[HeadType::NumberOfBiases];
    for (size_t o = 0; o < NUMBER_OF_OUTPUTS; ++o)
    {
        for (size_t k = 0; k < EMBEDDING; ++k)
        {
            shippedWeights[o * EMBEDDING + k] = net.getHiddenLayerWeightForNeuronAndConnection(0, k, o);
        }
        shippedBiases[o] = net.getHiddenLayerBiasNeuronWeightForConnection(0, o);
    }

    // Cache every sample's input and pre-activation under the batch-1 stats.
    ValueType input[NUMBER_OF_INPUTS];
    std::vector<Vec> pre(data.size(), Vec(EMBEDDING));
    std::vector<int> labels(data.size());
    for (size_t j = 0; j < data.size(); ++j)
    {
        toInput(data[j], input);
        backbone.preActivation(input, pre[j].data());
        labels[j] = data[j].label;
    }
    std::vector<Vec> valPre(val.size(), Vec(EMBEDDING));
    std::vector<int> valLabels(val.size());
    for (size_t j = 0; j < val.size(); ++j)
    {
        toInput(val[j], input);
        backbone.preActivation(input, valPre[j].data());
        valLabels[j] = val[j].label;
    }

    // The split must reproduce the network exactly: Q16.16 sums are exact,
    // so the hidden layer and the logits agree bit for bit with the MLP's.
    {
        HeadType head;
        head.loadWeights(shippedWeights, shippedBiases);
        ValueType h[EMBEDDING];
        size_t hiddenMismatches = 0;
        size_t argmaxMismatches = 0;
        size_t tiedMismatches = 0;
        ValueType learned[NUMBER_OF_OUTPUTS];
        for (size_t j = 0; j < data.size(); ++j)
        {
            toInput(data[j], input);
            net.feedForward(input);
            embed(pre[j].data(), backbone.biases, h);
            for (size_t k = 0; k < EMBEDDING; ++k)
            {
                hiddenMismatches += (net.getLastHiddenLayer().getOutputValueForNeuron(k) != h[k]) ? 1U : 0U;
            }
            net.getLearnedValues(learned);
            size_t best = 0;
            for (size_t o = 1; o < NUMBER_OF_OUTPUTS; ++o)
            {
                if (learned[o] > learned[best]) best = o;
            }
            const size_t headBest = head.predict(h);
            if (best != headBest)
            {
                ++argmaxMismatches;
                // The sigmoid table saturates, so two confident outputs can
                // both read 1.0 and the MLP takes the first; the head
                // compares the logits underneath.
                tiedMismatches += (learned[best] == learned[headBest]) ? 1U : 0U;
            }
        }
        std::cout << "\nBackbone + DenseHead vs the MLP over " << data.size()
                  << " samples: " << hiddenMismatches << " hidden values differ; "
                  << argmaxMismatches << " predictions differ, " << tiedMismatches
                  << " of them where two sigmoid outputs tie." << std::endl;
    }

    std::vector<size_t> byBatchClass[NUM_BATCHES + 1][NUM_CLASSES];
    for (size_t j = 0; j < data.size(); ++j)
    {
        byBatchClass[data[j].batch][data[j].label].push_back(j);
    }

    std::ofstream csv("gas_adapt.csv");
    csv << "method,batch,shots,accuracy_mean,accuracy_std,batch1_accuracy_mean" << std::endl;

    double batch10[NUM_METHODS][NUM_SHOTS];

    for (int b = 2; b <= NUM_BATCHES; ++b)
    {
        // renorm_all: statistics from the whole batch, independent of the draw.
        FeatureStats allStats;
        {
            std::vector<size_t> all;
            for (int c = 0; c < NUM_CLASSES; ++c)
                all.insert(all.end(), byBatchClass[b][c].begin(), byBatchClass[b][c].end());
            statsFrom(data, all, allStats);
        }

        for (size_t si = 0; si < NUM_SHOTS; ++si)
        {
            const int shots = SHOTS[si];
            double sum[NUM_METHODS] = {0.0};
            double sumSq[NUM_METHODS] = {0.0};
            double sumB1[NUM_METHODS] = {0.0};

            for (int r = 0; r < DRAWS; ++r)
            {
                std::mt19937 rng(static_cast<uint32_t>(b * 10000 + shots * 100 + r));

                std::vector<size_t> shotIdx;
                std::vector<size_t> testIdx;
                for (int c = 0; c < NUM_CLASSES; ++c)
                {
                    std::vector<size_t> pool = byBatchClass[b][c];
                    for (int k = 0; k < shots; ++k)
                    {
                        const size_t pick = static_cast<size_t>(k) + rng() % (pool.size() - static_cast<size_t>(k));
                        std::swap(pool[static_cast<size_t>(k)], pool[pick]);
                    }
                    shotIdx.insert(shotIdx.end(), pool.begin(), pool.begin() + shots);
                    testIdx.insert(testIdx.end(), pool.begin() + shots, pool.end());
                }

                std::vector<Vec> shotPre;
                std::vector<int> shotLabels;
                for (size_t j : shotIdx) { shotPre.push_back(pre[j]); shotLabels.push_back(labels[j]); }
                std::vector<Vec> testPre;
                std::vector<int> testLabels;
                for (size_t j : testIdx) { testPre.push_back(pre[j]); testLabels.push_back(labels[j]); }

                double acc[NUM_METHODS];
                double accB1[NUM_METHODS];

                HeadType shipped;
                shipped.loadWeights(shippedWeights, shippedBiases);

                acc[METHOD_NONE] = scoreHead(shipped, nullptr, testPre, testLabels, backbone.biases);
                accB1[METHOD_NONE] = scoreHead(shipped, nullptr, valPre, valLabels, backbone.biases);

                // renorm / renorm_all: new statistics, shipped weights.
                for (int m = METHOD_RENORM; m <= METHOD_RENORM_ALL; ++m)
                {
                    FeatureStats shotStats;
                    if (m == METHOD_RENORM) statsFrom(data, shotIdx, shotStats);
                    const FeatureStats& st = (m == METHOD_RENORM) ? shotStats : allStats;
                    std::vector<Vec> rp(testIdx.size(), Vec(EMBEDDING));
                    for (size_t j = 0; j < testIdx.size(); ++j)
                    {
                        toInputWith(data[testIdx[j]], st, input);
                        backbone.preActivation(input, rp[j].data());
                    }
                    acc[m] = scoreHead(shipped, nullptr, rp, testLabels, backbone.biases);
                    std::vector<Vec> vp(val.size(), Vec(EMBEDDING));
                    for (size_t j = 0; j < val.size(); ++j)
                    {
                        toInputWith(val[j], st, input);
                        backbone.preActivation(input, vp[j].data());
                    }
                    accB1[m] = scoreHead(shipped, nullptr, vp, valLabels, backbone.biases);
                }

                // prototype: class means of the shots' embeddings.
                {
                    PrototypeType proto;
                    ValueType h[EMBEDDING];
                    for (size_t j = 0; j < shotPre.size(); ++j)
                    {
                        embed(shotPre[j].data(), backbone.biases, h);
                        proto.learn(h, static_cast<size_t>(shotLabels[j]));
                    }
                    acc[METHOD_PROTOTYPE] = scoreHead(shipped, &proto, testPre, testLabels, backbone.biases);
                    accB1[METHOD_PROTOTYPE] = scoreHead(shipped, &proto, valPre, valLabels, backbone.biases);
                }

                // head / bias / bias_head: SGD on the shots.
                for (int m = METHOD_HEAD; m <= METHOD_BIAS_HEAD; ++m)
                {
                    HeadType head;
                    head.loadWeights(shippedWeights, shippedBiases);
                    ValueType bias[EMBEDDING];
                    for (size_t k = 0; k < EMBEDDING; ++k) bias[k] = backbone.biases[k];
                    std::mt19937 order(static_cast<uint32_t>(b * 10000 + shots * 100 + r + 7));
                    adapt(head, bias, shotPre, shotLabels, m != METHOD_HEAD, m != METHOD_BIAS, order);
                    acc[m] = scoreHead(head, nullptr, testPre, testLabels, bias);
                    accB1[m] = scoreHead(head, nullptr, valPre, valLabels, bias);
                }

                for (int m = 0; m < NUM_METHODS; ++m)
                {
                    sum[m] += acc[m];
                    sumSq[m] += acc[m] * acc[m];
                    sumB1[m] += accB1[m];
                }
            }

            for (int m = 0; m < NUM_METHODS; ++m)
            {
                const double mean = sum[m] / DRAWS;
                const double var = std::max(0.0, sumSq[m] / DRAWS - mean * mean);
                csv << METHOD_NAMES[m] << "," << b << "," << shots << ","
                    << std::fixed << std::setprecision(4) << mean << ","
                    << std::sqrt(var) << "," << (sumB1[m] / DRAWS) << std::endl;
                if (b == NUM_BATCHES) batch10[m][si] = mean;
            }
        }
    }
    csv.close();

    std::cout << "\nBatch 10 accuracy after adapting on k labelled samples per class"
              << " (mean of " << DRAWS << " draws):" << std::endl;
    std::cout << "  " << std::left << std::setw(11) << "method";
    for (size_t si = 0; si < NUM_SHOTS; ++si) std::cout << std::right << std::setw(8) << ("k=" + std::to_string(SHOTS[si]));
    std::cout << std::endl;
    for (int m = 0; m < NUM_METHODS; ++m)
    {
        std::cout << "  " << std::left << std::setw(11) << METHOD_NAMES[m];
        for (size_t si = 0; si < NUM_SHOTS; ++si)
        {
            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(3) << batch10[m][si];
        }
        std::cout << std::endl;
    }
}

static const unsigned WEIGHT_INIT_SEED = 1U; // see main()

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    // Two seeds: the initial weights come from WEIGHT_INIT_SEED and everything
    // else (data, shuffles, training samples) from the seed set just below.
    // WEIGHT_INIT_SEED = 1 reproduces this example's published results: the
    // network used to be a global, constructed before main() seeded rand(), so it
    // drew from glibc's default seed of 1.
    std::srand(WEIGHT_INIT_SEED);
    static NeuralNetworkType net;

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

        net.feedForward(input);
        const ValueType err = net.calculateError(target);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(err))
        {
            net.trainNetwork(target);
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
        net.feedForward(input);
        net.getLearnedValues(learned);
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

    runAdaptation(net, data, val);

    return 0;
}
