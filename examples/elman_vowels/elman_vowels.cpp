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

// Elman network speaker recognition on the UCI Japanese Vowels dataset --
// the offline-float-train / embedded-fixed-point-infer deployment story.
//
// Dataset: Japanese Vowels (UCI ML Repository, id 128)
//          https://archive.ics.uci.edu/dataset/128/japanese+vowels
//          Nine male speakers each utter the Japanese vowel /ae/; every
//          utterance is a short sequence (7..29 frames) of 12 LPC cepstrum
//          coefficients. Task: identify the speaker from the sequence.
//
// The task is genuinely temporal -- a single 12-coefficient frame is ambiguous
// between speakers, but the trajectory of the coefficients across an utterance
// is discriminative. An Elman network walks the frames one at a time, carrying
// what it has heard so far in its recurrent hidden context, and the per-frame
// class scores are summed over the utterance; the predicted speaker is argmax.
//
// Deployment flow demonstrated end to end:
//   1. Train an `ElmanNeuralNetwork<double, ...>` offline (host, floating point).
//   2. Copy the trained weights -- input, recurrent, output, biases -- into
//      inference-only (`IsTrainable = false`) fixed-point networks at Q16.16,
//      Q8.8, and Q4.4, saturating each weight to the target format's range.
//   3. Run the test split through every network and compare accuracy, looking
//      for the SMALLEST Q-format that matches the float model. The Q4.4 corner
//      stores the entire 617-weight network in 617 bytes.
//
// `ae.train` / `ae.test` are loaded from the working directory (`make run`
// cd's into ./output; the Makefile copies the data files there).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "qformat.hpp"
#include "constants.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"

static constexpr size_t NUM_FEATURES = 12;   // LPC cepstrum coefficients per frame
static constexpr size_t NUM_CLASSES = 9;     // speakers
static constexpr size_t HIDDEN_NEURONS = 16;
static constexpr unsigned TRAIN_EPOCHS = 60;
static constexpr unsigned RANDOM_SEED = 2;

// Per-feature z-score, then divide by 3 so virtually all inputs land in
// [-1, 1] -- representable even at Q4.4 resolution (1/16).
static constexpr double Z_SCALE = 3.0;

// ---------------------------------------------------------------------------
// Floating-point activation specializations (the LUT-backed fixed-point
// policies are only valid for Q-format types). Same pattern as
// lstm_sinusoid_float.
// ---------------------------------------------------------------------------
namespace tinymind {
    template<>
    struct TanhActivationPolicy<double>
    {
        static double activationFunction(const double& value) { return std::tanh(value); }
        static double activationFunctionDerivative(const double& value)
        {
            // tanh' expressed in terms of the activation output (1 - tanh^2).
            return (1.0 - (value * value));
        }
    };

    template<>
    struct SigmoidActivationPolicy<double>
    {
        static double activationFunction(const double& value) { return (1.0 / (1.0 + std::exp(-value))); }
        static double activationFunctionDerivative(const double& value)
        {
            // sigmoid' expressed in terms of the activation output.
            return (value * (1.0 - value));
        }
    };

    template<>
    struct ZeroToleranceCalculator<double>
    {
        static bool isWithinZeroTolerance(const double& value)
        {
            static const double zeroTolerance(0.0001);
            static const double negativeTolerance = (-1.0 * zeroTolerance);
            return ((0 == value) || ((value < zeroTolerance) && (value > negativeTolerance)));
        }
    };

    template<>
    struct Constants<double>
    {
        static double one() { return 1.0; }
        static double negativeOne() { return -1.0; }
        static double zero() { return 0.0; }
    };
}

// ---------------------------------------------------------------------------
// Floating-point transfer-function bundle (mirrors the one in nn_unit_test.cpp
// and lstm_sinusoid_float).
// ---------------------------------------------------------------------------
template<
        typename ValueType,
        template<typename> class TransferFunctionRandomNumberGeneratorPolicy,
        template<typename> class TransferFunctionHiddenNeuronActivationPolicy,
        template<typename> class TransferFunctionOutputNeuronActivationPolicy,
        template<typename> class TransferFunctionGatedNeuronActivationPolicy = tinymind::NullActivationPolicy,
        template<typename> class TransferFunctionZeroTolerancePolicy = tinymind::ZeroToleranceCalculator,
        unsigned NumberOfOutputNeurons = 1>
struct FloatingPointTransferFunctions
{
    typedef ValueType TransferFunctionsValueType;
    typedef TransferFunctionRandomNumberGeneratorPolicy<ValueType> RandomNumberGeneratorPolicy;
    typedef TransferFunctionHiddenNeuronActivationPolicy<ValueType> HiddenNeuronActivationPolicy;
    typedef TransferFunctionOutputNeuronActivationPolicy<ValueType> OutputNeuronActivationPolicy;
    typedef TransferFunctionGatedNeuronActivationPolicy<ValueType> GatedNeuronActivationPolicy;
    typedef TransferFunctionZeroTolerancePolicy<ValueType> ZeroToleranceCalculatorPolicy;

    static const unsigned NumberOfTransferFunctionsOutputNeurons = NumberOfOutputNeurons;

    static ValueType calculateError(ValueType const* const targetValues, ValueType const* const outputValues)
    {
        ValueType error(0);
        for (unsigned neuron = 0; neuron < NumberOfOutputNeurons; ++neuron)
        {
            const ValueType delta = (targetValues[neuron] - outputValues[neuron]);
            error += (delta * delta);
        }
        if (NumberOfOutputNeurons > 1)
        {
            error /= NumberOfOutputNeurons;
        }
        return error;
    }

    static ValueType calculateOutputGradient(const ValueType& targetValue, const ValueType& outputValue)
    {
        const ValueType delta = targetValue - outputValue;
        return (delta * OutputNeuronActivationPolicy::activationFunctionDerivative(outputValue));
    }

    static ValueType gateActivationFunction(const ValueType& value)
    {
        return GatedNeuronActivationPolicy::activationFunction(value);
    }

    static ValueType generateRandomWeight() { return RandomNumberGeneratorPolicy::generateRandomWeight(); }

    static ValueType hiddenNeuronActivationFunction(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunction(value);
    }
    static ValueType hiddenNeuronActivationFunctionDerivative(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunctionDerivative(value);
    }
    static ValueType outputNeuronActivationFunction(const ValueType& value)
    {
        return OutputNeuronActivationPolicy::activationFunction(value);
    }
    static ValueType outputNeuronActivationFunctionDerivative(const ValueType& value)
    {
        return OutputNeuronActivationPolicy::activationFunctionDerivative(value);
    }
    static ValueType neuronActivationFunction(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunction(value);
    }
    static ValueType neuronActivationFunctionDerivative(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunctionDerivative(value);
    }

    static ValueType initialAccelerationRate() { return ValueType(0.1); }
    static ValueType initialBiasOutputValue()  { return ValueType(1.0); }
    static ValueType initialDeltaWeight()       { return ValueType(0);   }
    static ValueType initialGradientValue()     { return ValueType(0);   }
    static ValueType initialLearningRate()      { return ValueType(0.01);}
    static ValueType initialMomentumRate()      { return ValueType(0.5); }
    static ValueType initialOutputValue()       { return ValueType(0);   }

    static bool isWithinZeroTolerance(const ValueType& value)
    {
        return ZeroToleranceCalculatorPolicy::isWithinZeroTolerance(value);
    }
    static ValueType negate(const ValueType& value) { return (-1.0 * value); }
    static ValueType noOpDeltaWeight() { return ValueType(1.0); }
    static ValueType noOpWeight()      { return ValueType(1.0); }
};

// ---------------------------------------------------------------------------
// Random weight generators.
// ---------------------------------------------------------------------------
template<typename VT>
struct FloatRandomGen
{
    static VT generateRandomWeight()
    {
        return static_cast<VT>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) - 0.5);
    }
};

template<typename VT>
struct QRandomGen
{
    static VT generateRandomWeight()
    {
        // Symmetric small init in [-0.5, 0.5] (raw Q-format units). The
        // inference networks overwrite every weight anyway; this only seeds
        // the constructor.
        const typename VT::FullWidthValueType half =
            tinymind::Constants<VT>::one().getValue() / 2;
        const typename VT::FullWidthValueType raw =
            (static_cast<typename VT::FullWidthValueType>(rand()) % (2 * half + 1)) - half;
        return VT(raw);
    }
};

// ---------------------------------------------------------------------------
// Network types. Float for offline training; Q-formats for inference only.
// ---------------------------------------------------------------------------
typedef FloatingPointTransferFunctions<
    double,
    FloatRandomGen,
    tinymind::TanhActivationPolicy,
    tinymind::SigmoidActivationPolicy,
    tinymind::NullActivationPolicy,
    tinymind::ZeroToleranceCalculator,
    NUM_CLASSES> FloatTransferFunctions;

typedef tinymind::ElmanNeuralNetwork<
    double, NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES,
    FloatTransferFunctions> FloatElmanNetwork;

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> Q16_16;
typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> Q8_8;
typedef tinymind::QValue<4, 4, true, tinymind::RoundUpPolicy> Q4_4;

template<typename QT>
struct QElmanTypes
{
    typedef tinymind::FixedPointTransferFunctions<
        QT,
        QRandomGen<QT>,
        tinymind::TanhActivationPolicy<QT>,
        tinymind::SigmoidActivationPolicy<QT>,
        NUM_CLASSES> TransferFunctionsType;

    // IsTrainable = false: inference-only neurons, no gradient/delta storage --
    // this is the network an MCU would instantiate.
    typedef tinymind::ElmanNeuralNetwork<
        QT, NUM_FEATURES, HIDDEN_NEURONS, NUM_CLASSES,
        TransferFunctionsType, false> NetworkType;
};

// ---------------------------------------------------------------------------
// Value conversion traits: double <-> network ValueType.
// ---------------------------------------------------------------------------
template<typename VT>
struct ValueIO
{
    static VT from(double v, size_t* clipped = nullptr)
    {
        typedef typename VT::FullWidthValueType FullWidthType;
        static const double scale = static_cast<double>(1ULL << VT::NumberOfFractionalBits);
        static const double maxRaw = static_cast<double>(std::numeric_limits<FullWidthType>::max());
        static const double minRaw = static_cast<double>(std::numeric_limits<FullWidthType>::min());

        double raw = std::round(v * scale);
        if (raw > maxRaw)
        {
            raw = maxRaw;
            if (clipped) ++(*clipped);
        }
        else if (raw < minRaw)
        {
            raw = minRaw;
            if (clipped) ++(*clipped);
        }

        VT q;
        q.setValue(static_cast<FullWidthType>(raw));
        return q;
    }

    static double to(const VT& v)
    {
        static const double scale = static_cast<double>(1ULL << VT::NumberOfFractionalBits);
        return static_cast<double>(v.getValue()) / scale;
    }
};

template<>
struct ValueIO<double>
{
    static double from(double v, size_t* clipped = nullptr)
    {
        (void)clipped;
        return v;
    }
    static double to(const double& v) { return v; }
};

// ---------------------------------------------------------------------------
// Dataset loading. ae.train / ae.test are blank-line separated blocks of
// frames; each frame line is 12 space-separated LPC cepstrum coefficients.
// Block k belongs to the speaker whose cumulative utterance count covers k.
// ---------------------------------------------------------------------------
struct Utterance
{
    std::vector<std::array<double, NUM_FEATURES> > frames;
    int label;
};

static bool loadUtterances(const std::string& path,
                           const size_t* utterancesPerSpeaker,
                           std::vector<Utterance>& out)
{
    std::ifstream file(path.c_str());
    if (!file)
    {
        std::cerr << "Cannot open " << path << " -- run via 'make run' so the data is copied next to the binary." << std::endl;
        return false;
    }

    out.clear();
    Utterance current;
    current.label = -1;
    std::string line;

    while (std::getline(file, line))
    {
        const bool blank = (line.find_first_not_of(" \t\r\n") == std::string::npos);
        if (blank)
        {
            if (!current.frames.empty())
            {
                out.push_back(current);
                current.frames.clear();
            }
            continue;
        }

        std::istringstream stream(line);
        std::array<double, NUM_FEATURES> frame;
        for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
        {
            if (!(stream >> frame[feature]))
            {
                std::cerr << "Malformed frame in " << path << std::endl;
                return false;
            }
        }
        current.frames.push_back(frame);
    }
    if (!current.frames.empty())
    {
        out.push_back(current);
    }

    size_t index = 0;
    for (size_t speaker = 0; speaker < NUM_CLASSES; ++speaker)
    {
        for (size_t utterance = 0; utterance < utterancesPerSpeaker[speaker]; ++utterance)
        {
            if (index >= out.size())
            {
                std::cerr << "Utterance count mismatch in " << path << std::endl;
                return false;
            }
            out[index++].label = static_cast<int>(speaker);
        }
    }
    if (index != out.size())
    {
        std::cerr << "Utterance count mismatch in " << path << " (" << out.size() << " blocks, " << index << " expected)" << std::endl;
        return false;
    }
    return true;
}

// Per-feature z-score (train statistics), then /Z_SCALE so inputs sit in
// roughly [-1, 1].
static void normalize(std::vector<Utterance>& train, std::vector<Utterance>& test)
{
    double mean[NUM_FEATURES] = {0};
    double var[NUM_FEATURES] = {0};
    size_t count = 0;

    for (size_t utterance = 0; utterance < train.size(); ++utterance)
    {
        for (size_t frame = 0; frame < train[utterance].frames.size(); ++frame)
        {
            for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
            {
                mean[feature] += train[utterance].frames[frame][feature];
            }
            ++count;
        }
    }
    for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
    {
        mean[feature] /= static_cast<double>(count);
    }
    for (size_t utterance = 0; utterance < train.size(); ++utterance)
    {
        for (size_t frame = 0; frame < train[utterance].frames.size(); ++frame)
        {
            for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
            {
                const double delta = train[utterance].frames[frame][feature] - mean[feature];
                var[feature] += (delta * delta);
            }
        }
    }
    for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
    {
        var[feature] = std::sqrt(var[feature] / static_cast<double>(count));
    }

    std::vector<Utterance>* splits[2] = { &train, &test };
    for (size_t split = 0; split < 2; ++split)
    {
        std::vector<Utterance>& utterances = *splits[split];
        for (size_t utterance = 0; utterance < utterances.size(); ++utterance)
        {
            for (size_t frame = 0; frame < utterances[utterance].frames.size(); ++frame)
            {
                for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
                {
                    utterances[utterance].frames[frame][feature] =
                        (utterances[utterance].frames[frame][feature] - mean[feature]) / (Z_SCALE * var[feature]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recurrent context reset. The plain Elman network exposes no resetState()
// (only the gated LSTM/GRU networks do), but the context is just the recurrent
// layer's neuron output values -- zero them between utterances so one
// speaker's trailing context never leaks into the next utterance.
// ---------------------------------------------------------------------------
template<typename NetworkType, typename VT>
static void resetContext(NetworkType& net)
{
    for (size_t neuron = 0; neuron < HIDDEN_NEURONS; ++neuron)
    {
        net.getRecurrentLayer().setOutputValueForOutgoingConnection(neuron, VT());
    }
}

// ---------------------------------------------------------------------------
// Evaluation: walk each utterance frame by frame, sum the per-frame class
// scores, argmax. Returns accuracy; optionally fills a 9x9 confusion matrix
// (row = true speaker, column = predicted).
// ---------------------------------------------------------------------------
template<typename NetworkType, typename VT>
static double evaluate(NetworkType& net,
                       const std::vector<Utterance>& utterances,
                       unsigned* confusion = nullptr)
{
    VT input[NUM_FEATURES];
    VT output[NUM_CLASSES];
    size_t correct = 0;

    for (size_t utterance = 0; utterance < utterances.size(); ++utterance)
    {
        resetContext<NetworkType, VT>(net);

        double scores[NUM_CLASSES] = {0};
        for (size_t frame = 0; frame < utterances[utterance].frames.size(); ++frame)
        {
            for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
            {
                input[feature] = ValueIO<VT>::from(utterances[utterance].frames[frame][feature]);
            }
            net.feedForward(&input[0]);
            net.getLearnedValues(&output[0]);
            for (size_t cls = 0; cls < NUM_CLASSES; ++cls)
            {
                scores[cls] += ValueIO<VT>::to(output[cls]);
            }
        }

        size_t predicted = 0;
        for (size_t cls = 1; cls < NUM_CLASSES; ++cls)
        {
            if (scores[cls] > scores[predicted])
            {
                predicted = cls;
            }
        }

        const size_t actual = static_cast<size_t>(utterances[utterance].label);
        if (predicted == actual)
        {
            ++correct;
        }
        if (confusion)
        {
            ++confusion[(actual * NUM_CLASSES) + predicted];
        }
    }

    return static_cast<double>(correct) / static_cast<double>(utterances.size());
}

// ---------------------------------------------------------------------------
// Offline float training: per-frame backprop against the utterance's one-hot
// speaker target, utterance order shuffled every epoch, context reset at every
// utterance boundary. Writes one MSE row per epoch to the loss CSV.
// ---------------------------------------------------------------------------
static void trainFloat(FloatElmanNetwork& net,
                       std::vector<Utterance>& train,
                       std::ofstream& lossCsv)
{
    double input[NUM_FEATURES];
    double target[NUM_CLASSES];

    std::vector<size_t> order(train.size());
    for (size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }

    for (unsigned epoch = 0; epoch < TRAIN_EPOCHS; ++epoch)
    {
        // Fisher-Yates with rand() -- deterministic under the fixed seed.
        for (size_t i = order.size() - 1; i > 0; --i)
        {
            const size_t j = static_cast<size_t>(rand()) % (i + 1);
            std::swap(order[i], order[j]);
        }

        double epochError = 0.0;
        size_t framesSeen = 0;

        for (size_t utteranceIndex = 0; utteranceIndex < order.size(); ++utteranceIndex)
        {
            const Utterance& utterance = train[order[utteranceIndex]];

            resetContext<FloatElmanNetwork, double>(net);

            for (size_t cls = 0; cls < NUM_CLASSES; ++cls)
            {
                target[cls] = (static_cast<int>(cls) == utterance.label) ? 1.0 : 0.0;
            }

            for (size_t frame = 0; frame < utterance.frames.size(); ++frame)
            {
                for (size_t feature = 0; feature < NUM_FEATURES; ++feature)
                {
                    input[feature] = utterance.frames[frame][feature];
                }
                net.feedForward(&input[0]);
                const double error = net.calculateError(&target[0]);
                epochError += error;
                ++framesSeen;
                if (!FloatTransferFunctions::isWithinZeroTolerance(error))
                {
                    net.trainNetwork(&target[0]);
                }
            }
        }

        const double mse = epochError / static_cast<double>(framesSeen);
        lossCsv << epoch << "," << mse << std::endl;
        if ((epoch % 5) == 0 || epoch == (TRAIN_EPOCHS - 1))
        {
            std::cout << "  epoch " << std::setw(2) << epoch << "  train MSE " << mse << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Weight transfer: float-trained network -> fixed-point inference network.
// Covers everything feedForward touches: input weights + bias, recurrent
// (context -> hidden) weights, hidden->output weights + bias. Each weight is
// saturated to the target format's representable range; returns the clip count
// so the caller can see when a format is too narrow for the trained weights.
// ---------------------------------------------------------------------------
template<typename QNetworkType, typename QT>
static size_t transferWeights(FloatElmanNetwork& source, QNetworkType& destination)
{
    size_t clipped = 0;

    for (size_t input = 0; input < NUM_FEATURES; ++input)
    {
        for (size_t hidden = 0; hidden < HIDDEN_NEURONS; ++hidden)
        {
            destination.setInputLayerWeightForNeuronAndConnection(
                input, hidden,
                ValueIO<QT>::from(source.getInputLayerWeightForNeuronAndConnection(input, hidden), &clipped));
        }
    }
    for (size_t hidden = 0; hidden < HIDDEN_NEURONS; ++hidden)
    {
        destination.setInputLayerBiasWeightForConnection(
            hidden,
            ValueIO<QT>::from(source.getInputLayerBiasNeuronWeightForConnection(hidden), &clipped));
    }

    // Recurrent context -> hidden weights. Note: NeuralNetwork::setWeights()
    // does not copy these, so the transfer is done explicitly.
    for (size_t context = 0; context < HIDDEN_NEURONS; ++context)
    {
        for (size_t hidden = 0; hidden < HIDDEN_NEURONS; ++hidden)
        {
            destination.getRecurrentLayer().setWeightForNeuronAndConnection(
                context, hidden,
                ValueIO<QT>::from(source.getRecurrentLayer().getWeightForNeuronAndConnection(context, hidden), &clipped));
        }
    }

    for (size_t hidden = 0; hidden < HIDDEN_NEURONS; ++hidden)
    {
        for (size_t output = 0; output < NUM_CLASSES; ++output)
        {
            destination.setHiddenLayerWeightForNeuronAndConnection(
                0, hidden, output,
                ValueIO<QT>::from(source.getHiddenLayerWeightForNeuronAndConnection(0, hidden, output), &clipped));
        }
    }
    for (size_t output = 0; output < NUM_CLASSES; ++output)
    {
        destination.setHiddenLayerBiasNeuronWeightForConnection(
            0, output,
            ValueIO<QT>::from(source.getHiddenLayerBiasNeuronWeightForConnection(0, output), &clipped));
    }

    return clipped;
}

static constexpr size_t NUM_WEIGHTS =
    (NUM_FEATURES * HIDDEN_NEURONS) + HIDDEN_NEURONS +      // input + bias
    (HIDDEN_NEURONS * HIDDEN_NEURONS) +                      // recurrent
    (HIDDEN_NEURONS * NUM_CLASSES) + NUM_CLASSES;            // output + bias

// ---------------------------------------------------------------------------
// One fixed-point inference corner: build the network, copy weights, evaluate.
// ---------------------------------------------------------------------------
template<typename QT>
static double runQuantizedCorner(const char* name,
                                 FloatElmanNetwork& trained,
                                 const std::vector<Utterance>& test,
                                 std::ofstream& accuracyCsv,
                                 unsigned* confusion = nullptr)
{
    typedef typename QElmanTypes<QT>::NetworkType QNetworkType;

    static QNetworkType net; // static: keeps large templates off the stack
    const size_t clipped = transferWeights<QNetworkType, QT>(trained, net);
    const double accuracy = evaluate<QNetworkType, QT>(net, test, confusion);

    const size_t weightBytes = NUM_WEIGHTS * sizeof(QT);
    std::cout << "  " << std::left << std::setw(8) << name
              << " accuracy " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%"
              << "  weights " << weightBytes << " bytes"
              << "  clipped " << clipped << "/" << NUM_WEIGHTS << std::endl;
    accuracyCsv << name << "," << (8 * sizeof(QT)) << "," << (accuracy * 100.0) << ","
                << weightBytes << "," << clipped << std::endl;

    return accuracy;
}

// ---------------------------------------------------------------------------
int main()
{
    static const size_t TRAIN_COUNTS[NUM_CLASSES] = { 30, 30, 30, 30, 30, 30, 30, 30, 30 };
    static const size_t TEST_COUNTS[NUM_CLASSES] = { 31, 35, 88, 44, 29, 24, 40, 50, 29 };

    std::vector<Utterance> train;
    std::vector<Utterance> test;
    if (!loadUtterances("ae.train", &TRAIN_COUNTS[0], train) ||
        !loadUtterances("ae.test", &TEST_COUNTS[0], test))
    {
        return 1;
    }
    std::cout << "Loaded " << train.size() << " train / " << test.size() << " test utterances." << std::endl;

    normalize(train, test);

    std::ofstream lossCsv("vowels_loss.csv");
    lossCsv << "epoch,mse" << std::endl;

    std::cout << "Training Elman network offline (double, " << TRAIN_EPOCHS << " epochs)..." << std::endl;
    srand(RANDOM_SEED);
    static FloatElmanNetwork floatNet;
    trainFloat(floatNet, train, lossCsv);

    std::ofstream accuracyCsv("vowels_accuracy.csv");
    accuracyCsv << "format,bits,accuracy,weight_bytes,clipped_weights" << std::endl;

    std::cout << "Test accuracy (370 utterances, 9 speakers):" << std::endl;

    const double floatAccuracy = evaluate<FloatElmanNetwork, double>(floatNet, test);
    std::cout << "  " << std::left << std::setw(8) << "float64"
              << " accuracy " << std::fixed << std::setprecision(2) << (floatAccuracy * 100.0) << "%"
              << "  weights " << (NUM_WEIGHTS * sizeof(double)) << " bytes" << std::endl;
    accuracyCsv << "float64," << (8 * sizeof(double)) << "," << (floatAccuracy * 100.0) << ","
                << (NUM_WEIGHTS * sizeof(double)) << ",0" << std::endl;

    unsigned confusionQ8[NUM_CLASSES * NUM_CLASSES] = {0};

    runQuantizedCorner<Q16_16>("Q16.16", floatNet, test, accuracyCsv);
    runQuantizedCorner<Q8_8>("Q8.8", floatNet, test, accuracyCsv, &confusionQ8[0]);
    runQuantizedCorner<Q4_4>("Q4.4", floatNet, test, accuracyCsv);

    // Confusion matrix for the headline embedded corner (Q8.8).
    std::ofstream confusionCsv("vowels_confusion.csv");
    confusionCsv << "true_speaker,predicted_speaker,count" << std::endl;
    for (size_t actual = 0; actual < NUM_CLASSES; ++actual)
    {
        for (size_t predicted = 0; predicted < NUM_CLASSES; ++predicted)
        {
            confusionCsv << actual << "," << predicted << "," << confusionQ8[(actual * NUM_CLASSES) + predicted] << std::endl;
        }
    }

    std::cout << "Results written to vowels_loss.csv, vowels_accuracy.csv, vowels_confusion.csv" << std::endl;
    return 0;
}
