/**
* LSTM sinusoid prediction -- floating-point vs Q16.16 fixed-point side by side.
*
* This is the floating-point counterpart to examples/lstm_sinusoid. The SAME
* LSTM (1 input -> 16 hidden -> 1 output), the SAME periodic training stream
* (sin[t] -> sin[t+1]), the SAME seed, topology, and epoch budget are trained
* twice: once with ValueType = double, once with ValueType = Q16.16 fixed-point.
* Both are then evaluated in two modes over the same horizon:
*
*   one_step  -- teacher forced: always fed the TRUE sin[t]. Measures how well
*                the dynamics were learned.
*   free_run  -- auto-regressive: fed its OWN previous prediction back. The
*                honest generation test; small one-step errors compound into
*                amplitude/phase drift.
*
* Writing both precisions to a single CSV lets the plot overlay them and show
* the quantization cost directly: where the float free-run holds phase, the
* Q16.16 free-run drifts a little more.
*
* The library ships only FixedPointTransferFunctions; the floating-point
* transfer-function bundle and the double activation specializations below are
* the same ones unit_test/nn/nn_unit_test.cpp uses to exercise the float path.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "qformat.hpp"
#include "constants.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"

#define TRAINING_EPOCHS 20000U
#define SAMPLES_PER_PERIOD 20U
#define WARMUP_STEPS SAMPLES_PER_PERIOD          // prime state with one true period
#define PREDICTION_STEPS (2U * SAMPLES_PER_PERIOD) // free-run two periods
#define RANDOM_SEED 7U
#define HIDDEN_NEURONS 16U

// ---------------------------------------------------------------------------
// Floating-point activation specializations.
//
// The fixed-point activation policies are LUT-backed and only valid for
// Q-format value types. The LSTM cell hardcodes SigmoidActivationPolicy and
// TanhActivationPolicy internally (gates + cell-state), so the double path
// needs native double specializations of both, plus Constants<double> and a
// zero-tolerance band.
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
            static const double zeroTolerance(0.004);
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
// Floating-point transfer-function bundle (mirrors the one in nn_unit_test.cpp).
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
    static ValueType initialLearningRate()      { return ValueType(0.15);}
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
// Random weight generators -- both draw from [-0.5, 0.5] so the two precisions
// start from comparable inits when seeded identically.
// ---------------------------------------------------------------------------
template<typename VT>
struct QRandomGen
{
    static VT generateRandomWeight()
    {
        const typename VT::FullWidthValueType half =
            tinymind::Constants<VT>::one().getValue() / 2;
        const typename VT::FullWidthValueType raw =
            (static_cast<typename VT::FullWidthValueType>(rand()) % (2 * half + 1)) - half;
        return VT(raw);
    }
};

template<typename VT>
struct FloatRandomGen
{
    static VT generateRandomWeight()
    {
        return static_cast<VT>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) - 0.5);
    }
};

// ---------------------------------------------------------------------------
// Q16.16 network type.
// ---------------------------------------------------------------------------
typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> QValueType;
typedef QValueType::FullWidthValueType QFullWidthValueType;

typedef tinymind::FixedPointTransferFunctions<
    QValueType,
    QRandomGen<QValueType>,
    tinymind::TanhActivationPolicy<QValueType>,
    tinymind::SigmoidActivationPolicy<QValueType>,
    1,
    tinymind::DefaultNetworkInitializer<QValueType>,
    tinymind::MeanSquaredErrorCalculator<QValueType, 1>,
    tinymind::ZeroToleranceCalculator<QValueType>,
    tinymind::GradientClipByValue<QValueType>> QTransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    QValueType, 1,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    1,
    QTransferFunctionsType> QLstmNetworkType;

// ---------------------------------------------------------------------------
// Floating-point (double) network type.
// ---------------------------------------------------------------------------
typedef double FValueType;

typedef FloatingPointTransferFunctions<
    FValueType,
    FloatRandomGen,
    tinymind::TanhActivationPolicy,
    tinymind::SigmoidActivationPolicy> FTransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    FValueType, 1,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    1,
    FTransferFunctionsType> FLstmNetworkType;

// ---------------------------------------------------------------------------
// Conversion helpers.
// ---------------------------------------------------------------------------
static inline QValueType toQ(double v)
{
    static const double scale = static_cast<double>(1ULL << QValueType::NumberOfFractionalBits);
    QValueType q;
    q.setValue(static_cast<QFullWidthValueType>(std::lround(v * scale)));
    return q;
}

static inline double fromQ(const QValueType& q)
{
    static const double scale = static_cast<double>(1ULL << QValueType::NumberOfFractionalBits);
    return static_cast<double>(q.getValue()) / scale;
}

// Sinusoid sample i, scaled to [0, 1] for the sigmoid output.
static inline double trueSample(size_t i)
{
    return (std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(SAMPLES_PER_PERIOD)) + 1.0) / 2.0;
}

// ---------------------------------------------------------------------------
// Train one network on the periodic stream, then evaluate one-step and
// free-run. toVal converts a real (double) sample into the network's value
// type; toReal converts a network output back to a double. Both result arrays
// are filled with PREDICTION_STEPS doubles.
// ---------------------------------------------------------------------------
template<typename NetType, typename VT, typename ToVal, typename ToReal>
static void trainAndEvaluate(NetType& net, ToVal toVal, ToReal toReal,
                             double* oneStep, double* freeRun)
{
    typedef typename NetType::NeuralNetworkTransferFunctionsPolicy TF;

    static const size_t NUM_SAMPLES = SAMPLES_PER_PERIOD;
    VT sinSamples[NUM_SAMPLES];
    for (size_t i = 0; i < NUM_SAMPLES; ++i)
    {
        sinSamples[i] = toVal(trueSample(i));
    }

    VT input[1];
    VT target[1];
    VT learnedValues[1];
    VT error = VT(0);

    // Continuous periodic stream: sin[t] -> sin[t+1]. Recurrent state carries
    // across epoch boundaries (sample N-1 wraps to sample 0).
    for (unsigned epoch = 0; epoch < TRAINING_EPOCHS; ++epoch)
    {
        for (size_t t = 0; t < NUM_SAMPLES; ++t)
        {
            input[0] = sinSamples[t];
            target[0] = sinSamples[(t + 1) % NUM_SAMPLES];

            net.feedForward(&input[0]);
            error = net.calculateError(&target[0]);

            if (!TF::isWithinZeroTolerance(error))
            {
                net.trainNetwork(&target[0]);
            }
        }
    }

    // --- one-step-ahead (teacher forced) ---
    net.resetState();
    for (size_t i = 0; i + 1 < WARMUP_STEPS; ++i)
    {
        input[0] = toVal(trueSample(i));
        net.feedForward(&input[0]);
    }
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = toVal(trueSample(WARMUP_STEPS - 1 + p));
        net.feedForward(&input[0]);
        net.getLearnedValues(&learnedValues[0]);
        oneStep[p] = toReal(learnedValues[0]);
    }

    // --- auto-regressive free-run ---
    net.resetState();
    for (size_t i = 0; i + 1 < WARMUP_STEPS; ++i)
    {
        input[0] = toVal(trueSample(i));
        net.feedForward(&input[0]);
    }
    VT curr = toVal(trueSample(WARMUP_STEPS - 1));
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = curr;
        net.feedForward(&input[0]);
        net.getLearnedValues(&learnedValues[0]);
        freeRun[p] = toReal(learnedValues[0]);
        curr = learnedValues[0]; // feed prediction back (auto-regressive)
    }
}

int main()
{
    double qOneStep[PREDICTION_STEPS];
    double qFreeRun[PREDICTION_STEPS];
    double fOneStep[PREDICTION_STEPS];
    double fFreeRun[PREDICTION_STEPS];

    // Seed identically before each network so both start from comparable inits.
    std::cout << "Training Q16.16 LSTM on sinusoid (" << SAMPLES_PER_PERIOD
              << " samples/period)..." << std::endl;
    srand(RANDOM_SEED);
    QLstmNetworkType qNet;
    trainAndEvaluate<QLstmNetworkType, QValueType>(
        qNet, toQ, fromQ, qOneStep, qFreeRun);

    std::cout << "Training double LSTM on sinusoid..." << std::endl;
    srand(RANDOM_SEED);
    FLstmNetworkType fNet;
    trainAndEvaluate<FLstmNetworkType, FValueType>(
        fNet,
        [](double v) { return v; },
        [](double v) { return v; },
        fOneStep, fFreeRun);

    std::ofstream results("lstm_sinusoid_float.csv");
    results << "step,true,float_one_step,float_free_run,q_one_step,q_free_run" << std::endl;
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        results << p << "," << trueSample(WARMUP_STEPS + p) << ","
                << fOneStep[p] << "," << fFreeRun[p] << ","
                << qOneStep[p] << "," << qFreeRun[p] << std::endl;
    }

    std::cout << "Prediction written to lstm_sinusoid_float.csv" << std::endl;
    return 0;
}
