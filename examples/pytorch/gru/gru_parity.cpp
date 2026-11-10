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

/**
* Loads a gru_export.py weight file into tinymind's GruNeuralNetwork and runs
* sequences read from stdin, so test_gru_parity.py can compare the result with
* PyTorch.
*
* Usage: gru_parity <weights file> <double|q16>
*   double -- the file holds decimal values (gru_weights_float.txt)
*   q16    -- the file holds raw Q16.16 integers (gru_weights_q16_16.txt)
*
* stdin: repeated records "T" followed by T lines "x0 x1". For each record the
* network state is reset, the T steps are fed forward, and the output after
* the last step is printed on its own line.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fstream>

#include "qformat.hpp"
#include "constants.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "nnproperties.hpp"

#define NUMBER_OF_INPUTS 2U
#define HIDDEN_NEURONS 4U
#define NUMBER_OF_OUTPUTS 1U

// ---------------------------------------------------------------------------
// Double-precision activation specializations (the library's activation
// policies are lookup tables for Q-format types). The GRU layer uses
// SigmoidActivationPolicy for its gates and TanhActivationPolicy for the
// candidate, so both need a double form, as in examples/lstm_sinusoid_float.
// ---------------------------------------------------------------------------
namespace tinymind {
    template<>
    struct TanhActivationPolicy<double>
    {
        static double activationFunction(const double& value)
        {
            return std::tanh(value);
        }

        static double activationFunctionDerivative(const double& value)
        {
            return (1.0 - (value * value));
        }
    };

    template<>
    struct SigmoidActivationPolicy<double>
    {
        static double activationFunction(const double& value)
        {
            return (1.0 / (1.0 + std::exp(-value)));
        }

        static double activationFunctionDerivative(const double& value)
        {
            return (value * (1.0 - value));
        }
    };

    template<>
    struct ZeroToleranceCalculator<double>
    {
        static bool isWithinZeroTolerance(const double& value)
        {
            return (0.0 == value);
        }
    };

    template<>
    struct Constants<double>
    {
        static double one()
        {
            return 1.0;
        }

        static double negativeOne()
        {
            return -1.0;
        }

        static double zero()
        {
            return 0.0;
        }
    };
}

// Inference only: weights come from the file, so no random initialisation.
template<typename ValueType>
struct NullRandomWeight
{
    static ValueType generateRandomWeight()
    {
        return ValueType(0);
    }
};

// Minimal transfer-function bundle for an inference-only double network.
struct DoubleTransferFunctions
{
    typedef double ValueType;
    typedef double TransferFunctionsValueType;
    typedef NullRandomWeight<double> RandomNumberGeneratorPolicy;
    typedef tinymind::TanhActivationPolicy<double> HiddenNeuronActivationPolicy;
    typedef tinymind::SigmoidActivationPolicy<double> OutputNeuronActivationPolicy;
    typedef tinymind::NullActivationPolicy<double> GatedNeuronActivationPolicy;
    typedef tinymind::ZeroToleranceCalculator<double> ZeroToleranceCalculatorPolicy;

    static const unsigned NumberOfTransferFunctionsOutputNeurons = NUMBER_OF_OUTPUTS;

    static ValueType calculateError(ValueType const* const targetValues, ValueType const* const outputValues)
    {
        ValueType error(0);
        for (unsigned neuron = 0; neuron < NUMBER_OF_OUTPUTS; ++neuron)
        {
            const ValueType delta = (targetValues[neuron] - outputValues[neuron]);
            error += (delta * delta);
        }
        return error;
    }

    static ValueType calculateOutputGradient(const ValueType& targetValue, const ValueType& outputValue)
    {
        return ((targetValue - outputValue) * OutputNeuronActivationPolicy::activationFunctionDerivative(outputValue));
    }

    static ValueType gateActivationFunction(const ValueType& value)
    {
        return GatedNeuronActivationPolicy::activationFunction(value);
    }

    static ValueType generateRandomWeight()
    {
        return RandomNumberGeneratorPolicy::generateRandomWeight();
    }

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

    static ValueType initialAccelerationRate()
    {
        return ValueType(0);
    }

    static ValueType initialBiasOutputValue()
    {
        return ValueType(1.0);
    }

    static ValueType initialLearningRate()
    {
        return ValueType(0);
    }

    static ValueType initialMomentumRate()
    {
        return ValueType(0);
    }

    static ValueType noOpDeltaWeight()
    {
        return ValueType(0);
    }

    static ValueType noOpWeight()
    {
        return ValueType(0);
    }

    static ValueType negate(const ValueType& value)
    {
        return -value;
    }

    static bool isWithinZeroTolerance(const ValueType& value)
    {
        return ZeroToleranceCalculatorPolicy::isWithinZeroTolerance(value);
    }
};

typedef tinymind::GruNeuralNetwork<
    double,
    NUMBER_OF_INPUTS,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    NUMBER_OF_OUTPUTS,
    DoubleTransferFunctions,
    false> DoubleGruType;

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> QValueType;

typedef tinymind::FixedPointTransferFunctions<
    QValueType,
    NullRandomWeight<QValueType>,
    tinymind::TanhActivationPolicy<QValueType>,
    tinymind::SigmoidActivationPolicy<QValueType>,
    NUMBER_OF_OUTPUTS> QTransferFunctionsType;

typedef tinymind::GruNeuralNetwork<
    QValueType,
    NUMBER_OF_INPUTS,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    NUMBER_OF_OUTPUTS,
    QTransferFunctionsType,
    false> QGruType;

static DoubleGruType doubleGru;
static QGruType qGru;

static QValueType toQ(const double value)
{
    const double scaled = std::floor((value * 65536.0) + 0.5);
    return QValueType(static_cast<QValueType::FullWidthValueType>(scaled));
}

static double fromQ(const QValueType& value)
{
    return (static_cast<double>(value.getValue()) / 65536.0);
}

template<typename NetworkType, typename ValueType>
static int runSequences(NetworkType& network, ValueType (*convertIn)(double), double (*convertOut)(const ValueType&))
{
    int steps;

    while (std::scanf("%d", &steps) == 1)
    {
        ValueType inputs[NUMBER_OF_INPUTS];
        ValueType outputs[NUMBER_OF_OUTPUTS];

        network.resetState();
        for (int step = 0; step < steps; ++step)
        {
            double x0;
            double x1;
            if (std::scanf("%lf %lf", &x0, &x1) != 2)
            {
                std::fprintf(stderr, "malformed sequence on stdin\n");
                return 3;
            }
            inputs[0] = convertIn(x0);
            inputs[1] = convertIn(x1);
            network.feedForward(&inputs[0]);
        }
        network.getLearnedValues(&outputs[0]);
        std::printf("%.12f\n", convertOut(outputs[0]));
    }

    return 0;
}

static double identityIn(double value)
{
    return value;
}

static double identityOut(const double& value)
{
    return value;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s <weights file> <double|q16>\n", argv[0]);
        return 2;
    }

    std::ifstream weightsFile(argv[1]);
    if (!weightsFile.is_open())
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    if (0 == std::strcmp(argv[2], "double"))
    {
        tinymind::RecurrentNetworkPropertiesFileManager<DoubleGruType>::template loadNetworkWeights<double, double>(doubleGru, weightsFile);
        return runSequences<DoubleGruType, double>(doubleGru, identityIn, identityOut);
    }
    else if (0 == std::strcmp(argv[2], "q16"))
    {
        tinymind::RecurrentNetworkPropertiesFileManager<QGruType>::template loadNetworkWeights<QValueType, QValueType>(qGru, weightsFile);
        return runSequences<QGruType, QValueType>(qGru, toQ, fromQ);
    }

    std::fprintf(stderr, "unknown mode %s (expected double or q16)\n", argv[2]);
    return 2;
}
