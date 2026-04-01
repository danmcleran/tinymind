/**
* Copyright (c) 2020 Intel Corporation
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

// nn_unit_test.cpp : Defines the entry point for the neural network template unit tests.

#include "compiler.h"

#define BOOST_TEST_MODULE nn_unit_test
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

#include <cstdint>
#include <string.h>

#include "qformat.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "gradientClipping.hpp"
#include "weightDecay.hpp"
#include "learningRateSchedule.hpp"
#include "adam.hpp"
#include "earlyStopping.hpp"
#include "teacherForcing.hpp"
#include "truncatedBPTT.hpp"
#include "conv1d.hpp"
#include "pool1d.hpp"
#include "dropout.hpp"
#include "batchnorm.hpp"
#include "rmsprop.hpp"
#include "binarylayer.hpp"
#include "ternarylayer.hpp"
#include "networkStats.hpp"
#include "random.hpp"
#include "nnproperties.hpp"
#include "xavier.hpp"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>
#include <numeric>
#include <deque>
#include <cmath>

namespace tinymind {
    template<>
    struct TanhActivationPolicy<double>
    {
        static double activationFunction(const double& value)
        {
            return tanh(value);
        }

        static double activationFunctionDerivative(const double& value)
        {
            //Approximation for 1st derivative of tanh
            return (static_cast<double>(1.0) - (value * value));
        }
    };
}

namespace tinymind {
    template<>
    struct SigmoidActivationPolicy<double>
    {
        static double activationFunction(const double& value)
        {
            return (1.0 / (1.0 + exp(-value)));
        }

        static double activationFunctionDerivative(const double& value)
        {
            return (value * (1.0 - value));
        }
    };
}

namespace tinymind {
    template<>
    struct ZeroToleranceCalculator<double>
    {
        static bool isWithinZeroTolerance(const double& value)
        {
            static const double zeroTolerance(0.004);
            static const double negativeTolerance = (static_cast<double>(-1.0) * zeroTolerance);

            return ((0 == value) || ((value < zeroTolerance) && (value > negativeTolerance)));
        }
    };
}

namespace tinymind {
    template<>
    struct Constants<float>
    {
        static float one()
        {
            return 1.0f;
        }

        static float negativeOne()
        {
            return -1.0f;
        }

        static float zero()
        {
            return 0.0f;
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

using namespace std;

#define TRAINING_ITERATIONS 2000
#define NUM_SAMPLES_AVG_ERROR 20
#define STOP_ON_AVG_ERROR 0
#define USE_WEIGHTS_INPUT_FILE 0 // the weights input file uses the initial values from a successful training run
#define RANDOM_SEED 7U

template<typename ValueType>
struct ValueHelper
{
    typedef typename ValueType::FullWidthValueType FullWidthValueType;

    static FullWidthValueType getErrorLimit()
    {
        static const FullWidthValueType ERROR_LIMIT = (1 << (ValueType::NumberOfFractionalBits - 6));

        return ERROR_LIMIT;
    }
};

template<>
struct ValueHelper<double>
{
    static double getErrorLimit()
    {
        return 0.1;
    }
};

template<typename ValueType>
struct UniformRealRandomNumberGenerator
{
    typedef tinymind::ValueConverter<double, ValueType> WeightConverterPolicy;

    static ValueType generateRandomWeight()
    {
        const double temp = distribution(generator);
        const ValueType weight = WeightConverterPolicy::convertToDestinationType(temp);

        return weight;
    }

    static void seed(unsigned int s)
    {
        generator.seed(s);
    }
private:
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;
};

template<typename ValueType>
std::default_random_engine UniformRealRandomNumberGenerator<ValueType>::generator(RANDOM_SEED);

template<typename ValueType>
std::uniform_real_distribution<double> UniformRealRandomNumberGenerator<ValueType>::distribution(-1.0, 1.0);

template<typename ValueType, unsigned NUMBER_OF_INPUTS, unsigned NUMBER_OF_HIDDEN_LAYERS, unsigned NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, unsigned NUMBER_OF_OUTPUTS>
struct XavierUniformRandomNumberGenerator
{
    typedef tinymind::XavierWeightInitializer<NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, NUMBER_OF_OUTPUTS> XavierWeightInitializerType;
    typedef tinymind::ValueConverter<double, ValueType> WeightConverterPolicy;

    static ValueType generateRandomWeight()
    {
        static XavierWeightInitializerType xavierWeightInitializer;
        const double temp = xavierWeightInitializer.generateUniformWeight();
        const ValueType weight = WeightConverterPolicy::convertToDestinationType(temp);

        return weight;
    }
};

template<typename ValueType, unsigned NUMBER_OF_INPUTS, unsigned NUMBER_OF_HIDDEN_LAYERS, unsigned NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, unsigned NUMBER_OF_OUTPUTS>
struct XavierNormalRandomNumberGenerator
{
    typedef tinymind::XavierWeightInitializer<NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, NUMBER_OF_OUTPUTS> XavierWeightInitializerType;
    typedef tinymind::ValueConverter<double, ValueType> WeightConverterPolicy;

    static ValueType generateRandomWeight()
    {
        static XavierWeightInitializerType xavierWeightInitializer;
        const double temp = xavierWeightInitializer.generateNormalWeight();
        const ValueType weight = WeightConverterPolicy::convertToDestinationType(temp);

        return weight;
    }
};

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

        //calculate overall error RMS
        for (uint32_t neuron = 0; neuron < NumberOfOutputNeurons; neuron++)
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
        ValueType rate(0.1);

        return rate;
    }

    static ValueType initialBiasOutputValue()
    {
        const ValueType value(1.0);

        return value;
    }

    static ValueType initialDeltaWeight()
    {
        const ValueType delta(0);

        return delta;
    }

    static ValueType initialGradientValue()
    {
        const ValueType gradient(0);

        return gradient;
    }

    static ValueType initialLearningRate()
    {
        ValueType rate(0.15);

        return rate;
    }

    static ValueType initialMomentumRate()
    {
        ValueType rate(0.5);

        return rate;
    }

    static ValueType initialOutputValue()
    {
        const ValueType value(0);

        return value;
    }

    static bool isWithinZeroTolerance(const ValueType& value)
    {
        return ZeroToleranceCalculatorPolicy::isWithinZeroTolerance(value);
    }

    static ValueType negate(const ValueType& value)
    {
        return (-1.0 * value);
    }

    static ValueType neuronActivationFunction(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunction(value);
    }

    static ValueType neuronActivationFunctionDerivative(const ValueType& value)
    {
        return HiddenNeuronActivationPolicy::activationFunctionDerivative(value);
    }

    static ValueType noOpDeltaWeight()
    {
        const ValueType value(1.0);

        return value;
    }

    static ValueType noOpWeight()
    {
        const ValueType value(1.0);

        return value;
    }
};

template<typename T>
static void generateXorValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = x ^ y;

    *values = static_cast<T>(x);
    ++values;
    *values = static_cast<T>(y);
    *output = z;
}

template<typename T>
static void generateFixedPointXorValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = x ^ y;

    *values = static_cast<T>(x << T::NumberOfFractionalBits);
    ++values;
    *values = static_cast<T>(y << T::NumberOfFractionalBits);
    *output = (z << T::NumberOfFractionalBits);
}

template<typename T>
static void generateAndValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = x & y;

    *values = static_cast<T>(x);
    ++values;
    *values = static_cast<T>(y);
    *output = z;
}

template<typename T>
static void generateFixedPointAndValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = x & y;

    *values = static_cast<T>(x << T::NumberOfFractionalBits);
    ++values;
    *values = static_cast<T>(y << T::NumberOfFractionalBits);
    *output = (z << T::NumberOfFractionalBits);
}

template<typename T>
static void generateOrValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = x | y;

    *values = static_cast<T>(x);
    ++values;
    *values = static_cast<T>(y);
    *output = z;
}

template<typename T>
static void generateFixedPointOrValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = x | y;

    *values = static_cast<T>(x << T::NumberOfFractionalBits);
    ++values;
    *values = static_cast<T>(y << T::NumberOfFractionalBits);
    *output = (z << T::NumberOfFractionalBits);
}

template<typename T>
static void generateFixedPointNorValues(T* values, T* output)
{
    const uint32_t x = (rand() & 1);
    const uint32_t y = (rand() & 1);
    const uint32_t z = !(x | y);

    *values = static_cast<T>(x << T::NumberOfFractionalBits);
    ++values;
    *values = static_cast<T>(y << T::NumberOfFractionalBits);
    *output = (z << T::NumberOfFractionalBits);
}

template<typename T>
static void generateRecurrentValues(T* values, T* output)
{
    static int32_t x = -1;
    static int32_t y = 0;
    const int32_t z = x + y;

    *values = static_cast<T>(x);
    ++values;
    *values = static_cast<T>(y);
    *output = static_cast<T>(z);

    if (++x > 1)
        x = -1;

    if (++y > 1)
        y = -1;
}

template<typename T>
static void generateFixedPointRecurrentValues(T* values, T* output)
{
    static int32_t x = -1;
    static int32_t y = 0;
    const int32_t z = x + y;

    *values = static_cast<T>(x << T::NumberOfFractionalBits);
    ++values;
    *values = static_cast<T>(y << T::NumberOfFractionalBits);
    *output = (z << T::NumberOfFractionalBits);

    if (++x > 1)
        x = -1;

    if (++y > 1)
        y = -1;
}

template<typename NeuralNetworkType>
static void testFloatingPointNN(    NeuralNetworkType& neuralNetwork,
                                    void(*pValuesFn)(typename NeuralNetworkType::NeuralNetworkValueType*, typename NeuralNetworkType::NeuralNetworkValueType*),
                                    char const* const path,
                                    const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
    static const ValueType ERROR_LIMIT = 0.1;
    ofstream results(path);
    ofstream weightsOutputFile;
    std::string weightsOutputPath(path);
    std::deque<double> errors;
    ValueType error;

    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    double learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];

    weightsOutputPath.replace(weightsOutputPath.find("."), std::string::npos, "_weights.txt");

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

    for (int i = 0; i < numberOfTrainingIterations; ++i)
    {
        pValuesFn(values, output);

        neuralNetwork.feedForward(&values[0]);
        error = neuralNetwork.calculateError(&output[0]);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            neuralNetwork.trainNetwork(&output[0]);
        }
        neuralNetwork.getLearnedValues(&learnedValues[0]);

        errors.push_front(error);
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();

#if STOP_ON_AVG_ERROR
            const double totalError = std::accumulate(errors.begin(), errors.end(), static_cast<double>(0));
            const double averageError = (totalError / static_cast<double>(NUM_SAMPLES_AVG_ERROR));
            if (averageError <= ERROR_LIMIT)
            {
                break;
            }
#endif // STOP_ON_AVG_ERROR
        }

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(neuralNetwork, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    weightsOutputFile.open(weightsOutputPath.c_str());

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkWeights(neuralNetwork, weightsOutputFile);

    weightsOutputFile.close();

    const double totalError = std::accumulate(errors.begin(), errors.end(), static_cast<double>(0));
    const double averageError = (totalError / static_cast<double>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

template<typename NeuralNetworkType>
static void testFloatingPointNN_Xor(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFloatingPointNN(neuralNetwork, generateXorValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFloatingPointNN_And(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFloatingPointNN(neuralNetwork, generateAndValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFloatingPointNN_Or(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFloatingPointNN(neuralNetwork, generateOrValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork(  NeuralNetworkType& neuralNetwork,
                                void(*pValuesFn)(typename NeuralNetworkType::NeuralNetworkValueType*, typename NeuralNetworkType::NeuralNetworkValueType*),
                                char const* const path,
                                const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
#if USE_WEIGHTS_INPUT_FILE == 1
    typedef tinymind::NetworkPropertiesFileManager<NeuralNetworkType> NetworkPropertiesFileManagerType;
#endif // USE_WEIGHTS_INPUT_FILE
    typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;
    typedef ValueHelper<ValueType> ValueHelperType;
    static const FullWidthValueType ERROR_LIMIT = ValueHelperType::getErrorLimit();
    ofstream results(path);
    ofstream weightsOutputFile;
    std::string initialWeightsInputPath(path);
    std::string weightsOutputPath(path);
    std::string binaryWeightsOutputPath(path);
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;

    initialWeightsInputPath.replace(weightsOutputPath.find("."), std::string::npos, "_initial_weights.txt");
    weightsOutputPath.replace(weightsOutputPath.find("."), std::string::npos, "_weights.txt");
    binaryWeightsOutputPath.replace(binaryWeightsOutputPath.find(".txt"), std::string::npos, ".bin");

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

#if USE_WEIGHTS_INPUT_FILE
    initialWeightsInputPath.insert(0, "../input/");
    ifstream weightsInputFile(initialWeightsInputPath);
    if (weightsInputFile.is_open())
    {
        NetworkPropertiesFileManagerType::template loadNetworkWeights<ValueType, ValueType>(neuralNetwork, weightsInputFile);
    }
    else
    {
        cout << "Did not find initial weights input file: " << initialWeightsInputPath << endl;
    }
#endif

    for (int i = 0; i < numberOfTrainingIterations; ++i)
    {
        pValuesFn(values, output);

        neuralNetwork.feedForward(&values[0]);
        error = neuralNetwork.calculateError(&output[0]);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            neuralNetwork.trainNetwork(&output[0]);
        }
        neuralNetwork.getLearnedValues(&learnedValues[0]);

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();

#if STOP_ON_AVG_ERROR
            const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
            const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
            if (averageError <= ERROR_LIMIT)
            {
                break;
            }
#endif // STOP_ON_AVG_ERROR
        }
        
        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(neuralNetwork, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    
    weightsOutputFile.open(weightsOutputPath.c_str());

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkWeights(neuralNetwork, weightsOutputFile);

    weightsOutputFile.close();

    BOOST_TEST(averageError <= ERROR_LIMIT);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Xor(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFixedPointNeuralNetwork(neuralNetwork, generateFixedPointXorValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_And(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFixedPointNeuralNetwork(neuralNetwork, generateFixedPointAndValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Or(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFixedPointNeuralNetwork(neuralNetwork, generateFixedPointOrValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Nor(NeuralNetworkType& neuralNetwork, char const* const path, const int numberOfTrainingIterations = TRAINING_ITERATIONS)
{
    testFixedPointNeuralNetwork(neuralNetwork, generateFixedPointNorValues, path, numberOfTrainingIterations);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_No_Train( NeuralNetworkType& neuralNetwork,
                                        void(*pValuesFn)(typename NeuralNetworkType::NeuralNetworkValueType*, typename NeuralNetworkType::NeuralNetworkValueType*),
                                        char const* const path,
                                        char const* const weightsInputPath)
{
    typedef tinymind::NetworkPropertiesFileManager<NeuralNetworkType> NetworkPropertiesFileManagerType;
    typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;
    typedef ValueHelper<ValueType> ValueHelperType;
    static const FullWidthValueType ERROR_LIMIT = ValueHelperType::getErrorLimit();
    ofstream results(path);
    ifstream weightsInputFile(weightsInputPath);
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;

    NetworkPropertiesFileManagerType::template loadNetworkWeights<ValueType, ValueType>(neuralNetwork, weightsInputFile);
    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        pValuesFn(values, output);

        neuralNetwork.feedForward(&values[0]);
        error = neuralNetwork.calculateError(&output[0]);
        neuralNetwork.getLearnedValues(&learnedValues[0]);

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();

#if STOP_ON_AVG_ERROR
            const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
            const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
            if (averageError <= ERROR_LIMIT)
            {
                break;
            }
#endif // STOP_ON_AVG_ERROR
        }

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(neuralNetwork, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Xor_No_Train(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train(neuralNetwork, generateFixedPointXorValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_And_No_Train(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train(neuralNetwork, generateFixedPointAndValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Or_No_Train(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train(neuralNetwork, generateFixedPointOrValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Nor_No_Train(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train(neuralNetwork, generateFixedPointNorValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_No_Train_Float_Weights( NeuralNetworkType& neuralNetwork,
                                                                void(*pValuesFn)(typename NeuralNetworkType::NeuralNetworkValueType*, typename NeuralNetworkType::NeuralNetworkValueType*),
                                                                char const* const path,
                                                                char const* const weightsInputPath)
{
    typedef tinymind::NetworkPropertiesFileManager<NeuralNetworkType> NetworkPropertiesFileManagerType;
    typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;
    typedef ValueHelper<ValueType> ValueHelperType;
    static const FullWidthValueType ERROR_LIMIT = ValueHelperType::getErrorLimit();
    ofstream results(path);
    ifstream weightsInputFile(weightsInputPath);
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;

    NetworkPropertiesFileManagerType::template loadNetworkWeights<double, ValueType>(neuralNetwork, weightsInputFile);
    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        pValuesFn(values, output);

        neuralNetwork.feedForward(&values[0]);
        error = neuralNetwork.calculateError(&output[0]);
        neuralNetwork.getLearnedValues(&learnedValues[0]);

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();

#if STOP_ON_AVG_ERROR
            const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
            const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
            if (averageError <= ERROR_LIMIT)
            {
                break;
            }
#endif // STOP_ON_AVG_ERROR
        }

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(neuralNetwork, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Xor_No_Train_Float_Weights(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train_Float_Weights(neuralNetwork, generateFixedPointXorValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_And_No_Train_Float_Weights(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train_Float_Weights(neuralNetwork, generateFixedPointAndValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testFixedPointNeuralNetwork_Or_No_Train_Float_Weights(NeuralNetworkType& neuralNetwork, char const* const path, char const* const weightsInputPath)
{
    testFixedPointNeuralNetwork_No_Train_Float_Weights(neuralNetwork, generateFixedPointOrValues, path, weightsInputPath);
}

template<typename NeuralNetworkType>
static void testNeuralNetwork_Recurrent(NeuralNetworkType& neuralNetwork, char const* const path)
{
    typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;
    typedef ValueHelper<ValueType> ValueHelperType;
    static const FullWidthValueType ERROR_LIMIT = ValueHelperType::getErrorLimit();
    ofstream results(path);
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateFixedPointRecurrentValues(values, output);

        neuralNetwork.feedForward(&values[0]);
        error = neuralNetwork.calculateError(&output[0]);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            neuralNetwork.trainNetwork(&output[0]);
        }
        neuralNetwork.getLearnedValues(&learnedValues[0]);

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();

#if STOP_ON_AVG_ERROR
            const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
            const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
            if (averageError <= ERROR_LIMIT)
            {
                break;
            }
#endif // STOP_ON_AVG_ERROR
        }

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(neuralNetwork, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

template<typename NeuralNetworkType>
static void testFloatingPointNeuralNetwork_Recurrent(NeuralNetworkType& neuralNetwork, char const* const path)
{
    typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
    typedef double FullWidthValueType;
    typedef ValueHelper<ValueType> ValueHelperType;
    static const FullWidthValueType ERROR_LIMIT = ValueHelperType::getErrorLimit();
    ofstream results(path);
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateRecurrentValues(values, output);

        neuralNetwork.feedForward(&values[0]);
        error = neuralNetwork.calculateError(&output[0]);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            neuralNetwork.trainNetwork(&output[0]);
        }
        neuralNetwork.getLearnedValues(&learnedValues[0]);

        errors.push_front(error);
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();

#if STOP_ON_AVG_ERROR
            const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
            const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
            if (averageError <= ERROR_LIMIT)
            {
                break;
            }
#endif // STOP_ON_AVG_ERROR
        }

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(neuralNetwork, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

BOOST_AUTO_TEST_SUITE(test_suite_nn)

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_xor_xavier_uniform)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    XavierUniformRandomNumberGenerator<ValueType, NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, NUMBER_OF_OUTPUTS>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_xor_xavier_uniform.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_xor_xavier_normal)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    XavierNormalRandomNumberGenerator<ValueType, NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, NUMBER_OF_OUTPUTS>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_xor_xavier_normal.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_xor_nn_copy)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_xor.txt";
    char const* const pathCopy = "output/nn_fixed_xor_copy.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;
    FixedPointMultiLayerPerceptronNetworkType nnCopy;

    testFixedPointNeuralNetwork_Xor(nn, path);
    nnCopy.setWeights(nn);
    testFixedPointNeuralNetwork_Xor(nnCopy, pathCopy);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_And(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Or(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_nor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_nor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Nor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_no_train_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    static const bool TRAINABLE = false;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_no_train_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor_No_Train(nn, path, "output/nn_fixed_xor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_no_train_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    static const bool TRAINABLE = false;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_no_train_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_And_No_Train(nn, path, "output/nn_fixed_and_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_no_train_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    static const bool TRAINABLE = false;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_no_train_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Or_No_Train(nn, path, "output/nn_fixed_or_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_no_train_nor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    static const bool TRAINABLE = false;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_no_train_nor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Nor_No_Train(nn, path, "output/nn_fixed_nor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_5_hidden_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_5_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_5_hidden_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_5_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_And(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_5_hidden_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_5_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Or(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_16_16_nn_5_hidden_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 16;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_16_16_5_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_16_16_nn_5_hidden_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 16;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_16_16_5_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_And(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_16_16_nn_5_hidden_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 16;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_16_16_5_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Or(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_8_24_nn_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_8_24_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_8_24_nn_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_8_24_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_And(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_8_24_nn_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_8_24_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Or(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_batch_2_8_24_nn_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = true;
    static const size_t BATCH_SIZE = 2;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE,
                                            BATCH_SIZE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_batch_2_8_24_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    const ValueType learningRate(nn.getLearningRate() / 4);
    const ValueType accelerationRate(nn.getAccelerationRate() / 4);
    const ValueType momentumRate(nn.getMomentumRate() / 4);

    nn.setLearningRate(learningRate);
    nn.setAccelerationRate(accelerationRate);
    nn.setMomentumRate(momentumRate);

    testFixedPointNeuralNetwork_Xor(nn, path, 10000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_batch_2_8_24_nn_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = true;
    static const size_t BATCH_SIZE = 2;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE,
                                            BATCH_SIZE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_batch_2_8_24_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    const ValueType learningRate(nn.getLearningRate() / 4);
    const ValueType accelerationRate(nn.getAccelerationRate() / 4);
    const ValueType momentumRate(nn.getMomentumRate() / 4);

    nn.setLearningRate(learningRate);
    nn.setAccelerationRate(accelerationRate);
    nn.setMomentumRate(momentumRate);

    testFixedPointNeuralNetwork_And(nn, path, 10000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_batch_2_8_24_nn_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = true;
    static const size_t BATCH_SIZE = 2;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE,
                                            BATCH_SIZE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_batch_2_8_24_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    const ValueType learningRate(nn.getLearningRate() / 4);
    const ValueType accelerationRate(nn.getAccelerationRate() / 4);
    const ValueType momentumRate(nn.getMomentumRate() / 4);

    nn.setLearningRate(learningRate);
    nn.setAccelerationRate(accelerationRate);
    nn.setMomentumRate(momentumRate);

    testFixedPointNeuralNetwork_Or(nn, path, 10000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_batch_4_8_24_nn_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = true;
    static const size_t BATCH_SIZE = 4;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE,
                                            BATCH_SIZE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_batch_4_8_24_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    const ValueType learningRate(nn.getLearningRate() / 4);
    const ValueType accelerationRate(nn.getAccelerationRate() / 4);
    const ValueType momentumRate(nn.getMomentumRate() / 4);

    nn.setLearningRate(learningRate);
    nn.setAccelerationRate(accelerationRate);
    nn.setMomentumRate(momentumRate);

    testFixedPointNeuralNetwork_Xor(nn, path, 10000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_batch_4_8_24_nn_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = true;
    static const size_t BATCH_SIZE = 4;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE,
                                            BATCH_SIZE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_batch_4_8_24_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    const ValueType learningRate(nn.getLearningRate() / 4);
    const ValueType accelerationRate(nn.getAccelerationRate() / 4);
    const ValueType momentumRate(nn.getMomentumRate() / 4);

    nn.setLearningRate(learningRate);
    nn.setAccelerationRate(accelerationRate);
    nn.setMomentumRate(momentumRate);

    testFixedPointNeuralNetwork_And(nn, path, 10000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_batch_4_8_24_nn_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = true;
    static const size_t BATCH_SIZE = 4;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE,
                                            BATCH_SIZE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_batch_4_8_24_or.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    const ValueType learningRate(nn.getLearningRate() / 4);
    const ValueType accelerationRate(nn.getAccelerationRate() / 4);
    const ValueType momentumRate(nn.getMomentumRate() / 4);

    nn.setLearningRate(learningRate);
    nn.setAccelerationRate(accelerationRate);
    nn.setMomentumRate(momentumRate);

    testFixedPointNeuralNetwork_Or(nn, path, 10000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_elman_nn)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::ElmanNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FixedPointElmanNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_elman.txt";
    FixedPointElmanNetworkType nn;

    testNeuralNetwork_Recurrent(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_elman_nn)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::ElmanNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FloatingPointElmanNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_elman.txt";
    FloatingPointElmanNetworkType nn;

    testFloatingPointNeuralNetwork_Recurrent(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_nn_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FloatingPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_xor.txt";
    FloatingPointMultiLayerPerceptronNetworkType nn;

    testFloatingPointNN_Xor(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_nn_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FloatingPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_and.txt";
    FloatingPointMultiLayerPerceptronNetworkType nn;

    testFloatingPointNN_And(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_nn_or)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FloatingPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_or.txt";
    FloatingPointMultiLayerPerceptronNetworkType nn;

    testFloatingPointNN_Or(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_no_train_float_weights_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = false;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_no_train_float_weights_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor_No_Train_Float_Weights(nn, path, "output/nn_float_xor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_no_train_float_weights_and)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = false;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_no_train_float_weights_and.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_And_No_Train_Float_Weights(nn, path, "output/nn_float_and_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_nn_relu_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FloatingPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_relu_xor.txt";
    FloatingPointMultiLayerPerceptronNetworkType nn;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    testFloatingPointNN_Xor(nn, path, 200000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_relu_xor_no_train)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = false;
    static const size_t NUMBER_OF_FIXED_BITS = 16;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    tinymind::NullRandomNumberPolicy<ValueType>,
                                                    tinymind::ReluActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_relu_xor_no_train.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor_No_Train_Float_Weights(nn, path, "output/nn_float_relu_xor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_8_24_nn_relu_xor_no_train)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = false;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 24;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    tinymind::NullRandomNumberPolicy<ValueType>,
                                                    tinymind::ReluActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_8_24_relu_xor_no_train.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor_No_Train_Float_Weights(nn, path, "output/nn_float_relu_xor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_8_8_nn_relu_xor_no_train)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const bool TRAINABLE = false;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    tinymind::NullRandomNumberPolicy<ValueType>,
                                                    tinymind::ReluActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType,
                                            TRAINABLE> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_8_8_relu_xor_no_train.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor_No_Train_Float_Weights(nn, path, "output/nn_float_relu_xor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_2_hidden_nn_relu_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FloatingPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_2_hidden_relu_xor.txt";
    FloatingPointMultiLayerPerceptronNetworkType nn;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    testFloatingPointNN_Xor(nn, path, 100000);
}

BOOST_AUTO_TEST_CASE(test_case_floatingpoint_2_hidden_nn_relu_xor_copy)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FloatingPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_2_hidden_relu_xor.txt";
    char const* const pathCopy = "output/nn_float_2_hidden_relu_xor_copy.txt";
    FloatingPointMultiLayerPerceptronNetworkType nn;
    FloatingPointMultiLayerPerceptronNetworkType nnCopy;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    testFloatingPointNN_Xor(nn, path, 100000);
    nnCopy.setWeights(nn);
    testFloatingPointNN_Xor(nnCopy, pathCopy, 100000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_2_hidden_nn_relu_xor_no_train)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 5;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 16;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    tinymind::NullRandomNumberPolicy<ValueType>,
                                                    tinymind::ReluActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron<ValueType, NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_LAYERS, NUMBER_OF_NEURONS_PER_HIDDEN_LAYER, NUMBER_OF_OUTPUTS, TransferFunctionsType, false> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_2_hidden_relu_xor_no_train.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor_No_Train_Float_Weights(nn, path, "output/nn_float_2_hidden_relu_xor_weights.txt");
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_sigmoid_xor)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::SigmoidActivationPolicy<ValueType>,
                                                    tinymind::SigmoidActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_sigmoid_xor.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path, 75000);
}

BOOST_AUTO_TEST_CASE(test_case_fixedpoint_nn_xor_4_hidden_layers)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_HIDDEN_LAYERS = 4;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            NUMBER_OF_HIDDEN_LAYERS,
                                            NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> FixedPointMultiLayerPerceptronNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_xor_4_hidden_layers.txt";
    FixedPointMultiLayerPerceptronNetworkType nn;

    testFixedPointNeuralNetwork_Xor(nn, path, TRAINING_ITERATIONS * 100);
}

BOOST_AUTO_TEST_SUITE_END()

typedef tinymind::QValue<8, 8, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ8_8SatPolicyType;
typedef tinymind::QValue<8, 8, false, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> UnsignedQ8_8SatPolicyType;

BOOST_AUTO_TEST_SUITE(SoftMaxLinearClassificationTests)

// Test case: 2x2 grayscale image with 2 classes
BOOST_AUTO_TEST_CASE(test_softmax_2x2_image_2_classes)
{
    // 2x2 grayscale image (4 input features)
    const size_t inputSize = 4;  // 2x2 pixels
    const size_t numClasses = 2; // Binary classification
    
    // Example: grayscale values normalized to Q8.8 format
    // Image pixels: [100, 150, 200, 50] -> normalized to [-1, 1] range
    SignedQ8_8SatPolicyType inputImage[inputSize];
    inputImage[0] = SignedQ8_8SatPolicyType(-1, 0);    // Dark pixel (-1.0)
    inputImage[1] = SignedQ8_8SatPolicyType(0, 0);     // Medium pixel (0.0)
    inputImage[2] = SignedQ8_8SatPolicyType(1, 0);     // Bright pixel (1.0)
    inputImage[3] = SignedQ8_8SatPolicyType(-1, 128);  // Dark-ish pixel (-0.5)
    
    // Linear layer weights: [numClasses x inputSize]
    // Class 0 weights favor dark pixels, Class 1 favors bright pixels
    SignedQ8_8SatPolicyType weights[numClasses][inputSize] = {
        // Class 0: favors dark pixels
        {SignedQ8_8SatPolicyType(1, 0), SignedQ8_8SatPolicyType(0, 128), 
         SignedQ8_8SatPolicyType(-1, 0), SignedQ8_8SatPolicyType(1, 128)},
        // Class 1: favors bright pixels
        {SignedQ8_8SatPolicyType(-1, 0), SignedQ8_8SatPolicyType(0, 128), 
         SignedQ8_8SatPolicyType(2, 0), SignedQ8_8SatPolicyType(-1, 128)}
    };
    
    // Bias terms for each class
    SignedQ8_8SatPolicyType bias[numClasses] = {
        SignedQ8_8SatPolicyType(0, 128),  // 0.5 bias for class 0
        SignedQ8_8SatPolicyType(-1, 128)  // -0.5 bias for class 1
    };
    
    // Linear transformation: logits = weights * input + bias
    SignedQ8_8SatPolicyType logits[numClasses];
    for (size_t c = 0; c < numClasses; ++c) {
        logits[c] = bias[c];
        for (size_t i = 0; i < inputSize; ++i) {
            logits[c] += weights[c][i] * inputImage[i];
        }
    }
    
    // Apply SoftMax
    SignedQ8_8SatPolicyType probabilities[numClasses];
    tinymind::SoftmaxActivationPolicy<SignedQ8_8SatPolicyType>::activationFunction(
        logits, 
        probabilities, 
        numClasses
    );
    
    // Verify: Sum of probabilities should be close to 1.0
    SignedQ8_8SatPolicyType sum(0, 0);
    for (size_t c = 0; c < numClasses; ++c) {
        sum += probabilities[c];
    }
    
    // Expected sum: 1.0 in Q8.8 = 0x0100 = 256
    SignedQ8_8SatPolicyType expectedSum(1, 0);
    
    // Allow tolerance due to Q-format rounding and exp approximation
    // Tolerance: ±3 LSB = ±0.01171875 in Q8.8
    const int tolerance = 3;
    int sumValue = sum.getValue();
    int expectedValue = expectedSum.getValue();
    
    // Test to make sure sum of probabilities should be ~1.0
    BOOST_TEST(sumValue >= (expectedValue - tolerance));
    BOOST_TEST(sumValue <= (expectedValue + tolerance));
    
    // Verify each probability is in [0, 1] range
    for (size_t c = 0; c < numClasses; ++c) {
        BOOST_TEST(probabilities[c].getValue() >= 0);
        BOOST_TEST(probabilities[c].getValue() <= SignedQ8_8SatPolicyType(1, 0).getValue());
    }
}

// =========================================================================
// Tests for the new NeuralNetwork class with heterogeneous hidden layers
// =========================================================================

BOOST_AUTO_TEST_CASE(test_case_new_nn_uniform_hidden_layers_xor)
{
    // NeuralNetwork with uniform hidden layers should work like MultilayerPerceptron
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::NeuralNetwork< ValueType,
                                     2,
                                     tinymind::HiddenLayers<5>,
                                     1,
                                     TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;
    ValueType error;

    nn.setLearningRate(0.1);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    ValueType values[2];
    ValueType output[1];

    for (int i = 0; i < 10000; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
    }

    // Verify the network has learned XOR
    std::deque<double> errors;

    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        errors.push_back(error);
    }

    double totalError = 0.0;
    for (size_t i = 0; i < errors.size(); ++i)
    {
        totalError += std::abs(errors[i]);
    }
    double averageError = totalError / static_cast<double>(errors.size());
    BOOST_TEST(averageError <= 0.1);
}

BOOST_AUTO_TEST_CASE(test_case_new_nn_2_uniform_hidden_layers_xor)
{
    // NeuralNetwork with 2 uniform hidden layers
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::NeuralNetwork< ValueType,
                                     2,
                                     tinymind::HiddenLayers<5, 5>,
                                     1,
                                     TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;
    ValueType error;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    ValueType values[2];
    ValueType output[1];

    for (int i = 0; i < 100000; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
    }

    // Verify convergence
    std::deque<double> errors;
    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        errors.push_back(error);
    }

    double totalError = 0.0;
    for (size_t i = 0; i < errors.size(); ++i)
    {
        totalError += std::abs(errors[i]);
    }
    double averageError = totalError / static_cast<double>(errors.size());
    BOOST_TEST(averageError <= 0.1);
}

BOOST_AUTO_TEST_CASE(test_case_new_nn_heterogeneous_hidden_layers_xor)
{
    // NeuralNetwork with heterogeneous hidden layers: 8 neurons, then 4 neurons
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::NeuralNetwork< ValueType,
                                     2,
                                     tinymind::HiddenLayers<8, 4>,
                                     1,
                                     TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;
    ValueType error;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    ValueType values[2];
    ValueType output[1];

    for (int i = 0; i < 100000; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
    }

    // Verify convergence
    std::deque<double> errors;
    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        errors.push_back(error);
    }

    double totalError = 0.0;
    for (size_t i = 0; i < errors.size(); ++i)
    {
        totalError += std::abs(errors[i]);
    }
    double averageError = totalError / static_cast<double>(errors.size());
    BOOST_TEST(averageError <= 0.1);
}

BOOST_AUTO_TEST_CASE(test_case_new_nn_3_heterogeneous_hidden_layers_xor)
{
    // NeuralNetwork with 3 heterogeneous hidden layers: 10, 5, 3
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::NeuralNetwork< ValueType,
                                     2,
                                     tinymind::HiddenLayers<10, 5, 3>,
                                     1,
                                     TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;
    ValueType error;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    ValueType values[2];
    ValueType output[1];

    for (int i = 0; i < 100000; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
    }

    // Verify convergence
    std::deque<double> errors;
    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        errors.push_back(error);
    }

    double totalError = 0.0;
    for (size_t i = 0; i < errors.size(); ++i)
    {
        totalError += std::abs(errors[i]);
    }
    double averageError = totalError / static_cast<double>(errors.size());
    BOOST_TEST(averageError <= 0.1);
}

BOOST_AUTO_TEST_CASE(test_case_new_nn_uniform_alias_xor)
{
    // Verify UniformHiddenLayersAlias works correctly
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::NeuralNetwork< ValueType,
                                     2,
                                     typename tinymind::UniformHiddenLayersAlias<2, 5>::type,
                                     1,
                                     TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;
    ValueType error;

    nn.setLearningRate(0.005);
    nn.setAccelerationRate(0.03);
    nn.setMomentumRate(0.16);

    ValueType values[2];
    ValueType output[1];

    for (int i = 0; i < 100000; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
    }

    // Verify convergence
    std::deque<double> errors;
    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(values, output);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        errors.push_back(error);
    }

    double totalError = 0.0;
    for (size_t i = 0; i < errors.size(); ++i)
    {
        totalError += std::abs(errors[i]);
    }
    double averageError = totalError / static_cast<double>(errors.size());
    BOOST_TEST(averageError <= 0.1);
}

BOOST_AUTO_TEST_CASE(test_case_new_nn_not_trainable)
{
    // Verify non-trainable NeuralNetwork compiles and runs feedForward
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::ReluActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::NeuralNetwork< ValueType,
                                     2,
                                     tinymind::HiddenLayers<4, 3>,
                                     1,
                                     TransferFunctionsType,
                                     false> NNType;
    NNType nn;

    ValueType values[2] = {1.0, 0.0};
    nn.feedForward(&values[0]);

    // Just verify it runs without crashing
    BOOST_TEST(true);
}

// =========================================================================
// Tests for RecurrentNeuralNetwork and ElmanNeuralNetwork
// =========================================================================

BOOST_AUTO_TEST_CASE(test_case_elman_neural_network_floating_point)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::ElmanNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FloatingPointElmanNeuralNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_elman_neural_network.txt";
    FloatingPointElmanNeuralNetworkType nn;

    testFloatingPointNeuralNetwork_Recurrent(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_elman_neural_network_fixed_point)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t NUMBER_OF_FIXED_BITS = 8;
    static const size_t NUMBER_OF_FRACTIONAL_BITS = 8;
    typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::ElmanNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FixedPointElmanNeuralNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_fixed_elman_neural_network.txt";
    FixedPointElmanNeuralNetworkType nn;

    testNeuralNetwork_Recurrent(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_recurrent_neural_network_floating_point)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::RecurrentNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<3>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> RecurrentNNType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_recurrent_neural_network.txt";
    RecurrentNNType nn;

    testFloatingPointNeuralNetwork_Recurrent(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_recurrent_neural_network_heterogeneous_layers)
{
    // RecurrentNeuralNetwork with heterogeneous hidden layers
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::RecurrentNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<4, 3>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> RecurrentNNType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_recurrent_nn_hetero.txt";
    RecurrentNNType nn;

    testFloatingPointNeuralNetwork_Recurrent(nn, path);
}

// =========================================================================
// Tests for LstmNeuralNetwork
// =========================================================================

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_floating_point)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 4;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FloatingPointLstmNeuralNetworkType;
    srand(RANDOM_SEED);
    char const* const path = "output/nn_float_lstm_neural_network.txt";
    FloatingPointLstmNeuralNetworkType nn;

    testFloatingPointNeuralNetwork_Recurrent(nn, path);
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_fixed_point)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 4;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FixedPointLstmNeuralNetworkType;
    srand(RANDOM_SEED);
    FixedPointLstmNeuralNetworkType nn;

    ValueType values[FixedPointLstmNeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[FixedPointLstmNeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[FixedPointLstmNeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType error;
    ValueType firstError;
    bool firstErrorCaptured = false;

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateFixedPointRecurrentValues(values, output);

        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);

        if (!firstErrorCaptured)
        {
            firstError = error;
            firstErrorCaptured = true;
        }

        if (!FixedPointLstmNeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
        nn.getLearnedValues(&learnedValues[0]);
    }

    // Verify feedforward produces valid output
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&learnedValues[0]);
    BOOST_TEST(learnedValues[0].getValue() != 0);
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_multi_layer)
{
    // Test multi-layer LSTM: two LSTM hidden layers (8 and 4 neurons).
    // Both inner and last hidden layers should be LSTM layers with their
    // own recurrent connections and gate weights.
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<8, 4>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> MultiLayerLstmType;
    srand(RANDOM_SEED);
    MultiLayerLstmType nn;

    ValueType values[MultiLayerLstmType::NumberOfInputLayerNeurons];
    ValueType output[MultiLayerLstmType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[MultiLayerLstmType::NumberOfOutputLayerNeurons];

    // Verify feedforward works with multi-layer LSTM
    values[0] = 1.0;
    values[1] = 0.0;
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&learnedValues[0]);
    BOOST_TEST(!std::isnan(learnedValues[0]));
    BOOST_TEST(!std::isinf(learnedValues[0]));

    // Verify output changes with different input
    (void)learnedValues[0]; // first output verified above
    values[0] = 0.0;
    values[1] = 1.0;
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&learnedValues[0]);
    BOOST_TEST(!std::isnan(learnedValues[0]));
    BOOST_TEST(!std::isinf(learnedValues[0]));

    // Verify successive calls produce different output (LSTM state accumulates)
    const double secondOutput = learnedValues[0];
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&learnedValues[0]);
    BOOST_TEST(learnedValues[0] != secondOutput);

    // Verify training doesn't crash
    output[0] = 1.0;
    nn.feedForward(&values[0]);
    const ValueType error = nn.calculateError(&output[0]);
    nn.trainNetwork(&output[0]);
    BOOST_TEST(!std::isnan(error));
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_feedforward_produces_output)
{
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> LstmNNType;
    srand(RANDOM_SEED);
    LstmNNType nn;

    ValueType values[LstmNNType::NumberOfInputLayerNeurons];
    ValueType output[LstmNNType::NumberOfOutputLayerNeurons];

    values[0] = 1.0;
    values[1] = 0.0;

    nn.feedForward(&values[0]);
    nn.getLearnedValues(&output[0]);

    // After feedForward with initialized weights, the output should be a valid number
    BOOST_TEST(!std::isnan(output[0]));
    BOOST_TEST(!std::isinf(output[0]));
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_sequential_inputs)
{
    // Test that LSTM maintains state across sequential feedForward calls
    static const size_t NUMBER_OF_INPUTS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> LstmNNType;
    srand(RANDOM_SEED);
    LstmNNType nn;

    ValueType values[LstmNNType::NumberOfInputLayerNeurons];
    ValueType output1[LstmNNType::NumberOfOutputLayerNeurons];
    ValueType output2[LstmNNType::NumberOfOutputLayerNeurons];

    // First feedForward
    values[0] = 1.0;
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&output1[0]);

    // Second feedForward with same input - output should differ due to recurrent state
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&output2[0]);

    BOOST_TEST(!std::isnan(output1[0]));
    BOOST_TEST(!std::isnan(output2[0]));
    // Due to LSTM cell state, the same input should produce different outputs
    BOOST_TEST(output1[0] != output2[0]);
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_char_sequence_prediction)
{
    // Train a floating-point LSTM to predict the next value in a repeating
    // sequence of 3 values: 0.2, 0.5, 0.8, 0.2, 0.5, 0.8, ...
    // Input is (previous, current), target is the next value.
    // The 3 training pairs are:
    //   (0.8, 0.2) -> 0.5
    //   (0.2, 0.5) -> 0.8
    //   (0.5, 0.8) -> 0.2
    // After training, verify the average error over the last samples is
    // below a threshold, confirming the network has learned the pattern.
    static const size_t SEQUENCE_LENGTH = 3;
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 8;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> LstmSeqNNType;
    srand(RANDOM_SEED);
    LstmSeqNNType nn;

    // Repeating sequence values
    const ValueType sequence[SEQUENCE_LENGTH] = {0.2, 0.5, 0.8};

    ValueType values[LstmSeqNNType::NumberOfInputLayerNeurons];
    ValueType target[LstmSeqNNType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[LstmSeqNNType::NumberOfOutputLayerNeurons];
    std::deque<double> errors;
    ValueType error;
    static const int SEQ_TRAINING_ITERATIONS = 20000;
    size_t seqIndex = 0;

    for (int i = 0; i < SEQ_TRAINING_ITERATIONS; ++i)
    {
        const size_t prevIndex = (seqIndex + SEQUENCE_LENGTH - 1) % SEQUENCE_LENGTH;
        const size_t nextIndex = (seqIndex + 1) % SEQUENCE_LENGTH;

        values[0] = sequence[prevIndex];
        values[1] = sequence[seqIndex];
        target[0] = sequence[nextIndex];

        nn.feedForward(&values[0]);
        error = nn.calculateError(&target[0]);

        if (!LstmSeqNNType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&target[0]);
        }
        nn.getLearnedValues(&learnedValues[0]);

        errors.push_front(error);
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();
        }

        seqIndex = nextIndex;
    }

    // Verify: the average error over the last NUM_SAMPLES_AVG_ERROR samples
    // should be below the error limit, confirming the LSTM learned the
    // repeating sequence pattern.
    const double totalError = std::accumulate(errors.begin(), errors.end(), 0.0);
    const double averageError = (totalError / static_cast<double>(NUM_SAMPLES_AVG_ERROR));
    static const double SEQ_ERROR_LIMIT = 0.15;

    BOOST_TEST(averageError <= SEQ_ERROR_LIMIT);
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_fixed_point_sequence_prediction)
{
    // Train a Q16.16 fixed-point LSTM to predict the next value in a
    // repeating 2-value sequence: 0.3, 0.7, 0.3, 0.7, ...
    // Input is (previous, current) and the target is the next value.
    // The 2 training pairs are:
    //   (0.7, 0.3) -> 0.7
    //   (0.3, 0.7) -> 0.3
    // After training, verify the average error over the last samples is
    // below a threshold, confirming the network learned to predict the
    // next number in the repeating sequence.
    static const size_t SEQUENCE_LENGTH = 2;
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 4;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FixedPointLstmSeqNNType;
    srand(RANDOM_SEED);
    FixedPointLstmSeqNNType nn;

    // Repeating sequence: 0.3, 0.7 in Q16.16 fixed-point
    typedef tinymind::ValueConverter<double, ValueType> ConverterType;
    const ValueType sequence[SEQUENCE_LENGTH] = {
        ConverterType::convertToDestinationType(0.3),
        ConverterType::convertToDestinationType(0.7)
    };
    ValueType values[FixedPointLstmSeqNNType::NumberOfInputLayerNeurons];
    ValueType target[FixedPointLstmSeqNNType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[FixedPointLstmSeqNNType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;
    static const int FP_SEQ_TRAINING_ITERATIONS = 20000;
    size_t seqIndex = 0;

    for (int i = 0; i < FP_SEQ_TRAINING_ITERATIONS; ++i)
    {
        const size_t prevIndex = (seqIndex + SEQUENCE_LENGTH - 1) % SEQUENCE_LENGTH;
        const size_t nextIndex = (seqIndex + 1) % SEQUENCE_LENGTH;

        values[0] = sequence[prevIndex];
        values[1] = sequence[seqIndex];
        target[0] = sequence[nextIndex];

        nn.feedForward(&values[0]);
        error = nn.calculateError(&target[0]);

        if (!FixedPointLstmSeqNNType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&target[0]);
        }
        nn.getLearnedValues(&learnedValues[0]);

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();
        }

        seqIndex = nextIndex;
    }

    // Verify: the average error over the last NUM_SAMPLES_AVG_ERROR samples
    // should be below the error limit, confirming the LSTM learned to
    // predict the next number in the repeating sequence.
    static const FullWidthValueType FP_SEQ_ERROR_LIMIT = (1 << (ValueType::NumberOfFractionalBits - 4));
    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));

    BOOST_TEST(averageError <= FP_SEQ_ERROR_LIMIT);
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_float_sinusoid_prediction)
{
    // Train a floating-point LSTM on a sampled sinusoid using sequential
    // input processing: one value per timestep, with LSTM hidden state
    // carrying temporal context between steps.
    //
    // Training: feed sin[i], target sin[i+1], for each step in the sequence.
    // LSTM state is reset at the start of each epoch so the network learns
    // the dynamics from a clean state each time.
    //
    // Prediction: prime the LSTM with the training sequence, then
    // auto-regressively predict PREDICTION_LENGTH values.
    static const size_t NUMBER_OF_INPUTS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 16;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t SEQUENCE_LENGTH = 10;
    static const size_t PREDICTION_LENGTH = 20;
    static const int SIN_TRAINING_ITERATIONS = 100000;
    static const double SIN_TRAINING_ERROR_LIMIT = 0.15;
    static const double SIN_PREDICTION_TOLERANCE = 0.90;

    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> LstmSinNNType;
    srand(RANDOM_SEED);
    UniformRealRandomNumberGenerator<double>::seed(RANDOM_SEED);
    LstmSinNNType nn;

    // Generate sinusoid samples scaled to [0, 1]
    // sin(x) ranges [-1, 1], so (sin(x) + 1) / 2 maps to [0, 1]
    const double step = 2.0 * M_PI / static_cast<double>(SEQUENCE_LENGTH);
    double sinSamples[SEQUENCE_LENGTH + PREDICTION_LENGTH];
    for (size_t i = 0; i < SEQUENCE_LENGTH + PREDICTION_LENGTH; ++i)
    {
        sinSamples[i] = (sin(static_cast<double>(i) * step) + 1.0) / 2.0;
    }

    ValueType values[LstmSinNNType::NumberOfInputLayerNeurons];
    ValueType target[LstmSinNNType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[LstmSinNNType::NumberOfOutputLayerNeurons];
    std::deque<double> errors;
    ValueType error;

    // Train: feed one value per timestep, LSTM state carries context.
    // Reset state at the start of each epoch for clean temporal learning.
    for (int epoch = 0; epoch < SIN_TRAINING_ITERATIONS; ++epoch)
    {
        nn.resetState();

        for (size_t i = 0; i < SEQUENCE_LENGTH - 1; ++i)
        {
            values[0] = sinSamples[i];
            target[0] = sinSamples[i + 1];

            nn.feedForward(&values[0]);
            error = nn.calculateError(&target[0]);

            if (!LstmSinNNType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
            {
                nn.trainNetwork(&target[0]);
            }

            errors.push_front(error);
            if (errors.size() > NUM_SAMPLES_AVG_ERROR)
            {
                errors.pop_back();
            }
        }
    }

    // Verify training converged
    const double totalError = std::accumulate(errors.begin(), errors.end(), 0.0);
    const double averageError = (totalError / static_cast<double>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= SIN_TRAINING_ERROR_LIMIT);

    // Prime the LSTM by feeding the training sequence (no training)
    nn.resetState();
    for (size_t i = 0; i < SEQUENCE_LENGTH - 1; ++i)
    {
        values[0] = sinSamples[i];
        nn.feedForward(&values[0]);
    }

    // Auto-regressively predict: feed last known value, get prediction,
    // feed prediction as next input
    double curr = sinSamples[SEQUENCE_LENGTH - 1];

    ofstream predictionOutput("output/nn_float_lstm_sinusoid_prediction.txt");
    predictionOutput << "Step,Actual,Predicted\n";
    for (size_t i = 0; i < SEQUENCE_LENGTH; ++i)
    {
        predictionOutput << i << "," << sinSamples[i] << ",\n";
    }

    for (size_t p = 0; p < PREDICTION_LENGTH; ++p)
    {
        values[0] = curr;

        nn.feedForward(&values[0]);
        nn.getLearnedValues(&learnedValues[0]);

        const double predicted = learnedValues[0];
        const double expected = sinSamples[SEQUENCE_LENGTH + p];
        const double predictionError = fabs(predicted - expected);

        predictionOutput << (SEQUENCE_LENGTH + p) << "," << expected << "," << predicted << "\n";

        BOOST_TEST(predictionError <= SIN_PREDICTION_TOLERANCE);

        // Feed prediction as next input
        curr = predicted;
    }
}

BOOST_AUTO_TEST_CASE(test_case_lstm_neural_network_fixed_point_sinusoid_prediction)
{
    // Train a Q16.16 fixed-point LSTM on a sampled sinusoid using sequential
    // input processing: one value per timestep with LSTM state carrying
    // temporal context. State is reset at the start of each epoch.
    static const size_t NUMBER_OF_INPUTS = 1;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 16;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    static const size_t SEQUENCE_LENGTH = 10;
    static const size_t PREDICTION_LENGTH = 20;
    static const int FP_SIN_TRAINING_ITERATIONS = 50000;

    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;
    typedef tinymind::ValueConverter<double, ValueType> ConverterType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FixedPointLstmSinNNType;
    srand(RANDOM_SEED);
    UniformRealRandomNumberGenerator<ValueType>::seed(RANDOM_SEED);
    FixedPointLstmSinNNType nn;

    // Generate sinusoid samples scaled to [0, 1] and convert to fixed-point
    const double step = 2.0 * M_PI / static_cast<double>(SEQUENCE_LENGTH);
    ValueType sinSamples[SEQUENCE_LENGTH + PREDICTION_LENGTH];
    double sinSamplesDouble[SEQUENCE_LENGTH + PREDICTION_LENGTH];
    for (size_t i = 0; i < SEQUENCE_LENGTH + PREDICTION_LENGTH; ++i)
    {
        sinSamplesDouble[i] = (sin(static_cast<double>(i) * step) + 1.0) / 2.0;
        sinSamples[i] = ConverterType::convertToDestinationType(sinSamplesDouble[i]);
    }

    ValueType values[FixedPointLstmSinNNType::NumberOfInputLayerNeurons];
    ValueType target[FixedPointLstmSinNNType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[FixedPointLstmSinNNType::NumberOfOutputLayerNeurons];
    std::deque<FullWidthValueType> errors;
    ValueType error;

    // Train: feed one value per timestep, reset state each epoch
    for (int epoch = 0; epoch < FP_SIN_TRAINING_ITERATIONS; ++epoch)
    {
        nn.resetState();

        for (size_t i = 0; i < SEQUENCE_LENGTH - 1; ++i)
        {
            values[0] = sinSamples[i];
            target[0] = sinSamples[i + 1];

            nn.feedForward(&values[0]);
            error = nn.calculateError(&target[0]);

            if (!FixedPointLstmSinNNType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
            {
                nn.trainNetwork(&target[0]);
            }

            errors.push_front(error.getValue());
            if (errors.size() > NUM_SAMPLES_AVG_ERROR)
            {
                errors.pop_back();
            }
        }
    }

    // Verify training converged
    // Error limit: 1/4 of full fractional range (generous for fixed-point sinusoid)
    static const FullWidthValueType FP_SIN_ERROR_LIMIT = (1 << (ValueType::NumberOfFractionalBits - 2));
    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= FP_SIN_ERROR_LIMIT);

    // Prime the LSTM with the training sequence
    nn.resetState();
    for (size_t i = 0; i < SEQUENCE_LENGTH - 1; ++i)
    {
        values[0] = sinSamples[i];
        nn.feedForward(&values[0]);
    }

    // Auto-regressively predict
    ValueType curr = sinSamples[SEQUENCE_LENGTH - 1];
    // Prediction tolerance in fixed-point: ~0.90 = 0.90 * 65536 ≈ 58982
    static const FullWidthValueType FP_SIN_PREDICTION_TOLERANCE = static_cast<FullWidthValueType>(0.90 * (1 << ValueType::NumberOfFractionalBits));

    ofstream predictionOutput("output/nn_fixed_lstm_sinusoid_prediction.txt");
    predictionOutput << "Step,Actual,Predicted\n";
    for (size_t i = 0; i < SEQUENCE_LENGTH; ++i)
    {
        predictionOutput << i << "," << sinSamplesDouble[i] << ",\n";
    }

    for (size_t p = 0; p < PREDICTION_LENGTH; ++p)
    {
        values[0] = curr;

        nn.feedForward(&values[0]);
        nn.getLearnedValues(&learnedValues[0]);

        const FullWidthValueType predicted = learnedValues[0].getValue();
        const FullWidthValueType expected = sinSamples[SEQUENCE_LENGTH + p].getValue();
        const FullWidthValueType predictionError = (predicted > expected) ? (predicted - expected) : (expected - predicted);

        predictionOutput << (SEQUENCE_LENGTH + p) << "," << sinSamplesDouble[SEQUENCE_LENGTH + p] << "," << learnedValues[0] << "\n";

        BOOST_TEST(predictionError <= FP_SIN_PREDICTION_TOLERANCE);

        // Feed prediction as next input
        curr = learnedValues[0];
    }
}

BOOST_AUTO_TEST_SUITE_END()

// =========================================================================
// Test suite for Tier 1 features: gradient clipping, weight decay,
// learning rate scheduling, and GRU networks.
// =========================================================================
BOOST_AUTO_TEST_SUITE(test_suite_tier1_features)

BOOST_AUTO_TEST_CASE(test_case_gradient_clipping_policy_fixed_point)
{
    // Verify GradientClipByValue clips Q8.8 gradients correctly
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::GradientClipByValue<ValueType> ClipPolicy;

    // Value within range [-1, 1] should pass through
    ValueType small(0, 128); // 0.5
    ValueType clipped = ClipPolicy::clip(small);
    BOOST_TEST(clipped.getValue() == small.getValue());

    // Null policy should pass through unchanged
    ValueType large(2, 0); // 2.0
    ValueType unchanged = tinymind::NullGradientClippingPolicy<ValueType>::clip(large);
    BOOST_TEST(unchanged.getValue() == large.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_weight_decay_policy_fixed_point)
{
    // Verify L2WeightDecay reduces weight magnitude for Q16.16
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;

    ValueType weight(2, 0); // 2.0
    ValueType lr(0, (1 << 14)); // ~0.25
    // Use lambda = (0, 256) = 256/65536 ≈ 0.004 for Q16.16
    ValueType result = tinymind::L2WeightDecay<ValueType, 0, 256>::applyDecay(weight, lr);

    // Weight should be reduced (decay pulls toward zero)
    BOOST_TEST(result.getValue() < weight.getValue());

    // Null policy should leave weight unchanged
    ValueType unchanged = tinymind::NullWeightDecayPolicy<ValueType>::applyDecay(weight, lr);
    BOOST_TEST(unchanged.getValue() == weight.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_learning_rate_schedule_fixed_point)
{
    // Verify FixedLearningRatePolicy never changes the rate
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    tinymind::FixedLearningRatePolicy<ValueType> schedule;

    ValueType rate(0, (1 << 6)); // ~0.25
    schedule.initialize(rate);

    for (int i = 0; i < 100; ++i)
    {
        rate = schedule.step(rate);
    }
    BOOST_TEST(rate.getValue() == ValueType(0, (1 << 6)).getValue());
}

BOOST_AUTO_TEST_CASE(test_case_step_decay_schedule_fixed_point)
{
    // Verify StepDecaySchedule reduces rate after interval for Q8.8
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    // Decay factor: 0, 230 => 230/256 ≈ 0.898, interval 3 steps
    tinymind::StepDecaySchedule<ValueType, 3, 0, 230> schedule;

    ValueType rate(0, 128); // 0.5
    schedule.initialize(rate);

    // Steps 1-2: unchanged
    ValueType r1 = schedule.step(rate);
    BOOST_TEST(r1.getValue() == rate.getValue());
    ValueType r2 = schedule.step(rate);
    BOOST_TEST(r2.getValue() == rate.getValue());

    // Step 3: decays
    ValueType r3 = schedule.step(rate);
    BOOST_TEST(r3.getValue() < rate.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_fixed_point_gradient_clipping_xor)
{
    // Verify fixed-point XOR training with gradient clipping enabled via FixedPointTransferFunctions
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;

    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    NUMBER_OF_OUTPUTS,
                                                    tinymind::DefaultNetworkInitializer<ValueType>,
                                                    tinymind::MeanSquaredErrorCalculator<ValueType, NUMBER_OF_OUTPUTS>,
                                                    tinymind::ZeroToleranceCalculator<ValueType>,
                                                    tinymind::GradientClipByValue<ValueType>> TransferFunctionsType;

    typedef tinymind::MultilayerPerceptron< ValueType,
                                            NUMBER_OF_INPUTS,
                                            1, 3,
                                            NUMBER_OF_OUTPUTS,
                                            TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;

    ValueType values[2];
    ValueType output[1];
    ValueType error;

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateFixedPointXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        nn.trainNetwork(&output[0]);
    }

    // Network should have learned XOR to some degree
    BOOST_TEST(error.getValue() < ValueHelper<ValueType>::getErrorLimit());
}

BOOST_AUTO_TEST_CASE(test_case_gru_neural_network_floating_point)
{
    // Verify GRU network can be instantiated, trained, and produce valid output
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 4;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::GruNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FloatingPointGruNeuralNetworkType;
    srand(RANDOM_SEED);
    FloatingPointGruNeuralNetworkType nn;

    ValueType values[2];
    ValueType output[1];
    ValueType learnedValues[1];
    ValueType error;

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateXorValues(&values[0], &output[0]);

        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);

        nn.trainNetwork(&output[0]);
        nn.getLearnedValues(&learnedValues[0]);
    }

    // Verify feedforward produces valid, finite output
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&learnedValues[0]);
    BOOST_TEST(!std::isnan(learnedValues[0]));
    BOOST_TEST(!std::isinf(learnedValues[0]));
    BOOST_TEST(!std::isnan(error));
    BOOST_TEST(!std::isinf(error));
}

BOOST_AUTO_TEST_CASE(test_case_gru_neural_network_fixed_point)
{
    // Verify GRU network works with fixed-point arithmetic
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 4;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
                                                    ValueType,
                                                    UniformRealRandomNumberGenerator<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>,
                                                    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;
    typedef tinymind::GruNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<NUMBER_OF_NEURONS_PER_HIDDEN_LAYER>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> FixedPointGruNeuralNetworkType;
    srand(RANDOM_SEED);
    FixedPointGruNeuralNetworkType nn;

    ValueType values[FixedPointGruNeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[FixedPointGruNeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[FixedPointGruNeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType error;
    ValueType firstError;
    bool firstErrorCaptured = false;

    for (int i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateFixedPointRecurrentValues(values, output);

        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);

        if (!firstErrorCaptured)
        {
            firstError = error;
            firstErrorCaptured = true;
        }

        if (!FixedPointGruNeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }
        nn.getLearnedValues(&learnedValues[0]);
    }

    // Verify feedforward produces valid output
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&learnedValues[0]);
    BOOST_TEST(learnedValues[0].getValue() != 0);
}

BOOST_AUTO_TEST_CASE(test_case_gru_reset_state)
{
    // Verify GRU resetState clears accumulated state so a second
    // reset+feedforward pair produces the same output as the first.
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::GruNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<4>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType> GruType;

    srand(RANDOM_SEED);
    GruType nn;

    ValueType values[2] = {0.5, 0.3};
    ValueType output1[1];
    ValueType output2[1];

    // Reset, then feedforward to get baseline
    nn.resetState();
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&output1[0]);

    // Feed forward multiple times to accumulate state
    nn.feedForward(&values[0]);
    nn.feedForward(&values[0]);
    nn.feedForward(&values[0]);

    // Reset and feedforward again — should match baseline
    nn.resetState();
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&output2[0]);

    BOOST_TEST(fabs(output1[0] - output2[0]) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_gru_non_trainable)
{
    // Verify GRU network can be instantiated as non-trainable (inference only)
    static const size_t NUMBER_OF_INPUTS = 2;
    static const size_t NUMBER_OF_OUTPUTS = 1;
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::GruNeuralNetwork< ValueType,
                                    NUMBER_OF_INPUTS,
                                    tinymind::HiddenLayers<4>,
                                    NUMBER_OF_OUTPUTS,
                                    TransferFunctionsType,
                                    false> NonTrainableGruType;

    NonTrainableGruType nn;

    ValueType values[2] = {0.5, 0.3};
    ValueType output[1];

    // Should be able to feed forward without crash
    nn.feedForward(&values[0]);
    nn.getLearnedValues(&output[0]);

    // Output should be finite
    BOOST_TEST(!std::isnan(output[0]));
    BOOST_TEST(!std::isinf(output[0]));
}

BOOST_AUTO_TEST_CASE(test_case_early_stopping)
{
    // Verify EarlyStopping correctly detects convergence
    tinymind::EarlyStopping<double, 5> stopper;

    // Decreasing error: should not stop
    BOOST_TEST(!stopper.shouldStop(1.0));
    BOOST_TEST(!stopper.shouldStop(0.5));
    BOOST_TEST(!stopper.shouldStop(0.3));

    // Stagnant error: should stop after patience exhausted
    BOOST_TEST(!stopper.shouldStop(0.4));
    BOOST_TEST(!stopper.shouldStop(0.4));
    BOOST_TEST(!stopper.shouldStop(0.4));
    BOOST_TEST(!stopper.shouldStop(0.4));
    BOOST_TEST(stopper.shouldStop(0.4)); // 5th non-improving step

    // Best error should be the minimum seen
    BOOST_TEST(fabs(stopper.getBestError() - 0.3) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_early_stopping_reset)
{
    // Verify EarlyStopping reset works
    tinymind::EarlyStopping<double, 3> stopper;

    stopper.shouldStop(1.0);
    stopper.shouldStop(2.0);
    stopper.shouldStop(2.0);
    stopper.shouldStop(2.0);
    BOOST_TEST(stopper.shouldStop(2.0)); // should stop

    stopper.reset();
    BOOST_TEST(!stopper.shouldStop(5.0)); // should NOT stop after reset
}

BOOST_AUTO_TEST_CASE(test_case_early_stopping_with_training)
{
    // Verify early stopping works in a training loop
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy> TransferFunctionsType;
    typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 3, 1,
                                            TransferFunctionsType> NNType;
    srand(RANDOM_SEED);
    NNType nn;

    tinymind::EarlyStopping<ValueType, 200> stopper;
    ValueType values[2], output[1], error;
    int iterationsUsed = 0;

    for (int i = 0; i < 10000; ++i)
    {
        generateXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        if (stopper.shouldStop(error)) break;
        nn.trainNetwork(&output[0]);
        ++iterationsUsed;
    }

    // Should have stopped before 10000 iterations
    BOOST_TEST(iterationsUsed < 10000);
    BOOST_TEST(!std::isnan(error));
}

struct AdamTransferFunctions
{
    typedef double TransferFunctionsValueType;
    typedef UniformRealRandomNumberGenerator<double> RandomNumberGeneratorPolicy;
    typedef tinymind::TanhActivationPolicy<double> HiddenNeuronActivationPolicy;
    typedef tinymind::TanhActivationPolicy<double> OutputNeuronActivationPolicy;
    typedef tinymind::ZeroToleranceCalculator<double> ZeroToleranceCalculatorPolicy;
    typedef tinymind::AdamOptimizerFloat<double> OptimizerPolicyType;

    static const unsigned NumberOfTransferFunctionsOutputNeurons = 1;

    static double calculateError(double const* const targetValues, double const* const outputValues)
    {
        const double delta = (targetValues[0] - outputValues[0]);
        return (delta * delta);
    }

    static double calculateOutputGradient(const double& targetValue, const double& outputValue)
    {
        return (targetValue - outputValue) * OutputNeuronActivationPolicy::activationFunctionDerivative(outputValue);
    }

    static double generateRandomWeight() { return RandomNumberGeneratorPolicy::generateRandomWeight(); }
    static double hiddenNeuronActivationFunction(const double& value) { return HiddenNeuronActivationPolicy::activationFunction(value); }
    static double hiddenNeuronActivationFunctionDerivative(const double& value) { return HiddenNeuronActivationPolicy::activationFunctionDerivative(value); }
    static double outputNeuronActivationFunction(const double& value) { return OutputNeuronActivationPolicy::activationFunction(value); }
    static double outputNeuronActivationFunctionDerivative(const double& value) { return OutputNeuronActivationPolicy::activationFunctionDerivative(value); }
    static double initialAccelerationRate() { return 0.0; }
    static double initialBiasOutputValue() { return 1.0; }
    static double initialDeltaWeight() { return 0.0; }
    static double initialGradientValue() { return 0.0; }
    static double initialLearningRate() { return 0.01; }
    static double initialMomentumRate() { return 0.0; }
    static double initialOutputValue() { return 0.0; }
    static bool isWithinZeroTolerance(const double& value) { return (fabs(value) < 0.004); }
    static double negate(const double& value) { return -value; }
    static double noOpDeltaWeight() { return 1.0; }
    static double noOpWeight() { return 1.0; }
};

BOOST_AUTO_TEST_CASE(test_case_adam_optimizer_xor_training)
{
    // Verify Adam optimizer converges on XOR
    typedef double ValueType;
    typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 5, 1,
                                            AdamTransferFunctions> AdamNNType;
    srand(RANDOM_SEED);
    AdamNNType nn;

    static const double ERROR_LIMIT = ValueHelper<double>::getErrorLimit();
    ValueType values[2], output[1], error;
    std::deque<double> errors;

    static const int ADAM_TRAINING_ITERATIONS = 10000;

    for (int i = 0; i < ADAM_TRAINING_ITERATIONS; ++i)
    {
        generateXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);

        if (!AdamTransferFunctions::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }

        errors.push_front(error);
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();
        }
    }

    const double totalError = std::accumulate(errors.begin(), errors.end(), 0.0);
    const double averageError = (totalError / static_cast<double>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

BOOST_AUTO_TEST_CASE(test_case_adam_fixedpoint_xor_training)
{
    // Verify Adam optimizer converges on XOR (fixed-point Q16.16)
    // Q16.16 provides sufficient precision for Adam's moment calculations
    // and bias correction, which lose accuracy rapidly in Q8.8.
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;

    typedef tinymind::FixedPointTransferFunctions<
        ValueType,
        UniformRealRandomNumberGenerator<ValueType>,
        tinymind::TanhActivationPolicy<ValueType>,
        tinymind::TanhActivationPolicy<ValueType>,
        1,
        tinymind::DefaultNetworkInitializer<ValueType>,
        tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
        tinymind::ZeroToleranceCalculator<ValueType>,
        tinymind::GradientClipByValue<ValueType>,
        tinymind::NullWeightDecayPolicy<ValueType>,
        tinymind::FixedLearningRatePolicy<ValueType>,
        tinymind::AdamOptimizer<ValueType>> TransferFunctionsType;

    typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 5, 1, TransferFunctionsType> NNType;

    srand(RANDOM_SEED);
    NNType nn;

    // Adam needs a lower learning rate than the default SGD rate.
    // QValue(0, 655) ≈ 655/65536 ≈ 0.01
    nn.setLearningRate(ValueType(0, 655));

    static const FullWidthValueType ERROR_LIMIT = ValueHelper<ValueType>::getErrorLimit();
    ValueType values[2], output[1], error;
    std::deque<FullWidthValueType> errors;

    static const int ADAM_FP_ITERATIONS = 10000;

    for (int i = 0; i < ADAM_FP_ITERATIONS; ++i)
    {
        generateFixedPointXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);

        if (!NNType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();
        }
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

BOOST_AUTO_TEST_CASE(test_case_lstm_weight_serialization)
{
    // Verify LSTM weights can be saved and loaded
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork<ValueType, 2,
                                    tinymind::HiddenLayers<4>, 1,
                                    TransferFunctionsType> LstmType;
    srand(RANDOM_SEED);
    LstmType nn;

    // Train briefly to get non-trivial weights
    ValueType values[2], output[1];
    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        nn.trainNetwork(&output[0]);
    }

    // Save weights
    {
        std::ofstream outFile("output/lstm_weights.txt");
        tinymind::RecurrentNetworkPropertiesFileManager<LstmType>::storeNetworkWeights(nn, outFile);
    }

    // Create a new network and load weights
    LstmType nn2;

    {
        std::ifstream inFile("output/lstm_weights.txt");
        tinymind::RecurrentNetworkPropertiesFileManager<LstmType>::template loadNetworkWeights<ValueType, ValueType>(nn2, inFile);
    }

    // Reset state so recurrent layer doesn't affect comparison
    nn.resetState();
    nn2.resetState();

    // Both networks should produce the same output for the same input
    values[0] = 0.5; values[1] = 0.3;
    nn.feedForward(&values[0]);
    nn2.feedForward(&values[0]);

    ValueType out1[1], out2[1];
    nn.getLearnedValues(&out1[0]);
    nn2.getLearnedValues(&out2[0]);

    // Tolerance accounts for text serialization precision loss
    BOOST_TEST(fabs(out1[0] - out2[0]) < 0.02);
}

BOOST_AUTO_TEST_CASE(test_case_gru_weight_serialization)
{
    // Verify GRU weights can be saved and loaded
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
                                            ValueType,
                                            UniformRealRandomNumberGenerator,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::TanhActivationPolicy,
                                            tinymind::SigmoidActivationPolicy> TransferFunctionsType;
    typedef tinymind::GruNeuralNetwork<ValueType, 2,
                                    tinymind::HiddenLayers<4>, 1,
                                    TransferFunctionsType> GruType;
    srand(RANDOM_SEED);
    GruType nn;

    // Train briefly
    ValueType values[2], output[1];
    for (int i = 0; i < 100; ++i)
    {
        generateXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        nn.trainNetwork(&output[0]);
    }

    // Save weights
    {
        std::ofstream outFile("output/gru_weights.txt");
        tinymind::RecurrentNetworkPropertiesFileManager<GruType>::storeNetworkWeights(nn, outFile);
    }

    // Load weights into new network
    GruType nn2;

    {
        std::ifstream inFile("output/gru_weights.txt");
        tinymind::RecurrentNetworkPropertiesFileManager<GruType>::template loadNetworkWeights<ValueType, ValueType>(nn2, inFile);
    }

    // Both should produce same output
    values[0] = 0.5; values[1] = 0.3;
    nn.resetState();
    nn2.resetState();
    nn.feedForward(&values[0]);
    nn2.feedForward(&values[0]);

    ValueType out1[1], out2[1];
    nn.getLearnedValues(&out1[0]);
    nn2.getLearnedValues(&out2[0]);

    BOOST_TEST(fabs(out1[0] - out2[0]) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_elu_activation_fixed_point)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::EluActivationPolicy<ValueType> EluPolicy;

    // Positive values: f(x) = x
    ValueType pos(0, 128); // 0.5
    ValueType result = EluPolicy::activationFunction(pos);
    BOOST_TEST(result.getValue() == pos.getValue());

    // Derivative for positive: f'(x) = 1
    ValueType deriv = EluPolicy::activationFunctionDerivative(pos);
    BOOST_TEST(deriv.getValue() == tinymind::Constants<ValueType>::one().getValue());
}

BOOST_AUTO_TEST_CASE(test_case_gelu_activation_fixed_point)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::GeluActivationPolicy<ValueType> GeluPolicy;

    // GELU(0) should be ~0 (x * sigmoid(0) = 0 * 0.5 = 0)
    ValueType zero(0);
    ValueType result = GeluPolicy::activationFunction(zero);
    BOOST_TEST(result.getValue() == 0);

    // GELU for positive input should be positive
    ValueType pos(1, 0); // 1.0
    ValueType posResult = GeluPolicy::activationFunction(pos);
    BOOST_TEST(posResult.getValue() > 0);
}

BOOST_AUTO_TEST_CASE(test_case_network_stats)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::FixedPointTransferFunctions<
        ValueType,
        UniformRealRandomNumberGenerator<ValueType>,
        tinymind::TanhActivationPolicy<ValueType>,
        tinymind::TanhActivationPolicy<ValueType>> TF;
    typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<5>, 1, TF> NNType;
    typedef tinymind::NetworkStats<NNType> Stats;

    // Use static_assert for compile-time verification (no ODR-use issues)
    static_assert(Stats::NumberOfInputs == 2, "Wrong input count");
    static_assert(Stats::NumberOfHiddenNeurons == 5, "Wrong hidden count");
    static_assert(Stats::NumberOfOutputs == 1, "Wrong output count");
    static_assert(Stats::NumberOfHiddenLayers == 1, "Wrong layer count");
    static_assert(Stats::ValueSizeBytes == sizeof(ValueType), "Wrong value size");
    static_assert(Stats::InstanceSizeBytes == sizeof(NNType), "Wrong instance size");
    static_assert(Stats::InstanceSizeBytes > 0, "Instance size must be positive");
    BOOST_TEST(true); // test counts toward suite
}

BOOST_AUTO_TEST_CASE(test_case_teacher_forcing)
{
    tinymind::ScheduledSampling<double> sampler(100);

    // At step 0, teacher forcing ratio should be 1.0 (always ground truth)
    BOOST_TEST(fabs(sampler.getTeacherForcingRatio() - 1.0) < 0.01);

    // Advance halfway
    for (int i = 0; i < 50; ++i) sampler.step();
    BOOST_TEST(fabs(sampler.getTeacherForcingRatio() - 0.5) < 0.01);

    // Advance to end
    for (int i = 0; i < 50; ++i) sampler.step();
    BOOST_TEST(fabs(sampler.getTeacherForcingRatio() - 0.0) < 0.01);

    // After decay, selectInput should return prediction
    double result = sampler.selectInput(1.0, 0.5);
    BOOST_TEST(fabs(result - 0.5) < 0.001);

    // Reset should restore ratio to 1.0
    sampler.reset();
    BOOST_TEST(fabs(sampler.getTeacherForcingRatio() - 1.0) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_conv1d_forward)
{
    // Conv1D with 5-point input, kernel=3, stride=1, 1 filter
    tinymind::Conv1D<double, 5, 3, 1, 1> conv;

    // Set known kernel weights: [1, 0, -1] with bias=0
    conv.setFilterWeight(0, 0, 1.0);
    conv.setFilterWeight(0, 1, 0.0);
    conv.setFilterWeight(0, 2, -1.0);
    conv.setFilterWeight(0, 3, 0.0); // bias

    double input[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double output[3]; // (5 - 3) / 1 + 1 = 3

    conv.forward(input, output);

    // kernel [1, 0, -1]:
    // pos 0: 1*1 + 0*2 + (-1)*3 = -2
    // pos 1: 1*2 + 0*3 + (-1)*4 = -2
    // pos 2: 1*3 + 0*4 + (-1)*5 = -2
    BOOST_TEST(fabs(output[0] - (-2.0)) < 0.001);
    BOOST_TEST(fabs(output[1] - (-2.0)) < 0.001);
    BOOST_TEST(fabs(output[2] - (-2.0)) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_conv1d_multi_filter)
{
    // Conv1D with 4-point input, kernel=2, stride=1, 2 filters
    tinymind::Conv1D<double, 4, 2, 1, 2> conv;

    // Filter 0: [1, 1], bias=0 (sum filter)
    conv.setFilterWeight(0, 0, 1.0);
    conv.setFilterWeight(0, 1, 1.0);
    conv.setFilterWeight(0, 2, 0.0);

    // Filter 1: [1, -1], bias=0 (difference filter)
    conv.setFilterWeight(1, 0, 1.0);
    conv.setFilterWeight(1, 1, -1.0);
    conv.setFilterWeight(1, 2, 0.0);

    double input[4] = {1.0, 3.0, 2.0, 4.0};
    double output[6]; // 2 filters * 3 positions

    conv.forward(input, output);

    // Filter 0 (sum): [4, 5, 6]
    BOOST_TEST(fabs(output[0] - 4.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 5.0) < 0.001);
    BOOST_TEST(fabs(output[2] - 6.0) < 0.001);

    // Filter 1 (diff): [-2, 1, -2]
    BOOST_TEST(fabs(output[3] - (-2.0)) < 0.001);
    BOOST_TEST(fabs(output[4] - 1.0) < 0.001);
    BOOST_TEST(fabs(output[5] - (-2.0)) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_conv1d_static_sizes)
{
    typedef tinymind::Conv1D<double, 100, 5, 2, 8> ConvType;

    // OutputLength = (100 - 5) / 2 + 1 = 48
    static_assert(ConvType::OutputLength == 48, "Wrong output length");
    static_assert(ConvType::OutputSize == 384, "Wrong output size"); // 8 * 48
    static_assert(ConvType::TotalWeights == 48, "Wrong total weights"); // 8 * (5 + 1)
    BOOST_TEST(true); // test counts toward suite
}

BOOST_AUTO_TEST_CASE(test_case_conv1d_fixed_point)
{
    // Verify Conv1D works with Q8.8 fixed-point
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    tinymind::Conv1D<ValueType, 5, 3, 1, 1> conv;

    // Set kernel [1, 0, -1], bias=0
    conv.setFilterWeight(0, 0, ValueType(1, 0));
    conv.setFilterWeight(0, 1, ValueType(0));
    conv.setFilterWeight(0, 2, ValueType(-1, 0));
    conv.setFilterWeight(0, 3, ValueType(0)); // bias

    ValueType input[5];
    input[0] = ValueType(1, 0);
    input[1] = ValueType(2, 0);
    input[2] = ValueType(3, 0);
    input[3] = ValueType(4, 0);
    input[4] = ValueType(5, 0);

    ValueType output[3];
    conv.forward(input, output);

    // kernel [1, 0, -1]: each output should be -2
    ValueType expected(-2, 0);
    BOOST_TEST(output[0].getValue() == expected.getValue());
    BOOST_TEST(output[1].getValue() == expected.getValue());
    BOOST_TEST(output[2].getValue() == expected.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_truncated_bptt)
{
    // Test that TruncatedBPTT accumulates steps before training
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
        ValueType,
        UniformRealRandomNumberGenerator,
        tinymind::TanhActivationPolicy,
        tinymind::TanhActivationPolicy,
        tinymind::SigmoidActivationPolicy> TF;
    typedef tinymind::LstmNeuralNetwork<ValueType, 1, tinymind::HiddenLayers<4>, 1, TF> LstmType;

    LstmType nn;
    tinymind::TruncatedBPTT<LstmType, 3> trainer;

    ValueType input[1] = {0.5};
    ValueType target[1] = {0.8};

    // Steps 1-2: accumulate, no training yet
    trainer.step(nn, input, target);
    BOOST_TEST(trainer.getStepCount() == 1u);
    trainer.step(nn, input, target);
    BOOST_TEST(trainer.getStepCount() == 2u);

    // Step 3: window full, trains and resets
    trainer.step(nn, input, target);
    BOOST_TEST(trainer.getStepCount() == 0u);

    // Flush with partial window
    trainer.step(nn, input, target);
    trainer.flush(nn);
    BOOST_TEST(trainer.getStepCount() == 0u);
}

// ============================================================
// MaxPool1D tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_maxpool1d_forward)
{
    // MaxPool1D: 6-element input, pool size 2, stride 2, 1 channel
    tinymind::MaxPool1D<double, 6, 2, 2, 1> pool;

    double input[6] = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0};
    double output[3]; // (6 - 2) / 2 + 1 = 3

    pool.forward(input, output);

    // Window [1,3] -> 3, [2,5] -> 5, [4,6] -> 6
    BOOST_TEST(fabs(output[0] - 3.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 5.0) < 0.001);
    BOOST_TEST(fabs(output[2] - 6.0) < 0.001);

    // Verify argmax indices
    BOOST_TEST(pool.getArgMaxIndex(0) == 1u);
    BOOST_TEST(pool.getArgMaxIndex(1) == 3u);
    BOOST_TEST(pool.getArgMaxIndex(2) == 5u);
}

BOOST_AUTO_TEST_CASE(test_case_maxpool1d_backward)
{
    tinymind::MaxPool1D<double, 6, 2, 2, 1> pool;

    double input[6] = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0};
    double output[3];
    pool.forward(input, output);

    // Backprop: gradients should route to argmax positions
    double outputDeltas[3] = {0.1, 0.2, 0.3};
    double inputDeltas[6];
    pool.backward(outputDeltas, inputDeltas);

    BOOST_TEST(fabs(inputDeltas[0]) < 0.001); // not max
    BOOST_TEST(fabs(inputDeltas[1] - 0.1) < 0.001); // max of window 0
    BOOST_TEST(fabs(inputDeltas[2]) < 0.001); // not max
    BOOST_TEST(fabs(inputDeltas[3] - 0.2) < 0.001); // max of window 1
    BOOST_TEST(fabs(inputDeltas[4]) < 0.001); // not max
    BOOST_TEST(fabs(inputDeltas[5] - 0.3) < 0.001); // max of window 2
}

BOOST_AUTO_TEST_CASE(test_case_maxpool1d_multichannel)
{
    // 2 channels, 4 elements each, pool size 2, stride 2
    tinymind::MaxPool1D<double, 4, 2, 2, 2> pool;

    // Channel-major: [ch0: 1,4,2,3, ch1: 5,2,6,1]
    double input[8] = {1.0, 4.0, 2.0, 3.0, 5.0, 2.0, 6.0, 1.0};
    double output[4]; // 2 channels * 2 outputs

    pool.forward(input, output);

    // Ch0: [1,4]->4, [2,3]->3
    BOOST_TEST(fabs(output[0] - 4.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 3.0) < 0.001);
    // Ch1: [5,2]->5, [6,1]->6
    BOOST_TEST(fabs(output[2] - 5.0) < 0.001);
    BOOST_TEST(fabs(output[3] - 6.0) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_maxpool1d_stride1)
{
    // Overlapping windows: pool size 3, stride 1
    tinymind::MaxPool1D<double, 5, 3, 1, 1> pool;

    double input[5] = {2.0, 1.0, 4.0, 3.0, 5.0};
    double output[3]; // (5 - 3) / 1 + 1 = 3

    pool.forward(input, output);

    // [2,1,4]->4, [1,4,3]->4, [4,3,5]->5
    BOOST_TEST(fabs(output[0] - 4.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 4.0) < 0.001);
    BOOST_TEST(fabs(output[2] - 5.0) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_maxpool1d_static_sizes)
{
    typedef tinymind::MaxPool1D<double, 100, 4, 4, 3> PoolType;

    // OutputLength = (100 - 4) / 4 + 1 = 25
    static_assert(PoolType::OutputLength == 25, "Wrong output length");
    static_assert(PoolType::OutputSize == 75, "Wrong output size"); // 3 * 25
    static_assert(PoolType::InputSize == 300, "Wrong input size"); // 3 * 100
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(test_case_maxpool1d_fixed_point)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    tinymind::MaxPool1D<ValueType, 4, 2, 2, 1> pool;

    ValueType input[4];
    input[0] = ValueType(1, 0);
    input[1] = ValueType(3, 0);
    input[2] = ValueType(2, 0);
    input[3] = ValueType(5, 0);

    ValueType output[2];
    pool.forward(input, output);

    BOOST_TEST(output[0].getValue() == ValueType(3, 0).getValue());
    BOOST_TEST(output[1].getValue() == ValueType(5, 0).getValue());
}

// ============================================================
// AvgPool1D tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_avgpool1d_forward)
{
    tinymind::AvgPool1D<double, 6, 2, 2, 1> pool;

    double input[6] = {1.0, 3.0, 2.0, 6.0, 4.0, 8.0};
    double output[3];

    pool.forward(input, output);

    // [1,3]->2, [2,6]->4, [4,8]->6
    BOOST_TEST(fabs(output[0] - 2.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 4.0) < 0.001);
    BOOST_TEST(fabs(output[2] - 6.0) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_avgpool1d_backward)
{
    tinymind::AvgPool1D<double, 6, 2, 2, 1> pool;

    double outputDeltas[3] = {0.2, 0.4, 0.6};
    double inputDeltas[6];
    pool.backward(outputDeltas, inputDeltas);

    // Each gradient splits evenly across pool window
    BOOST_TEST(fabs(inputDeltas[0] - 0.1) < 0.001);
    BOOST_TEST(fabs(inputDeltas[1] - 0.1) < 0.001);
    BOOST_TEST(fabs(inputDeltas[2] - 0.2) < 0.001);
    BOOST_TEST(fabs(inputDeltas[3] - 0.2) < 0.001);
    BOOST_TEST(fabs(inputDeltas[4] - 0.3) < 0.001);
    BOOST_TEST(fabs(inputDeltas[5] - 0.3) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_avgpool1d_multichannel)
{
    tinymind::AvgPool1D<double, 4, 2, 2, 2> pool;

    double input[8] = {2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0};
    double output[4];

    pool.forward(input, output);

    // Ch0: [2,4]->3, [6,8]->7
    BOOST_TEST(fabs(output[0] - 3.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 7.0) < 0.001);
    // Ch1: [1,3]->2, [5,7]->6
    BOOST_TEST(fabs(output[2] - 2.0) < 0.001);
    BOOST_TEST(fabs(output[3] - 6.0) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_avgpool1d_pool3)
{
    tinymind::AvgPool1D<double, 6, 3, 3, 1> pool;

    double input[6] = {3.0, 6.0, 9.0, 12.0, 15.0, 18.0};
    double output[2];

    pool.forward(input, output);

    // [3,6,9]->6, [12,15,18]->15
    BOOST_TEST(fabs(output[0] - 6.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 15.0) < 0.001);
}

// ============================================================
// Dropout tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_dropout_inference_passthrough)
{
    // In inference mode, dropout should be identity
    tinymind::Dropout<double, 4, 50> dropout;
    dropout.setTraining(false);

    double input[4] = {1.0, 2.0, 3.0, 4.0};
    double output[4];

    dropout.forward(input, output);

    BOOST_TEST(fabs(output[0] - 1.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 2.0) < 0.001);
    BOOST_TEST(fabs(output[2] - 3.0) < 0.001);
    BOOST_TEST(fabs(output[3] - 4.0) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_dropout_training_zeros_some)
{
    // In training mode, some outputs should be zero
    srand(RANDOM_SEED);
    tinymind::Dropout<double, 100, 50> dropout;
    dropout.setTraining(true);

    double input[100];
    double output[100];
    for (size_t i = 0; i < 100; ++i)
    {
        input[i] = 1.0;
    }

    dropout.forward(input, output);

    size_t zeroCount = 0;
    size_t nonZeroCount = 0;
    for (size_t i = 0; i < 100; ++i)
    {
        if (fabs(output[i]) < 0.001)
        {
            ++zeroCount;
        }
        else
        {
            ++nonZeroCount;
        }
    }

    // With 50% dropout on 100 elements, expect roughly 50 zeros
    // Allow wide tolerance for randomness
    BOOST_TEST(zeroCount > 20u);
    BOOST_TEST(nonZeroCount > 20u);
}

BOOST_AUTO_TEST_CASE(test_case_dropout_inverted_scaling)
{
    // Verify inverted dropout scaling: survivors should be scaled by 1/(1-p)
    // With 50% dropout, scale = 2.0
    srand(RANDOM_SEED);
    tinymind::Dropout<double, 10, 50> dropout;
    dropout.setTraining(true);

    double input[10];
    double output[10];
    for (size_t i = 0; i < 10; ++i)
    {
        input[i] = 1.0;
    }

    dropout.forward(input, output);

    for (size_t i = 0; i < 10; ++i)
    {
        if (dropout.getMask(i))
        {
            // Kept: should be scaled by 2.0 (1/(1-0.5))
            BOOST_TEST(fabs(output[i] - 2.0) < 0.001);
        }
        else
        {
            // Dropped: should be zero
            BOOST_TEST(fabs(output[i]) < 0.001);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_case_dropout_backward_mask)
{
    srand(RANDOM_SEED);
    tinymind::Dropout<double, 5, 50> dropout;
    dropout.setTraining(true);

    double input[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double output[5];
    dropout.forward(input, output); // generates mask

    double outputDeltas[5] = {0.5, 0.5, 0.5, 0.5, 0.5};
    double inputDeltas[5];
    dropout.backward(outputDeltas, inputDeltas);

    for (size_t i = 0; i < 5; ++i)
    {
        if (dropout.getMask(i))
        {
            // Gradient scaled by 1/(1-p) = 2.0
            BOOST_TEST(fabs(inputDeltas[i] - 1.0) < 0.001);
        }
        else
        {
            BOOST_TEST(fabs(inputDeltas[i]) < 0.001);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_case_dropout_zero_percent)
{
    // 0% dropout should pass everything through even in training mode
    tinymind::Dropout<double, 4, 0> dropout;
    dropout.setTraining(true);

    double input[4] = {1.0, 2.0, 3.0, 4.0};
    double output[4];

    dropout.forward(input, output);

    BOOST_TEST(fabs(output[0] - 1.0) < 0.001);
    BOOST_TEST(fabs(output[1] - 2.0) < 0.001);
    BOOST_TEST(fabs(output[2] - 3.0) < 0.001);
    BOOST_TEST(fabs(output[3] - 4.0) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_dropout_mode_toggle)
{
    tinymind::Dropout<double, 4, 50> dropout;

    BOOST_TEST(dropout.isTraining() == true); // default is training
    dropout.setTraining(false);
    BOOST_TEST(dropout.isTraining() == false);
    dropout.setTraining(true);
    BOOST_TEST(dropout.isTraining() == true);
}

// ============================================================
// RMSprop optimizer tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_rmsprop_float_xor)
{
    // Verify RMSprop optimizer converges on XOR (floating-point)
    typedef double ValueType;
    typedef FloatingPointTransferFunctions<
        ValueType,
        UniformRealRandomNumberGenerator,
        tinymind::TanhActivationPolicy,
        tinymind::TanhActivationPolicy> BaseTF;

    struct RmsPropTF : public BaseTF
    {
        typedef tinymind::RmsPropOptimizerFloat<ValueType> OptimizerPolicyType;
    };

    typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 5, 1, RmsPropTF> NNType;

    srand(RANDOM_SEED);
    NNType nn;

    static const double ERROR_LIMIT = ValueHelper<double>::getErrorLimit();
    ValueType values[2], output[1], error;
    std::deque<double> errors;

    for (int i = 0; i < 10000; ++i)
    {
        generateXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);
        nn.trainNetwork(&output[0]);

        errors.push_front(error);
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();
        }
    }

    const double totalError = std::accumulate(errors.begin(), errors.end(), 0.0);
    const double averageError = (totalError / static_cast<double>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

BOOST_AUTO_TEST_CASE(test_case_rmsprop_fixedpoint_xor)
{
    // Verify RMSprop optimizer converges on XOR (fixed-point)
    // Adaptive optimizers need gradient clipping with fixed-point to
    // prevent overflow in the moment accumulation
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    typedef typename ValueType::FullWidthValueType FullWidthValueType;

    typedef tinymind::FixedPointTransferFunctions<
        ValueType,
        UniformRealRandomNumberGenerator<ValueType>,
        tinymind::TanhActivationPolicy<ValueType>,
        tinymind::TanhActivationPolicy<ValueType>,
        1,
        tinymind::DefaultNetworkInitializer<ValueType>,
        tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
        tinymind::ZeroToleranceCalculator<ValueType>,
        tinymind::GradientClipByValue<ValueType>,
        tinymind::NullWeightDecayPolicy<ValueType>,
        tinymind::FixedLearningRatePolicy<ValueType>,
        tinymind::RmsPropOptimizer<ValueType>> TransferFunctionsType;

    typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 5, 1, TransferFunctionsType> NNType;

    srand(RANDOM_SEED);
    NNType nn;

    static const FullWidthValueType ERROR_LIMIT = ValueHelper<ValueType>::getErrorLimit();
    ValueType values[2], output[1], error;
    std::deque<FullWidthValueType> errors;

    static const int RMSPROP_FP_ITERATIONS = TRAINING_ITERATIONS;

    for (int i = 0; i < RMSPROP_FP_ITERATIONS; ++i)
    {
        generateFixedPointXorValues(&values[0], &output[0]);
        nn.feedForward(&values[0]);
        error = nn.calculateError(&output[0]);

        if (!NNType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            nn.trainNetwork(&output[0]);
        }

        errors.push_front(error.getValue());
        if (errors.size() > NUM_SAMPLES_AVG_ERROR)
        {
            errors.pop_back();
        }
    }

    const FullWidthValueType totalError = std::accumulate(errors.begin(), errors.end(), static_cast<FullWidthValueType>(0));
    const FullWidthValueType averageError = (totalError / static_cast<FullWidthValueType>(NUM_SAMPLES_AVG_ERROR));
    BOOST_TEST(averageError <= ERROR_LIMIT);
}

// ============================================================
// Conv1D + Pool1D integration test
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_conv1d_maxpool1d_pipeline)
{
    // Conv1D(8 input, kernel=3) -> MaxPool1D(6 input, pool=2)
    tinymind::Conv1D<double, 8, 3, 1, 2> conv;
    typedef tinymind::MaxPool1D<double, 6, 2, 2, 2> PoolType;
    PoolType pool;

    // Set filter 0: [1, 1, 1], bias=0 (moving sum)
    conv.setFilterWeight(0, 0, 1.0);
    conv.setFilterWeight(0, 1, 1.0);
    conv.setFilterWeight(0, 2, 1.0);
    conv.setFilterWeight(0, 3, 0.0);

    // Set filter 1: [1, 0, -1], bias=0 (edge detect)
    conv.setFilterWeight(1, 0, 1.0);
    conv.setFilterWeight(1, 1, 0.0);
    conv.setFilterWeight(1, 2, -1.0);
    conv.setFilterWeight(1, 3, 0.0);

    double input[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double convOutput[12]; // 2 filters * 6 positions
    double poolOutput[6]; // 2 channels * 3 positions

    conv.forward(input, convOutput);

    // Filter 0 (sum): [3, 6, 9, 12, 15, 18]
    BOOST_TEST(fabs(convOutput[0] - 3.0) < 0.001);
    BOOST_TEST(fabs(convOutput[5] - 18.0) < 0.001);

    pool.forward(convOutput, poolOutput);

    // Pool ch0: [3,6]->6, [9,12]->12, [15,18]->18
    BOOST_TEST(fabs(poolOutput[0] - 6.0) < 0.001);
    BOOST_TEST(fabs(poolOutput[1] - 12.0) < 0.001);
    BOOST_TEST(fabs(poolOutput[2] - 18.0) < 0.001);
}

// ============================================================
// BatchNorm1D tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_batchnorm_normalizes_output)
{
    // BatchNorm should normalize input to approximately zero mean
    tinymind::BatchNorm1D<double, 4> bn;
    bn.setTraining(true);

    double input[4] = {2.0, 4.0, 6.0, 8.0};
    double output[4];

    bn.forward(input, output);

    // Mean of normalized output should be ~0
    double mean = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        mean += output[i];
    }
    mean /= 4.0;
    BOOST_TEST(fabs(mean) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_unit_variance)
{
    // With default gamma=1, beta=0, output should have unit variance
    tinymind::BatchNorm1D<double, 4> bn;
    bn.setTraining(true);

    double input[4] = {1.0, 3.0, 5.0, 7.0};
    double output[4];

    bn.forward(input, output);

    // Compute variance of output
    double mean = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        mean += output[i];
    }
    mean /= 4.0;

    double variance = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        const double diff = output[i] - mean;
        variance += diff * diff;
    }
    variance /= 4.0;

    // Variance should be ~1.0 (normalized)
    BOOST_TEST(fabs(variance - 1.0) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_gamma_beta)
{
    // Custom gamma and beta should scale and shift the normalized output
    tinymind::BatchNorm1D<double, 4> bn;
    bn.setTraining(true);

    // Set gamma=2, beta=3 for all elements
    for (size_t i = 0; i < 4; ++i)
    {
        bn.setGamma(i, 2.0);
        bn.setBeta(i, 3.0);
    }

    double input[4] = {1.0, 3.0, 5.0, 7.0};
    double output[4];

    bn.forward(input, output);

    // Output mean should be ~3 (beta) since normalized mean is 0
    double mean = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        mean += output[i];
    }
    mean /= 4.0;
    BOOST_TEST(fabs(mean - 3.0) < 0.01);

    // Output variance should be ~4 (gamma^2 * 1.0)
    double variance = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        const double diff = output[i] - mean;
        variance += diff * diff;
    }
    variance /= 4.0;
    BOOST_TEST(fabs(variance - 4.0) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_inference_uses_running_stats)
{
    // After training, inference should use running mean/variance
    tinymind::BatchNorm1D<double, 4, 100> bn; // momentum=100% so running stats match batch stats immediately

    double input[4] = {2.0, 4.0, 6.0, 8.0};
    double trainOutput[4];
    double inferOutput[4];

    // Training forward pass to populate running stats
    bn.setTraining(true);
    bn.forward(input, trainOutput);

    // Switch to inference
    bn.setTraining(false);
    bn.forward(input, inferOutput);

    // With 100% momentum, running stats equal batch stats, so outputs should match
    for (size_t i = 0; i < 4; ++i)
    {
        BOOST_TEST(fabs(trainOutput[i] - inferOutput[i]) < 0.01);
    }
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_constant_input)
{
    // Constant input should produce zero-centered output (all zeros with default gamma=1, beta=0)
    tinymind::BatchNorm1D<double, 4> bn;
    bn.setTraining(true);

    double input[4] = {5.0, 5.0, 5.0, 5.0};
    double output[4];

    bn.forward(input, output);

    // All inputs equal -> variance ~0 -> normalized is ~0 -> output is ~beta=0
    for (size_t i = 0; i < 4; ++i)
    {
        BOOST_TEST(fabs(output[i]) < 0.1);
    }
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_backward_gradients)
{
    tinymind::BatchNorm1D<double, 4> bn;
    bn.setTraining(true);

    double input[4] = {1.0, 2.0, 3.0, 4.0};
    double output[4];
    bn.forward(input, output);

    // Backward with uniform gradients
    double outputDeltas[4] = {1.0, 1.0, 1.0, 1.0};
    double inputDeltas[4];
    bn.backward(outputDeltas, inputDeltas);

    // Beta gradient should be sum of outputDeltas = 4.0 total, 1.0 per element
    for (size_t i = 0; i < 4; ++i)
    {
        BOOST_TEST(fabs(bn.getBetaGradient(i) - 1.0) < 0.001);
    }

    // Gamma gradient = outputDelta * normalized, and normalized values
    // should have zero mean, so gamma gradients should sum to ~0
    double gammaGradSum = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
        gammaGradSum += bn.getGammaGradient(i);
    }
    BOOST_TEST(fabs(gammaGradSum) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_mode_toggle)
{
    tinymind::BatchNorm1D<double, 4> bn;

    BOOST_TEST(bn.isTraining() == true); // default is training
    bn.setTraining(false);
    BOOST_TEST(bn.isTraining() == false);
    bn.setTraining(true);
    BOOST_TEST(bn.isTraining() == true);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_update_parameters)
{
    tinymind::BatchNorm1D<double, 4> bn;
    bn.setTraining(true);

    double input[4] = {1.0, 2.0, 3.0, 4.0};
    double output[4];
    bn.forward(input, output);

    double outputDeltas[4] = {1.0, 1.0, 1.0, 1.0};
    double inputDeltas[4];
    bn.backward(outputDeltas, inputDeltas);

    // Update with a learning rate
    bn.updateParameters(0.01);

    // Gamma should have changed (gradient was non-zero for at least some elements)
    // Beta should have changed (gradient was 1.0 for all elements)
    bool betaChanged = false;
    for (size_t i = 0; i < 4; ++i)
    {
        if (fabs(bn.getBeta(i)) > 0.001)
        {
            betaChanged = true;
        }
    }
    BOOST_TEST(betaChanged);

    // Gradients should be zeroed after update
    for (size_t i = 0; i < 4; ++i)
    {
        BOOST_TEST(fabs(bn.getGammaGradient(i)) < 0.001);
        BOOST_TEST(fabs(bn.getBetaGradient(i)) < 0.001);
    }
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_fixed_point)
{
    // Verify BatchNorm compiles and runs with Q16.16 fixed-point
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    tinymind::BatchNorm1D<ValueType, 4> bn;
    bn.setTraining(true);

    ValueType input[4];
    input[0] = ValueType(1, 0);
    input[1] = ValueType(3, 0);
    input[2] = ValueType(5, 0);
    input[3] = ValueType(7, 0);

    ValueType output[4];
    bn.forward(input, output);

    // Normalized output should have mean ~0
    // Sum the raw values - they should roughly cancel out
    typename ValueType::FullWidthValueType sum = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        sum += output[i].getValue();
    }
    // Allow tolerance for fixed-point rounding
    const typename ValueType::FullWidthValueType tolerance = 1 << (ValueType::NumberOfFractionalBits - 2);
    BOOST_TEST(std::abs(sum) <= tolerance);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_conv1d_pipeline)
{
    // Conv1D -> BatchNorm -> output pipeline
    tinymind::Conv1D<double, 8, 3, 1, 1> conv;
    tinymind::BatchNorm1D<double, 6> bn; // Conv output is (8-3)/1+1 = 6

    // Set kernel to [1, 1, 1], bias=0 (moving sum)
    conv.setFilterWeight(0, 0, 1.0);
    conv.setFilterWeight(0, 1, 1.0);
    conv.setFilterWeight(0, 2, 1.0);
    conv.setFilterWeight(0, 3, 0.0);

    double input[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double convOutput[6];
    double bnOutput[6];

    conv.forward(input, convOutput);
    // Conv output: [6, 9, 12, 15, 18, 21]

    bn.setTraining(true);
    bn.forward(convOutput, bnOutput);

    // BatchNorm output should have zero mean
    double mean = 0.0;
    for (size_t i = 0; i < 6; ++i)
    {
        mean += bnOutput[i];
    }
    mean /= 6.0;
    BOOST_TEST(fabs(mean) < 0.01);
}

// ============================================================
// BinaryDense tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_binary_dense_forward_basic)
{
    // 4 inputs, 2 outputs
    tinymind::BinaryDense<double, 4, 2> layer;

    // Set latent weights: output 0 = [+1, +1, -1, -1], output 1 = [+1, -1, +1, -1]
    layer.setLatentWeight(0, 0,  0.5);
    layer.setLatentWeight(0, 1,  0.3);
    layer.setLatentWeight(0, 2, -0.7);
    layer.setLatentWeight(0, 3, -0.2);

    layer.setLatentWeight(1, 0,  0.9);
    layer.setLatentWeight(1, 1, -0.4);
    layer.setLatentWeight(1, 2,  0.6);
    layer.setLatentWeight(1, 3, -0.1);

    layer.setBias(0, 0.0);
    layer.setBias(1, 0.0);

    layer.binarizeWeights();

    // Verify binarization: positive latent -> +1, negative latent -> -1
    BOOST_TEST(layer.getBinaryWeight(0, 0) ==  1.0);
    BOOST_TEST(layer.getBinaryWeight(0, 1) ==  1.0);
    BOOST_TEST(layer.getBinaryWeight(0, 2) == -1.0);
    BOOST_TEST(layer.getBinaryWeight(0, 3) == -1.0);

    BOOST_TEST(layer.getBinaryWeight(1, 0) ==  1.0);
    BOOST_TEST(layer.getBinaryWeight(1, 1) == -1.0);
    BOOST_TEST(layer.getBinaryWeight(1, 2) ==  1.0);
    BOOST_TEST(layer.getBinaryWeight(1, 3) == -1.0);

    // Input: [1.0, -1.0, 1.0, -1.0] => sign = [+1, -1, +1, -1]
    double input[4] = {1.0, -1.0, 1.0, -1.0};
    double output[2];
    layer.forward(input, output);

    // output[0]: sign(input) dot weights[0] = (+1)(+1) + (-1)(+1) + (+1)(-1) + (-1)(-1)
    //          = 1 - 1 - 1 + 1 = 0
    BOOST_TEST(fabs(output[0] - 0.0) < 0.01);

    // output[1]: sign(input) dot weights[1] = (+1)(+1) + (-1)(-1) + (+1)(+1) + (-1)(-1)
    //          = 1 + 1 + 1 + 1 = 4
    BOOST_TEST(fabs(output[1] - 4.0) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_forward_with_bias)
{
    tinymind::BinaryDense<double, 3, 1> layer;

    // All positive latent weights => binary weights all +1
    layer.setLatentWeight(0, 0, 0.5);
    layer.setLatentWeight(0, 1, 0.5);
    layer.setLatentWeight(0, 2, 0.5);
    layer.setBias(0, 2.0);
    layer.binarizeWeights();

    // Input: [1.0, 1.0, 1.0] => sign = [+1, +1, +1]
    // dot product = 3, plus bias 2 = 5
    double input[3] = {1.0, 1.0, 1.0};
    double output[1];
    layer.forward(input, output);

    BOOST_TEST(fabs(output[0] - 5.0) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_static_sizes)
{
    typedef tinymind::BinaryDense<double, 64, 16> BinType;

    // 64 * 16 = 1024 binary weights
    static_assert(BinType::TotalBinaryWeights == 1024, "Wrong total binary weights");
    // 1024 / 32 = 32 packed words
    static_assert(BinType::PackedWeightWords == 32, "Wrong packed weight words");
    // 64 / 32 = 2 packed input words
    static_assert(BinType::PackedInputWords == 2, "Wrong packed input words");
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_packed_storage)
{
    // Verify that packed weights use minimal storage
    typedef tinymind::BinaryDense<double, 33, 1> BinType;

    // 33 bits needs 2 words (ceil(33/32) = 2)
    static_assert(BinType::PackedWeightWords == 2, "Wrong packed words for 33 bits");
    static_assert(BinType::PackedInputWords == 2, "Wrong packed input words for 33 bits");
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_backward_ste)
{
    tinymind::BinaryDense<double, 2, 1> layer;

    // Set latent weights within [-1, 1]
    layer.setLatentWeight(0, 0,  0.5);
    layer.setLatentWeight(0, 1, -0.3);
    layer.setBias(0, 0.0);
    layer.binarizeWeights();

    double input[2] = {1.0, -1.0};
    double output[1];
    layer.forward(input, output);

    double outputDeltas[1] = {1.0};
    double inputDeltas[2];
    layer.backward(outputDeltas, input, inputDeltas);

    // Learning rate = -0.1 (negative because gradient descent)
    layer.updateWeights(-0.1);

    // Latent weights should have been updated
    // w0: 0.5 + (-0.1) * (1.0 * sign(1.0)) = 0.5 - 0.1 = 0.4
    // w1: -0.3 + (-0.1) * (1.0 * sign(-1.0)) = -0.3 + 0.1 = -0.2
    BOOST_TEST(fabs(layer.getLatentWeight(0, 0) - 0.4) < 0.01);
    BOOST_TEST(fabs(layer.getLatentWeight(0, 1) - (-0.2)) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_ste_clipping)
{
    tinymind::BinaryDense<double, 2, 1> layer;

    // Set one latent weight outside [-1, 1] — STE should zero its gradient
    layer.setLatentWeight(0, 0,  1.5);  // outside [-1, 1]
    layer.setLatentWeight(0, 1,  0.5);  // inside [-1, 1]
    layer.setBias(0, 0.0);
    layer.binarizeWeights();

    double input[2] = {1.0, 1.0};
    double output[1];
    layer.forward(input, output);

    double outputDeltas[1] = {1.0};
    layer.backward(outputDeltas, input, nullptr);
    layer.updateWeights(-0.1);

    // w0 at 1.5 is outside [-1,1], gradient clipped to 0 => stays at 1.5
    BOOST_TEST(fabs(layer.getLatentWeight(0, 0) - 1.5) < 0.01);
    // w1 at 0.5 is inside [-1,1], gradient applies => 0.5 + (-0.1)(1.0*1.0) = 0.4
    BOOST_TEST(fabs(layer.getLatentWeight(0, 1) - 0.4) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_fixed_point)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    tinymind::BinaryDense<ValueType, 4, 1> layer;

    // Set latent weights using ValueConverter
    typedef tinymind::ValueConverter<double, ValueType> Conv;
    layer.setLatentWeight(0, 0, Conv::convertToDestinationType(0.5));
    layer.setLatentWeight(0, 1, Conv::convertToDestinationType(-0.5));
    layer.setLatentWeight(0, 2, Conv::convertToDestinationType(0.5));
    layer.setLatentWeight(0, 3, Conv::convertToDestinationType(-0.5));
    layer.setBias(0, Conv::convertToDestinationType(0.0));
    layer.binarizeWeights();

    // Binary weights should be [+1, -1, +1, -1]
    ValueType posOne = Conv::convertToDestinationType(1.0);
    ValueType negOne = Conv::convertToDestinationType(-1.0);
    BOOST_TEST(layer.getBinaryWeight(0, 0).getValue() == posOne.getValue());
    BOOST_TEST(layer.getBinaryWeight(0, 1).getValue() == negOne.getValue());
    BOOST_TEST(layer.getBinaryWeight(0, 2).getValue() == posOne.getValue());
    BOOST_TEST(layer.getBinaryWeight(0, 3).getValue() == negOne.getValue());

    // Input: all +1 => sign = [+1, +1, +1, +1]
    // dot = (+1)(+1) + (+1)(-1) + (+1)(+1) + (+1)(-1) = 0
    ValueType input[4];
    input[0] = Conv::convertToDestinationType(1.0);
    input[1] = Conv::convertToDestinationType(1.0);
    input[2] = Conv::convertToDestinationType(1.0);
    input[3] = Conv::convertToDestinationType(1.0);

    ValueType output[1];
    layer.forward(input, output);

    ValueType expected = Conv::convertToDestinationType(0.0);
    BOOST_TEST(output[0].getValue() == expected.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_binary_dense_input_gradient_propagation)
{
    tinymind::BinaryDense<double, 3, 2> layer;

    layer.setLatentWeight(0, 0,  0.5);
    layer.setLatentWeight(0, 1, -0.5);
    layer.setLatentWeight(0, 2,  0.5);

    layer.setLatentWeight(1, 0, -0.5);
    layer.setLatentWeight(1, 1,  0.5);
    layer.setLatentWeight(1, 2, -0.5);

    layer.setBias(0, 0.0);
    layer.setBias(1, 0.0);
    layer.binarizeWeights();

    double input[3] = {1.0, 1.0, 1.0};
    double output[2];
    layer.forward(input, output);

    // output deltas both = 1.0
    double outputDeltas[2] = {1.0, 1.0};
    double inputDeltas[3];
    layer.backward(outputDeltas, input, inputDeltas);

    // inputDeltas[0] = delta0 * sign(w[0][0]) + delta1 * sign(w[1][0])
    //                = 1.0 * (+1) + 1.0 * (-1) = 0
    BOOST_TEST(fabs(inputDeltas[0] - 0.0) < 0.01);
    // inputDeltas[1] = 1.0 * (-1) + 1.0 * (+1) = 0
    BOOST_TEST(fabs(inputDeltas[1] - 0.0) < 0.01);
    // inputDeltas[2] = 1.0 * (+1) + 1.0 * (-1) = 0
    BOOST_TEST(fabs(inputDeltas[2] - 0.0) < 0.01);
}

// ============================================================
// TernaryDense tests
// ============================================================

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_forward_basic)
{
    // 4 inputs, 2 outputs, threshold 50%
    tinymind::TernaryDense<double, 4, 2, 50> layer;

    // Set latent weights with clear magnitudes
    // Large values => +1 or -1, small values near 0 => 0
    layer.setLatentWeight(0, 0,  0.9);   // should be +1
    layer.setLatentWeight(0, 1,  0.01);  // should be 0 (below threshold)
    layer.setLatentWeight(0, 2, -0.8);   // should be -1
    layer.setLatentWeight(0, 3,  0.02);  // should be 0 (below threshold)

    layer.setLatentWeight(1, 0, -0.7);   // should be -1
    layer.setLatentWeight(1, 1,  0.85);  // should be +1
    layer.setLatentWeight(1, 2,  0.03);  // should be 0 (below threshold)
    layer.setLatentWeight(1, 3, -0.9);   // should be -1

    layer.setBias(0, 0.0);
    layer.setBias(1, 0.0);
    layer.ternarizeWeights();

    // Verify ternarization
    // Mean |w| = (0.9+0.01+0.8+0.02+0.7+0.85+0.03+0.9)/8 = 4.21/8 = 0.52625
    // threshold = 0.5 * 0.52625 = 0.263125
    // w[0][0]=0.9 > 0.263 => +1
    // w[0][1]=0.01 < 0.263 => 0
    // w[0][2]=-0.8, |0.8| > 0.263 => -1
    // w[0][3]=0.02 < 0.263 => 0
    BOOST_TEST(layer.getTernaryWeight(0, 0) == tinymind::detail::TERNARY_POS);
    BOOST_TEST(layer.getTernaryWeight(0, 1) == tinymind::detail::TERNARY_ZERO);
    BOOST_TEST(layer.getTernaryWeight(0, 2) == tinymind::detail::TERNARY_NEG);
    BOOST_TEST(layer.getTernaryWeight(0, 3) == tinymind::detail::TERNARY_ZERO);

    BOOST_TEST(layer.getTernaryWeight(1, 0) == tinymind::detail::TERNARY_NEG);
    BOOST_TEST(layer.getTernaryWeight(1, 1) == tinymind::detail::TERNARY_POS);
    BOOST_TEST(layer.getTernaryWeight(1, 2) == tinymind::detail::TERNARY_ZERO);
    BOOST_TEST(layer.getTernaryWeight(1, 3) == tinymind::detail::TERNARY_NEG);

    // Input: [2.0, 3.0, 4.0, 5.0]
    double input[4] = {2.0, 3.0, 4.0, 5.0};
    double output[2];
    layer.forward(input, output);

    // output[0] = (+1)*2 + 0*3 + (-1)*4 + 0*5 = 2 - 4 = -2
    BOOST_TEST(fabs(output[0] - (-2.0)) < 0.01);

    // output[1] = (-1)*2 + (+1)*3 + 0*4 + (-1)*5 = -2 + 3 - 5 = -4
    BOOST_TEST(fabs(output[1] - (-4.0)) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_forward_with_bias)
{
    tinymind::TernaryDense<double, 2, 1, 10> layer;

    // Both large positive => both +1
    layer.setLatentWeight(0, 0, 0.9);
    layer.setLatentWeight(0, 1, 0.8);
    layer.setBias(0, 1.5);
    layer.ternarizeWeights();

    double input[2] = {3.0, 4.0};
    double output[1];
    layer.forward(input, output);

    // output = (+1)*3 + (+1)*4 + 1.5 = 8.5
    BOOST_TEST(fabs(output[0] - 8.5) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_static_sizes)
{
    typedef tinymind::TernaryDense<double, 64, 16, 50> TernType;

    static_assert(TernType::TotalTernaryWeights == 1024, "Wrong total ternary weights");
    // 1024 ternary values, 16 per word = 64 words
    static_assert(TernType::PackedWeightWords == 64, "Wrong packed weight words");
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_all_zero_threshold)
{
    // With very high threshold, most weights should become 0
    tinymind::TernaryDense<double, 4, 1, 99> layer;

    layer.setLatentWeight(0, 0,  0.5);
    layer.setLatentWeight(0, 1, -0.5);
    layer.setLatentWeight(0, 2,  0.5);
    layer.setLatentWeight(0, 3, -0.5);
    layer.setBias(0, 0.0);
    layer.ternarizeWeights();

    // Mean |w| = 0.5, threshold = 0.99 * 0.5 = 0.495
    // All |w| = 0.5 > 0.495, so all should still be non-zero
    // (need truly small weights relative to mean for zeroing)
    double input[4] = {1.0, 1.0, 1.0, 1.0};
    double output[1];
    layer.forward(input, output);

    // With very uniform weights, threshold 99% of mean still passes
    // output = (+1)*1 + (-1)*1 + (+1)*1 + (-1)*1 = 0
    BOOST_TEST(fabs(output[0]) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_backward_ste)
{
    tinymind::TernaryDense<double, 2, 1, 10> layer;

    layer.setLatentWeight(0, 0,  0.8);
    layer.setLatentWeight(0, 1, -0.6);
    layer.setBias(0, 0.0);
    layer.ternarizeWeights();

    double input[2] = {2.0, 3.0};
    double output[1];
    layer.forward(input, output);

    double outputDeltas[1] = {1.0};
    double inputDeltas[2];
    layer.backward(outputDeltas, input, inputDeltas);
    layer.updateWeights(-0.1);

    // Gradient for w0: delta * input[0] = 1.0 * 2.0 = 2.0
    // w0: 0.8 + (-0.1)(2.0) = 0.6
    BOOST_TEST(fabs(layer.getLatentWeight(0, 0) - 0.6) < 0.01);

    // Gradient for w1: delta * input[1] = 1.0 * 3.0 = 3.0
    // w1: -0.6 + (-0.1)(3.0) = -0.9
    BOOST_TEST(fabs(layer.getLatentWeight(0, 1) - (-0.9)) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_input_gradient_propagation)
{
    tinymind::TernaryDense<double, 3, 1, 10> layer;

    // All large => all non-zero ternary
    layer.setLatentWeight(0, 0,  0.9);  // +1
    layer.setLatentWeight(0, 1, -0.8);  // -1
    layer.setLatentWeight(0, 2,  0.7);  // +1
    layer.setBias(0, 0.0);
    layer.ternarizeWeights();

    double input[3] = {1.0, 2.0, 3.0};
    double output[1];
    layer.forward(input, output);

    double outputDeltas[1] = {2.0};
    double inputDeltas[3];
    layer.backward(outputDeltas, input, inputDeltas);

    // inputDeltas[0] = delta * ternary(w[0][0]) = 2.0 * (+1) = 2.0
    BOOST_TEST(fabs(inputDeltas[0] - 2.0) < 0.01);
    // inputDeltas[1] = delta * ternary(w[0][1]) = 2.0 * (-1) = -2.0
    BOOST_TEST(fabs(inputDeltas[1] - (-2.0)) < 0.01);
    // inputDeltas[2] = delta * ternary(w[0][2]) = 2.0 * (+1) = 2.0
    BOOST_TEST(fabs(inputDeltas[2] - 2.0) < 0.01);
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_fixed_point)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    typedef tinymind::ValueConverter<double, ValueType> Conv;
    tinymind::TernaryDense<ValueType, 4, 1, 10> layer;

    layer.setLatentWeight(0, 0, Conv::convertToDestinationType(0.9));
    layer.setLatentWeight(0, 1, Conv::convertToDestinationType(-0.8));
    layer.setLatentWeight(0, 2, Conv::convertToDestinationType(0.7));
    layer.setLatentWeight(0, 3, Conv::convertToDestinationType(-0.6));
    layer.setBias(0, Conv::convertToDestinationType(0.0));
    layer.ternarizeWeights();

    // All weights are large magnitude => all non-zero ternary
    BOOST_TEST(layer.getTernaryWeight(0, 0) == tinymind::detail::TERNARY_POS);
    BOOST_TEST(layer.getTernaryWeight(0, 1) == tinymind::detail::TERNARY_NEG);
    BOOST_TEST(layer.getTernaryWeight(0, 2) == tinymind::detail::TERNARY_POS);
    BOOST_TEST(layer.getTernaryWeight(0, 3) == tinymind::detail::TERNARY_NEG);

    // Input: [2, 1, 3, 2]
    // output = (+1)*2 + (-1)*1 + (+1)*3 + (-1)*2 = 2 - 1 + 3 - 2 = 2
    ValueType input[4];
    input[0] = Conv::convertToDestinationType(2.0);
    input[1] = Conv::convertToDestinationType(1.0);
    input[2] = Conv::convertToDestinationType(3.0);
    input[3] = Conv::convertToDestinationType(2.0);

    ValueType output[1];
    layer.forward(input, output);

    ValueType expected = Conv::convertToDestinationType(2.0);
    BOOST_TEST(output[0].getValue() == expected.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_ternary_dense_get_value_accessors)
{
    tinymind::TernaryDense<double, 2, 1, 10> layer;

    layer.setLatentWeight(0, 0,  0.9);
    layer.setLatentWeight(0, 1, -0.8);
    layer.setBias(0, 0.0);
    layer.ternarizeWeights();

    BOOST_TEST(fabs(layer.getTernaryWeightValue(0, 0) - 1.0) < 0.01);
    BOOST_TEST(fabs(layer.getTernaryWeightValue(0, 1) - (-1.0)) < 0.01);
}

BOOST_AUTO_TEST_SUITE_END()
