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

#include <iostream>
#include <fstream>
#include <string>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "random.hpp"
#include "neuralnet.hpp"
#include "constants.hpp"
#include "nnproperties.hpp"

// Q-Format value type
static const size_t NUMBER_OF_FIXED_BITS = 16;
static const size_t NUMBER_OF_FRACTIONAL_BITS = 16;
typedef tinymind::QValue<NUMBER_OF_FIXED_BITS, NUMBER_OF_FRACTIONAL_BITS, true> ValueType;

// Neural network architecture
static const size_t NUMBER_OF_INPUTS = 2;
static const size_t NUMBER_OF_HIDDEN_LAYERS = 1;
static const size_t NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 3;
static const size_t NUMBER_OF_OUTPUTS = 1;

// Typedef of transfer functions for the fixed-point neural network
typedef tinymind::FixedPointTransferFunctions<  ValueType,
                                                tinymind::NullRandomNumberPolicy<ValueType>,
                                                tinymind::ReluActivationPolicy<ValueType>,
                                                tinymind::SigmoidActivationPolicy<ValueType>> TransferFunctionsType;

// typedef the neural network itself
typedef tinymind::MultilayerPerceptron< ValueType,
                                        NUMBER_OF_INPUTS,
                                        NUMBER_OF_HIDDEN_LAYERS,
                                        NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
                                        NUMBER_OF_OUTPUTS,
                                        TransferFunctionsType> NeuralNetworkType;

typedef tinymind::NetworkPropertiesFileManager<NeuralNetworkType> NetworkPropertiesFileManagerType;

static NeuralNetworkType testNeuralNet;

#define TESTING_ITERATIONS 1000U
#define RANDOM_SEED 7U

static void generateXorTrainingValue(ValueType& x, ValueType& y, ValueType& z)
{
    const int randX = rand() & 0x1;
    const int randY = rand() & 0x1;
    const int result = (randX ^ randY);

    x = ValueType(randX, 0);
    y = ValueType(randY, 0);
    z = ValueType(result, 0);
}

int main(const int argc, char *argv[])
{
    (void)argc; // suppress unused parameter warning
    (void)argv; // suppress unused parameter warning

    using namespace std;

    srand(RANDOM_SEED); // seed random number generator

    char const* const inPath = "xor_weights_q16.bin";
    char const* const outPath = "nn_fixed_xor.txt";
    ifstream weightsInputFile(inPath, ios::in | ios::binary);
    ofstream results(outPath);
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType error;
    ValueType avgError(0U);
    ValueType lastAvgError(0U);

    NetworkPropertiesFileManagerType::template loadNetworkWeights<ValueType, ValueType>(testNeuralNet, weightsInputFile);

    NetworkPropertiesFileManagerType::writeHeader(results);

    for (unsigned i = 0U; i < TESTING_ITERATIONS ; ++i)
    {
        generateXorTrainingValue(values[0], values[1], output[0]);

        testNeuralNet.feedForward(&values[0]);
        error = testNeuralNet.calculateError(&output[0]);
        testNeuralNet.getLearnedValues(&learnedValues[0]);

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(testNeuralNet, results, &output[0], &learnedValues[0]);
        results << error << std::endl;
    }

    results.close();

    return 0;
}
