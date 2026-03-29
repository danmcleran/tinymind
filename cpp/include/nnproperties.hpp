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

#pragma once

/**
 * @file nnproperties.hpp
 * @brief Neural network weight serialization for offline-trained model import.
 *
 * NetworkPropertiesFileManager stores and restores neural network weights in
 * both text and binary formats. This enables a workflow where a network is
 * trained offline (e.g. in PyTorch) and the learned weights are loaded into
 * an untrainable TinyMind network for inference on a target device.
 *
 * === Weight File Format (text) ===
 *
 * One value per line. Fixed-point values are stored as their raw integer
 * representation (e.g. Q16.16 value 2.5 is stored as 163840). Floating-point
 * values are stored as decimal strings.
 *
 * The values appear in the following order:
 *
 * 1. Input-to-first-hidden-layer weights
 *    For each input neuron i (0 .. NumberOfInputs-1):
 *      For each hidden neuron h (0 .. NumberOfHiddenNeurons-1):
 *        weight[i][h]
 *    Count: NumberOfInputs * NumberOfHiddenNeurons
 *
 * 2. Input layer bias weights (one per first-hidden-layer neuron)
 *    For each hidden neuron h (0 .. NumberOfHiddenNeurons-1):
 *      bias[h]
 *    Count: NumberOfHiddenNeurons
 *
 * 3. Hidden-to-hidden layer weights (only when NumberOfHiddenLayers > 1)
 *    For each inner hidden layer l (0 .. NumberOfHiddenLayers-2):
 *      For each neuron h in layer l (0 .. NumberOfHiddenNeurons-1):
 *        For each neuron h1 in layer l+1 (0 .. NumberOfHiddenNeurons-1):
 *          weight[l][h][h1]
 *      Bias weights for layer l:
 *        For each neuron h1 in layer l+1 (0 .. NumberOfHiddenNeurons-1):
 *          bias[l][h1]
 *    Count per layer: NumberOfHiddenNeurons^2 + NumberOfHiddenNeurons
 *
 * 4. Last-hidden-layer-to-output weights
 *    For each hidden neuron h (0 .. NumberOfHiddenNeurons-1):
 *      For each output neuron o (0 .. NumberOfOutputs-1):
 *        weight[h][o]
 *    Count: NumberOfHiddenNeurons * NumberOfOutputs
 *
 * 5. Output layer bias weights
 *    For each output neuron o (0 .. NumberOfOutputs-1):
 *      bias[o]
 *    Count: NumberOfOutputs
 *
 * === Weight File Format (binary) ===
 *
 * Same value order as text format. Each value is written as a FullWidthValueType
 * in platform-native byte order using sizeof(FullWidthValueType) bytes per value.
 *
 * === Example: 2-input, 1-hidden-layer (3 neurons), 1-output network ===
 *
 * Line  1: weight input[0] -> hidden[0]
 * Line  2: weight input[0] -> hidden[1]
 * Line  3: weight input[0] -> hidden[2]
 * Line  4: weight input[1] -> hidden[0]
 * Line  5: weight input[1] -> hidden[1]
 * Line  6: weight input[1] -> hidden[2]
 * Line  7: bias -> hidden[0]
 * Line  8: bias -> hidden[1]
 * Line  9: bias -> hidden[2]
 * Line 10: weight hidden[0] -> output[0]
 * Line 11: weight hidden[1] -> output[0]
 * Line 12: weight hidden[2] -> output[0]
 * Line 13: bias -> output[0]
 *
 * Total values = I*H + H + H*O + O  (single hidden layer)
 * Total values = I*H + H + (L-1)*(H^2 + H) + H*O + O  (L hidden layers)
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstdio>

namespace tinymind {
    template<typename SourceType>
    struct ValueParser
    {
        typedef typename SourceType::FullWidthValueType FullWidthValueType;
        typedef int64_t ParsedValueType;

        static int64_t parseValue(char const* const buffer)
        {
            int64_t value;
            char* endPtr;
            value = strtoll(buffer, &endPtr, 10);
            return value;
        }
    };

    template<>
    struct ValueParser<double>
    {
        typedef double FullWidthValueType;
        typedef double ParsedValueType;

        static double parseValue(char const* const buffer)
        {
            return atof(buffer);
        }
    };
    
    template<>
    struct ValueParser<float>
    {
        typedef float FullWidthValueType;
        typedef float ParsedValueType;

        static float parseValue(char const* const buffer)
        {
            return static_cast<float>(atof(buffer));
        }
    };

    template<typename SourceType, typename DestinationType>
    struct ValueConverter
    {
        static DestinationType convertToDestinationType(const SourceType& value)
        {
            return value;
        }
    };

    template<typename DestinationType>
    struct ValueConverter<double, DestinationType>
    {
        typedef typename DestinationType::FullWidthValueType FullWidthValueType;

        static DestinationType convertToDestinationType(const double& value)
        {
            const FullWidthValueType fullWidthValue = static_cast<FullWidthValueType>(value * static_cast<double>(1ULL << DestinationType::NumberOfFractionalBits));
            const DestinationType weight(fullWidthValue);

            return weight;
        }
    };

    template<typename SourceType>
    struct ValueConverter<SourceType, double>
    {
        static double convertToDestinationType(const SourceType& value)
        {
            static const double factor = pow(2, -1.0 * SourceType::NumberOfFractionalBits);
            const double result = (static_cast<double>(value.getValue()) * factor);

            return result;
        }
    };

    template<>
    struct ValueConverter<double, double>
    {
        static double convertToDestinationType(const double& value)
        {
            return value;
        }
    };

    template<typename DestinationType>
    struct ValueConverter<float, DestinationType>
    {
        typedef typename DestinationType::FullWidthValueType FullWidthValueType;

        static DestinationType convertToDestinationType(const float& value)
        {
            const FullWidthValueType fullWidthValue = static_cast<FullWidthValueType>(value * static_cast<float>(1ULL << DestinationType::NumberOfFractionalBits));
            const DestinationType weight(fullWidthValue);

            return weight;
        }
    };

    template<typename SourceType>
    struct ValueConverter<SourceType, float>
    {
        static float convertToDestinationType(const SourceType& value)
        {
            static const float factor = pow(2, -1.0f * SourceType::NumberOfFractionalBits);
            const float result = (static_cast<float>(value.getValue()) * factor);

            return result;
        }
    };

    template<>
    struct ValueConverter<float, float>
    {
        static float convertToDestinationType(const float& value)
        {
            return value;
        }
    };

    template<typename NeuralNetworkType>
    struct NetworkPropertiesFileManager
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        static const size_t NumberOfInputLayerNeurons = NeuralNetworkType::NumberOfInputLayerNeurons;
        static const size_t NumberOfHiddenLayers = NeuralNetworkType::NeuralNetworkNumberOfHiddenLayers;
        static const size_t NumberOfHiddenLayerNeurons = NeuralNetworkType::NumberOfHiddenLayerNeurons;
        static const size_t NumberOfOutputLayerNeurons = NeuralNetworkType::NumberOfOutputLayerNeurons;

        template<typename SourceType, typename DestinationType>
        static void loadNetworkWeights(NeuralNetworkType& neuralNetwork, std::ifstream& inFile)
        {
            typedef ValueParser<SourceType> ValueParserType;
            typedef ValueConverter<SourceType, DestinationType> ValueConverterType;
            typedef typename ValueParserType::ParsedValueType ParsedValueType;
            char buffer[256];
            ParsedValueType weight;
            ValueType weightValue;
            unsigned hiddenLayer = 0;

            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    inFile.getline(buffer, 255);
                    weight = ValueParserType::parseValue(buffer);
                    weightValue = ValueConverterType::convertToDestinationType(weight);
                    neuralNetwork.setInputLayerWeightForNeuronAndConnection(i, h, weightValue);
                }
            }

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                inFile.getline(buffer, 255);
                weight = ValueParserType::parseValue(buffer);
                weightValue = ValueConverterType::convertToDestinationType(weight);
                neuralNetwork.setInputLayerBiasWeightForConnection(h, weightValue);
            }

            if (NumberOfHiddenLayers > 1)
            {
                for (hiddenLayer = 0; hiddenLayer < (NumberOfHiddenLayers - 1); ++hiddenLayer)
                {
                    for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                    {
                        for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                        {
                            inFile.getline(buffer, 255);
                            weight = ValueParserType::parseValue(buffer);
                            weightValue = ValueConverterType::convertToDestinationType(weight);
                            neuralNetwork.setHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1, weightValue);
                        }
                    }

                    for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                    {
                        inFile.getline(buffer, 255);
                        weight = ValueParserType::parseValue(buffer);
                        weightValue = ValueConverterType::convertToDestinationType(weight);
                        neuralNetwork.setHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1, weightValue);
                    }
                }
            }

            hiddenLayer = NumberOfHiddenLayers - 1;

            for (uint32_t hiddenNeuron = 0; hiddenNeuron < NumberOfHiddenLayerNeurons; ++hiddenNeuron)
            {
                for (uint32_t outputNeuron = 0; outputNeuron < NumberOfOutputLayerNeurons; ++outputNeuron)
                {
                    inFile.getline(buffer, 255);
                    weight = ValueParserType::parseValue(buffer);
                    weightValue = ValueConverterType::convertToDestinationType(weight);
                    neuralNetwork.setHiddenLayerWeightForNeuronAndConnection(hiddenLayer, hiddenNeuron, outputNeuron, weightValue);
                }
            }

            for (uint32_t outputNeuron = 0; outputNeuron < NumberOfOutputLayerNeurons; ++outputNeuron)
            {
                inFile.getline(buffer, 255);
                weight = ValueParserType::parseValue(buffer);
                weightValue = ValueConverterType::convertToDestinationType(weight);
                neuralNetwork.setHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, outputNeuron, weightValue);
            }
        }

        template<typename SourceType, typename DestinationType>
        static void loadStates(
                               NeuralNetworkType& neuralNetwork,
                               std::ifstream& inFile,
                               std::vector<SourceType>& sourceTypeStates,
                               std::vector<DestinationType>& destinationTypeStates)
        {
            typedef ValueParser<SourceType> ValueParserType;
            typedef ValueConverter<SourceType, DestinationType> ValueConverterType;
            typedef typename ValueParserType::ParsedValueType ParsedValueType;
            char buffer[256];
            ParsedValueType parsedValue;
            ValueType destinationValue;

            while(!inFile.eof())
            {
                inFile.getline(buffer, 255);
                parsedValue = ValueParserType::parseValue(buffer);
                destinationValue = ValueConverterType::convertToDestinationType(parsedValue);

                sourceTypeStates.push_back(parsedValue);
                destinationTypeStates.push_back(destinationValue);
            }
        }

        static void storeNetworkProperties(NeuralNetworkType& neuralNetwork, std::ofstream& outFile, ValueType const* const output, ValueType const* const learnedValues)
        {
            storeNetworkWeights(neuralNetwork, outFile, ",");

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << output[o] << ",";
            }

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << learnedValues[o] << ",";
            }
        }

        static void storeNetworkWeights(NeuralNetworkType& neuralNetwork, std::ofstream& outFile, char const* const delimiter = "\n")
        {
            unsigned hiddenLayer = 0;

            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    outFile << neuralNetwork.getInputLayerWeightForNeuronAndConnection(i, h) << delimiter;
                }
            }

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                outFile << neuralNetwork.getInputLayerBiasNeuronWeightForConnection(h) << delimiter;
            }

            if (NumberOfHiddenLayers > 1)
            {
                for (hiddenLayer = 0; hiddenLayer < (NumberOfHiddenLayers - 1); ++hiddenLayer)
                {
                    for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                    {
                        for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                        {
                            outFile << neuralNetwork.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1) << delimiter;
                        }
                    }

                    for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                    {
                        outFile << neuralNetwork.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1) << delimiter;
                    }
                }
            }

            hiddenLayer = NumberOfHiddenLayers - 1;

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    outFile << neuralNetwork.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, o) << delimiter;
                }
            }

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << neuralNetwork.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, o) << delimiter;
            }
        }

        static void storeNetworkWeights(NeuralNetworkType& neuralNetwork, char const* const binaryFilePath)
        {
            typedef typename NeuralNetworkType::NeuralNetworkValueType::FullWidthValueType FullWidthValueType;
            std::ofstream outFile(binaryFilePath, std::ios::binary);
            FullWidthValueType value;
            unsigned hiddenLayer = 0;

            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    value = neuralNetwork.getInputLayerWeightForNeuronAndConnection(i, h).getValue();
                    outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
                }
            }

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                value = neuralNetwork.getInputLayerBiasNeuronWeightForConnection(h).getValue();
                outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
            }

            if (NumberOfHiddenLayers > 1)
            {
                for (hiddenLayer = 0; hiddenLayer < (NumberOfHiddenLayers - 1); ++hiddenLayer)
                {
                    for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                    {
                        for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                        {
                            value = neuralNetwork.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, h1).getValue();
                            outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
                        }
                    }

                    for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                    {
                        value = neuralNetwork.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, h1).getValue();
                        outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
                    }
                }
            }

            hiddenLayer = NumberOfHiddenLayers - 1;

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    value = neuralNetwork.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, o).getValue();
                    outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
                }
            }

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                value = neuralNetwork.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, o).getValue();
                outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
            }
        }

        static void writeHeader(std::ofstream& outFile)
        {
            unsigned hiddenLayer = 0;

            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    outFile << "Input" << i << h << "Weight,";
                }
            }

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                outFile << "InputBias" << hiddenLayer << h << "Weight,";
            }

            if (NumberOfHiddenLayers > 1)
            {
                for (hiddenLayer = 0; hiddenLayer < (NumberOfHiddenLayers - 1); ++hiddenLayer)
                {
                    for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                    {
                        for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                        {
                            outFile << "Hidden" << hiddenLayer << h << h1 << "Weight,";
                        }
                    }

                    for (uint32_t h1 = 0; h1 < NumberOfHiddenLayerNeurons; ++h1)
                    {
                        outFile << "Hidden" << hiddenLayer << "ToHidden" << (hiddenLayer + 1) << "Bias" << h1 << "Weight,";
                    }
                }
            }

            hiddenLayer = NumberOfHiddenLayers - 1;

            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    outFile << "Hidden" << hiddenLayer << h << o << "Weight,";
                }
            }

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << "Hidden" << hiddenLayer << "Bias" << o << "Weight,";
            }

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << "Expected" << o << ",";
            }

            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << "Learned" << o << ",";
            }

            outFile << "Error" << std::endl;
            outFile << std::dec;
        }

    };

    /**
     * Weight serialization for gated recurrent networks (LSTM, GRU).
     *
     * Weight file format (text, one value per line):
     *
     * 1. Input-to-hidden weights (gated: each input neuron has H*G connections)
     *    Count: NumberOfInputs * NumberOfHiddenNeurons * GateMultiplier
     *
     * 2. Input layer bias weights (gated)
     *    Count: NumberOfHiddenNeurons * GateMultiplier
     *
     * 3. Recurrent-to-hidden weights (gated: each recurrent neuron has H*G connections)
     *    Count: NumberOfHiddenNeurons * NumberOfHiddenNeurons * GateMultiplier
     *
     * 4. Last-hidden-to-output weights (NOT gated)
     *    Count: NumberOfHiddenNeurons * NumberOfOutputs
     *
     * 5. Output layer bias weights (from last hidden layer bias, if present)
     *    Count: NumberOfOutputs
     */
    template<typename NeuralNetworkType>
    struct RecurrentNetworkPropertiesFileManager
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        static const size_t NumberOfInputLayerNeurons = NeuralNetworkType::NumberOfInputLayerNeurons;
        static const size_t NumberOfHiddenLayerNeurons = NeuralNetworkType::NumberOfHiddenLayerNeurons;
        static const size_t NumberOfOutputLayerNeurons = NeuralNetworkType::NumberOfOutputLayerNeurons;
        static const size_t InputToHiddenConnections = NeuralNetworkType::InputToHiddenConnections;
        static const size_t RecurrentToHiddenConnections = NeuralNetworkType::RecurrentToHiddenConnections;
        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        template<typename SourceType, typename DestinationType>
        static void loadNetworkWeights(NeuralNetworkType& neuralNetwork, std::ifstream& inFile)
        {
            typedef ValueParser<SourceType> ValueParserType;
            typedef ValueConverter<SourceType, DestinationType> ValueConverterType;
            typedef typename ValueParserType::ParsedValueType ParsedValueType;
            char buffer[256];
            ParsedValueType weight;
            ValueType weightValue;

            // 1. Input-to-hidden weights (gated)
            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t conn = 0; conn < InputToHiddenConnections; ++conn)
                {
                    inFile.getline(buffer, 255);
                    weight = ValueParserType::parseValue(buffer);
                    weightValue = ValueConverterType::convertToDestinationType(weight);
                    neuralNetwork.setInputLayerWeightForNeuronAndConnection(i, conn, weightValue);
                }
            }

            // 2. Input layer bias weights (gated)
            for (uint32_t conn = 0; conn < InputToHiddenConnections; ++conn)
            {
                inFile.getline(buffer, 255);
                weight = ValueParserType::parseValue(buffer);
                weightValue = ValueConverterType::convertToDestinationType(weight);
                neuralNetwork.setInputLayerBiasWeightForConnection(conn, weightValue);
            }

            // 3. Recurrent-to-hidden weights (gated)
            for (uint32_t r = 0; r < NumberOfHiddenLayerNeurons; ++r)
            {
                for (uint32_t conn = 0; conn < RecurrentToHiddenConnections; ++conn)
                {
                    inFile.getline(buffer, 255);
                    weight = ValueParserType::parseValue(buffer);
                    weightValue = ValueConverterType::convertToDestinationType(weight);
                    neuralNetwork.getRecurrentLayer().setWeightForNeuronAndConnection(r, conn, weightValue);
                }
            }

            // 4. Last-hidden-to-output weights (not gated)
            const size_t hiddenLayer = neuralNetwork.NeuralNetworkNumberOfHiddenLayers - 1;
            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    inFile.getline(buffer, 255);
                    weight = ValueParserType::parseValue(buffer);
                    weightValue = ValueConverterType::convertToDestinationType(weight);
                    neuralNetwork.setHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, o, weightValue);
                }
            }

            // 5. Output bias weights
            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                inFile.getline(buffer, 255);
                weight = ValueParserType::parseValue(buffer);
                weightValue = ValueConverterType::convertToDestinationType(weight);
                neuralNetwork.setHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, o, weightValue);
            }
        }

        static void storeNetworkWeights(NeuralNetworkType& neuralNetwork, std::ofstream& outFile, char const* const delimiter = "\n")
        {
            // 1. Input-to-hidden weights (gated)
            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t conn = 0; conn < InputToHiddenConnections; ++conn)
                {
                    outFile << neuralNetwork.getInputLayerWeightForNeuronAndConnection(i, conn) << delimiter;
                }
            }

            // 2. Input layer bias weights (gated)
            for (uint32_t conn = 0; conn < InputToHiddenConnections; ++conn)
            {
                outFile << neuralNetwork.getInputLayerBiasNeuronWeightForConnection(conn) << delimiter;
            }

            // 3. Recurrent-to-hidden weights (gated)
            for (uint32_t r = 0; r < NumberOfHiddenLayerNeurons; ++r)
            {
                for (uint32_t conn = 0; conn < RecurrentToHiddenConnections; ++conn)
                {
                    outFile << neuralNetwork.getRecurrentLayer().getWeightForNeuronAndConnection(r, conn) << delimiter;
                }
            }

            // 4. Last-hidden-to-output weights (not gated)
            const size_t hiddenLayer = NeuralNetworkType::NeuralNetworkNumberOfHiddenLayers - 1;
            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    outFile << neuralNetwork.getHiddenLayerWeightForNeuronAndConnection(hiddenLayer, h, o) << delimiter;
                }
            }

            // 5. Output bias weights
            for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
            {
                outFile << neuralNetwork.getHiddenLayerBiasNeuronWeightForConnection(hiddenLayer, o) << delimiter;
            }
        }
    };

    /**
     * Weight serialization for Kolmogorov-Arnold Networks (KAN).
     *
     * Weight file format (text, one value per line):
     *
     * For each layer (input, inner hidden layers, last hidden):
     *   For each neuron in the layer:
     *     For each outgoing KAN connection:
     *       - Base weight (w_b)
     *       - Spline weight (w_s)
     *       - Spline coefficients (GridSize + SplineDegree values)
     *
     * Values per connection: 2 + GridSize + SplineDegree
     */
    template<typename KanNetworkType>
    struct KanNetworkPropertiesFileManager
    {
        typedef typename KanNetworkType::KanValueType ValueType;
        static const size_t NumberOfInputLayerNeurons = KanNetworkType::NumberOfInputLayerNeurons;
        static const size_t NumberOfHiddenLayerNeurons = KanNetworkType::NumberOfHiddenLayerNeurons;
        static const size_t NumberOfOutputLayerNeurons = KanNetworkType::NumberOfOutputLayerNeurons;
        static const size_t NumberOfInnerHiddenLayers = KanNetworkType::NumberOfInnerHiddenLayers;
        static const size_t GridSize = KanNetworkType::KanGridSize;
        static const size_t SplineDegree = KanNetworkType::KanSplineDegree;
        static const size_t NumberOfCoefficients = GridSize + SplineDegree;

        template<typename SourceType, typename DestinationType>
        static void loadNetworkWeights(KanNetworkType& network, std::ifstream& inFile)
        {
            char buffer[256];

            // Input layer: each neuron has NumberOfHiddenLayerNeurons outgoing KAN connections
            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    loadKanConnection(network.getInputLayer().getNeuron(i)->getConnection(h),
                                     inFile, buffer);
                }
            }

            // Last hidden layer: each neuron has NumberOfOutputLayerNeurons outgoing KAN connections
            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    loadKanConnection(network.getLastHiddenLayer().getNeuron(h)->getConnection(o),
                                     inFile, buffer);
                }
            }
        }

        static void storeNetworkWeights(KanNetworkType& network, std::ofstream& outFile, char const* const delimiter = "\n")
        {
            // Input layer
            for (uint32_t i = 0; i < NumberOfInputLayerNeurons; ++i)
            {
                for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
                {
                    storeKanConnection(network.getInputLayer().getNeuron(i)->getConnection(h),
                                      outFile, delimiter);
                }
            }

            // Last hidden layer
            for (uint32_t h = 0; h < NumberOfHiddenLayerNeurons; ++h)
            {
                for (uint32_t o = 0; o < NumberOfOutputLayerNeurons; ++o)
                {
                    storeKanConnection(network.getLastHiddenLayer().getNeuron(h)->getConnection(o),
                                      outFile, delimiter);
                }
            }
        }

    private:
        template<typename ConnectionType>
        static void loadKanConnection(ConnectionType* conn, std::ifstream& inFile, char* buffer)
        {
            typedef ValueParser<ValueType> ValueParserType;

            // Base weight
            inFile.getline(buffer, 255);
            conn->setBaseWeight(ValueParserType::parseValue(buffer));

            // Spline weight
            inFile.getline(buffer, 255);
            conn->setSplineWeight(ValueParserType::parseValue(buffer));

            // Spline coefficients
            for (uint32_t c = 0; c < NumberOfCoefficients; ++c)
            {
                inFile.getline(buffer, 255);
                conn->setCoefficient(c, ValueParserType::parseValue(buffer));
            }
        }

        template<typename ConnectionType>
        static void storeKanConnection(const ConnectionType* conn, std::ofstream& outFile, char const* const delimiter)
        {
            outFile << conn->getBaseWeight() << delimiter;
            outFile << conn->getSplineWeight() << delimiter;
            for (uint32_t c = 0; c < NumberOfCoefficients; ++c)
            {
                outFile << conn->getCoefficient(c) << delimiter;
            }
        }
    };
}
