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
 * an untrainable tinymind network for inference on a target device.
 *
 * === Weight File Format (text) ===
 *
 * One value per line. Fixed-point values are stored as their raw integer
 * representation (e.g. Q16.16 value 2.5 is stored as 163840). Floating-point
 * values are stored as decimal strings with 17 significant digits, so a
 * double reads back exactly.
 *
 * The values appear in the following order (S_l = neurons in hidden layer l,
 * G = connections per next-layer neuron: 1, or 4 for LSTM and 3 for GRU):
 *
 * 1. Input-to-first-hidden-layer weights
 *    For each input neuron i (0 .. NumberOfInputs-1):
 *      For each connection h (0 .. G*S_0-1):
 *        weight[i][h]
 *    Count: NumberOfInputs * G * S_0
 *
 * 2. Input layer bias weights (one per first-hidden-layer connection)
 *    Count: G * S_0
 *
 * 3. Hidden-to-hidden layer weights (only when NumberOfHiddenLayers > 1)
 *    For each inner hidden layer l (0 .. NumberOfHiddenLayers-2):
 *      For each neuron h in layer l (0 .. S_l-1):
 *        For each connection h1 into layer l+1 (0 .. G*S_{l+1}-1):
 *          weight[l][h][h1]
 *      Bias weights for layer l:
 *        For each connection h1 into layer l+1 (0 .. G*S_{l+1}-1):
 *          bias[l][h1]
 *    Count per layer: S_l * G * S_{l+1} + G * S_{l+1}
 *
 * 4. Last-hidden-layer-to-output weights
 *    For each hidden neuron h (0 .. S_last-1):
 *      For each output neuron o (0 .. NumberOfOutputs-1):
 *        weight[h][o]
 *    Count: S_last * NumberOfOutputs
 *
 * 5. Output layer bias weights
 *    For each output neuron o (0 .. NumberOfOutputs-1):
 *      bias[o]
 *    Count: NumberOfOutputs
 *
 * 6. Recurrent weights (recurrent networks only)
 *    For each inner hidden layer l, then the last hidden layer:
 *      For each neuron h (0 .. S_l-1):
 *        For each connection h1 back into the same layer (0 .. G*S_l-1):
 *          recurrent[l][h][h1]
 *    Count per layer: S_l * G * S_l
 *
 * For a feed-forward network with equal hidden layer sizes this is the
 * original layout. RecurrentNetworkPropertiesFileManager uses its own order
 * (see below).
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
 * Total values = I*H + H + (L-1)*(H^2 + H) + H*O + O  (L feed-forward hidden layers of H neurons)
 */

#include "tinymind_platform.hpp"

#include <cstdint>

#if TINYMIND_ENABLE_HOSTED_IO
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#endif

namespace tinymind {
#if TINYMIND_ENABLE_HOSTED_IO
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

#if TINYMIND_ENABLE_FLOAT
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
#endif // TINYMIND_ENABLE_FLOAT
#endif // TINYMIND_ENABLE_HOSTED_IO

    template<typename SourceType, typename DestinationType>
    struct ValueConverter
    {
        static DestinationType convertToDestinationType(const SourceType& value)
        {
            return value;
        }
    };

#if TINYMIND_ENABLE_FLOAT
    // Q -> float scale factor: 2^-NumberOfFractionalBits computed by repeated
    // halving so we don't depend on <cmath> / std::pow. Initialized once per
    // (SourceType, FloatType) pair as a function-local static.
    namespace detail {
        template<typename FloatType>
        static FloatType pow2NegativeFloat(unsigned exponent)
        {
            FloatType result = static_cast<FloatType>(1);
            for (unsigned i = 0; i < exponent; ++i)
            {
                result *= static_cast<FloatType>(0.5);
            }
            return result;
        }
    }

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
            static const double factor = detail::pow2NegativeFloat<double>(SourceType::NumberOfFractionalBits);
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
            static const float factor = detail::pow2NegativeFloat<float>(SourceType::NumberOfFractionalBits);
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
#endif // TINYMIND_ENABLE_FLOAT

#if TINYMIND_ENABLE_HOSTED_IO
    namespace detail
    {
        // Where one value of a weight file lives in the network.
        enum NetworkWeightKind
        {
            InputLayerWeight,
            InputLayerBiasWeight,
            HiddenLayerWeight,          // hidden layer `layer` -> next hidden layer or the output layer
            HiddenLayerBiasWeight,
            InnerRecurrentLayerWeight,  // h_{t-1} of inner hidden layer `layer` into its own gates
            RecurrentLayerWeight        // h_{t-1} of the last hidden layer into its own gates
        };

        struct NetworkWeightLocation
        {
            NetworkWeightKind kind;
            size_t layer;
            size_t neuron;
            size_t connection;
        };

        // Recurrent layer weights exist only on a recurrent network; a
        // feed-forward network's NullRecurrentLayer has none.
        template<bool HasRecurrentLayer>
        struct RecurrentWeightAccess
        {
            template<typename NeuralNetworkType, typename ValueType>
            static ValueType get(NeuralNetworkType& nn, const NetworkWeightLocation& loc)
            {
                if (InnerRecurrentLayerWeight == loc.kind)
                {
                    return nn.getInnerRecurrentLayerWeightForNeuronAndConnection(loc.layer, loc.neuron, loc.connection);
                }
                return nn.getRecurrentLayer().getWeightForNeuronAndConnection(loc.neuron, loc.connection);
            }

            template<typename NeuralNetworkType, typename ValueType>
            static void set(NeuralNetworkType& nn, const NetworkWeightLocation& loc, const ValueType& value)
            {
                if (InnerRecurrentLayerWeight == loc.kind)
                {
                    nn.setInnerRecurrentLayerWeightForNeuronAndConnection(loc.layer, loc.neuron, loc.connection, value);
                    return;
                }
                nn.getRecurrentLayer().setWeightForNeuronAndConnection(loc.neuron, loc.connection, value);
            }
        };

        template<>
        struct RecurrentWeightAccess<false>
        {
            template<typename NeuralNetworkType, typename ValueType>
            static ValueType get(NeuralNetworkType&, const NetworkWeightLocation&)
            {
                return ValueType();
            }

            template<typename NeuralNetworkType, typename ValueType>
            static void set(NeuralNetworkType&, const NetworkWeightLocation&, const ValueType&)
            {
            }
        };

        /**
         * Order of the values in a weight file: every trainable weight of the
         * network exactly once, using each hidden layer's own size and, for
         * LSTM/GRU layers, one connection per gate.
         *
         * RecurrentWeightsLast = true (NetworkPropertiesFileManager):
         *   input weights, input bias, [per inner layer: weights, bias],
         *   last-hidden-to-output weights, output bias, then (recurrent nets)
         *   [per inner layer: recurrent weights], last recurrent weights.
         *   A feed-forward network keeps the long-standing layout.
         *
         * RecurrentWeightsLast = false (RecurrentNetworkPropertiesFileManager):
         *   input weights, input bias, [per inner layer: weights, bias,
         *   recurrent weights], last recurrent weights, last-hidden-to-output
         *   weights, output bias. A single hidden layer keeps the long-standing
         *   LSTM/GRU layout (input, input bias, recurrent, output, output bias).
         */
        template<typename NeuralNetworkType, bool RecurrentWeightsLast>
        struct NetworkWeightLayout
        {
            typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
            static const size_t NumberOfInputs = NeuralNetworkType::NumberOfInputLayerNeurons;
            static const size_t NumberOfOutputs = NeuralNetworkType::NumberOfOutputLayerNeurons;
            static const size_t NumberOfHiddenLayers = NeuralNetworkType::NeuralNetworkNumberOfHiddenLayers;
            static const size_t Gates = NeuralNetworkType::HiddenLayerGateMultiplier;
            static const bool HasRecurrentLayer = NeuralNetworkType::NeuralNetworkHasRecurrentLayer;

            static size_t count()
            {
                size_t total = 0;
                NetworkWeightLocation loc;
                while (locate(total, loc))
                {
                    ++total;
                }
                return total;
            }

            // Map the k-th value of the file to its weight; false past the end.
            static bool locate(size_t k, NetworkWeightLocation& loc)
            {
                const size_t lastLayer = NumberOfHiddenLayers - 1;
                const size_t firstConnections = Gates * NeuralNetworkType::getHiddenLayerSize(0);

                if (take(k, loc, InputLayerWeight, 0, NumberOfInputs, firstConnections)) return true;
                if (take(k, loc, InputLayerBiasWeight, 0, 1, firstConnections)) return true;

                for (size_t layer = 0; layer < lastLayer; ++layer)
                {
                    const size_t size = NeuralNetworkType::getHiddenLayerSize(layer);
                    const size_t nextConnections = Gates * NeuralNetworkType::getHiddenLayerSize(layer + 1);

                    if (take(k, loc, HiddenLayerWeight, layer, size, nextConnections)) return true;
                    if (take(k, loc, HiddenLayerBiasWeight, layer, 1, nextConnections)) return true;
                    if (!RecurrentWeightsLast && HasRecurrentLayer)
                    {
                        if (take(k, loc, InnerRecurrentLayerWeight, layer, size, Gates * size)) return true;
                    }
                }

                const size_t lastSize = NeuralNetworkType::getHiddenLayerSize(lastLayer);

                if (!RecurrentWeightsLast && HasRecurrentLayer)
                {
                    if (take(k, loc, RecurrentLayerWeight, lastLayer, lastSize, Gates * lastSize)) return true;
                }

                if (take(k, loc, HiddenLayerWeight, lastLayer, lastSize, NumberOfOutputs)) return true;
                if (take(k, loc, HiddenLayerBiasWeight, lastLayer, 1, NumberOfOutputs)) return true;

                if (RecurrentWeightsLast && HasRecurrentLayer)
                {
                    for (size_t layer = 0; layer < lastLayer; ++layer)
                    {
                        const size_t size = NeuralNetworkType::getHiddenLayerSize(layer);
                        if (take(k, loc, InnerRecurrentLayerWeight, layer, size, Gates * size)) return true;
                    }
                    if (take(k, loc, RecurrentLayerWeight, lastLayer, lastSize, Gates * lastSize)) return true;
                }

                return false;
            }

            static ValueType get(NeuralNetworkType& nn, const NetworkWeightLocation& loc)
            {
                switch (loc.kind)
                {
                    case InputLayerWeight:
                        return nn.getInputLayerWeightForNeuronAndConnection(loc.neuron, loc.connection);
                    case InputLayerBiasWeight:
                        return nn.getInputLayerBiasNeuronWeightForConnection(loc.connection);
                    case HiddenLayerWeight:
                        return nn.getHiddenLayerWeightForNeuronAndConnection(loc.layer, loc.neuron, loc.connection);
                    case HiddenLayerBiasWeight:
                        return nn.getHiddenLayerBiasNeuronWeightForConnection(loc.layer, loc.connection);
                    default:
                        return RecurrentWeightAccess<HasRecurrentLayer>::template get<NeuralNetworkType, ValueType>(nn, loc);
                }
            }

            static void set(NeuralNetworkType& nn, const NetworkWeightLocation& loc, const ValueType& value)
            {
                switch (loc.kind)
                {
                    case InputLayerWeight:
                        nn.setInputLayerWeightForNeuronAndConnection(loc.neuron, loc.connection, value);
                        break;
                    case InputLayerBiasWeight:
                        nn.setInputLayerBiasWeightForConnection(loc.connection, value);
                        break;
                    case HiddenLayerWeight:
                        nn.setHiddenLayerWeightForNeuronAndConnection(loc.layer, loc.neuron, loc.connection, value);
                        break;
                    case HiddenLayerBiasWeight:
                        nn.setHiddenLayerBiasNeuronWeightForConnection(loc.layer, loc.connection, value);
                        break;
                    default:
                        RecurrentWeightAccess<HasRecurrentLayer>::template set<NeuralNetworkType, ValueType>(nn, loc, value);
                        break;
                }
            }

            // Column name for the CSV header (the long-standing names for
            // feed-forward weights).
            static void writeName(std::ofstream& outFile, const NetworkWeightLocation& loc)
            {
                const size_t lastLayer = NumberOfHiddenLayers - 1;
                switch (loc.kind)
                {
                    case InputLayerWeight:
                        outFile << "Input" << loc.neuron << loc.connection << "Weight,";
                        break;
                    case InputLayerBiasWeight:
                        outFile << "InputBias" << 0 << loc.connection << "Weight,";
                        break;
                    case HiddenLayerWeight:
                        outFile << "Hidden" << loc.layer << loc.neuron << loc.connection << "Weight,";
                        break;
                    case HiddenLayerBiasWeight:
                        if (loc.layer == lastLayer)
                        {
                            outFile << "Hidden" << loc.layer << "Bias" << loc.connection << "Weight,";
                        }
                        else
                        {
                            outFile << "Hidden" << loc.layer << "ToHidden" << (loc.layer + 1) << "Bias" << loc.connection << "Weight,";
                        }
                        break;
                    default:
                        outFile << "Recurrent" << loc.layer << loc.neuron << loc.connection << "Weight,";
                        break;
                }
            }

        private:
            // Consume a rows x columns block: if k falls inside it, fill loc,
            // otherwise skip past it.
            static bool take(size_t& k, NetworkWeightLocation& loc, const NetworkWeightKind kind, const size_t layer, const size_t rows, const size_t columns)
            {
                const size_t blockSize = rows * columns;
                if (k < blockSize)
                {
                    loc.kind = kind;
                    loc.layer = layer;
                    loc.neuron = k / columns;
                    loc.connection = k % columns;
                    return true;
                }
                k -= blockSize;
                return false;
            }
        };

        // Floating-point weights are written with enough digits to read back
        // the same value (max_digits10 of double); the stream's precision is
        // restored afterwards. Fixed-point values print as raw integers.
        struct FullPrecisionStreamGuard
        {
            explicit FullPrecisionStreamGuard(std::ofstream& stream) : mStream(stream), mPrecision(stream.precision())
            {
                mStream.precision(17);
            }

            ~FullPrecisionStreamGuard()
            {
                mStream.precision(mPrecision);
            }

        private:
            std::ofstream& mStream;
            std::streamsize mPrecision;

            FullPrecisionStreamGuard(const FullPrecisionStreamGuard&) = delete;
            FullPrecisionStreamGuard& operator=(const FullPrecisionStreamGuard&) = delete;
        };
    } // namespace detail

    /**
     * Text and binary weight files for any NeuralNetwork: every weight of
     * every hidden layer (with its own size, and one connection per gate for
     * LSTM/GRU), then, for a recurrent network, every recurrent layer's
     * weights. See detail::NetworkWeightLayout for the order.
     */
    template<typename NeuralNetworkType>
    struct NetworkPropertiesFileManager
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef detail::NetworkWeightLayout<NeuralNetworkType, true> LayoutType;
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
            detail::NetworkWeightLocation loc;

            for (size_t k = 0; LayoutType::locate(k, loc); ++k)
            {
                inFile.getline(buffer, 255);
                weight = ValueParserType::parseValue(buffer);
                LayoutType::set(neuralNetwork, loc, ValueConverterType::convertToDestinationType(weight));
            }
        }

        template<typename SourceType, typename DestinationType>
        static void loadStates(
                               NeuralNetworkType&,
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
            const detail::FullPrecisionStreamGuard precisionGuard(outFile);
            detail::NetworkWeightLocation loc;

            for (size_t k = 0; LayoutType::locate(k, loc); ++k)
            {
                outFile << LayoutType::get(neuralNetwork, loc) << delimiter;
            }
        }

        static void storeNetworkWeights(NeuralNetworkType& neuralNetwork, char const* const binaryFilePath)
        {
            typedef typename NeuralNetworkType::NeuralNetworkValueType::FullWidthValueType FullWidthValueType;
            std::ofstream outFile(binaryFilePath, std::ios::binary);
            FullWidthValueType value;
            detail::NetworkWeightLocation loc;

            for (size_t k = 0; LayoutType::locate(k, loc); ++k)
            {
                value = LayoutType::get(neuralNetwork, loc).getValue();
                outFile.write(reinterpret_cast<char*>(&value), sizeof(value));
            }
        }

        static void writeHeader(std::ofstream& outFile)
        {
            detail::NetworkWeightLocation loc;

            for (size_t k = 0; LayoutType::locate(k, loc); ++k)
            {
                LayoutType::writeName(outFile, loc);
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
     * Weight serialization for recurrent networks (LSTM, GRU, Elman).
     *
     * Weight file format (text, one value per line; G = gates per neuron:
     * 4 for LSTM, 3 for GRU, 1 for Elman; S_l = size of hidden layer l):
     *
     * 1. Input-to-first-hidden weights: NumberOfInputs * G * S_0
     * 2. Input layer bias weights: G * S_0
     * 3. For each inner hidden layer l (stacked networks only):
     *      weights into hidden layer l+1: S_l * G * S_{l+1}
     *      bias weights into hidden layer l+1: G * S_{l+1}
     *      recurrent weights of layer l: S_l * G * S_l
     * 4. Recurrent weights of the last hidden layer: S_last * G * S_last
     * 5. Last-hidden-to-output weights (not gated): S_last * NumberOfOutputs
     * 6. Output bias weights: NumberOfOutputs
     *
     * A single hidden layer has no step 3, which is the original layout.
     */
    template<typename NeuralNetworkType>
    struct RecurrentNetworkPropertiesFileManager
    {
        typedef typename NeuralNetworkType::NeuralNetworkValueType ValueType;
        typedef detail::NetworkWeightLayout<NeuralNetworkType, false> LayoutType;
        static const size_t NumberOfInputLayerNeurons = NeuralNetworkType::NumberOfInputLayerNeurons;
        static const size_t NumberOfHiddenLayerNeurons = NeuralNetworkType::NumberOfHiddenLayerNeurons;
        static const size_t NumberOfOutputLayerNeurons = NeuralNetworkType::NumberOfOutputLayerNeurons;
        static const size_t InputToHiddenConnections = NeuralNetworkType::InputToHiddenConnections;
        static const size_t RecurrentToHiddenConnections = NeuralNetworkType::RecurrentToHiddenConnections;
        static const size_t GateMultiplier = NeuralNetworkType::HiddenLayerGateMultiplier;

        static_assert(NeuralNetworkType::NeuralNetworkHasRecurrentLayer, "RecurrentNetworkPropertiesFileManager needs a recurrent network; use NetworkPropertiesFileManager.");

        template<typename SourceType, typename DestinationType>
        static void loadNetworkWeights(NeuralNetworkType& neuralNetwork, std::ifstream& inFile)
        {
            typedef ValueParser<SourceType> ValueParserType;
            typedef ValueConverter<SourceType, DestinationType> ValueConverterType;
            typedef typename ValueParserType::ParsedValueType ParsedValueType;
            char buffer[256];
            ParsedValueType weight;
            detail::NetworkWeightLocation loc;

            for (size_t k = 0; LayoutType::locate(k, loc); ++k)
            {
                inFile.getline(buffer, 255);
                weight = ValueParserType::parseValue(buffer);
                LayoutType::set(neuralNetwork, loc, ValueConverterType::convertToDestinationType(weight));
            }
        }

        static void storeNetworkWeights(NeuralNetworkType& neuralNetwork, std::ofstream& outFile, char const* const delimiter = "\n")
        {
            const detail::FullPrecisionStreamGuard precisionGuard(outFile);
            detail::NetworkWeightLocation loc;

            for (size_t k = 0; LayoutType::locate(k, loc); ++k)
            {
                outFile << LayoutType::get(neuralNetwork, loc) << delimiter;
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
#endif // TINYMIND_ENABLE_HOSTED_IO
}
