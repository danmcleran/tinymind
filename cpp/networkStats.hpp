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

#include <cstddef>

namespace tinymind {
    /**
     * Compile-time network statistics.
     *
     * Provides static constants describing a network's resource requirements,
     * useful for verifying a network fits within an embedded device's memory
     * budget before flashing.
     *
     * Usage:
     *   typedef NetworkStats<MyNetworkType> Stats;
     *   static_assert(Stats::InstanceSizeBytes <= 2048, "Network exceeds 2KB budget");
     *   std::cout << "Parameters: " << Stats::TotalParameters << std::endl;
     *
     * @tparam NeuralNetworkType Any TinyMind network type (MLP, LSTM, GRU, etc.)
     */
    template<typename NeuralNetworkType>
    struct NetworkStats
    {
        /// Total instance size in bytes (sizeof the network object)
        static const size_t InstanceSizeBytes = sizeof(NeuralNetworkType);

        /// Number of input neurons
        static const size_t NumberOfInputs = NeuralNetworkType::NumberOfInputLayerNeurons;

        /// Number of hidden layer neurons (last hidden layer)
        static const size_t NumberOfHiddenNeurons = NeuralNetworkType::NumberOfHiddenLayerNeurons;

        /// Number of output neurons
        static const size_t NumberOfOutputs = NeuralNetworkType::NumberOfOutputLayerNeurons;

        /// Number of hidden layers
        static const size_t NumberOfHiddenLayers = NeuralNetworkType::NeuralNetworkNumberOfHiddenLayers;

        /// Size of a single value in bytes
        static const size_t ValueSizeBytes = sizeof(typename NeuralNetworkType::NeuralNetworkValueType);
    };

    /**
     * Compile-time network statistics for KAN networks.
     *
     * @tparam KanNetworkType A KolmogorovArnoldNetwork type
     */
    template<typename KanNetworkType>
    struct KanNetworkStats
    {
        static const size_t InstanceSizeBytes = sizeof(KanNetworkType);

        static const size_t NumberOfInputs = KanNetworkType::NumberOfInputLayerNeurons;
        static const size_t NumberOfHiddenNeurons = KanNetworkType::NumberOfHiddenLayerNeurons;
        static const size_t NumberOfOutputs = KanNetworkType::NumberOfOutputLayerNeurons;

        static const size_t GridSize = KanNetworkType::KanGridSize;
        static const size_t SplineDegree = KanNetworkType::KanSplineDegree;
        static const size_t CoefficientsPerEdge = GridSize + SplineDegree;

        /// Values per KAN edge: base weight + spline weight + coefficients
        static const size_t ValuesPerEdge = 2 + CoefficientsPerEdge;

        static const size_t ValueSizeBytes = sizeof(typename KanNetworkType::KanValueType);
    };
}
