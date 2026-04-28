/**
* Copyright (c) 2025 Dan McLeran
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

#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace tinymind {

// Forward declaration: definition lives in neuralnet.hpp.
template<size_t...> struct HiddenLayers;

namespace detail {

/**
 * XavierStages computes per-stage metrics for a network with NumberOfInputs
 * inputs, the given HiddenLayers<...> descriptor, and NumberOfOutputs outputs.
 *
 * A "stage" is the set of weights between two adjacent layers. For L hidden
 * layers there are L+1 stages (input->H[0], H[0]->H[1], ..., H[L-1]->O).
 *
 * Each source layer carries a bias neuron, so the weight count for stage k is
 * (LayerSize(k) + 1) * LayerSize(k+1). The Xavier fan-sum at stage k is
 * LayerSize(k) + LayerSize(k+1).
 */
template<size_t NumberOfInputs, typename HiddenLayersDesc, size_t NumberOfOutputs>
struct XavierStages;

template<size_t NumberOfInputs, size_t NumberOfOutputs, size_t... Sizes>
struct XavierStages<NumberOfInputs, HiddenLayers<Sizes...>, NumberOfOutputs>
{
    static constexpr size_t Count = sizeof...(Sizes) + 1;

    static constexpr size_t layerSize(const size_t k)
    {
        constexpr size_t sizes[] = { NumberOfInputs, Sizes..., NumberOfOutputs };
        return sizes[k];
    }

    static constexpr size_t stageWeightCount(const size_t k)
    {
        return (layerSize(k) + 1) * layerSize(k + 1);
    }

    static constexpr size_t stageFanSum(const size_t k)
    {
        return layerSize(k) + layerSize(k + 1);
    }
};

template<size_t Count, size_t Size, size_t... Accumulated>
struct UniformHiddenLayersForXavier
{
    typedef typename UniformHiddenLayersForXavier<Count - 1, Size, Size, Accumulated...>::type type;
};

template<size_t Size, size_t... Accumulated>
struct UniformHiddenLayersForXavier<0, Size, Accumulated...>
{
    typedef HiddenLayers<Accumulated...> type;
};

} // namespace detail

/**
 * XavierWeightInitializerForLayers — Xavier weight initializer that supports
 * heterogeneous hidden layer widths via the same HiddenLayers<S0, S1, ...>
 * descriptor used by NeuralNetwork in neuralnet.hpp.
 *
 * Each call to generateUniformWeight()/generateNormalWeight() emits one weight
 * for the next outgoing connection, advancing through the layer pairs in the
 * same order the network's initializeWeights() chain visits them:
 *   input layer -> first hidden, first hidden -> second hidden, ...,
 *   last hidden -> output. Both regular neurons and per-layer bias neurons
 *   contribute to each stage's weight count.
 */
template<size_t NumberOfInputs, typename HiddenLayersDesc, size_t NumberOfOutputs>
struct XavierWeightInitializerForLayers;

template<size_t NumberOfInputs, size_t NumberOfOutputs, size_t... Sizes>
struct XavierWeightInitializerForLayers<NumberOfInputs, HiddenLayers<Sizes...>, NumberOfOutputs>
{
private:
    typedef detail::XavierStages<NumberOfInputs, HiddenLayers<Sizes...>, NumberOfOutputs> Stages;

    size_t mWeightInStage;
    size_t mStage;

    void advance()
    {
        ++mWeightInStage;
        if (mWeightInStage >= Stages::stageWeightCount(mStage))
        {
            mWeightInStage = 0;
            ++mStage;
            if (mStage >= Stages::Count)
            {
                mStage = 0;
            }
        }
    }

public:
    XavierWeightInitializerForLayers() : mWeightInStage(0), mStage(0)
    {
    }

    double generateUniformWeight()
    {
        const double fanSum = static_cast<double>(Stages::stageFanSum(mStage));
        const double limit = std::sqrt(6.0 / fanSum);
        const double randomValue = ((static_cast<double>(rand()) / RAND_MAX) * 2.0 * limit) - limit;

        advance();

        return randomValue;
    }

    double generateNormalWeight()
    {
        const double fanSum = static_cast<double>(Stages::stageFanSum(mStage));
        const double limit = std::sqrt(2.0 / fanSum);
        const double randomValue = ((static_cast<double>(rand()) / RAND_MAX) * 2.0 * limit) - limit;

        advance();

        return randomValue;
    }
};

/**
 * XavierWeightInitializer — backward-compatible alias for the uniform-width
 * case. NumberOfNeuronsInHiddenLayers is used for every hidden layer.
 */
template<
            size_t NumberOfInputs,
            size_t NumberOfHiddenLayers,
            size_t NumberOfNeuronsInHiddenLayers,
            size_t NumberOfOutputs>
using XavierWeightInitializer = XavierWeightInitializerForLayers<
        NumberOfInputs,
        typename detail::UniformHiddenLayersForXavier<NumberOfHiddenLayers, NumberOfNeuronsInHiddenLayers>::type,
        NumberOfOutputs>;

} // namespace tinymind
