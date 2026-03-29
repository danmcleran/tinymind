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

#pragma once

#include <cstddef>

namespace tinymind {
    /**
     * 1D convolution layer for time-series feature extraction.
     *
     * Designed for embedded sensor data processing (accelerometers, IMUs,
     * vibration, ECG). All dimensions are compile-time template parameters
     * with zero dynamic allocation.
     *
     * Output feeds into any TinyMind network as input:
     *   Conv1D<double, 100, 3, 1, 4> conv;    // 100 samples, kernel=3, 4 filters
     *   NeuralNetwork<double, 4*98, ...> mlp;  // flattened conv output
     *
     *   conv.forward(sensorData, features);
     *   mlp.feedForward(features);
     *
     * @tparam ValueType     Numeric type (QValue or float/double)
     * @tparam InputLength   Number of input time steps
     * @tparam KernelSize    Convolution kernel width
     * @tparam Stride        Step size between kernel positions (default 1)
     * @tparam NumFilters    Number of output feature channels (default 1)
     */
    template<
        typename ValueType,
        size_t InputLength,
        size_t KernelSize,
        size_t Stride = 1,
        size_t NumFilters = 1>
    class Conv1D
    {
    public:
        static const size_t OutputLength = (InputLength - KernelSize) / Stride + 1;
        static const size_t OutputSize = NumFilters * OutputLength;
        static const size_t WeightsPerFilter = KernelSize + 1; // kernel + bias
        static const size_t TotalWeights = NumFilters * WeightsPerFilter;

        Conv1D()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = ValueType(0);
                mGradients[i] = ValueType(0);
            }
        }

        /**
         * Initialize weights with values from a random number generator.
         */
        template<typename RandomNumberGeneratorPolicy>
        void initializeWeights()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = RandomNumberGeneratorPolicy::generateRandomWeight();
                mGradients[i] = ValueType(0);
            }
        }

        /**
         * Forward pass: convolve input with filters.
         * @param input  Array of InputLength values
         * @param output Array of OutputSize values (NumFilters * OutputLength)
         *               Layout: filter-major [f0_pos0, f0_pos1, ..., f1_pos0, ...]
         */
        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t f = 0; f < NumFilters; ++f)
            {
                const size_t weightOffset = f * WeightsPerFilter;
                const size_t outputOffset = f * OutputLength;

                for (size_t pos = 0; pos < OutputLength; ++pos)
                {
                    const size_t inputStart = pos * Stride;
                    ValueType sum = mWeights[weightOffset + KernelSize]; // bias

                    for (size_t k = 0; k < KernelSize; ++k)
                    {
                        sum += mWeights[weightOffset + k] * input[inputStart + k];
                    }

                    output[outputOffset + pos] = sum;
                }
            }
        }

        /**
         * Compute gradients given output deltas and the input that produced them.
         * Call after forward pass and error computation.
         * @param outputDeltas Array of OutputSize gradient values from the next layer
         * @param input        The same input array used in the forward pass
         */
        void computeGradients(ValueType const* const outputDeltas, ValueType const* const input)
        {
            // Zero gradients
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mGradients[i] = ValueType(0);
            }

            for (size_t f = 0; f < NumFilters; ++f)
            {
                const size_t weightOffset = f * WeightsPerFilter;
                const size_t outputOffset = f * OutputLength;

                for (size_t pos = 0; pos < OutputLength; ++pos)
                {
                    const size_t inputStart = pos * Stride;
                    const ValueType delta = outputDeltas[outputOffset + pos];

                    // Kernel weight gradients
                    for (size_t k = 0; k < KernelSize; ++k)
                    {
                        mGradients[weightOffset + k] += delta * input[inputStart + k];
                    }

                    // Bias gradient
                    mGradients[weightOffset + KernelSize] += delta;
                }
            }
        }

        /**
         * Update weights using SGD with learning rate.
         * @param learningRate Step size for weight update
         */
        void updateWeights(const ValueType& learningRate)
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] += learningRate * mGradients[i];
            }
        }

        // Weight accessors for serialization
        ValueType getWeight(const size_t index) const { return mWeights[index]; }
        void setWeight(const size_t index, const ValueType& value) { mWeights[index] = value; }
        ValueType getGradient(const size_t index) const { return mGradients[index]; }

        /**
         * Get kernel weight for a specific filter and position.
         * @param filter Filter index [0, NumFilters)
         * @param position Kernel position [0, KernelSize) or KernelSize for bias
         */
        ValueType getFilterWeight(const size_t filter, const size_t position) const
        {
            return mWeights[filter * WeightsPerFilter + position];
        }

        void setFilterWeight(const size_t filter, const size_t position, const ValueType& value)
        {
            mWeights[filter * WeightsPerFilter + position] = value;
        }

    private:
        ValueType mWeights[TotalWeights];
        ValueType mGradients[TotalWeights];

        static_assert(InputLength >= KernelSize, "Input length must be >= kernel size.");
        static_assert(Stride > 0, "Stride must be > 0.");
        static_assert(NumFilters > 0, "Number of filters must be > 0.");
        static_assert(KernelSize > 0, "Kernel size must be > 0.");
    };
}
