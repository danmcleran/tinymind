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
     * 1D max pooling layer for downsampling time-series features.
     *
     * Pairs with Conv1D for embedded signal processing pipelines:
     *   Conv1D → MaxPool1D → Dense network
     *
     * All dimensions are compile-time template parameters with zero
     * dynamic allocation. Tracks argmax indices for backpropagation.
     *
     * @tparam ValueType     Numeric type (QValue or float/double)
     * @tparam InputLength   Number of input elements per channel
     * @tparam PoolSize      Width of the pooling window
     * @tparam Stride        Step size between pooling windows (default = PoolSize)
     * @tparam NumChannels   Number of input channels (default 1)
     */
    template<
        typename ValueType,
        size_t InputLength,
        size_t PoolSize,
        size_t Stride = PoolSize,
        size_t NumChannels = 1>
    class MaxPool1D
    {
    public:
        static const size_t OutputLength = (InputLength - PoolSize) / Stride + 1;
        static const size_t OutputSize = NumChannels * OutputLength;
        static const size_t InputSize = NumChannels * InputLength;

        MaxPool1D()
        {
            for (size_t i = 0; i < OutputSize; ++i)
            {
                mArgMaxIndices[i] = 0;
            }
        }

        /**
         * Forward pass: select the maximum value within each pooling window.
         *
         * @param input  Array of InputSize values (channel-major layout:
         *               [ch0_t0, ch0_t1, ..., ch1_t0, ch1_t1, ...])
         * @param output Array of OutputSize values (same channel-major layout)
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            for (size_t ch = 0; ch < NumChannels; ++ch)
            {
                const size_t inputOffset = ch * InputLength;
                const size_t outputOffset = ch * OutputLength;

                for (size_t pos = 0; pos < OutputLength; ++pos)
                {
                    const size_t inputStart = inputOffset + pos * Stride;
                    ValueType maxVal = input[inputStart];
                    size_t maxIdx = 0;

                    for (size_t k = 1; k < PoolSize; ++k)
                    {
                        if (input[inputStart + k] > maxVal)
                        {
                            maxVal = input[inputStart + k];
                            maxIdx = k;
                        }
                    }

                    output[outputOffset + pos] = maxVal;
                    mArgMaxIndices[outputOffset + pos] = inputStart + maxIdx;
                }
            }
        }

        /**
         * Backward pass: route gradients to the positions that produced
         * the maximum values during the forward pass.
         *
         * @param outputDeltas Array of OutputSize gradient values
         * @param inputDeltas  Array of InputSize gradient values (zeroed then populated)
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t i = 0; i < InputSize; ++i)
            {
                inputDeltas[i] = ValueType(0);
            }

            for (size_t i = 0; i < OutputSize; ++i)
            {
                inputDeltas[mArgMaxIndices[i]] += outputDeltas[i];
            }
        }

        /**
         * Get the input index that produced the max for a given output position.
         */
        size_t getArgMaxIndex(const size_t outputIndex) const
        {
            return mArgMaxIndices[outputIndex];
        }

    private:
        size_t mArgMaxIndices[OutputSize];

        static_assert(InputLength >= PoolSize, "Input length must be >= pool size.");
        static_assert(Stride > 0, "Stride must be > 0.");
        static_assert(PoolSize > 0, "Pool size must be > 0.");
        static_assert(NumChannels > 0, "Number of channels must be > 0.");
    };

    /**
     * 1D average pooling layer for downsampling time-series features.
     *
     * Computes the mean value within each pooling window. Simpler
     * gradient flow than max pooling (uniform distribution to all
     * positions in the window).
     *
     * @tparam ValueType     Numeric type (QValue or float/double)
     * @tparam InputLength   Number of input elements per channel
     * @tparam PoolSize      Width of the pooling window
     * @tparam Stride        Step size between pooling windows (default = PoolSize)
     * @tparam NumChannels   Number of input channels (default 1)
     */
    template<
        typename ValueType,
        size_t InputLength,
        size_t PoolSize,
        size_t Stride = PoolSize,
        size_t NumChannels = 1>
    class AvgPool1D
    {
    public:
        static const size_t OutputLength = (InputLength - PoolSize) / Stride + 1;
        static const size_t OutputSize = NumChannels * OutputLength;
        static const size_t InputSize = NumChannels * InputLength;

        /**
         * Forward pass: compute the average value within each pooling window.
         *
         * @param input  Array of InputSize values (channel-major layout)
         * @param output Array of OutputSize values (channel-major layout)
         */
        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t ch = 0; ch < NumChannels; ++ch)
            {
                const size_t inputOffset = ch * InputLength;
                const size_t outputOffset = ch * OutputLength;

                for (size_t pos = 0; pos < OutputLength; ++pos)
                {
                    const size_t inputStart = inputOffset + pos * Stride;
                    ValueType sum = ValueType(0);

                    for (size_t k = 0; k < PoolSize; ++k)
                    {
                        sum += input[inputStart + k];
                    }

                    output[outputOffset + pos] = sum / static_cast<ValueType>(PoolSize);
                }
            }
        }

        /**
         * Backward pass: distribute gradients uniformly across the pooling window.
         *
         * @param outputDeltas Array of OutputSize gradient values
         * @param inputDeltas  Array of InputSize gradient values (zeroed then populated)
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t i = 0; i < InputSize; ++i)
            {
                inputDeltas[i] = ValueType(0);
            }

            for (size_t ch = 0; ch < NumChannels; ++ch)
            {
                const size_t inputOffset = ch * InputLength;
                const size_t outputOffset = ch * OutputLength;

                for (size_t pos = 0; pos < OutputLength; ++pos)
                {
                    const size_t inputStart = inputOffset + pos * Stride;
                    const ValueType grad = outputDeltas[outputOffset + pos] / static_cast<ValueType>(PoolSize);

                    for (size_t k = 0; k < PoolSize; ++k)
                    {
                        inputDeltas[inputStart + k] += grad;
                    }
                }
            }
        }

    private:
        static_assert(InputLength >= PoolSize, "Input length must be >= pool size.");
        static_assert(Stride > 0, "Stride must be > 0.");
        static_assert(PoolSize > 0, "Pool size must be > 0.");
        static_assert(NumChannels > 0, "Number of channels must be > 0.");
    };
}
