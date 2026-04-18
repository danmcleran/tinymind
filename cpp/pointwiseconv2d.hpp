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
     * Pointwise (1x1) 2D convolution: a per-pixel linear combination
     * across input channels producing Cout output channels. Together
     * with DepthwiseConv2D this forms a depthwise-separable block;
     * standalone it is the cheapest way to mix channel information
     * between depthwise layers.
     *
     * NHWC layout. Kernel is implicitly 1x1 so StrideH/StrideW do not
     * change output size; spatial dimensions pass through unchanged.
     *
     * Weights: per-filter [Cin values, bias].
     *
     * @tparam ValueType    Numeric type (QValue or float/double)
     * @tparam H            Input height (passes through)
     * @tparam W            Input width  (passes through)
     * @tparam InChannels   Input channel count
     * @tparam NumFilters   Output channel count
     */
    template<
        typename ValueType,
        size_t H,
        size_t W,
        size_t InChannels,
        size_t NumFilters>
    class PointwiseConv2D
    {
    public:
        static const size_t OutputHeight = H;
        static const size_t OutputWidth  = W;
        static const size_t OutputSize   = H * W * NumFilters;
        static const size_t InputSize    = H * W * InChannels;

        static const size_t WeightsPerFilter = InChannels + 1; // +1 bias
        static const size_t TotalWeights     = NumFilters * WeightsPerFilter;

        PointwiseConv2D()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = ValueType(0);
                mGradients[i] = ValueType(0);
            }
        }

        template<typename RandomNumberGeneratorPolicy>
        void initializeWeights()
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] = RandomNumberGeneratorPolicy::generateRandomWeight();
                mGradients[i] = ValueType(0);
            }
        }

        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    const size_t inPixelOffset  = (h * W + w) * InChannels;
                    const size_t outPixelOffset = (h * W + w) * NumFilters;

                    for (size_t f = 0; f < NumFilters; ++f)
                    {
                        const size_t weightOffset = f * WeightsPerFilter;
                        ValueType sum = mWeights[weightOffset + InChannels]; // bias

                        for (size_t ci = 0; ci < InChannels; ++ci)
                        {
                            sum += mWeights[weightOffset + ci] * input[inPixelOffset + ci];
                        }

                        output[outPixelOffset + f] = sum;
                    }
                }
            }
        }

        void computeGradients(ValueType const* const outputDeltas, ValueType const* const input)
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mGradients[i] = ValueType(0);
            }

            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    const size_t inPixelOffset  = (h * W + w) * InChannels;
                    const size_t outPixelOffset = (h * W + w) * NumFilters;

                    for (size_t f = 0; f < NumFilters; ++f)
                    {
                        const size_t weightOffset = f * WeightsPerFilter;
                        const ValueType delta = outputDeltas[outPixelOffset + f];

                        for (size_t ci = 0; ci < InChannels; ++ci)
                        {
                            mGradients[weightOffset + ci] += delta * input[inPixelOffset + ci];
                        }

                        mGradients[weightOffset + InChannels] += delta;
                    }
                }
            }
        }

        void updateWeights(const ValueType& learningRate)
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mWeights[i] += learningRate * mGradients[i];
            }
        }

        ValueType getWeight(size_t index) const { return mWeights[index]; }
        void setWeight(size_t index, const ValueType& value) { mWeights[index] = value; }
        ValueType getGradient(size_t index) const { return mGradients[index]; }

        ValueType getFilterWeight(size_t filter, size_t inChannel) const
        {
            return mWeights[filter * WeightsPerFilter + inChannel];
        }

        void setFilterWeight(size_t filter, size_t inChannel, const ValueType& value)
        {
            mWeights[filter * WeightsPerFilter + inChannel] = value;
        }

        ValueType getFilterBias(size_t filter) const
        {
            return mWeights[filter * WeightsPerFilter + InChannels];
        }

        void setFilterBias(size_t filter, const ValueType& value)
        {
            mWeights[filter * WeightsPerFilter + InChannels] = value;
        }

    private:
        ValueType mWeights[TotalWeights];
        ValueType mGradients[TotalWeights];

        static_assert(H > 0 && W > 0, "Spatial dimensions must be > 0.");
        static_assert(InChannels > 0, "Input channels must be > 0.");
        static_assert(NumFilters > 0, "Number of filters must be > 0.");
    };
}
