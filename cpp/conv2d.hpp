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
     * 2D convolution layer for spatial feature extraction (images,
     * spectrograms, time-frequency tiles).
     *
     * Layout is NHWC (channel-last) to match CMSIS-NN / TFLite Micro
     * and to give good loop locality when iterating kernel positions
     * across input channels.
     *
     * Padding is VALID only (no zero padding at borders). Output
     * spatial size follows the same formula as Conv1D.
     *
     * Weights are packed per-filter:
     *   [filter0: KH*KW*Cin kernel values, bias, filter1: ...]
     *
     * All dimensions are compile-time template parameters with zero
     * dynamic allocation.
     *
     * @tparam ValueType   Numeric type (QValue or float/double)
     * @tparam H           Input height
     * @tparam W           Input width
     * @tparam InChannels  Input channel count
     * @tparam KH          Kernel height
     * @tparam KW          Kernel width
     * @tparam StrideH     Vertical stride (default 1)
     * @tparam StrideW     Horizontal stride (default 1)
     * @tparam NumFilters  Output channel count (default 1)
     */
    template<
        typename ValueType,
        size_t H,
        size_t W,
        size_t InChannels,
        size_t KH,
        size_t KW,
        size_t StrideH = 1,
        size_t StrideW = 1,
        size_t NumFilters = 1>
    class Conv2D
    {
    public:
        static const size_t OutputHeight = (H - KH) / StrideH + 1;
        static const size_t OutputWidth  = (W - KW) / StrideW + 1;
        static const size_t OutputSize   = OutputHeight * OutputWidth * NumFilters;
        static const size_t InputSize    = H * W * InChannels;

        static const size_t WeightsPerFilter = KH * KW * InChannels + 1; // +1 bias
        static const size_t TotalWeights     = NumFilters * WeightsPerFilter;

        Conv2D()
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

        /**
         * Forward pass. Input is NHWC: pixel (h, w, c) lives at
         * input[h*W*InChannels + w*InChannels + c]. Output uses the
         * same convention with NumFilters channels.
         */
        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const size_t ihStart = oh * StrideH;
                for (size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const size_t iwStart = ow * StrideW;
                    const size_t outPixelOffset = (oh * OutputWidth + ow) * NumFilters;

                    for (size_t f = 0; f < NumFilters; ++f)
                    {
                        const size_t weightOffset = f * WeightsPerFilter;
                        ValueType sum = mWeights[weightOffset + KH * KW * InChannels]; // bias

                        for (size_t kh = 0; kh < KH; ++kh)
                        {
                            const size_t ih = ihStart + kh;
                            for (size_t kw = 0; kw < KW; ++kw)
                            {
                                const size_t iw = iwStart + kw;
                                const size_t inPixelOffset = (ih * W + iw) * InChannels;
                                const size_t kPixelOffset = (kh * KW + kw) * InChannels;

                                for (size_t ci = 0; ci < InChannels; ++ci)
                                {
                                    sum += mWeights[weightOffset + kPixelOffset + ci]
                                         * input[inPixelOffset + ci];
                                }
                            }
                        }

                        output[outPixelOffset + f] = sum;
                    }
                }
            }
        }

        /**
         * Compute weight and bias gradients given output deltas and the
         * input that produced them. Does not compute input deltas; add a
         * separate backward() if upstream layers need them.
         */
        void computeGradients(ValueType const* const outputDeltas, ValueType const* const input)
        {
            for (size_t i = 0; i < TotalWeights; ++i)
            {
                mGradients[i] = ValueType(0);
            }

            for (size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const size_t ihStart = oh * StrideH;
                for (size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const size_t iwStart = ow * StrideW;
                    const size_t outPixelOffset = (oh * OutputWidth + ow) * NumFilters;

                    for (size_t f = 0; f < NumFilters; ++f)
                    {
                        const size_t weightOffset = f * WeightsPerFilter;
                        const ValueType delta = outputDeltas[outPixelOffset + f];

                        for (size_t kh = 0; kh < KH; ++kh)
                        {
                            const size_t ih = ihStart + kh;
                            for (size_t kw = 0; kw < KW; ++kw)
                            {
                                const size_t iw = iwStart + kw;
                                const size_t inPixelOffset = (ih * W + iw) * InChannels;
                                const size_t kPixelOffset = (kh * KW + kw) * InChannels;

                                for (size_t ci = 0; ci < InChannels; ++ci)
                                {
                                    mGradients[weightOffset + kPixelOffset + ci]
                                        += delta * input[inPixelOffset + ci];
                                }
                            }
                        }

                        mGradients[weightOffset + KH * KW * InChannels] += delta;
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

        ValueType getWeight(const size_t index) const { return mWeights[index]; }
        void setWeight(const size_t index, const ValueType& value) { mWeights[index] = value; }
        ValueType getGradient(const size_t index) const { return mGradients[index]; }

        /**
         * Access kernel weight at (filter, kh, kw, ci). Pass ci < 0 is
         * not supported; use getFilterBias() for the bias term.
         */
        ValueType getFilterWeight(size_t filter, size_t kh, size_t kw, size_t ci) const
        {
            return mWeights[filter * WeightsPerFilter + (kh * KW + kw) * InChannels + ci];
        }

        void setFilterWeight(size_t filter, size_t kh, size_t kw, size_t ci, const ValueType& value)
        {
            mWeights[filter * WeightsPerFilter + (kh * KW + kw) * InChannels + ci] = value;
        }

        ValueType getFilterBias(size_t filter) const
        {
            return mWeights[filter * WeightsPerFilter + KH * KW * InChannels];
        }

        void setFilterBias(size_t filter, const ValueType& value)
        {
            mWeights[filter * WeightsPerFilter + KH * KW * InChannels] = value;
        }

    private:
        ValueType mWeights[TotalWeights];
        ValueType mGradients[TotalWeights];

        static_assert(H >= KH, "Input height must be >= kernel height.");
        static_assert(W >= KW, "Input width must be >= kernel width.");
        static_assert(StrideH > 0, "Vertical stride must be > 0.");
        static_assert(StrideW > 0, "Horizontal stride must be > 0.");
        static_assert(KH > 0 && KW > 0, "Kernel dimensions must be > 0.");
        static_assert(InChannels > 0, "Input channels must be > 0.");
        static_assert(NumFilters > 0, "Number of filters must be > 0.");
    };
}
