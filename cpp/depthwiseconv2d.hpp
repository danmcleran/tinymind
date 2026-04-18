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
     * Depthwise 2D convolution: one kernel per input channel with no
     * cross-channel mixing. Output channel count equals input channel
     * count. Pairs with PointwiseConv2D to form a MobileNet-style
     * separable block at a fraction of the MACs of a full Conv2D.
     *
     * NHWC layout (channel-last), VALID padding.
     *
     * Weights are packed per-channel:
     *   [ch0: KH*KW kernel values, bias, ch1: ...]
     *
     * @tparam ValueType   Numeric type (QValue or float/double)
     * @tparam H           Input height
     * @tparam W           Input width
     * @tparam Channels    Input/output channel count
     * @tparam KH          Kernel height
     * @tparam KW          Kernel width
     * @tparam StrideH     Vertical stride (default 1)
     * @tparam StrideW     Horizontal stride (default 1)
     */
    template<
        typename ValueType,
        size_t H,
        size_t W,
        size_t Channels,
        size_t KH,
        size_t KW,
        size_t StrideH = 1,
        size_t StrideW = 1>
    class DepthwiseConv2D
    {
    public:
        static const size_t OutputHeight = (H - KH) / StrideH + 1;
        static const size_t OutputWidth  = (W - KW) / StrideW + 1;
        static const size_t OutputSize   = OutputHeight * OutputWidth * Channels;
        static const size_t InputSize    = H * W * Channels;

        static const size_t WeightsPerChannel = KH * KW + 1; // +1 bias
        static const size_t TotalWeights      = Channels * WeightsPerChannel;

        DepthwiseConv2D()
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
            for (size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const size_t ihStart = oh * StrideH;
                for (size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const size_t iwStart = ow * StrideW;
                    const size_t outPixelOffset = (oh * OutputWidth + ow) * Channels;

                    for (size_t c = 0; c < Channels; ++c)
                    {
                        const size_t weightOffset = c * WeightsPerChannel;
                        ValueType sum = mWeights[weightOffset + KH * KW]; // bias

                        for (size_t kh = 0; kh < KH; ++kh)
                        {
                            const size_t ih = ihStart + kh;
                            for (size_t kw = 0; kw < KW; ++kw)
                            {
                                const size_t iw = iwStart + kw;
                                const size_t inIdx = (ih * W + iw) * Channels + c;
                                sum += mWeights[weightOffset + kh * KW + kw] * input[inIdx];
                            }
                        }

                        output[outPixelOffset + c] = sum;
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

            for (size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const size_t ihStart = oh * StrideH;
                for (size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const size_t iwStart = ow * StrideW;
                    const size_t outPixelOffset = (oh * OutputWidth + ow) * Channels;

                    for (size_t c = 0; c < Channels; ++c)
                    {
                        const size_t weightOffset = c * WeightsPerChannel;
                        const ValueType delta = outputDeltas[outPixelOffset + c];

                        for (size_t kh = 0; kh < KH; ++kh)
                        {
                            const size_t ih = ihStart + kh;
                            for (size_t kw = 0; kw < KW; ++kw)
                            {
                                const size_t iw = iwStart + kw;
                                const size_t inIdx = (ih * W + iw) * Channels + c;
                                mGradients[weightOffset + kh * KW + kw] += delta * input[inIdx];
                            }
                        }

                        mGradients[weightOffset + KH * KW] += delta;
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

        ValueType getChannelWeight(size_t channel, size_t kh, size_t kw) const
        {
            return mWeights[channel * WeightsPerChannel + kh * KW + kw];
        }

        void setChannelWeight(size_t channel, size_t kh, size_t kw, const ValueType& value)
        {
            mWeights[channel * WeightsPerChannel + kh * KW + kw] = value;
        }

        ValueType getChannelBias(size_t channel) const
        {
            return mWeights[channel * WeightsPerChannel + KH * KW];
        }

        void setChannelBias(size_t channel, const ValueType& value)
        {
            mWeights[channel * WeightsPerChannel + KH * KW] = value;
        }

    private:
        ValueType mWeights[TotalWeights];
        ValueType mGradients[TotalWeights];

        static_assert(H >= KH, "Input height must be >= kernel height.");
        static_assert(W >= KW, "Input width must be >= kernel width.");
        static_assert(StrideH > 0, "Vertical stride must be > 0.");
        static_assert(StrideW > 0, "Horizontal stride must be > 0.");
        static_assert(KH > 0 && KW > 0, "Kernel dimensions must be > 0.");
        static_assert(Channels > 0, "Channels must be > 0.");
    };
}
