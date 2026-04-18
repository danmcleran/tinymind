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
     * 2D max pooling layer for spatial downsampling.
     *
     * NHWC layout. All dimensions compile-time, zero dynamic allocation.
     * Tracks argmax indices per output element for backpropagation.
     *
     * @tparam ValueType   Numeric type (QValue or float/double)
     * @tparam H           Input height
     * @tparam W           Input width
     * @tparam Channels    Channel count
     * @tparam PoolH       Pool window height
     * @tparam PoolW       Pool window width
     * @tparam StrideH     Vertical stride (default PoolH)
     * @tparam StrideW     Horizontal stride (default PoolW)
     */
    template<
        typename ValueType,
        size_t H,
        size_t W,
        size_t Channels,
        size_t PoolH,
        size_t PoolW,
        size_t StrideH = PoolH,
        size_t StrideW = PoolW>
    class MaxPool2D
    {
    public:
        static const size_t OutputHeight = (H - PoolH) / StrideH + 1;
        static const size_t OutputWidth  = (W - PoolW) / StrideW + 1;
        static const size_t OutputSize   = OutputHeight * OutputWidth * Channels;
        static const size_t InputSize    = H * W * Channels;

        MaxPool2D()
        {
            for (size_t i = 0; i < OutputSize; ++i)
            {
                mArgMaxIndices[i] = 0;
            }
        }

        void forward(ValueType const* const input, ValueType* output)
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
                        const size_t firstInIdx = (ihStart * W + iwStart) * Channels + c;
                        ValueType maxVal = input[firstInIdx];
                        size_t maxIdx = firstInIdx;

                        for (size_t kh = 0; kh < PoolH; ++kh)
                        {
                            for (size_t kw = 0; kw < PoolW; ++kw)
                            {
                                const size_t inIdx = ((ihStart + kh) * W + (iwStart + kw)) * Channels + c;
                                if (input[inIdx] > maxVal)
                                {
                                    maxVal = input[inIdx];
                                    maxIdx = inIdx;
                                }
                            }
                        }

                        output[outPixelOffset + c] = maxVal;
                        mArgMaxIndices[outPixelOffset + c] = maxIdx;
                    }
                }
            }
        }

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

        size_t getArgMaxIndex(size_t outputIndex) const { return mArgMaxIndices[outputIndex]; }

    private:
        size_t mArgMaxIndices[OutputSize];

        static_assert(H >= PoolH, "Input height must be >= pool height.");
        static_assert(W >= PoolW, "Input width must be >= pool width.");
        static_assert(StrideH > 0 && StrideW > 0, "Strides must be > 0.");
        static_assert(PoolH > 0 && PoolW > 0, "Pool dims must be > 0.");
        static_assert(Channels > 0, "Channels must be > 0.");
    };

    /**
     * 2D average pooling layer. Uniform gradient distribution across
     * the window.
     */
    template<
        typename ValueType,
        size_t H,
        size_t W,
        size_t Channels,
        size_t PoolH,
        size_t PoolW,
        size_t StrideH = PoolH,
        size_t StrideW = PoolW>
    class AvgPool2D
    {
    public:
        static const size_t OutputHeight = (H - PoolH) / StrideH + 1;
        static const size_t OutputWidth  = (W - PoolW) / StrideW + 1;
        static const size_t OutputSize   = OutputHeight * OutputWidth * Channels;
        static const size_t InputSize    = H * W * Channels;
        static const size_t WindowArea   = PoolH * PoolW;

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
                        ValueType sum = ValueType(0);
                        for (size_t kh = 0; kh < PoolH; ++kh)
                        {
                            for (size_t kw = 0; kw < PoolW; ++kw)
                            {
                                const size_t inIdx = ((ihStart + kh) * W + (iwStart + kw)) * Channels + c;
                                sum += input[inIdx];
                            }
                        }
                        output[outPixelOffset + c] = sum / static_cast<ValueType>(WindowArea);
                    }
                }
            }
        }

        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t i = 0; i < InputSize; ++i)
            {
                inputDeltas[i] = ValueType(0);
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
                        const ValueType grad = outputDeltas[outPixelOffset + c]
                                             / static_cast<ValueType>(WindowArea);
                        for (size_t kh = 0; kh < PoolH; ++kh)
                        {
                            for (size_t kw = 0; kw < PoolW; ++kw)
                            {
                                const size_t inIdx = ((ihStart + kh) * W + (iwStart + kw)) * Channels + c;
                                inputDeltas[inIdx] += grad;
                            }
                        }
                    }
                }
            }
        }

    private:
        static_assert(H >= PoolH, "Input height must be >= pool height.");
        static_assert(W >= PoolW, "Input width must be >= pool width.");
        static_assert(StrideH > 0 && StrideW > 0, "Strides must be > 0.");
        static_assert(PoolH > 0 && PoolW > 0, "Pool dims must be > 0.");
        static_assert(Channels > 0, "Channels must be > 0.");
    };

    /**
     * Global average pooling: collapse H and W into a single value
     * per channel. Produces a Channels-length vector suitable for
     * feeding a final Dense layer. Replaces the big flatten-to-dense
     * matrix that dominates flash in small CNNs.
     */
    template<
        typename ValueType,
        size_t H,
        size_t W,
        size_t Channels>
    class GlobalAvgPool2D
    {
    public:
        static const size_t OutputSize = Channels;
        static const size_t InputSize  = H * W * Channels;
        static const size_t Area       = H * W;

        void forward(ValueType const* const input, ValueType* output) const
        {
            for (size_t c = 0; c < Channels; ++c)
            {
                output[c] = ValueType(0);
            }

            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    const size_t pixelOffset = (h * W + w) * Channels;
                    for (size_t c = 0; c < Channels; ++c)
                    {
                        output[c] += input[pixelOffset + c];
                    }
                }
            }

            for (size_t c = 0; c < Channels; ++c)
            {
                output[c] = output[c] / static_cast<ValueType>(Area);
            }
        }

        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t c = 0; c < Channels; ++c)
            {
                const ValueType grad = outputDeltas[c] / static_cast<ValueType>(Area);
                for (size_t h = 0; h < H; ++h)
                {
                    for (size_t w = 0; w < W; ++w)
                    {
                        inputDeltas[(h * W + w) * Channels + c] = grad;
                    }
                }
            }
        }

    private:
        static_assert(H > 0 && W > 0, "Spatial dimensions must be > 0.");
        static_assert(Channels > 0, "Channels must be > 0.");
    };
}
