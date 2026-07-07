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

#include "include/tinymind_traits.hpp"

#include <cstddef>

namespace tinymind {
    /**
     * 1D nearest-neighbour upsampling layer.
     *
     * Grows a sequence by an integer factor, repeating each input element
     * ScaleFactor times. The transpose (decoder) counterpart to MaxPool1D /
     * AvgPool1D in encoder-decoder pipelines (1D autoencoders, U-Net-1D
     * segmentation, denoising) where the decoder must restore the original
     * time resolution:
     *   Conv1D -> Pool1D -> ... -> Upsample1D -> Conv1D
     *
     * A cheaper, artifact-free alternative to a transposed convolution
     * (Upsample + Conv1D avoids the checkerboard patterns learned upsampling
     * can introduce). Uses no arithmetic, so it holds for every ValueType at
     * TINYMIND_ENABLE_FLOAT=0 / STD=0.
     *
     * All dimensions are compile-time template parameters with zero dynamic
     * allocation. Layout is channel-major, matching Conv1D / Pool1D:
     * [ch0_t0, ch0_t1, ..., ch1_t0, ch1_t1, ...].
     *
     * @tparam ValueType     Numeric type (QValue or float/double)
     * @tparam InputLength   Number of input elements per channel
     * @tparam ScaleFactor   Integer upsampling factor (output length = InputLength * ScaleFactor)
     * @tparam NumChannels   Number of input channels (default 1)
     */
    template<
        typename ValueType,
        size_t InputLength,
        size_t ScaleFactor,
        size_t NumChannels = 1>
    class UpsampleNearest1D
    {
    public:
        static const size_t OutputLength = InputLength * ScaleFactor;
        static const size_t OutputSize = NumChannels * OutputLength;
        static const size_t InputSize = NumChannels * InputLength;

        /**
         * Forward pass: repeat each input element ScaleFactor times.
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

                for (size_t i = 0; i < InputLength; ++i)
                {
                    const ValueType value = input[inputOffset + i];

                    for (size_t k = 0; k < ScaleFactor; ++k)
                    {
                        output[outputOffset + i * ScaleFactor + k] = value;
                    }
                }
            }
        }

        /**
         * Backward pass: accumulate the gradients of the ScaleFactor outputs
         * that each input element produced back onto that element.
         *
         * @param outputDeltas Array of OutputSize gradient values
         * @param inputDeltas  Array of InputSize gradient values (zeroed then populated)
         */
        void backward(ValueType const* const outputDeltas, ValueType* inputDeltas) const
        {
            for (size_t ch = 0; ch < NumChannels; ++ch)
            {
                const size_t inputOffset = ch * InputLength;
                const size_t outputOffset = ch * OutputLength;

                for (size_t i = 0; i < InputLength; ++i)
                {
                    ValueType sum = ValueType(0);

                    for (size_t k = 0; k < ScaleFactor; ++k)
                    {
                        sum += outputDeltas[outputOffset + i * ScaleFactor + k];
                    }

                    inputDeltas[inputOffset + i] = sum;
                }
            }
        }

    private:
        static_assert(InputLength > 0, "Input length must be > 0.");
        static_assert(ScaleFactor > 0, "Scale factor must be > 0.");
        static_assert(NumChannels > 0, "Number of channels must be > 0.");
    };

    /**
     * 1D linear-interpolation upsampling layer.
     *
     * Grows a sequence by an integer factor, filling the gaps with a straight
     * ramp between neighbouring input elements instead of repeating them.
     * Input element k lands exactly on output index k * ScaleFactor; the
     * ScaleFactor-1 outputs between two input nodes are their weighted blend.
     * The tail beyond the last input node holds the last value (edge clamp),
     * so the output length stays InputLength * ScaleFactor.
     *
     * Smoother than UpsampleNearest1D at the cost of a multiply per output
     * element; requires ValueType arithmetic (QValue or float/double), the
     * same class of type Conv1D / AvgPool1D operate on.
     *
     * @tparam ValueType     Numeric type (QValue or float/double)
     * @tparam InputLength   Number of input elements per channel
     * @tparam ScaleFactor   Integer upsampling factor (output length = InputLength * ScaleFactor)
     * @tparam NumChannels   Number of input channels (default 1)
     */
    template<
        typename ValueType,
        size_t InputLength,
        size_t ScaleFactor,
        size_t NumChannels = 1>
    class UpsampleLinear1D
    {
    public:
        static const size_t OutputLength = InputLength * ScaleFactor;
        static const size_t OutputSize = NumChannels * OutputLength;
        static const size_t InputSize = NumChannels * InputLength;

        /**
         * Forward pass: linearly interpolate between neighbouring input nodes.
         *
         * output[o] = input[i0] + (input[i1] - input[i0]) * (k / ScaleFactor)
         * where i0 = o / ScaleFactor, k = o % ScaleFactor, and i1 is i0 + 1
         * clamped to the last input element.
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

                for (size_t o = 0; o < OutputLength; ++o)
                {
                    const size_t i0 = o / ScaleFactor;
                    const size_t k = o % ScaleFactor;
                    const size_t i1 = (i0 + 1 < InputLength) ? (i0 + 1) : i0;

                    const ValueType lo = input[inputOffset + i0];
                    const ValueType hi = input[inputOffset + i1];

                    output[outputOffset + o] = lo + (hi - lo) * weight(k);
                }
            }
        }

        /**
         * Backward pass: split each output gradient between the two input
         * nodes it interpolated, by the same weights used in the forward pass
         * (the transpose of forward).
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

                for (size_t o = 0; o < OutputLength; ++o)
                {
                    const size_t i0 = o / ScaleFactor;
                    const size_t k = o % ScaleFactor;
                    const size_t i1 = (i0 + 1 < InputLength) ? (i0 + 1) : i0;

                    const ValueType w = weight(k);
                    const ValueType grad = outputDeltas[outputOffset + o];

                    inputDeltas[inputOffset + i0] += grad * (one() - w);
                    inputDeltas[inputOffset + i1] += grad * w;
                }
            }
        }

    private:
        // Interpolation weight k / ScaleFactor as ValueType. For floating-point
        // a cast is enough; for QValue the (FixedPart, FractionalPart)
        // constructor is required because QValue(int) treats its argument as
        // raw bits -- so the whole numbers are built first, then divided.
        template<typename T = ValueType>
        static typename tinymind::enable_if<tinymind::is_floating_point<T>::value, T>::type
        weight(const size_t k)
        {
            return static_cast<T>(k) / static_cast<T>(ScaleFactor);
        }

        template<typename T = ValueType>
        static typename tinymind::enable_if<!tinymind::is_floating_point<T>::value, T>::type
        weight(const size_t k)
        {
            return T(static_cast<typename T::FixedPartFieldType>(k), 0u) /
                   T(static_cast<typename T::FixedPartFieldType>(ScaleFactor), 0u);
        }

        template<typename T = ValueType>
        static typename tinymind::enable_if<tinymind::is_floating_point<T>::value, T>::type
        one()
        {
            return static_cast<T>(1);
        }

        template<typename T = ValueType>
        static typename tinymind::enable_if<!tinymind::is_floating_point<T>::value, T>::type
        one()
        {
            return T(static_cast<typename T::FixedPartFieldType>(1), 0u);
        }

        static_assert(InputLength > 0, "Input length must be > 0.");
        static_assert(ScaleFactor > 0, "Scale factor must be > 0.");
        static_assert(NumChannels > 0, "Number of channels must be > 0.");
    };
}
