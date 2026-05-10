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

#include "include/tinymind_platform.hpp"
#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Per-tensor quantized 1x1 ("pointwise") convolution (NHWC).
 *
 * Pointwise conv is the channel-mixing half of a MobileNet depthwise-
 * separable block: for each output spatial location it reduces over
 * InChannels with a learnable [NumFilters][InChannels] weight matrix.
 * Strides and kernel size do not apply (the kernel is 1x1).
 *
 * TFLite allows per-tensor or per-channel weight scales here; per-tensor
 * is the default and what this layer implements. Per-channel pointwise
 * can be added later by templating over a Requantizer-array variant
 * analogous to QDepthwiseConv2D.
 *
 * Tensor layout:
 *
 *   input    NHWC               [H][W][InChannels]
 *   weights  [NumFilters][InChannels]  row-major, one row per output
 *                                       channel
 *   biases   [NumFilters]       int32 with effective scale
 *                                input_scale * weight_scale
 *   output   NHWC               [H][W][NumFilters]
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumType_,
        typename OutputStorage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t InChannels_,
        std::size_t NumFilters_>
    struct QPointwiseConv2D
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputHeight   = H_;
        static constexpr std::size_t InputWidth    = W_;
        static constexpr std::size_t InputChannels = InChannels_;
        static constexpr std::size_t NumFilters    = NumFilters_;

        static constexpr std::size_t OutputHeight = H_;
        static constexpr std::size_t OutputWidth  = W_;
        static constexpr std::size_t OutputSize   = H_ * W_ * NumFilters_;
        static constexpr std::size_t InputSize    = H_ * W_ * InChannels_;
        static constexpr std::size_t WeightsPerFilter = InChannels_;
        static constexpr std::size_t TotalWeights     = NumFilters_ * WeightsPerFilter;

        static_assert(H_ > 0 && W_ > 0, "Spatial dims must be > 0.");
        static_assert(InChannels_ > 0, "Input channels must be > 0.");
        static_assert(NumFilters_ > 0, "Number of filters must be > 0.");

        const WeightType* weights;
        const AccumulatorType* biases;
        InputType input_zero_point;
        Requantizer<AccumulatorType, OutputType> requantizer;

        void forward(const InputType* input, OutputType* output) const
        {
            for (std::size_t h = 0; h < H_; ++h)
            {
                for (std::size_t w = 0; w < W_; ++w)
                {
                    const std::size_t inPixelOffset  = (h * W_ + w) * InChannels_;
                    const std::size_t outPixelOffset = (h * W_ + w) * NumFilters_;

                    for (std::size_t f = 0; f < NumFilters_; ++f)
                    {
                        const std::size_t weightOffset = f * WeightsPerFilter;
                        AccumulatorType acc = (biases != nullptr)
                            ? biases[f]
                            : static_cast<AccumulatorType>(0);

                        for (std::size_t ci = 0; ci < InChannels_; ++ci)
                        {
                            const AccumulatorType x =
                                static_cast<AccumulatorType>(input[inPixelOffset + ci]) -
                                static_cast<AccumulatorType>(input_zero_point);
                            const AccumulatorType wv =
                                static_cast<AccumulatorType>(weights[weightOffset + ci]);
                            acc += wv * x;
                        }

                        output[outPixelOffset + f] = requantizer.apply(acc);
                    }
                }
            }
        }
    };

} // namespace tinymind
