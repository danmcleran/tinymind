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
#include "include/simd/simd_dispatch.hpp"

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
 * The InChannels reduction is contiguous in both the input pixel and the
 * weight row, so it goes through simd::dotProductWithZeroPoint and picks
 * up whichever TINYMIND_ENABLE_SIMD_* backend is enabled (scalar, and
 * byte-identical to the plain loop, when none is).
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
            // Snapshot the fields into locals. Every output store is an int8_t write,
            // which may alias *this, so member reads inside the loops would be
            // reloaded -- and the requantizer's shift re-decoded -- once per output.
            const WeightType* const layerWeights = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const Requantizer<AccumulatorType, OutputType> rq = requantizer;

            if (H_ * W_ == 1)
            {
                // A single pixel (a dense layer): nothing to amortize across
                // pixels, and holding the requantizer across the loop only
                // takes registers from the dot product.
                for (std::size_t f = 0; f < NumFilters_; ++f)
                {
                    const AccumulatorType bias = (layerBiases != nullptr)
                        ? layerBiases[f]
                        : static_cast<AccumulatorType>(0);
                    const AccumulatorType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumulatorType>(
                            input, layerWeights + f * WeightsPerFilter, InChannels_,
                            inputZeroPoint);
                    output[f] = rq.apply(acc);
                }
                return;
            }

            // Filter-outer: one filter's weight row, bias and requantizer stay
            // loaded while it sweeps every pixel, instead of being re-fetched
            // for each output. Same products and sums, so bit-exact with the
            // pixel-outer order.
            for (std::size_t f = 0; f < NumFilters_; ++f)
            {
                const WeightType* const filterWeights = layerWeights + f * WeightsPerFilter;
                const AccumulatorType bias = (layerBiases != nullptr)
                    ? layerBiases[f]
                    : static_cast<AccumulatorType>(0);

                for (std::size_t p = 0; p < H_ * W_; ++p)
                {
                    const AccumulatorType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumulatorType>(
                            input + p * InChannels_, filterWeights, InChannels_, inputZeroPoint);

                    output[p * NumFilters_ + f] = rq.apply(acc);
                }
            }
        }
    };

    /**
     * Per-output-channel variant of QPointwiseConv2D.
     *
     * 1x1 convolution with per-filter Requantizer (one per output
     * channel). The bias for filter f carries effective scale
     * input_scale * weight_scale[f]. Matches the TFLite per-channel
     * convention; same calibration code as QDepthwiseConv2D and
     * QConv2DPerChannel populates the requantizers array.
     */
    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumType_,
        typename OutputStorage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t InChannels_,
        std::size_t NumFilters_>
    struct QPointwiseConv2DPerChannel
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
        const Requantizer<AccumulatorType, OutputType>* requantizers; // [NumFilters]

        void forward(const InputType* input, OutputType* output) const
        {
            // Snapshot the fields into locals. Every output store is an int8_t write,
            // which may alias *this, so member reads inside the loops would be
            // reloaded -- and the requantizer's shift re-decoded -- once per output.
            const WeightType* const layerWeights = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const Requantizer<AccumulatorType, OutputType>* const rqs = requantizers;

            if (H_ * W_ == 1)
            {
                // A single pixel (a dense layer): nothing to amortize across
                // pixels, and holding the requantizer across the loop only
                // takes registers from the dot product.
                for (std::size_t f = 0; f < NumFilters_; ++f)
                {
                    const AccumulatorType bias = (layerBiases != nullptr)
                        ? layerBiases[f]
                        : static_cast<AccumulatorType>(0);
                    const AccumulatorType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumulatorType>(
                            input, layerWeights + f * WeightsPerFilter, InChannels_,
                            inputZeroPoint);
                    output[f] = rqs[f].apply(acc);
                }
                return;
            }

            // Filter-outer: one filter's weight row, bias and requantizer stay
            // loaded while it sweeps every pixel, instead of being re-fetched
            // for each output. Same products and sums, so bit-exact with the
            // pixel-outer order.
            for (std::size_t f = 0; f < NumFilters_; ++f)
            {
                const WeightType* const filterWeights = layerWeights + f * WeightsPerFilter;
                const AccumulatorType bias = (layerBiases != nullptr)
                    ? layerBiases[f]
                    : static_cast<AccumulatorType>(0);
                const Requantizer<AccumulatorType, OutputType> rq = rqs[f];

                for (std::size_t p = 0; p < H_ * W_; ++p)
                {
                    const AccumulatorType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumulatorType>(
                            input + p * InChannels_, filterWeights, InChannels_, inputZeroPoint);

                    output[p * NumFilters_ + f] = rq.apply(acc);
                }
            }
        }
    };

} // namespace tinymind
