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

    namespace detail {

        // One filter of a multi-pixel pointwise layer, swept over every
        // pixel. The output loop is written out twice, once with the
        // requantizer's FastRequantizer form and once with the Requantizer
        // itself, chosen per filter; both live in this one function so that
        // a model whose filters all take one form leaves nothing unexecuted.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t Pixels_, std::size_t InChannels_, std::size_t NumFilters_>
        inline void pointwiseFilterSweep(const InputType* input, const WeightType* filterWeights,
                                         AccumType bias, InputType zeroPoint,
                                         const Requantizer<AccumType, OutputType>& rq,
                                         OutputType* out)
        {
            FastRequantizer<OutputType> fast;
            if (fast.init(rq))
            {
                const FilterWeights<WeightType, InChannels_> w(filterWeights);
                for (std::size_t p = 0; p < Pixels_; ++p)
                {
                    const AccumType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                            input + p * InChannels_, w.data(), InChannels_, zeroPoint);
                    out[p * NumFilters_] = fast.apply(acc);
                }
            }
            else
            {
                const FilterWeights<WeightType, InChannels_> w(filterWeights);
                for (std::size_t p = 0; p < Pixels_; ++p)
                {
                    const AccumType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                            input + p * InChannels_, w.data(), InChannels_, zeroPoint);
                    out[p * NumFilters_] = rq.apply(acc);
                }
            }
        }

        /*
         * Input channels from which a multi-pixel pointwise layer runs filter
         * pairs over pixel pairs through simd::ConvRows2x2 (each load feeds
         * two multiplies), when the backend has a blocked kernel for it
         * (simd::kBlockedConvRows2x2). Below it, or without one, the
         * one-filter sweep above, with its weights held in a local, is as
         * fast or faster and its tuning is kept.
         */
        constexpr std::size_t kPointwise2x2MinChannels = 32;

        // Tag for the choice between the two multi-pixel forms, made at
        // compile time so the form a layer does not use is never instantiated.
        template<bool Blocked>
        struct PointwiseBlockedTag
        {
        };

        // Two filters over every pixel, two pixels at a time; an odd last
        // pixel takes two single dot products. As in convRowsFilterPairRow,
        // the pixel loop is written out once per requantizer form in this one
        // function. Bit-exact.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t Pixels_, std::size_t InChannels_, std::size_t NumFilters_>
        inline void pointwiseFilterPairSweep(const InputType* input, const WeightType* w0,
                                             AccumType bias0, AccumType bias1, InputType zeroPoint,
                                             const Requantizer<AccumType, OutputType>& rq0,
                                             const Requantizer<AccumType, OutputType>& rq1,
                                             OutputType* out)
        {
            constexpr std::size_t pairedPixels = Pixels_ & ~static_cast<std::size_t>(1u);
            typedef simd::ConvRows2x2<InputType, WeightType, AccumType, 1, InChannels_, InChannels_, InChannels_>
                Kernel;

            // Value-initialized: GCC cannot see that both are set whenever
            // the fast forms are used, and warns.
            FastRequantizer<OutputType> fast0 = {};
            FastRequantizer<OutputType> fast1 = {};
            const bool fast0Ready = fast0.init(rq0);
            const bool fast1Ready = fast1.init(rq1);
            if (fast0Ready && fast1Ready)
            {
                for (std::size_t p = 0; p < pairedPixels; p += 2)
                {
                    const InputType* const rows[1] = {input + p * InChannels_};
                    AccumType acc[4] = {bias0, bias1, bias0, bias1};
                    Kernel::apply(rows, 0, w0, zeroPoint, acc);
                    OutputType* const o = out + p * NumFilters_;
                    o[0] = fast0.apply(acc[0]);
                    o[1] = fast1.apply(acc[1]);
                    o[NumFilters_] = fast0.apply(acc[2]);
                    o[NumFilters_ + 1] = fast1.apply(acc[3]);
                }
            }
            else
            {
                for (std::size_t p = 0; p < pairedPixels; p += 2)
                {
                    const InputType* const rows[1] = {input + p * InChannels_};
                    AccumType acc[4] = {bias0, bias1, bias0, bias1};
                    Kernel::apply(rows, 0, w0, zeroPoint, acc);
                    OutputType* const o = out + p * NumFilters_;
                    o[0] = rq0.apply(acc[0]);
                    o[1] = rq1.apply(acc[1]);
                    o[NumFilters_] = rq0.apply(acc[2]);
                    o[NumFilters_ + 1] = rq1.apply(acc[3]);
                }
            }
            if (pairedPixels < Pixels_)
            {
                const InputType* const x = input + pairedPixels * InChannels_;
                OutputType* const o = out + pairedPixels * NumFilters_;
                o[0] = rq0.apply(bias0 + simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                                             x, w0, InChannels_, zeroPoint));
                o[1] = rq1.apply(bias1 + simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                                             x, w0 + InChannels_, InChannels_, zeroPoint));
            }
        }

        // Every filter of a multi-pixel pointwise layer: filter pairs through
        // pointwiseFilterPairSweep, an odd last filter through the one-filter
        // sweep. Rqs holds one Requantizer per filter, or a single per-tensor one.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t Pixels_, std::size_t InChannels_, std::size_t NumFilters_, bool PerChannel>
        inline void pointwiseForward2x2(const InputType* input, OutputType* output,
                                        const WeightType* weights, const AccumType* biases,
                                        InputType zeroPoint, const Requantizer<AccumType, OutputType>* rqs)
        {
            for (std::size_t pair = 0; pair < NumFilters_ / 2; ++pair)
            {
                const std::size_t f = 2 * pair;
                const AccumType bias0 = (biases != nullptr) ? biases[f] : static_cast<AccumType>(0);
                const AccumType bias1 = (biases != nullptr) ? biases[f + 1] : static_cast<AccumType>(0);
                const Requantizer<AccumType, OutputType> rq0 = rqs[PerChannel ? f : 0];
                const Requantizer<AccumType, OutputType> rq1 = rqs[PerChannel ? f + 1 : 0];
                pointwiseFilterPairSweep<InputType, WeightType, AccumType, OutputType,
                                         Pixels_, InChannels_, NumFilters_>(
                    input, weights + f * InChannels_, bias0, bias1, zeroPoint, rq0, rq1, output + f);
            }
            if ((NumFilters_ & 1u) != 0)
            {
                const std::size_t f = NumFilters_ - 1;
                const AccumType bias = (biases != nullptr) ? biases[f] : static_cast<AccumType>(0);
                pointwiseFilterSweep<InputType, WeightType, AccumType, OutputType,
                                     Pixels_, InChannels_, NumFilters_>(
                    input, weights + f * InChannels_, bias, zeroPoint, rqs[PerChannel ? f : 0], output + f);
            }
        }

    } // namespace detail

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

            forwardPixels(input, output, layerWeights, layerBiases, inputZeroPoint, rq,
                          detail::PointwiseBlockedTag<Blocked>());
        }

    private:
        static constexpr bool Blocked = simd::kBlockedConvRows2x2 &&
            (InChannels_ >= detail::kPointwise2x2MinChannels) && (NumFilters_ >= 2) && (H_ * W_ >= 2);

        // Filter pairs over pixel pairs through the backend's blocked kernel.
        void forwardPixels(const InputType* input, OutputType* output, const WeightType* layerWeights,
                           const AccumulatorType* layerBiases, InputType inputZeroPoint,
                           const Requantizer<AccumulatorType, OutputType>& rq,
                           detail::PointwiseBlockedTag<true>) const
        {
            detail::pointwiseForward2x2<InputType, WeightType, AccumulatorType, OutputType,
                                        H_ * W_, InChannels_, NumFilters_, false>(
                input, output, layerWeights, layerBiases, inputZeroPoint, &rq);
        }

        void forwardPixels(const InputType* input, OutputType* output, const WeightType* layerWeights,
                           const AccumulatorType* layerBiases, InputType inputZeroPoint,
                           const Requantizer<AccumulatorType, OutputType>& rq,
                           detail::PointwiseBlockedTag<false>) const
        {
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

                detail::pointwiseFilterSweep<InputType, WeightType, AccumulatorType, OutputType,
                                             H_ * W_, InChannels_, NumFilters_>(
                    input, filterWeights, bias, inputZeroPoint, rq, output + f);
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

            forwardPixels(input, output, layerWeights, layerBiases, inputZeroPoint, rqs,
                          detail::PointwiseBlockedTag<Blocked>());
        }

    private:
        static constexpr bool Blocked = simd::kBlockedConvRows2x2 &&
            (InChannels_ >= detail::kPointwise2x2MinChannels) && (NumFilters_ >= 2) && (H_ * W_ >= 2);

        // Filter pairs over pixel pairs through the backend's blocked kernel.
        void forwardPixels(const InputType* input, OutputType* output, const WeightType* layerWeights,
                           const AccumulatorType* layerBiases, InputType inputZeroPoint,
                           const Requantizer<AccumulatorType, OutputType>* rqs,
                           detail::PointwiseBlockedTag<true>) const
        {
            detail::pointwiseForward2x2<InputType, WeightType, AccumulatorType, OutputType,
                                        H_ * W_, InChannels_, NumFilters_, true>(
                input, output, layerWeights, layerBiases, inputZeroPoint, rqs);
        }

        void forwardPixels(const InputType* input, OutputType* output, const WeightType* layerWeights,
                           const AccumulatorType* layerBiases, InputType inputZeroPoint,
                           const Requantizer<AccumulatorType, OutputType>* rqs,
                           detail::PointwiseBlockedTag<false>) const
        {
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

                detail::pointwiseFilterSweep<InputType, WeightType, AccumulatorType, OutputType,
                                             H_ * W_, InChannels_, NumFilters_>(
                    input, filterWeights, bias, inputZeroPoint, rq, output + f);
            }
        }
    };

} // namespace tinymind
