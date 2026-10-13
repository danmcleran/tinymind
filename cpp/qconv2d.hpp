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
#include "include/simd/simd_dispatch.hpp"
#include "include/threading.hpp"
#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized 2D convolution (per-tensor weight scale, VALID padding, NHWC).
 *
 * Mirrors the float Conv2D shape parameters (H, W, InChannels, KH, KW,
 * StrideH, StrideW, NumFilters) but threads four storage types through
 * the template list so int8 inputs can multiply into int32 accumulators
 * without intermediate widening boilerplate at the call site.
 *
 * Tensor layout:
 *
 *   input    NHWC                  [H][W][InChannels]
 *   weights  OHWI                  [NumFilters][KH][KW][InChannels]
 *   biases   per output channel    [NumFilters] in AccumType (typically
 *                                   int32 with effective scale equal to
 *                                   input_scale * weight_scale)
 *   output   NHWC                  [OutputHeight][OutputWidth][NumFilters]
 *
 * Per-output-channel quantization is deferred to qdepthwiseconv2d.hpp /
 * qpointwiseconv2d.hpp in the next phase. This layer carries a single
 * Requantizer (per-tensor weight scale) sized to the destination storage
 * type. Input zero_point is subtracted during the MAC; weight zero_point
 * is implicitly 0 (symmetric weights, the TFLite convention).
 *
 * Pure integer at runtime: no float, no <cmath>, no stdlib. Compiles in
 * the freestanding (FLOAT=0, STD=0) configuration.
 */

namespace tinymind {

    namespace detail {

        /*
         * Largest kernel window (KH * KW * InChannels bytes) that the conv
         * layers copy into a contiguous on-stack patch. Under one kernel
         * window the NHWC input is KH runs of KW * InChannels contiguous
         * elements, so a window up to this size is gathered into a patch
         * and every filter runs a single dot product over it. Bigger
         * windows skip the copy (bounding the stack cost) and run one dot
         * product per kernel row instead.
         */
        constexpr std::size_t kConvPatchMaxBytes = 64;

        /*
         * Stack budget for a tile of gathered patches. The patch path gathers
         * the windows of up to a whole output row at once, then sweeps the
         * tile filter by filter so each filter's weights, bias and requantizer
         * stay loaded across the tile instead of being re-fetched per output.
         * A tile that spans the whole output row is fastest (every tile then
         * has the same compile-time length); the default fits a whole row for
         * small stems such as an 18-wide 3x3x1 layer in 12-byte slots. Define
         * TINYMIND_CONV_TILE_MAX_BYTES lower on stack-tight parts: 64 keeps
         * most of the -O3 gain, and 0 (one pixel per tile) is the smallest
         * stack but slower than no tiling at all.
         */
#ifndef TINYMIND_CONV_TILE_MAX_BYTES
#define TINYMIND_CONV_TILE_MAX_BYTES 256
#endif
        constexpr std::size_t kConvTileMaxBytes = TINYMIND_CONV_TILE_MAX_BYTES;

        template<typename InputType, std::size_t W_, std::size_t InChannels_,
                 std::size_t KH_, std::size_t KW_>
        inline void gatherConvPatch(const InputType* window, InputType* patch)
        {
            constexpr std::size_t rowLength = KW_ * InChannels_;
            for (std::size_t kh = 0; kh < KH_; ++kh)
            {
                const InputType* row = window + kh * W_ * InChannels_;
                for (std::size_t i = 0; i < rowLength; ++i)
                {
                    patch[kh * rowLength + i] = row[i];
                }
            }
        }

        // Patch path: kernel windows of one output-row tile gathered into
        // 4-byte-aligned slots, then a filter-outer sweep of the tile. Rqs
        // holds one Requantizer per filter, or a single per-tensor one.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t W_, std::size_t InChannels_, std::size_t KH_, std::size_t KW_,
                 std::size_t StrideH_, std::size_t StrideW_, std::size_t NumFilters_,
                 std::size_t OutputHeight, std::size_t OutputWidth, bool PerChannel>
        inline void convForwardPatchTiles(const InputType* input, OutputType* output,
                                          const WeightType* weights, const AccumType* biases,
                                          InputType zeroPoint,
                                          const Requantizer<AccumType, OutputType>* rqs)
        {
            constexpr std::size_t windowLength = KH_ * KW_ * InChannels_;
            constexpr std::size_t slot = (windowLength + 3u) & ~static_cast<std::size_t>(3u);
            constexpr std::size_t tileCap = kConvTileMaxBytes / (slot * sizeof(InputType));
            constexpr std::size_t tileMax =
                (tileCap == 0) ? 1 : ((OutputWidth < tileCap) ? OutputWidth : tileCap);

            alignas(alignof(InputType) > 4 ? alignof(InputType) : 4) InputType patches[tileMax * slot];

            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                for (std::size_t ow0 = 0; ow0 < OutputWidth; ow0 += tileMax)
                {
                    const std::size_t count =
                        (OutputWidth - ow0 < tileMax) ? (OutputWidth - ow0) : tileMax;

                    for (std::size_t t = 0; t < count; ++t)
                    {
                        const InputType* const window =
                            input + ((oh * StrideH_) * W_ + (ow0 + t) * StrideW_) * InChannels_;
                        gatherConvPatch<InputType, W_, InChannels_, KH_, KW_>(
                            window, patches + t * slot);
                    }

                    OutputType* const tileOut = output + (oh * OutputWidth + ow0) * NumFilters_;

                    TINYMIND_PARALLEL_FOR_OUTER
                    for (std::size_t f = 0; f < NumFilters_; ++f)
                    {
                        const WeightType* const filterWeights = weights + f * windowLength;
                        const AccumType bias = (biases != nullptr)
                            ? biases[f]
                            : static_cast<AccumType>(0);
                        const Requantizer<AccumType, OutputType> rq = rqs[PerChannel ? f : 0];

                        for (std::size_t t = 0; t < count; ++t)
                        {
                            const AccumType acc = bias +
                                simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                                    patches + t * slot, filterWeights, windowLength, zeroPoint);
                            tileOut[t * NumFilters_ + f] = rq.apply(acc);
                        }
                    }
                }
            }
        }

        // Row path: one dot product per kernel row, read in place. Same int32
        // products as the per-tap loop, only the summation order differs, so
        // the result is bit-exact.
        template<typename InputType, typename WeightType, typename AccumType,
                 std::size_t W_, std::size_t InChannels_, std::size_t KH_, std::size_t KW_>
        inline AccumType convRowsDot(const InputType* window, const WeightType* filterWeights,
                                     InputType zeroPoint)
        {
            constexpr std::size_t rowLength = KW_ * InChannels_;
            AccumType acc = static_cast<AccumType>(0);
            for (std::size_t kh = 0; kh < KH_; ++kh)
            {
                acc += simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                    window + kh * W_ * InChannels_, filterWeights + kh * rowLength,
                    rowLength, zeroPoint);
            }
            return acc;
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
        std::size_t KH_,
        std::size_t KW_,
        std::size_t StrideH_ = 1,
        std::size_t StrideW_ = 1,
        std::size_t NumFilters_ = 1>
    struct QConv2D
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputHeight   = H_;
        static constexpr std::size_t InputWidth    = W_;
        static constexpr std::size_t InputChannels = InChannels_;
        static constexpr std::size_t KernelHeight  = KH_;
        static constexpr std::size_t KernelWidth   = KW_;
        static constexpr std::size_t StrideH       = StrideH_;
        static constexpr std::size_t StrideW       = StrideW_;
        static constexpr std::size_t NumFilters    = NumFilters_;

        static constexpr std::size_t OutputHeight = (H_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * NumFilters_;
        static constexpr std::size_t InputSize    = H_ * W_ * InChannels_;
        static constexpr std::size_t WeightsPerFilter = KH_ * KW_ * InChannels_;
        static constexpr std::size_t TotalWeights     = NumFilters_ * WeightsPerFilter;
        static constexpr bool UsePatch =
            WeightsPerFilter * sizeof(InputStorage_) <= detail::kConvPatchMaxBytes;

        static_assert(H_ >= KH_, "Input height must be >= kernel height.");
        static_assert(W_ >= KW_, "Input width must be >= kernel width.");
        static_assert(StrideH_ > 0, "Vertical stride must be > 0.");
        static_assert(StrideW_ > 0, "Horizontal stride must be > 0.");
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

            if (UsePatch)
            {
                detail::convForwardPatchTiles<InputType, WeightType, AccumulatorType, OutputType,
                                              W_, InChannels_, KH_, KW_, StrideH_, StrideW_,
                                              NumFilters_, OutputHeight, OutputWidth, false>(
                    input, output, layerWeights, layerBiases, inputZeroPoint, &rq);
                return;
            }

            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const std::size_t ihStart = oh * StrideH_;
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t iwStart = ow * StrideW_;
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * NumFilters_;

                    const InputType* const window =
                        input + (ihStart * W_ + iwStart) * InChannels_;

                    TINYMIND_PARALLEL_FOR_OUTER
                    for (std::size_t f = 0; f < NumFilters_; ++f)
                    {
                        AccumulatorType acc = (layerBiases != nullptr)
                            ? layerBiases[f]
                            : static_cast<AccumulatorType>(0);

                        acc += detail::convRowsDot<InputType, WeightType, AccumulatorType,
                                                   W_, InChannels_, KH_, KW_>(
                            window, layerWeights + f * WeightsPerFilter, inputZeroPoint);

                        output[outPixelOffset + f] = rq.apply(acc);
                    }
                }
            }
        }
    };

    /**
     * Per-output-channel weight scale variant of QConv2D.
     *
     * Identical shape/template parameters; the only difference is the
     * requantizers field (an array of NumFilters Requantizers, one per
     * output channel) replacing the single requantizer field. This
     * matches the TFLite per-channel convention for full conv: each
     * filter has its own scale because trained weight ranges vary
     * dramatically across filters, and a single per-tensor scale would
     * crush dynamic range. The bias for filter f carries effective
     * scale input_scale * weight_scale[f] so it still adds cleanly into
     * the int32 accumulator before requantization.
     *
     * Pointer-to-array layout matches QDepthwiseConv2D so the same
     * host-side calibration helpers (computePerChannelSymmetricScales +
     * buildRequantizer per channel) drive both layers.
     */
    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumType_,
        typename OutputStorage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t InChannels_,
        std::size_t KH_,
        std::size_t KW_,
        std::size_t StrideH_ = 1,
        std::size_t StrideW_ = 1,
        std::size_t NumFilters_ = 1>
    struct QConv2DPerChannel
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputHeight   = H_;
        static constexpr std::size_t InputWidth    = W_;
        static constexpr std::size_t InputChannels = InChannels_;
        static constexpr std::size_t KernelHeight  = KH_;
        static constexpr std::size_t KernelWidth   = KW_;
        static constexpr std::size_t StrideH       = StrideH_;
        static constexpr std::size_t StrideW       = StrideW_;
        static constexpr std::size_t NumFilters    = NumFilters_;

        static constexpr std::size_t OutputHeight = (H_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * NumFilters_;
        static constexpr std::size_t InputSize    = H_ * W_ * InChannels_;
        static constexpr std::size_t WeightsPerFilter = KH_ * KW_ * InChannels_;
        static constexpr std::size_t TotalWeights     = NumFilters_ * WeightsPerFilter;
        static constexpr bool UsePatch =
            WeightsPerFilter * sizeof(InputStorage_) <= detail::kConvPatchMaxBytes;

        static_assert(H_ >= KH_, "Input height must be >= kernel height.");
        static_assert(W_ >= KW_, "Input width must be >= kernel width.");
        static_assert(StrideH_ > 0, "Vertical stride must be > 0.");
        static_assert(StrideW_ > 0, "Horizontal stride must be > 0.");
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

            if (UsePatch)
            {
                detail::convForwardPatchTiles<InputType, WeightType, AccumulatorType, OutputType,
                                              W_, InChannels_, KH_, KW_, StrideH_, StrideW_,
                                              NumFilters_, OutputHeight, OutputWidth, true>(
                    input, output, layerWeights, layerBiases, inputZeroPoint, rqs);
                return;
            }

            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const std::size_t ihStart = oh * StrideH_;
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t iwStart = ow * StrideW_;
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * NumFilters_;

                    const InputType* const window =
                        input + (ihStart * W_ + iwStart) * InChannels_;

                    TINYMIND_PARALLEL_FOR_OUTER
                    for (std::size_t f = 0; f < NumFilters_; ++f)
                    {
                        AccumulatorType acc = (layerBiases != nullptr)
                            ? layerBiases[f]
                            : static_cast<AccumulatorType>(0);

                        acc += detail::convRowsDot<InputType, WeightType, AccumulatorType,
                                                   W_, InChannels_, KH_, KW_>(
                            window, layerWeights + f * WeightsPerFilter, inputZeroPoint);

                        output[outPixelOffset + f] = rqs[f].apply(acc);
                    }
                }
            }
        }
    };

} // namespace tinymind
