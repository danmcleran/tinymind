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
 * Quantized 2D convolution (per-tensor weight scale, NHWC). VALID by
 * default; PadTop/PadBottom/PadLeft/PadRight pad the input with its zero
 * point (SAME padding, as TFLite does) without a separate QPad2D pass.
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

        /*
         * Stack budget for the band of padded input rows a padded convolution
         * keeps (KH rows of (W + PadLeft + PadRight) * InChannels elements).
         * A layer whose band is larger fails to compile; pad it with a
         * QPad2D pass instead, or raise the limit.
         */
#ifndef TINYMIND_CONV_PAD_BAND_MAX_BYTES
#define TINYMIND_CONV_PAD_BAND_MAX_BYTES 4096
#endif
        constexpr std::size_t kConvPadBandMaxBytes = TINYMIND_CONV_PAD_BAND_MAX_BYTES;

        /*
         * The KH input rows under one output row, as row pointers, so the
         * kernels below index window columns within a row and never step
         * from row to row themselves. Unpadded, they point into the input.
         * Padded, they point into a ring of KH padded rows filled from the
         * input as the output rows advance: each padded row is built once,
         * with the input zero point in the border (the value whose
         * contribution w * (x - zp) is zero, which is what TFLite pads
         * with), so the layer needs no padded copy of the whole input.
         */
        template<typename InputType, std::size_t W_, std::size_t C_, std::size_t KH_, std::size_t StrideH_>
        struct ConvRowsDirect
        {
            const InputType* input;

            ConvRowsDirect(const InputType* in, InputType) : input(in)
            {
            }

            void rows(std::size_t oh, const InputType** rowPointers) const
            {
                const InputType* const top = input + (oh * StrideH_) * W_ * C_;
                for (std::size_t kh = 0; kh < KH_; ++kh)
                {
                    rowPointers[kh] = top + kh * W_ * C_;
                }
            }
        };

        template<typename InputType, std::size_t H_, std::size_t W_, std::size_t C_,
                 std::size_t KH_, std::size_t StrideH_,
                 std::size_t PadTop_, std::size_t PadLeft_, std::size_t PadRight_>
        struct ConvRowsPadded
        {
            static constexpr std::size_t RowLength = (W_ + PadLeft_ + PadRight_) * C_;

            static_assert(KH_ * RowLength * sizeof(InputType) <= kConvPadBandMaxBytes,
                          "The padded-row band of this convolution exceeds TINYMIND_CONV_PAD_BAND_MAX_BYTES: "
                          "raise the limit, or pad with QPad2D and use an unpadded layer.");

            InputType band[KH_ * RowLength];
            const InputType* unpadded; // the layer input the padded rows are built from
            InputType zeroPoint;
            std::size_t next; // the next padded row to build

            ConvRowsPadded(const InputType* in, InputType zp) : unpadded(in), zeroPoint(zp), next(0)
            {
            }

            void fill(std::size_t r)
            {
                InputType* const dst = band + (r % KH_) * RowLength;
                if ((r < PadTop_) || (r >= PadTop_ + H_))
                {
                    for (std::size_t i = 0; i < RowLength; ++i)
                    {
                        dst[i] = zeroPoint;
                    }
                    return;
                }
                const InputType* const src = unpadded + (r - PadTop_) * W_ * C_;
                for (std::size_t i = 0; i < PadLeft_ * C_; ++i)
                {
                    dst[i] = zeroPoint;
                }
                for (std::size_t i = 0; i < W_ * C_; ++i)
                {
                    dst[PadLeft_ * C_ + i] = src[i];
                }
                for (std::size_t i = 0; i < PadRight_ * C_; ++i)
                {
                    dst[(PadLeft_ + W_) * C_ + i] = zeroPoint;
                }
            }

            void rows(std::size_t oh, const InputType** rowPointers)
            {
                const std::size_t top = oh * StrideH_;
                if (next < top)
                {
                    next = top; // a stride above KH skips rows no window reads
                }
                while (next < top + KH_)
                {
                    fill(next);
                    ++next;
                }
                for (std::size_t kh = 0; kh < KH_; ++kh)
                {
                    rowPointers[kh] = band + ((top + kh) % KH_) * RowLength;
                }
            }
        };

        // ConvRowsDirect for an unpadded layer, ConvRowsPadded otherwise. (Two
        // class names rather than a specialization: cppcheck attributes a
        // specialization's constructor to the primary template's members.)
        template<typename InputType, std::size_t H_, std::size_t W_, std::size_t C_,
                 std::size_t KH_, std::size_t StrideH_,
                 std::size_t PadTop_, std::size_t PadBottom_, std::size_t PadLeft_, std::size_t PadRight_,
                 bool Padded = ((PadTop_ | PadBottom_ | PadLeft_ | PadRight_) != 0)>
        struct ConvRowSource
        {
            typedef ConvRowsDirect<InputType, W_, C_, KH_, StrideH_> type;
        };

        template<typename InputType, std::size_t H_, std::size_t W_, std::size_t C_,
                 std::size_t KH_, std::size_t StrideH_,
                 std::size_t PadTop_, std::size_t PadBottom_, std::size_t PadLeft_, std::size_t PadRight_>
        struct ConvRowSource<InputType, H_, W_, C_, KH_, StrideH_, PadTop_, PadBottom_, PadLeft_, PadRight_, true>
        {
            typedef ConvRowsPadded<InputType, H_, W_, C_, KH_, StrideH_, PadTop_, PadLeft_, PadRight_> type;
        };

        // Gather from a window whose kernel rows are RowStride apart (an
        // unpadded input). The compile-time stride matters on small windows:
        // reading row pointers from the band's table instead cost the 20x20x1
        // 3x3 stem 2-14% at three bytes per row.
        template<typename InputType, std::size_t InChannels_, std::size_t KH_, std::size_t KW_,
                 std::size_t RowStride>
        inline void gatherConvPatchStrided(const InputType* window, InputType* patch)
        {
            constexpr std::size_t rowLength = KW_ * InChannels_;
            for (std::size_t kh = 0; kh < KH_; ++kh)
            {
                const InputType* const row = window + kh * RowStride;
                for (std::size_t i = 0; i < rowLength; ++i)
                {
                    patch[kh * rowLength + i] = row[i];
                }
            }
        }

        // Gather from KH row pointers (a padded layer's ring of rows).
        template<typename InputType, std::size_t InChannels_, std::size_t KH_, std::size_t KW_>
        inline void gatherConvPatch(const InputType* const* rows, std::size_t offset, InputType* patch)
        {
            constexpr std::size_t rowLength = KW_ * InChannels_;
            for (std::size_t kh = 0; kh < KH_; ++kh)
            {
                const InputType* const row = rows[kh] + offset;
                for (std::size_t i = 0; i < rowLength; ++i)
                {
                    patch[kh * rowLength + i] = row[i];
                }
            }
        }

        // One filter over a gathered tile: count outputs, NumFilters_ apart.
        // The output loop is written out twice, once with the requantizer's
        // FastRequantizer form and once with the Requantizer itself, and
        // chosen per filter, so neither relies on the optimizer unswitching
        // a branch. (Both live in one function rather than two template
        // instantiations so that a model whose filters all take one form
        // leaves no instantiated function unexecuted.)
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t NumFilters_, std::size_t WindowLength_, std::size_t Slot_>
        inline void convTileFilterSweep(const InputType* patches, std::size_t count,
                                        const WeightType* filterWeights, AccumType bias,
                                        InputType zeroPoint,
                                        const Requantizer<AccumType, OutputType>& rq,
                                        OutputType* out)
        {
            // The filter's weights go in a local: an int8 store through out
            // may alias the caller's weight array, and a local it cannot, so
            // the weights can stay in registers across the outputs.
            WeightType w[WindowLength_];

            FastRequantizer<OutputType> fast;
            if (fast.init(rq))
            {
                for (std::size_t i = 0; i < WindowLength_; ++i)
                {
                    w[i] = filterWeights[i];
                }
                for (std::size_t t = 0; t < count; ++t)
                {
                    const AccumType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                            patches + t * Slot_, w, WindowLength_, zeroPoint);
                    out[t * NumFilters_] = fast.apply(acc);
                }
            }
            else
            {
                for (std::size_t i = 0; i < WindowLength_; ++i)
                {
                    w[i] = filterWeights[i];
                }
                for (std::size_t t = 0; t < count; ++t)
                {
                    const AccumType acc = bias +
                        simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                            patches + t * Slot_, w, WindowLength_, zeroPoint);
                    out[t * NumFilters_] = rq.apply(acc);
                }
            }
        }

        // Patch path: kernel windows of one output-row tile gathered into
        // 4-byte-aligned slots, then a filter-outer sweep of the tile. Rqs
        // holds one Requantizer per filter, or a single per-tensor one.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t H_, std::size_t W_, std::size_t InChannels_, std::size_t KH_, std::size_t KW_,
                 std::size_t StrideH_, std::size_t StrideW_, std::size_t NumFilters_,
                 std::size_t OutputHeight, std::size_t OutputWidth,
                 std::size_t PadTop_, std::size_t PadBottom_, std::size_t PadLeft_, std::size_t PadRight_,
                 bool PerChannel>
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

            constexpr bool padded = (PadTop_ | PadBottom_ | PadLeft_ | PadRight_) != 0;

            alignas(alignof(InputType) > 4 ? alignof(InputType) : 4) InputType patches[tileMax * slot];
            typename ConvRowSource<InputType, H_, W_, InChannels_, KH_, StrideH_,
                                   PadTop_, PadBottom_, PadLeft_, PadRight_>::type source(input, zeroPoint);
            const InputType* rows[KH_];

            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                source.rows(oh, rows);
                for (std::size_t ow0 = 0; ow0 < OutputWidth; ow0 += tileMax)
                {
                    const std::size_t count =
                        (OutputWidth - ow0 < tileMax) ? (OutputWidth - ow0) : tileMax;

                    for (std::size_t t = 0; t < count; ++t)
                    {
                        const std::size_t offset = (ow0 + t) * StrideW_ * InChannels_;
                        if (padded)
                        {
                            gatherConvPatch<InputType, InChannels_, KH_, KW_>(rows, offset, patches + t * slot);
                        }
                        else
                        {
                            gatherConvPatchStrided<InputType, InChannels_, KH_, KW_, W_ * InChannels_>(
                                rows[0] + offset, patches + t * slot);
                        }
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

                        convTileFilterSweep<InputType, WeightType, AccumType, OutputType,
                                            NumFilters_, windowLength, slot>(
                            patches, count, filterWeights, bias, zeroPoint, rq, tileOut + f);
                    }
                }
            }
        }

        // Row path: one dot product per kernel row, read in place. Same int32
        // products as the per-tap loop, only the summation order differs, so
        // the result is bit-exact.
        template<typename InputType, typename WeightType, typename AccumType,
                 std::size_t InChannels_, std::size_t KH_, std::size_t KW_>
        inline AccumType convRowsDot(const InputType* const* rows, std::size_t offset,
                                     const WeightType* filterWeights, InputType zeroPoint)
        {
            constexpr std::size_t rowLength = KW_ * InChannels_;
            AccumType acc = static_cast<AccumType>(0);
            for (std::size_t kh = 0; kh < KH_; ++kh)
            {
                acc += simd::dotProductWithZeroPoint<InputType, WeightType, AccumType>(
                    rows[kh] + offset, filterWeights + kh * rowLength, rowLength, zeroPoint);
            }
            return acc;
        }

        // Two filters over one output row, two pixels at a time, through
        // simd::ConvRows2x2: one pass over the kernel rows computes four
        // outputs, so each load feeds two multiplies. The second window and
        // the second filter sit at compile-time distances from the first, so
        // the kernel addresses them as immediate offsets. An odd last pixel
        // takes the single-output row dot. Same products and sums, so
        // bit-exact.
        //
        // The pixel loop is written out twice, once with the filters'
        // FastRequantizer forms and once with their Requantizers, so neither
        // relies on the optimizer unswitching a branch. Both live in this one
        // function, not two instantiations, so that a model whose filters all
        // take one form leaves no instantiated function unexecuted.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t InChannels_, std::size_t KH_, std::size_t KW_,
                 std::size_t StrideW_, std::size_t NumFilters_, std::size_t OutputWidth>
        inline void convRowsFilterPairRow(const InputType* const* rows, OutputType* rowOut,
                                          const WeightType* w0,
                                          AccumType bias0, AccumType bias1, InputType zeroPoint,
                                          const Requantizer<AccumType, OutputType>& rq0,
                                          const Requantizer<AccumType, OutputType>& rq1)
        {
            constexpr std::size_t rowLength = KW_ * InChannels_;
            constexpr std::size_t windowLength = KH_ * rowLength;
            constexpr std::size_t pixelStep = StrideW_ * InChannels_;
            constexpr std::size_t pairedWidth = OutputWidth & ~static_cast<std::size_t>(1u);
            typedef simd::ConvRows2x2<InputType, WeightType, AccumType, KH_, rowLength, pixelStep, windowLength>
                Kernel;

            // Value-initialized: GCC cannot see that both are set whenever
            // the fast forms are used, and warns.
            FastRequantizer<OutputType> fast0 = {};
            FastRequantizer<OutputType> fast1 = {};
            const bool fast0Ready = fast0.init(rq0);
            const bool fast1Ready = fast1.init(rq1);
            if (fast0Ready && fast1Ready)
            {
                for (std::size_t ow = 0; ow < pairedWidth; ow += 2)
                {
                    AccumType acc[4] = {bias0, bias1, bias0, bias1};
                    Kernel::apply(rows, ow * pixelStep, w0, zeroPoint, acc);
                    OutputType* const o = rowOut + ow * NumFilters_;
                    o[0] = fast0.apply(acc[0]);
                    o[1] = fast1.apply(acc[1]);
                    o[NumFilters_] = fast0.apply(acc[2]);
                    o[NumFilters_ + 1] = fast1.apply(acc[3]);
                }
            }
            else
            {
                for (std::size_t ow = 0; ow < pairedWidth; ow += 2)
                {
                    AccumType acc[4] = {bias0, bias1, bias0, bias1};
                    Kernel::apply(rows, ow * pixelStep, w0, zeroPoint, acc);
                    OutputType* const o = rowOut + ow * NumFilters_;
                    o[0] = rq0.apply(acc[0]);
                    o[1] = rq1.apply(acc[1]);
                    o[NumFilters_] = rq0.apply(acc[2]);
                    o[NumFilters_ + 1] = rq1.apply(acc[3]);
                }
            }
            if (pairedWidth < OutputWidth)
            {
                const std::size_t ow = pairedWidth;
                OutputType* const o = rowOut + ow * NumFilters_;
                o[0] = rq0.apply(bias0 + convRowsDot<InputType, WeightType, AccumType, InChannels_, KH_, KW_>(
                                             rows, ow * pixelStep, w0, zeroPoint));
                o[1] = rq1.apply(bias1 + convRowsDot<InputType, WeightType, AccumType, InChannels_, KH_, KW_>(
                                             rows, ow * pixelStep, w0 + windowLength, zeroPoint));
            }
        }

        // Row path: kernel windows too big to gather into a patch are read
        // in place, one dot product per kernel row, filters in pairs (see
        // convRowsFilterPairRow). Rqs holds one Requantizer per filter, or a
        // single per-tensor one.
        template<typename InputType, typename WeightType, typename AccumType, typename OutputType,
                 std::size_t H_, std::size_t W_, std::size_t InChannels_, std::size_t KH_, std::size_t KW_,
                 std::size_t StrideH_, std::size_t StrideW_, std::size_t NumFilters_,
                 std::size_t OutputHeight, std::size_t OutputWidth,
                 std::size_t PadTop_, std::size_t PadBottom_, std::size_t PadLeft_, std::size_t PadRight_,
                 bool PerChannel>
        inline void convForwardRows(const InputType* input, OutputType* output,
                                    const WeightType* weights, const AccumType* biases,
                                    InputType zeroPoint,
                                    const Requantizer<AccumType, OutputType>* rqs)
        {
            constexpr std::size_t windowLength = KH_ * KW_ * InChannels_;
            constexpr std::size_t pixelStep = StrideW_ * InChannels_;

            typename ConvRowSource<InputType, H_, W_, InChannels_, KH_, StrideH_,
                                   PadTop_, PadBottom_, PadLeft_, PadRight_>::type source(input, zeroPoint);
            const InputType* rows[KH_];

            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                source.rows(oh, rows);
                OutputType* const rowOut = output + oh * OutputWidth * NumFilters_;

                // Filter pairs are independent, so the OpenMP build splits them across threads.
                TINYMIND_PARALLEL_FOR_OUTER
                for (std::size_t pair = 0; pair < NumFilters_ / 2; ++pair)
                {
                    const std::size_t f = 2 * pair;
                    const AccumType bias0 = (biases != nullptr) ? biases[f] : static_cast<AccumType>(0);
                    const AccumType bias1 = (biases != nullptr) ? biases[f + 1] : static_cast<AccumType>(0);
                    const Requantizer<AccumType, OutputType> rq0 = rqs[PerChannel ? f : 0];
                    const Requantizer<AccumType, OutputType> rq1 = rqs[PerChannel ? f + 1 : 0];
                    const WeightType* const w0 = weights + f * windowLength;

                    convRowsFilterPairRow<InputType, WeightType, AccumType, OutputType, InChannels_,
                                          KH_, KW_, StrideW_, NumFilters_, OutputWidth>(
                        rows, rowOut + f, w0, bias0, bias1, zeroPoint, rq0, rq1);
                }
                if ((NumFilters_ & 1u) != 0)
                {
                    // Odd filter count: the last filter alone.
                    const std::size_t f = NumFilters_ - 1;
                    const AccumType bias = (biases != nullptr) ? biases[f] : static_cast<AccumType>(0);
                    const Requantizer<AccumType, OutputType> rq = rqs[PerChannel ? f : 0];
                    for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                    {
                        rowOut[ow * NumFilters_ + f] = rq.apply(bias +
                            convRowsDot<InputType, WeightType, AccumType, InChannels_, KH_, KW_>(
                                rows, ow * pixelStep, weights + f * windowLength, zeroPoint));
                    }
                }
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
        std::size_t KH_,
        std::size_t KW_,
        std::size_t StrideH_ = 1,
        std::size_t StrideW_ = 1,
        std::size_t NumFilters_ = 1,
        std::size_t PadTop_ = 0,
        std::size_t PadBottom_ = 0,
        std::size_t PadLeft_ = 0,
        std::size_t PadRight_ = 0>
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
        static constexpr std::size_t PadTop        = PadTop_;
        static constexpr std::size_t PadBottom     = PadBottom_;
        static constexpr std::size_t PadLeft       = PadLeft_;
        static constexpr std::size_t PadRight      = PadRight_;

        static constexpr std::size_t OutputHeight = (H_ + PadTop_ + PadBottom_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ + PadLeft_ + PadRight_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * NumFilters_;
        static constexpr std::size_t InputSize    = H_ * W_ * InChannels_;
        static constexpr std::size_t WeightsPerFilter = KH_ * KW_ * InChannels_;
        static constexpr std::size_t TotalWeights     = NumFilters_ * WeightsPerFilter;
        static constexpr bool UsePatch =
            WeightsPerFilter * sizeof(InputStorage_) <= detail::kConvPatchMaxBytes;

        static_assert(H_ + PadTop_ + PadBottom_ >= KH_, "Padded input height must be >= kernel height.");
        static_assert(W_ + PadLeft_ + PadRight_ >= KW_, "Padded input width must be >= kernel width.");
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
                                              H_, W_, InChannels_, KH_, KW_, StrideH_, StrideW_,
                                              NumFilters_, OutputHeight, OutputWidth,
                                              PadTop_, PadBottom_, PadLeft_, PadRight_, false>(
                    input, output, layerWeights, layerBiases, inputZeroPoint, &rq);
                return;
            }

            detail::convForwardRows<InputType, WeightType, AccumulatorType, OutputType,
                                    H_, W_, InChannels_, KH_, KW_, StrideH_, StrideW_,
                                    NumFilters_, OutputHeight, OutputWidth,
                                    PadTop_, PadBottom_, PadLeft_, PadRight_, false>(
                input, output, layerWeights, layerBiases, inputZeroPoint, &rq);
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
        std::size_t NumFilters_ = 1,
        std::size_t PadTop_ = 0,
        std::size_t PadBottom_ = 0,
        std::size_t PadLeft_ = 0,
        std::size_t PadRight_ = 0>
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
        static constexpr std::size_t PadTop        = PadTop_;
        static constexpr std::size_t PadBottom     = PadBottom_;
        static constexpr std::size_t PadLeft       = PadLeft_;
        static constexpr std::size_t PadRight      = PadRight_;

        static constexpr std::size_t OutputHeight = (H_ + PadTop_ + PadBottom_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ + PadLeft_ + PadRight_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * NumFilters_;
        static constexpr std::size_t InputSize    = H_ * W_ * InChannels_;
        static constexpr std::size_t WeightsPerFilter = KH_ * KW_ * InChannels_;
        static constexpr std::size_t TotalWeights     = NumFilters_ * WeightsPerFilter;
        static constexpr bool UsePatch =
            WeightsPerFilter * sizeof(InputStorage_) <= detail::kConvPatchMaxBytes;

        static_assert(H_ + PadTop_ + PadBottom_ >= KH_, "Padded input height must be >= kernel height.");
        static_assert(W_ + PadLeft_ + PadRight_ >= KW_, "Padded input width must be >= kernel width.");
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
                                              H_, W_, InChannels_, KH_, KW_, StrideH_, StrideW_,
                                              NumFilters_, OutputHeight, OutputWidth,
                                              PadTop_, PadBottom_, PadLeft_, PadRight_, true>(
                    input, output, layerWeights, layerBiases, inputZeroPoint, rqs);
                return;
            }

            detail::convForwardRows<InputType, WeightType, AccumulatorType, OutputType,
                                    H_, W_, InChannels_, KH_, KW_, StrideH_, StrideW_,
                                    NumFilters_, OutputHeight, OutputWidth,
                                    PadTop_, PadBottom_, PadLeft_, PadRight_, true>(
                input, output, layerWeights, layerBiases, inputZeroPoint, rqs);
        }
    };

} // namespace tinymind
