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
            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const std::size_t ihStart = oh * StrideH_;
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t iwStart = ow * StrideW_;
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * NumFilters_;

                    for (std::size_t f = 0; f < NumFilters_; ++f)
                    {
                        const std::size_t weightOffset = f * WeightsPerFilter;
                        AccumulatorType acc = (biases != nullptr)
                            ? biases[f]
                            : static_cast<AccumulatorType>(0);

                        for (std::size_t kh = 0; kh < KH_; ++kh)
                        {
                            const std::size_t ih = ihStart + kh;
                            for (std::size_t kw = 0; kw < KW_; ++kw)
                            {
                                const std::size_t iw = iwStart + kw;
                                const std::size_t inPixelOffset =
                                    (ih * W_ + iw) * InChannels_;
                                const std::size_t kPixelOffset =
                                    (kh * KW_ + kw) * InChannels_;

                                for (std::size_t ci = 0; ci < InChannels_; ++ci)
                                {
                                    const AccumulatorType x =
                                        static_cast<AccumulatorType>(input[inPixelOffset + ci]) -
                                        static_cast<AccumulatorType>(input_zero_point);
                                    const AccumulatorType w =
                                        static_cast<AccumulatorType>(
                                            weights[weightOffset + kPixelOffset + ci]);
                                    acc += w * x;
                                }
                            }
                        }

                        output[outPixelOffset + f] = requantizer.apply(acc);
                    }
                }
            }
        }
    };

} // namespace tinymind
