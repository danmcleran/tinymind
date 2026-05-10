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
 * Per-channel quantized depthwise 2D convolution (NHWC, VALID padding).
 *
 * Depthwise conv applies one KH*KW filter per channel and does NOT mix
 * channels. TFLite's int8 spec mandates per-channel weight scales here
 * because the dynamic range varies dramatically between channels in
 * trained MobileNet-style filters; collapsing to a single per-tensor
 * scale destroys accuracy.
 *
 * Per-channel means each channel carries its own (multiplier, shift,
 * output_zero_point, qmin, qmax) - exactly the Requantizer type from
 * qaffine.hpp - so the caller provides an array of Channels Requantizers
 * sized to the destination storage type. Calibration code in
 * qcalibration.hpp can build this array from per-channel weight scales
 * via buildRequantizer(input_scale, weight_scale[c], output_scale, ...).
 *
 * Tensor layout:
 *
 *   input    NHWC                  [H][W][Channels]
 *   weights  [Channels][KH][KW]    one filter per channel
 *   biases   [Channels]            int32 with effective scale
 *                                   input_scale * weight_scale[c]
 *   output   NHWC                  [OutputHeight][OutputWidth][Channels]
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
        std::size_t Channels_,
        std::size_t KH_,
        std::size_t KW_,
        std::size_t StrideH_ = 1,
        std::size_t StrideW_ = 1>
    struct QDepthwiseConv2D
    {
        typedef InputStorage_ InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumType_ AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t InputHeight  = H_;
        static constexpr std::size_t InputWidth   = W_;
        static constexpr std::size_t Channels     = Channels_;
        static constexpr std::size_t KernelHeight = KH_;
        static constexpr std::size_t KernelWidth  = KW_;
        static constexpr std::size_t StrideH      = StrideH_;
        static constexpr std::size_t StrideW      = StrideW_;

        static constexpr std::size_t OutputHeight = (H_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * Channels_;
        static constexpr std::size_t InputSize    = H_ * W_ * Channels_;
        static constexpr std::size_t WeightsPerChannel = KH_ * KW_;
        static constexpr std::size_t TotalWeights      = Channels_ * WeightsPerChannel;

        static_assert(H_ >= KH_, "Input height must be >= kernel height.");
        static_assert(W_ >= KW_, "Input width must be >= kernel width.");
        static_assert(StrideH_ > 0, "Vertical stride must be > 0.");
        static_assert(StrideW_ > 0, "Horizontal stride must be > 0.");
        static_assert(Channels_ > 0, "Channels must be > 0.");

        const WeightType* weights;
        const AccumulatorType* biases;
        InputType input_zero_point;
        const Requantizer<AccumulatorType, OutputType>* requantizers; // [Channels]

        void forward(const InputType* input, OutputType* output) const
        {
            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const std::size_t ihStart = oh * StrideH_;
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const std::size_t iwStart = ow * StrideW_;
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * Channels_;

                    for (std::size_t c = 0; c < Channels_; ++c)
                    {
                        const std::size_t weightOffset = c * WeightsPerChannel;
                        AccumulatorType acc = (biases != nullptr)
                            ? biases[c]
                            : static_cast<AccumulatorType>(0);

                        for (std::size_t kh = 0; kh < KH_; ++kh)
                        {
                            const std::size_t ih = ihStart + kh;
                            for (std::size_t kw = 0; kw < KW_; ++kw)
                            {
                                const std::size_t iw = iwStart + kw;
                                const std::size_t inIdx =
                                    (ih * W_ + iw) * Channels_ + c;
                                const AccumulatorType x =
                                    static_cast<AccumulatorType>(input[inIdx]) -
                                    static_cast<AccumulatorType>(input_zero_point);
                                const AccumulatorType w =
                                    static_cast<AccumulatorType>(
                                        weights[weightOffset + kh * KW_ + kw]);
                                acc += w * x;
                            }
                        }

                        output[outPixelOffset + c] = requantizers[c].apply(acc);
                    }
                }
            }
        }
    };

} // namespace tinymind
