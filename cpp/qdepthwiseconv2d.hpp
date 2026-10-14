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
            // Snapshot the fields into locals. Every output store is an int8_t write,
            // which may alias *this, so member reads inside the loops would be
            // reloaded -- and the requantizer's shift re-decoded -- once per output.
            const WeightType* const layerWeights = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const Requantizer<AccumulatorType, OutputType>* const rqs = requantizers;

            // Channel-outer: one channel's kernel taps and requantizer stay in
            // locals while it sweeps every output pixel. The input zero point
            // is folded into the bias once per channel, since
            //   sum(w * (x - zp)) == sum(w * x) - zp * sum(w),
            // so the tap loop is a plain multiply-accumulate. Integer-exact,
            // so bit-exact with the per-pixel, per-tap form.
            for (std::size_t c = 0; c < Channels_; ++c)
            {
                AccumulatorType taps[WeightsPerChannel];
                AccumulatorType weightSum = static_cast<AccumulatorType>(0);
                for (std::size_t k = 0; k < WeightsPerChannel; ++k)
                {
                    taps[k] = static_cast<AccumulatorType>(layerWeights[c * WeightsPerChannel + k]);
                    weightSum += taps[k];
                }

                const AccumulatorType bias = ((layerBiases != nullptr)
                    ? layerBiases[c]
                    : static_cast<AccumulatorType>(0)) -
                    static_cast<AccumulatorType>(inputZeroPoint) * weightSum;
                const Requantizer<AccumulatorType, OutputType> rq = rqs[c];

                // The pixel loop is written out twice, once with the
                // requantizer's FastRequantizer form and once with the
                // Requantizer itself, chosen per channel.
                detail::FastRequantizer<OutputType> fast;
                if (fast.init(rq))
                {
                    for (std::size_t oh = 0; oh < OutputHeight; ++oh)
                    {
                        for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                        {
                            const InputType* const window =
                                input + ((oh * StrideH_) * W_ + ow * StrideW_) * Channels_ + c;

                            AccumulatorType acc = bias;
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                            for (std::size_t kh = 0; kh < KH_; ++kh)
                            {
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                                for (std::size_t kw = 0; kw < KW_; ++kw)
                                {
                                    acc += taps[kh * KW_ + kw] *
                                           static_cast<AccumulatorType>(
                                               window[(kh * W_ + kw) * Channels_]);
                                }
                            }

                            output[(oh * OutputWidth + ow) * Channels_ + c] = fast.apply(acc);
                        }
                    }
                }
                else
                {
                    for (std::size_t oh = 0; oh < OutputHeight; ++oh)
                    {
                        for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                        {
                            const InputType* const window =
                                input + ((oh * StrideH_) * W_ + ow * StrideW_) * Channels_ + c;

                            AccumulatorType acc = bias;
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                            for (std::size_t kh = 0; kh < KH_; ++kh)
                            {
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                                for (std::size_t kw = 0; kw < KW_; ++kw)
                                {
                                    acc += taps[kh * KW_ + kw] *
                                           static_cast<AccumulatorType>(
                                               window[(kh * W_ + kw) * Channels_]);
                                }
                            }

                            output[(oh * OutputWidth + ow) * Channels_ + c] = rq.apply(acc);
                        }
                    }
                }
            }
        }
    };

} // namespace tinymind
