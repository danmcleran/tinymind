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
 * Per-channel quantized depthwise 2D convolution (NHWC). VALID by default;
 * PadTop/PadBottom/PadLeft/PadRight pad the input with its zero point (SAME
 * padding, as TFLite does) without a separate QPad2D pass.
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
 *   weights  [Channels][KH][KW]    one filter per channel (with
 *                                   WeightStorage Int4Weights, the same
 *                                   tensor packed two per byte)
 *   biases   [Channels]            int32 with effective scale
 *                                   input_scale * weight_scale[c]
 *   output   NHWC                  [OutputHeight][OutputWidth][Channels]
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    namespace detail {

        template<bool Padded>
        struct DepthwisePadTag
        {
        };

        template<bool OneOutput>
        struct DepthwiseOneOutputTag
        {
        };

        /*
         * Stack budget for the band of padded rows a padded depthwise layer
         * keeps for one channel (KH rows of W + PadLeft + PadRight elements).
         * Shared with qconv2d.hpp, whose bands span every input channel.
         */
#ifndef TINYMIND_CONV_PAD_BAND_MAX_BYTES
#define TINYMIND_CONV_PAD_BAND_MAX_BYTES 4096
#endif

        /**
         * The type a depthwise window is summed in. Normally the accumulator,
         * starting from the bias. For int16 inputs and int8 weights (16x8)
         * with an int64 accumulator, a window of up to 256 taps cannot leave
         * int32 (|x| <= 32768 and |w| <= 128, since the zero point is folded
         * into the bias, so each term is at most 2^22), so it is summed in
         * int32 and widened once: a 64-bit multiply per tap is a libgcc call
         * on Cortex-M0+ and twice the work on Cortex-M4. start() is the
         * window's initial value and finish() the accumulator it yields; for
         * the plain case they are the identity, so that code is unchanged.
         */
        template<typename Input, typename Weight, typename Accum, std::size_t Taps,
                 bool Narrow = (Taps <= 256)>
        struct DepthwiseWindow
        {
            typedef Accum type;
            static Accum start(Accum bias)
            {
                return bias;
            }
            static Accum finish(Accum, Accum window)
            {
                return window;
            }
        };

        template<std::size_t Taps>
        struct DepthwiseWindow<int16_t, int8_t, int64_t, Taps, true>
        {
            typedef int32_t type;
            static int32_t start(int64_t)
            {
                return 0;
            }
            static int64_t finish(int64_t bias, int32_t window)
            {
                return bias + static_cast<int64_t>(window);
            }
        };

    } // namespace detail

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
        std::size_t StrideW_ = 1,
        std::size_t PadTop_ = 0,
        std::size_t PadBottom_ = 0,
        std::size_t PadLeft_ = 0,
        std::size_t PadRight_ = 0>
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
        static constexpr std::size_t PadTop       = PadTop_;
        static constexpr std::size_t PadBottom    = PadBottom_;
        static constexpr std::size_t PadLeft      = PadLeft_;
        static constexpr std::size_t PadRight     = PadRight_;

        static constexpr std::size_t OutputHeight = (H_ + PadTop_ + PadBottom_ - KH_) / StrideH_ + 1;
        static constexpr std::size_t OutputWidth  = (W_ + PadLeft_ + PadRight_ - KW_) / StrideW_ + 1;
        static constexpr std::size_t OutputSize   = OutputHeight * OutputWidth * Channels_;
        static constexpr std::size_t InputSize    = H_ * W_ * Channels_;
        static constexpr std::size_t WeightsPerChannel = KH_ * KW_;
        static constexpr std::size_t TotalWeights      = Channels_ * WeightsPerChannel;

        typedef detail::DepthwiseWindow<InputStorage_, WeightStorage_, AccumType_, KH_ * KW_> Window;
        typedef typename Window::type WindowType;
        // How the weights are held: int8 elements, or packed nibbles for
        // Int4Weights (QDepthwiseConv2DInt4 in qint4.hpp).
        typedef detail::WeightArray<WeightStorage_> WeightAccess;

        static_assert(H_ + PadTop_ + PadBottom_ >= KH_, "Padded input height must be >= kernel height.");
        static_assert(W_ + PadLeft_ + PadRight_ >= KW_, "Padded input width must be >= kernel width.");
        static_assert(StrideH_ > 0, "Vertical stride must be > 0.");
        static_assert(StrideW_ > 0, "Horizontal stride must be > 0.");
        static_assert(Channels_ > 0, "Channels must be > 0.");

        typename WeightAccess::pointer weights; // const WeightType*, or packed bytes for Int4Weights
        const AccumulatorType* biases;
        InputType input_zero_point;
        const Requantizer<AccumulatorType, OutputType>* requantizers; // [Channels]

        void forward(const InputType* input, OutputType* output) const
        {
            forwardImpl(input, output, detail::DepthwisePadTag<Padded>(),
                        detail::DepthwiseOneOutputTag<!Padded && (OutputHeight * OutputWidth == 1)>());
        }

    private:
        // Padded: each channel keeps a ring of KH padded rows of its own
        // plane, built once per row with the zero point in the border (whose
        // contribution w * (x - zp) is zero, as TFLite pads), and the same
        // tap loop runs over it. The zero point is folded into the bias, as
        // below, which the padding values satisfy: a border tap adds
        // w * zp, which the folded bias cancels exactly as for an interior x.
        static constexpr bool Padded = (PadTop_ | PadBottom_ | PadLeft_ | PadRight_) != 0;

        void forwardImpl(const InputType* input, OutputType* output, detail::DepthwisePadTag<true>,
                         detail::DepthwiseOneOutputTag<false>) const
        {
            constexpr std::size_t PaddedWidth = W_ + PadLeft_ + PadRight_;
            static_assert(KH_ * PaddedWidth * sizeof(InputType) <= TINYMIND_CONV_PAD_BAND_MAX_BYTES,
                          "The padded-row band of this depthwise layer exceeds TINYMIND_CONV_PAD_BAND_MAX_BYTES: "
                          "raise the limit, or pad with QPad2D and use an unpadded layer.");

            const typename WeightAccess::pointer layerWeights = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const Requantizer<AccumulatorType, OutputType>* const rqs = requantizers;

            InputType band[KH_ * PaddedWidth];
            for (std::size_t c = 0; c < Channels_; ++c)
            {
                WindowType taps[WeightsPerChannel];
                AccumulatorType weightSum = static_cast<AccumulatorType>(0);
                for (std::size_t k = 0; k < WeightsPerChannel; ++k)
                {
                    taps[k] = static_cast<WindowType>(WeightAccess::at(layerWeights, c * WeightsPerChannel + k));
                    weightSum += static_cast<AccumulatorType>(taps[k]);
                }
                const AccumulatorType bias = ((layerBiases != nullptr)
                    ? layerBiases[c]
                    : static_cast<AccumulatorType>(0)) -
                    static_cast<AccumulatorType>(inputZeroPoint) * weightSum;
                const Requantizer<AccumulatorType, OutputType> rq = rqs[c];
                // The zero point is in the bias, so the window's products are
                // of x by w, and their bound is that of a zero point of 0.
                typename detail::FastRequantizerFor<AccumulatorType, OutputType>::type fast;
                const bool useFast = fast.template initWindow<WeightType>(
                    rq, bias, WeightsPerChannel, static_cast<InputType>(0));

                std::size_t next = 0; // the next padded row to build
                for (std::size_t oh = 0; oh < OutputHeight; ++oh)
                {
                    const std::size_t top = oh * StrideH_;
                    if (next < top)
                    {
                        next = top;
                    }
                    for (; next < top + KH_; ++next)
                    {
                        InputType* const dst = band + (next % KH_) * PaddedWidth;
                        if ((next < PadTop_) || (next >= PadTop_ + H_))
                        {
                            for (std::size_t i = 0; i < PaddedWidth; ++i)
                            {
                                dst[i] = inputZeroPoint;
                            }
                            continue;
                        }
                        const InputType* const src = input + (next - PadTop_) * W_ * Channels_ + c;
                        for (std::size_t i = 0; i < PadLeft_; ++i)
                        {
                            dst[i] = inputZeroPoint;
                        }
                        for (std::size_t w = 0; w < W_; ++w)
                        {
                            dst[PadLeft_ + w] = src[w * Channels_];
                        }
                        for (std::size_t i = 0; i < PadRight_; ++i)
                        {
                            dst[PadLeft_ + W_ + i] = inputZeroPoint;
                        }
                    }

                    const InputType* rows[KH_];
                    for (std::size_t kh = 0; kh < KH_; ++kh)
                    {
                        rows[kh] = band + ((top + kh) % KH_) * PaddedWidth;
                    }
                    OutputType* const rowOut = output + oh * OutputWidth * Channels_ + c;
                    for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                    {
                        WindowType acc = Window::start(bias);
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
                                       static_cast<WindowType>(rows[kh][ow * StrideW_ + kw]);
                            }
                        }
                        const AccumulatorType sum = Window::finish(bias, acc);
                        rowOut[ow * Channels_] = useFast ? fast.apply(sum) : rq.apply(sum);
                    }
                }
            }
        }

        void forwardImpl(const InputType* input, OutputType* output, detail::DepthwisePadTag<false>,
                         detail::DepthwiseOneOutputTag<false>) const
        {
            // Snapshot the fields into locals. Every output store is an int8_t write,
            // which may alias *this, so member reads inside the loops would be
            // reloaded -- and the requantizer's shift re-decoded -- once per output.
            const typename WeightAccess::pointer layerWeights = weights;
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
                WindowType taps[WeightsPerChannel];
                AccumulatorType weightSum = static_cast<AccumulatorType>(0);
                for (std::size_t k = 0; k < WeightsPerChannel; ++k)
                {
                    taps[k] = static_cast<WindowType>(WeightAccess::at(layerWeights, c * WeightsPerChannel + k));
                    weightSum += static_cast<AccumulatorType>(taps[k]);
                }

                const AccumulatorType bias = ((layerBiases != nullptr)
                    ? layerBiases[c]
                    : static_cast<AccumulatorType>(0)) -
                    static_cast<AccumulatorType>(inputZeroPoint) * weightSum;
                const Requantizer<AccumulatorType, OutputType> rq = rqs[c];

                // The pixel loop is written out twice, once with the
                // requantizer's FastRequantizer form and once with the
                // Requantizer itself, chosen per channel.
                typename detail::FastRequantizerFor<AccumulatorType, OutputType>::type fast;
                if (fast.template initWindow<WeightType>(rq, bias, WeightsPerChannel, static_cast<InputType>(0)))
                {
                    for (std::size_t oh = 0; oh < OutputHeight; ++oh)
                    {
                        for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                        {
                            const InputType* const window =
                                input + ((oh * StrideH_) * W_ + ow * StrideW_) * Channels_ + c;

                            WindowType acc = Window::start(bias);
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 16
#endif
                            // Up to 16 kernel rows unrolled in full. At 8, GCC
                            // kept a 10-row kernel's loop rolled in some -O3
                            // builds of the MLPerf Tiny streaming wake word
                            // model, and that layer took 227k cycles on the
                            // F411 instead of 123k.
                            for (std::size_t kh = 0; kh < KH_; ++kh)
                            {
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                                for (std::size_t kw = 0; kw < KW_; ++kw)
                                {
                                    acc += taps[kh * KW_ + kw] *
                                           static_cast<WindowType>(
                                               window[(kh * W_ + kw) * Channels_]);
                                }
                            }

                            output[(oh * OutputWidth + ow) * Channels_ + c] = fast.apply(Window::finish(bias, acc));
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

                            WindowType acc = Window::start(bias);
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 16
#endif
                            for (std::size_t kh = 0; kh < KH_; ++kh)
                            {
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                                for (std::size_t kw = 0; kw < KW_; ++kw)
                                {
                                    acc += taps[kh * KW_ + kw] *
                                           static_cast<WindowType>(
                                               window[(kh * W_ + kw) * Channels_]);
                                }
                            }

                            output[(oh * OutputWidth + ow) * Channels_ + c] = rq.apply(Window::finish(bias, acc));
                        }
                    }
                }
            }
        }

        // Unpadded with one output pixel per channel (a window as large as the
        // input, as in a streaming model's step()).
        void forwardImpl(const InputType* input, OutputType* output, detail::DepthwisePadTag<false>,
                         detail::DepthwiseOneOutputTag<true>) const
        {
            const typename WeightAccess::pointer layerWeights = weights;
            const AccumulatorType* const layerBiases = biases;
            const InputType inputZeroPoint = input_zero_point;
            const Requantizer<AccumulatorType, OutputType>* const rqs = requantizers;

            // One output pixel per channel (a window as large as the
            // input, as in a streaming model's step()): copying the taps
            // and preparing the FastRequantizer cost more per channel
            // than they save on a single output, so each channel is one
            // window sum and one Requantizer::apply, with the zero point
            // folded into the bias as in the general loop. The same sums and the same
            // requantizer, so bit-exact with the general loop.
            for (std::size_t c = 0; c < Channels_; ++c)
            {
                AccumulatorType weightSum = static_cast<AccumulatorType>(0);
                for (std::size_t k = 0; k < WeightsPerChannel; ++k)
                {
                    weightSum += static_cast<AccumulatorType>(
                        WeightAccess::at(layerWeights, c * WeightsPerChannel + k));
                }
                const AccumulatorType bias = ((layerBiases != nullptr)
                    ? layerBiases[c]
                    : static_cast<AccumulatorType>(0)) -
                    static_cast<AccumulatorType>(inputZeroPoint) * weightSum;

                WindowType acc = Window::start(bias);
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 16
#endif
                for (std::size_t kh = 0; kh < KH_; ++kh)
                {
#if !defined(__OPTIMIZE_SIZE__)
#pragma GCC unroll 8
#endif
                    for (std::size_t kw = 0; kw < KW_; ++kw)
                    {
                        acc += static_cast<WindowType>(
                                   WeightAccess::at(layerWeights, c * WeightsPerChannel + kh * KW_ + kw)) *
                               static_cast<WindowType>(input[(kh * W_ + kw) * Channels_ + c]);
                    }
                }
                output[c] = rqs[c].apply(Window::finish(bias, acc));
            }
        }
    };

} // namespace tinymind
