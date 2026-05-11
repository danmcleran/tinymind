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
 * Quantized 2D BatchNorm (standalone, per-channel affine).
 *
 * Phase 11 covers two BN deployment shapes:
 *
 *   1. Fold BN into the upstream conv (preferred when conv->bn->activation
 *      is contiguous). Done host-side by foldBatchNorm in qcalibration.hpp;
 *      no runtime op needed.
 *
 *   2. Standalone BN at runtime, for cases where the float graph has a BN
 *      that cannot be folded (post-pool BN, BN between two non-conv ops,
 *      or any topology that does not match the conv->bn pattern). This
 *      header implements that case as a per-channel affine map.
 *
 * Math (NHWC, per output channel c):
 *
 *   sigma_eff[c] = gamma[c] / sqrt(variance[c] + epsilon)
 *   y_real       = sigma_eff[c] * (x_real - mean[c]) + beta[c]
 *                = (sigma_eff[c] * x_real) + (beta[c] - sigma_eff[c] * mean[c])
 *
 * At runtime each channel carries (multiplier, shift, bias_addend) built
 * by buildQBatchNormChannelParams (qcalibration.hpp). The forward path
 * computes
 *
 *   centered  = q_in - zp_in                                 // int32
 *   scaled    = MultiplyByQuantizedMultiplier(centered,
 *                                             mult[c], shift[c])
 *   q_out     = saturate(scaled + bias_addend[c] + zp_out)
 *
 * The bias_addend term is the per-channel integer offset translated into
 * the output scale; it captures (beta - sigma_eff * mean) so the runtime
 * pass is a single MAC + add per element. Pure integer; freestanding-safe.
 *
 * Layout matches QConv2D / QDepthwiseConv2D so this layer drops into the
 * same NHWC pipeline.
 */

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include "include/qcalibration.hpp"
#endif

namespace tinymind {

#if !(TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD)
    /**
     * Minimal stand-in for the calibration-built channel parameter triple
     * when the deployable inference build is freestanding. The host-side
     * builder lives in qcalibration.hpp and produces values of this exact
     * shape, embedded as a caller-owned constant array.
     */
    struct QBatchNormChannelParams
    {
        int32_t multiplier;
        int32_t shift;
        int32_t bias_addend;
    };
#endif

    template<
        typename InStorage_,
        typename OutStorage_,
        std::size_t H_,
        std::size_t W_,
        std::size_t Channels_>
    struct QBatchNorm2D
    {
        typedef InStorage_  InputType;
        typedef OutStorage_ OutputType;

        static constexpr std::size_t Height   = H_;
        static constexpr std::size_t Width    = W_;
        static constexpr std::size_t Channels = Channels_;
        static constexpr std::size_t Size     = H_ * W_ * Channels_;

        static_assert(H_ > 0, "Height must be > 0.");
        static_assert(W_ > 0, "Width must be > 0.");
        static_assert(Channels_ > 0, "Channels must be > 0.");

        const QBatchNormChannelParams* params; // [Channels]
        InputType  input_zero_point;
        OutputType output_zero_point;
        OutputType qmin;
        OutputType qmax;

        void forward(const InputType* input, OutputType* output) const
        {
            const int32_t zp_in  = static_cast<int32_t>(input_zero_point);
            const int32_t zp_out = static_cast<int32_t>(output_zero_point);
            const int32_t lo     = static_cast<int32_t>(qmin);
            const int32_t hi     = static_cast<int32_t>(qmax);

            for (std::size_t h = 0; h < H_; ++h)
            {
                for (std::size_t w = 0; w < W_; ++w)
                {
                    const std::size_t pixelOffset =
                        (h * W_ + w) * Channels_;
                    for (std::size_t c = 0; c < Channels_; ++c)
                    {
                        const int32_t centered =
                            static_cast<int32_t>(input[pixelOffset + c]) -
                            zp_in;
                        const int32_t scaled =
                            multiplyByQuantizedMultiplier(
                                centered, params[c].multiplier,
                                params[c].shift);
                        int32_t with_bias =
                            scaled + params[c].bias_addend + zp_out;
                        if (with_bias < lo) with_bias = lo;
                        if (with_bias > hi) with_bias = hi;
                        output[pixelOffset + c] =
                            static_cast<OutputType>(with_bias);
                    }
                }
            }
        }
    };

    /**
     * 1D variant: rank-2 tensor [Length][Channels] in channel-last layout.
     *
     * Identical math; loop is flattened over the leading axis. Useful for
     * post-pool BN in time-series / KWS pipelines where the spatial axis
     * has collapsed.
     */
    template<
        typename InStorage_,
        typename OutStorage_,
        std::size_t Length_,
        std::size_t Channels_>
    struct QBatchNorm1D
    {
        typedef InStorage_  InputType;
        typedef OutStorage_ OutputType;

        static constexpr std::size_t Length   = Length_;
        static constexpr std::size_t Channels = Channels_;
        static constexpr std::size_t Size     = Length_ * Channels_;

        static_assert(Length_ > 0, "Length must be > 0.");
        static_assert(Channels_ > 0, "Channels must be > 0.");

        const QBatchNormChannelParams* params; // [Channels]
        InputType  input_zero_point;
        OutputType output_zero_point;
        OutputType qmin;
        OutputType qmax;

        void forward(const InputType* input, OutputType* output) const
        {
            const int32_t zp_in  = static_cast<int32_t>(input_zero_point);
            const int32_t zp_out = static_cast<int32_t>(output_zero_point);
            const int32_t lo     = static_cast<int32_t>(qmin);
            const int32_t hi     = static_cast<int32_t>(qmax);

            for (std::size_t l = 0; l < Length_; ++l)
            {
                const std::size_t rowOffset = l * Channels_;
                for (std::size_t c = 0; c < Channels_; ++c)
                {
                    const int32_t centered =
                        static_cast<int32_t>(input[rowOffset + c]) -
                        zp_in;
                    const int32_t scaled =
                        multiplyByQuantizedMultiplier(
                            centered, params[c].multiplier,
                            params[c].shift);
                    int32_t with_bias =
                        scaled + params[c].bias_addend + zp_out;
                    if (with_bias < lo) with_bias = lo;
                    if (with_bias > hi) with_bias = hi;
                    output[rowOffset + c] =
                        static_cast<OutputType>(with_bias);
                }
            }
        }
    };

} // namespace tinymind
