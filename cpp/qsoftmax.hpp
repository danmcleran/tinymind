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

#include <cstddef>
#include <cstdint>

/*
 * Quantized softmax (1D, last-axis).
 *
 * For a rank-2 tensor [Rows][Features], each row is independently
 * softmaxed along the Features axis:
 *
 *   softmax(x)_i = exp(x_i) / sum_j exp(x_j)
 *
 * To avoid overflow, the canonical implementation subtracts the per-row
 * max from each x_i before exponentiating. Because softmax is invariant
 * to a shared additive constant, this affects neither the numerator nor
 * the denominator's ratio — but it keeps each exp argument <= 0, so the
 * exp values stay in (0, 1] and a small int domain is enough.
 *
 * Runtime path is pure integer:
 *
 *   1. Find max_q across the row (int8 comparison; pure integer).
 *   2. For each i, exp_q[i] = exp_lut[(q_in[i] - max_q) + 255], where
 *      exp_lut is a 256-entry int32 table built host-side by
 *      buildQSoftmaxExpLUT in qcalibration.hpp. The lookup is keyed by
 *      the integer difference q_in[i] - max_q (which is in [-255, 0] for
 *      int8) shifted into [0, 255]; zero_point cancels.
 *   3. sum = sum_i exp_q[i] (int64 to keep head room for Features up to
 *      a few thousand).
 *   4. For each i, y_q[i] = round(exp_q[i] / sum * 256) + zp_out, with
 *      saturation to [qmin, qmax]. The (* 256) factor is the standard
 *      TFLite convention: output_scale = 1/256, output_zero_point = -128,
 *      so the full [0, 1] probability range maps onto int8 [-128, 127].
 *
 * Two-pass (no per-row stack allocation): the first pass computes max
 * and sum; the second pass re-applies the lookup and emits the output.
 * The LUT load is cheap on every target tier that ships caches; the
 * trade-off is bounded stack regardless of Features.
 *
 * Pure integer; freestanding-safe. The host-side LUT builder lives in
 * qcalibration.hpp.
 */

namespace tinymind {

    /**
     * Convert (q_in, q_max) into the LUT index used by QSoftmax1D.
     *
     * q_in - q_max is in [-255, 0] for any int8 pair; adding 255 lands
     * the index in [0, 255]. Centralized so the layer and the host LUT
     * builder agree on the convention bit-for-bit.
     */
    inline std::size_t qSoftmaxLUTIndex(int32_t q_in, int32_t q_max)
    {
        int32_t diff = q_in - q_max;
        if (diff > 0)   diff = 0;     // defensive; should not happen
        if (diff < -255) diff = -255; // defensive
        return static_cast<std::size_t>(diff + 255);
    }

    template<
        typename InStorage_,
        typename OutStorage_,
        std::size_t Rows_,
        std::size_t Features_>
    struct QSoftmax1D
    {
        typedef InStorage_  InputType;
        typedef OutStorage_ OutputType;

        static constexpr std::size_t Rows     = Rows_;
        static constexpr std::size_t Features = Features_;
        static constexpr std::size_t Size     = Rows_ * Features_;

        static_assert(Rows_ > 0, "Rows must be > 0.");
        static_assert(Features_ > 0, "Features must be > 0.");

        const int32_t* exp_lut;     // [256]
        OutputType     output_zero_point; // typically -128 for int8 output
        OutputType     qmin;              // typically -128
        OutputType     qmax;              // typically  127

        void forward(const InputType* input, OutputType* output) const
        {
            const int32_t zp_out = static_cast<int32_t>(output_zero_point);
            const int32_t lo     = static_cast<int32_t>(qmin);
            const int32_t hi     = static_cast<int32_t>(qmax);

            for (std::size_t r = 0; r < Rows_; ++r)
            {
                const InputType* row_in  = input  + r * Features_;
                OutputType*      row_out = output + r * Features_;

                // Per-row maximum (used to shift the exp argument into
                // (-inf, 0] before lookup).
                int32_t max_q = static_cast<int32_t>(row_in[0]);
                for (std::size_t i = 1; i < Features_; ++i)
                {
                    const int32_t v = static_cast<int32_t>(row_in[i]);
                    if (v > max_q) max_q = v;
                }

                // Sum of exp(q - max) over the row.
                int64_t sum = 0;
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const std::size_t idx = qSoftmaxLUTIndex(
                        static_cast<int32_t>(row_in[i]), max_q);
                    sum += static_cast<int64_t>(exp_lut[idx]);
                }

                if (sum <= 0)
                {
                    // All-zero LUT row (would only happen if input_scale
                    // was wildly mis-calibrated); emit zero-probability
                    // saturating output instead of dividing by zero.
                    for (std::size_t i = 0; i < Features_; ++i)
                    {
                        int32_t q = zp_out;
                        if (q < lo) q = lo;
                        if (q > hi) q = hi;
                        row_out[i] = static_cast<OutputType>(q);
                    }
                    continue;
                }

                const int64_t half_sum = sum >> 1;
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const std::size_t idx = qSoftmaxLUTIndex(
                        static_cast<int32_t>(row_in[i]), max_q);
                    const int64_t numerator =
                        (static_cast<int64_t>(exp_lut[idx]) << 8) + half_sum;
                    int32_t scaled =
                        static_cast<int32_t>(numerator / sum);
                    // scaled is round(prob * 256) in [0, 256]; the upper
                    // value can equal 256 when one class fully dominates,
                    // which the saturation clamps to qmax.
                    int32_t q = scaled + zp_out;
                    if (q < lo) q = lo;
                    if (q > hi) q = hi;
                    row_out[i] = static_cast<OutputType>(q);
                }
            }
        }
    };

} // namespace tinymind
