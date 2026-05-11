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
 * Quantized LayerNorm (1D, last-axis).
 *
 * For a rank-2 tensor [Rows][Features], each row is normalized
 * independently along the Features axis:
 *
 *   mean_real     = scale_in * (mean_q - zp_in)
 *   variance_real = scale_in^2 * var_q
 *   normalized    = (x_real - mean_real) / sqrt(variance_real + eps)
 *                 = (q_in - mean_q) / sqrt(var_q + eps_q')
 *   y_real        = gamma[i] * normalized + beta[i]
 *
 * LayerNorm cancels both scale_in and zp_in in the centering / normalizing
 * step (the input range scale-cancels in numerator and denominator), so
 * the layer needs no input affine parameters at runtime. eps is translated
 * by the host calibrator to the integer var_q domain.
 *
 * Storage convention:
 *
 *   gamma           int16[Features]  per-feature affine scale, Q1.14
 *                                    (gamma_int = round(gamma_real * 2^14))
 *   beta            int32[Features]  per-feature offset in output scale
 *                                    (beta_int = round(beta_real / scale_out))
 *   epsilon_q       int32            eps quantized to var_q domain
 *                                    (round(eps / scale_in^2)); typically 1+
 *                                    to avoid degenerate-row division-by-zero
 *   output_multiplier, output_shift
 *                   Q0.31            captures ratio = 1 / (2^14 * scale_out);
 *                                    decomposed host-side by
 *                                    quantizeMultiplier
 *
 * Runtime path is pure integer: integer mean / variance, an inv-sqrt
 * computed by integer Newton iteration (no <cmath>), and the existing
 * MultiplyByQuantizedMultiplier primitive for the gamma scaling. Pure
 * integer; freestanding-safe.
 */

namespace tinymind {

    /**
     * Integer square root via Newton iteration. Returns floor(sqrt(x)).
     *
     * Pure 64-bit integer; converges in O(log x) steps. No <cmath>; safe at
     * STD=0. Used by invSqrtQ30 to compute 1/sqrt(var_q) at runtime in
     * QLayerNorm.
     */
    inline uint64_t qIntegerSqrt64(uint64_t x)
    {
        if (x < 2u) return x;
        uint64_t y = x;
        uint64_t z = (y + x / y) >> 1;
        while (z < y)
        {
            y = z;
            z = (y + x / y) >> 1;
        }
        return y;
    }

    /**
     * 1 / sqrt(x) in Q1.30 fixed point.
     *
     * Scales x by 2^28 before the sqrt so the result carries ~14 bits of
     * fractional precision below the integer sqrt boundary, then divides
     * (1 << 44) by the scaled sqrt to land in Q1.30. Saturates to the
     * Q1.30 max when x is degenerate.
     */
    inline uint32_t qInvSqrtQ30(uint32_t x)
    {
        if (x == 0u) return 0x7FFFFFFFu;
        const uint64_t scaled = static_cast<uint64_t>(x) << 28;
        const uint64_t s = qIntegerSqrt64(scaled);
        if (s == 0u) return 0x7FFFFFFFu;
        const uint64_t inv = (static_cast<uint64_t>(1) << 44) / s;
        if (inv > 0x7FFFFFFFu) return 0x7FFFFFFFu;
        return static_cast<uint32_t>(inv);
    }

    template<
        typename InStorage_,
        typename OutStorage_,
        std::size_t Rows_,
        std::size_t Features_>
    struct QLayerNorm1D
    {
        typedef InStorage_  InputType;
        typedef OutStorage_ OutputType;

        static constexpr std::size_t Rows     = Rows_;
        static constexpr std::size_t Features = Features_;
        static constexpr std::size_t Size     = Rows_ * Features_;

        static_assert(Rows_ > 0, "Rows must be > 0.");
        static_assert(Features_ > 0, "Features must be > 0.");

        const int16_t* gamma;       // [Features], Q1.14
        const int32_t* beta;        // [Features], output-scale integer
        int32_t        epsilon_q;   // var_q domain
        int32_t        output_multiplier; // Q0.31
        int32_t        output_shift;
        OutputType     output_zero_point;
        OutputType     qmin;
        OutputType     qmax;

        void forward(const InputType* input, OutputType* output) const
        {
            const int32_t zp_out = static_cast<int32_t>(output_zero_point);
            const int32_t lo     = static_cast<int32_t>(qmin);
            const int32_t hi     = static_cast<int32_t>(qmax);

            for (std::size_t r = 0; r < Rows_; ++r)
            {
                const InputType* row_in  = input  + r * Features_;
                OutputType*      row_out = output + r * Features_;

                // mean of the row in q-domain (no zero_point subtraction;
                // it cancels in the centering step below).
                int64_t sum = 0;
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    sum += static_cast<int64_t>(row_in[i]);
                }
                const int32_t N = static_cast<int32_t>(Features_);
                int32_t mean_q;
                {
                    const int64_t bias = (sum >= 0)
                        ? static_cast<int64_t>(N) / 2
                        : -(static_cast<int64_t>(N) / 2);
                    mean_q = static_cast<int32_t>((sum + bias) / N);
                }

                // sum of squared deviations.
                int64_t ssum = 0;
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const int32_t dev =
                        static_cast<int32_t>(row_in[i]) - mean_q;
                    ssum += static_cast<int64_t>(dev) *
                            static_cast<int64_t>(dev);
                }
                int32_t var_q;
                {
                    const int64_t bias_v = static_cast<int64_t>(N) / 2;
                    int64_t v = (ssum + bias_v) / N;
                    if (v > static_cast<int64_t>(0x7FFFFFFF))
                        v = 0x7FFFFFFF;
                    var_q = static_cast<int32_t>(v);
                }

                const int64_t denom_64 =
                    static_cast<int64_t>(var_q) +
                    static_cast<int64_t>(epsilon_q);
                const uint32_t denom_u32 = (denom_64 < 0)
                    ? 0u
                    : (denom_64 > static_cast<int64_t>(0xFFFFFFFFu)
                        ? 0xFFFFFFFFu
                        : static_cast<uint32_t>(denom_64));
                const uint32_t inv_stddev_q30 = qInvSqrtQ30(denom_u32);

                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const int32_t dev =
                        static_cast<int32_t>(row_in[i]) - mean_q;
                    // normalized in Q?.14: keep 14 fractional bits below
                    // the integer point so the next-stage multiply by
                    // gamma (also Q1.14) lands in Q?.28 with adequate
                    // precision for the output requantization.
                    const int64_t prod_norm =
                        static_cast<int64_t>(dev) *
                        static_cast<int64_t>(inv_stddev_q30);
                    const int32_t normalized_q14 =
                        static_cast<int32_t>(prod_norm >> 16);

                    // Multiply by per-feature gamma (Q1.14). Both factors
                    // fit comfortably inside int32: normalized_q14 is
                    // bounded by ~2^15 for typical input variance, and
                    // gamma_int <= 32767.
                    const int32_t weighted_q28 = normalized_q14 *
                        static_cast<int32_t>(gamma[i]);

                    // Requantize the Q?.28 weighted accumulator to the
                    // output grid using the host-side multiplier/shift,
                    // which captures 1 / (2^28 * scale_out).
                    const int32_t rescaled = multiplyByQuantizedMultiplier(
                        weighted_q28, output_multiplier, output_shift);

                    int32_t with_bias = rescaled + beta[i] + zp_out;
                    if (with_bias < lo) with_bias = lo;
                    if (with_bias > hi) with_bias = hi;
                    row_out[i] = static_cast<OutputType>(with_bias);
                }
            }
        }
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

    /**
     * Host-side: quantize a float gamma vector to int16 Q1.14.
     *
     * Used to populate the QLayerNorm1D.gamma buffer. Saturates to int16
     * bounds; in practice gamma in (-1, 1) trivially fits.
     */
    inline void quantizeLayerNormGamma(const float* gamma_real,
                                       std::size_t n,
                                       int16_t* gamma_int_out);

    /**
     * Host-side: quantize a float beta vector to int32 in the output scale.
     *
     * beta_int[i] = round(beta_real[i] / output_scale).
     */
    inline void quantizeLayerNormBeta(const float* beta_real,
                                      std::size_t n,
                                      float output_scale,
                                      int32_t* beta_int_out);

    /**
     * Host-side: build (output_multiplier, output_shift) for QLayerNorm1D.
     *
     * Effective ratio = 1 / ((1 << 28) * output_scale); the (1 << 28)
     * factor unwinds the combined gamma * normalized Q?.28 fixed-point
     * representation produced inside the forward pass.
     */
    inline void buildQLayerNormOutputParams(float output_scale,
                                            int32_t& output_multiplier,
                                            int32_t& output_shift);

    /**
     * Host-side: translate a float epsilon to the var_q integer domain
     * (var_q = variance of raw q_in values; eps_q = round(eps / scale_in^2)).
     *
     * Returns at least 1 so a degenerate (constant) row never hits
     * division by zero in the inv-sqrt step.
     */
    inline int32_t quantizeLayerNormEpsilon(float epsilon,
                                            float input_scale);

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    inline void quantizeLayerNormGamma(const float* gamma_real,
                                       std::size_t n,
                                       int16_t* gamma_int_out)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            long q = std::lround(static_cast<double>(gamma_real[i]) *
                                 static_cast<double>(1 << 14));
            if (q < -32768) q = -32768;
            if (q >  32767) q =  32767;
            gamma_int_out[i] = static_cast<int16_t>(q);
        }
    }

    inline void quantizeLayerNormBeta(const float* beta_real,
                                      std::size_t n,
                                      float output_scale,
                                      int32_t* beta_int_out)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const long q = std::lround(
                static_cast<double>(beta_real[i]) /
                static_cast<double>(output_scale));
            beta_int_out[i] = static_cast<int32_t>(q);
        }
    }

    inline void buildQLayerNormOutputParams(float output_scale,
                                            int32_t& output_multiplier,
                                            int32_t& output_shift)
    {
        const double ratio = 1.0 /
            (static_cast<double>(static_cast<int64_t>(1) << 28) *
             static_cast<double>(output_scale));
        quantizeMultiplier(ratio, output_multiplier, output_shift);
    }

    inline int32_t quantizeLayerNormEpsilon(float epsilon,
                                            float input_scale)
    {
        const double s2 = static_cast<double>(input_scale) *
                          static_cast<double>(input_scale);
        if (s2 <= 0.0) return 1;
        const long q = std::lround(static_cast<double>(epsilon) / s2);
        return (q < 1) ? 1 : static_cast<int32_t>(q);
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
