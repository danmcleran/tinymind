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

#include "tinymind_platform.hpp"

/*
 * Host-side calibration helpers for the affine quantization path.
 *
 * Strictly post-training: a float reference model produces tensor ranges
 * (min/max) over a representative dataset, those ranges are converted to
 * affine (scale, zero_point) parameters here, and the result is fed into
 * the integer Requantizer in qaffine.hpp at deployment time.
 *
 * Pulled in only when both float and stdlib are available because the
 * calibration math leans on std::lround and std::frexp. The deployable
 * inference binary does not need this header — it consumes the int32
 * (multiplier, shift, zero_point) triples that calibration emits.
 *
 * Conventions follow TFLite / CMSIS-NN:
 *   * Asymmetric activations: int8 storage with arbitrary zero_point in
 *     [qmin, qmax]; the observed float range is "nudged" to include 0
 *     so the zero_point can land on a real grid point.
 *   * Symmetric weights: int8 storage with zero_point = 0 and qmin
 *     restricted to -qmax_signed (the -128 slot is excluded so weights
 *     can be negated without overflow). Per-channel scale is supported
 *     for depthwise weights, which TFLite mandates.
 */

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

#include "qaffine.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tinymind {

    /**
     * Result of fitting an affine map to an observed float range.
     *
     * scale is always strictly positive (a degenerate zero range falls back
     * to scale = 1.0). zero_point is held as int32 to keep arithmetic head
     * room; callers narrow it to the destination storage type when writing
     * the per-tensor metadata.
     */
    struct AffineParams
    {
        float scale;
        int32_t zero_point;
    };

    /**
     * Streaming min/max observer.
     *
     * Drop calibration samples through observe(); query the accumulated
     * range afterwards. has_data flips on the first sample so callers can
     * tell an "all zeros" tensor apart from "no samples yet".
     */
    struct RangeObserver
    {
        float min_value;
        float max_value;
        bool has_data;

        RangeObserver()
            : min_value(0.0f), max_value(0.0f), has_data(false)
        {
        }

        void reset()
        {
            min_value = 0.0f;
            max_value = 0.0f;
            has_data = false;
        }

        void observe(float x)
        {
            if (!has_data)
            {
                min_value = x;
                max_value = x;
                has_data = true;
            }
            else
            {
                if (x < min_value) min_value = x;
                if (x > max_value) max_value = x;
            }
        }

        void observe(const float* xs, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                observe(xs[i]);
            }
        }
    };

    /**
     * Asymmetric (signed or unsigned) per-tensor parameters.
     *
     * Maps [fmin, fmax] (after extending to include 0) onto [qmin, qmax]
     * such that the float zero is exactly representable. The zero_point is
     * the integer value that decodes back to 0.0; it is clamped into
     * [qmin, qmax] so it is always a legal storage value.
     */
    inline AffineParams computeAffineParamsAsymmetric(float fmin, float fmax,
                                                      int32_t qmin, int32_t qmax)
    {
        AffineParams out;

        // Extend the range to include 0 so the zero_point lands on the grid.
        if (fmin > 0.0f) fmin = 0.0f;
        if (fmax < 0.0f) fmax = 0.0f;

        const float qrange = static_cast<float>(qmax - qmin);
        if (fmax == fmin || qrange == 0.0f)
        {
            out.scale = 1.0f;
            out.zero_point = qmin;
            return out;
        }

        out.scale = (fmax - fmin) / qrange;

        const double zp = static_cast<double>(qmin) -
                          static_cast<double>(fmin) / static_cast<double>(out.scale);
        long zp_round = std::lround(zp);
        if (zp_round < qmin) zp_round = qmin;
        if (zp_round > qmax) zp_round = qmax;
        out.zero_point = static_cast<int32_t>(zp_round);
        return out;
    }

    /**
     * Symmetric per-tensor parameters (zero_point = 0).
     *
     * Picks scale so that max(|fmin|, |fmax|) maps to qmax_signed. Weights
     * are typically calibrated this way with qmax_signed = 127, which leaves
     * -128 unused but allows safe negation in the multiply path.
     */
    inline AffineParams computeAffineParamsSymmetric(float fmin, float fmax,
                                                     int32_t qmax_signed)
    {
        AffineParams out;
        out.zero_point = 0;

        const float abs_min = (fmin < 0.0f) ? -fmin : fmin;
        const float abs_max = (fmax < 0.0f) ? -fmax : fmax;
        const float absmax = (abs_min > abs_max) ? abs_min : abs_max;

        if (absmax == 0.0f || qmax_signed <= 0)
        {
            out.scale = 1.0f;
            return out;
        }

        out.scale = absmax / static_cast<float>(qmax_signed);
        return out;
    }

    /**
     * Quantize a float to the destination storage type.
     *
     * Saturates to [qmin, qmax]. Rounding follows std::lround (ties away
     * from zero), matching TFLite's reference implementation.
     */
    template<typename DstStorage>
    inline DstStorage quantize(float x, float scale, int32_t zero_point,
                               int32_t qmin, int32_t qmax)
    {
        const double q = std::lround(static_cast<double>(x) / static_cast<double>(scale)) +
                         static_cast<double>(zero_point);
        long qi = static_cast<long>(q);
        if (qi < qmin) qi = qmin;
        if (qi > qmax) qi = qmax;
        return static_cast<DstStorage>(qi);
    }

    /**
     * Inverse of quantize(). No saturation; the math is exact up to float
     * precision.
     */
    template<typename SrcStorage>
    inline float dequantize(SrcStorage q, float scale, int32_t zero_point)
    {
        return scale * static_cast<float>(static_cast<int32_t>(q) - zero_point);
    }

    /**
     * Bulk variants. Provided so callers don't need their own loops; they
     * also document the expected element-wise contract.
     */
    template<typename DstStorage>
    inline void quantizeBuffer(const float* src, DstStorage* dst, std::size_t n,
                               float scale, int32_t zero_point,
                               int32_t qmin, int32_t qmax)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = quantize<DstStorage>(src[i], scale, zero_point, qmin, qmax);
        }
    }

    template<typename SrcStorage>
    inline void dequantizeBuffer(const SrcStorage* src, float* dst, std::size_t n,
                                 float scale, int32_t zero_point)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            dst[i] = dequantize<SrcStorage>(src[i], scale, zero_point);
        }
    }

    /**
     * Per-channel symmetric scale fitting for weight tensors.
     *
     * weights is a flat [num_channels * per_channel_count] buffer in
     * channel-major layout. Each channel gets its own absmax/qmax_signed
     * scale; zero_point is implicitly 0. scales_out must hold num_channels
     * floats.
     */
    inline void computePerChannelSymmetricScales(const float* weights,
                                                 std::size_t num_channels,
                                                 std::size_t per_channel_count,
                                                 int32_t qmax_signed,
                                                 float* scales_out)
    {
        for (std::size_t c = 0; c < num_channels; ++c)
        {
            const float* base = weights + c * per_channel_count;
            float absmax = 0.0f;
            for (std::size_t i = 0; i < per_channel_count; ++i)
            {
                const float v = base[i];
                const float a = (v < 0.0f) ? -v : v;
                if (a > absmax) absmax = a;
            }

            if (absmax == 0.0f || qmax_signed <= 0)
            {
                scales_out[c] = 1.0f;
            }
            else
            {
                scales_out[c] = absmax / static_cast<float>(qmax_signed);
            }
        }
    }

    /**
     * Build a Requantizer from calibration scales.
     *
     * effective_scale = (input_scale * weight_scale) / output_scale.
     * Decomposes that ratio with quantizeMultiplier, then attaches the
     * destination zero_point and saturation bounds. The returned object
     * is plain data and can be embedded in a layer's constant table.
     */
    template<typename DstStorage>
    inline Requantizer<int32_t, DstStorage> buildRequantizer(
        float input_scale, float weight_scale, float output_scale,
        int32_t output_zero_point, int32_t qmin, int32_t qmax)
    {
        Requantizer<int32_t, DstStorage> r;
        const double ratio = (static_cast<double>(input_scale) *
                              static_cast<double>(weight_scale)) /
                             static_cast<double>(output_scale);
        int32_t multiplier = 0;
        int32_t shift = 0;
        quantizeMultiplier(ratio, multiplier, shift);
        r.multiplier = multiplier;
        r.shift = shift;
        r.zero_point = static_cast<DstStorage>(output_zero_point);
        r.qmin = static_cast<DstStorage>(qmin);
        r.qmax = static_cast<DstStorage>(qmax);
        return r;
    }

} // namespace tinymind

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
