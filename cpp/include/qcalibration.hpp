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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

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

    /**
     * Pure rescale (no weight component): ratio = input_scale / output_scale.
     *
     * Used by QConcat2_2D, QPad2D rescaling and any other op that copies
     * affine data from one (scale, zero_point) grid to another without an
     * intervening MAC. Equivalent to buildRequantizer with weight_scale=1.
     */
    template<typename DstStorage>
    inline Requantizer<int32_t, DstStorage> buildRescaler(
        float input_scale, float output_scale,
        int32_t output_zero_point, int32_t qmin, int32_t qmax)
    {
        return buildRequantizer<DstStorage>(input_scale, 1.0f, output_scale,
                                            output_zero_point, qmin, qmax);
    }

    /**
     * Build calibration constants for QAdd (TFLite ADD semantics).
     *
     * Decomposes the three scale ratios (real_a, real_b, real_out) into
     * Q0.31 multiplier / shift pairs around a fixed left_shift (default
     * 20 — matches TFLite). twice_max_input_scale is shared between the
     * two input rescalers and the output rescaler so the int32
     * intermediate has enough head room without overflowing.
     *
     * Output: writes into the int32 fields of `out`. The struct shape
     * matches QAdd<>; caller copies these values into the layer.
     */
    struct QAddParams
    {
        int32_t left_shift;
        int32_t input_a_multiplier;
        int32_t input_a_shift;
        int32_t input_b_multiplier;
        int32_t input_b_shift;
        int32_t output_multiplier;
        int32_t output_shift;
    };

    inline QAddParams buildQAddParams(float input_a_scale, float input_b_scale,
                                      float output_scale, int32_t left_shift = 20)
    {
        QAddParams out;
        out.left_shift = left_shift;

        const float max_in = (input_a_scale > input_b_scale)
            ? input_a_scale : input_b_scale;
        const double twice_max_input_scale =
            2.0 * static_cast<double>(max_in);

        const double real_a = static_cast<double>(input_a_scale) /
                              twice_max_input_scale;
        const double real_b = static_cast<double>(input_b_scale) /
                              twice_max_input_scale;
        const double real_out = twice_max_input_scale /
            ((static_cast<double>(static_cast<int64_t>(1) << left_shift)) *
             static_cast<double>(output_scale));

        quantizeMultiplier(real_a, out.input_a_multiplier, out.input_a_shift);
        quantizeMultiplier(real_b, out.input_b_multiplier, out.input_b_shift);
        quantizeMultiplier(real_out, out.output_multiplier, out.output_shift);
        return out;
    }

    /**
     * Build a Requantizer for QMul.
     *
     * Mul effective scale ratio is (scale_A * scale_B) / scale_Y, with no
     * extra left_shift. Convenience wrapper around quantizeMultiplier
     * that returns a populated Requantizer with the destination zero
     * point and saturation bounds attached.
     */
    template<typename DstStorage>
    inline Requantizer<int32_t, DstStorage> buildQMulRequantizer(
        float input_a_scale, float input_b_scale, float output_scale,
        int32_t output_zero_point, int32_t qmin, int32_t qmax)
    {
        Requantizer<int32_t, DstStorage> r;
        const double ratio = (static_cast<double>(input_a_scale) *
                              static_cast<double>(input_b_scale)) /
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

    /**
     * Fold a Conv2D + BatchNorm pair into a fused Conv2D.
     *
     * Pre-quantization pass. Operates on float weight / bias / BN parameter
     * buffers, writing fused float buffers that can then go through the
     * existing per-tensor / per-channel quantize path.
     *
     * Weight layout matches QConv2D's OHWI ordering: weights are
     * [NumFilters][KH][KW][InChannels] in channel-major. Biases are
     * [NumFilters]. BN parameters (gamma, beta, mean, variance) are all
     * [NumFilters] in NHWC-channel order. epsilon is the BN numerical
     * stability term, mirroring the BatchNorm forward formula
     * y = gamma * (x - mean) / sqrt(var + eps) + beta.
     *
     * The fused parameters are:
     *
     *   sigma_eff = gamma / sqrt(var + eps)
     *   fused_w[f, kh, kw, ci] = conv_w[f, kh, kw, ci] * sigma_eff[f]
     *   fused_b[f]             = (conv_b[f] - mean[f]) * sigma_eff[f] + beta[f]
     *
     * conv_b may be null; in that case the fused bias is just
     *   sigma_eff[f] * (-mean[f]) + beta[f].
     *
     * fused_w / fused_b may alias conv_w / conv_b (in-place fold is fine).
     */
    inline void foldBatchNorm(const float* conv_w, const float* conv_b,
                              const float* bn_gamma, const float* bn_beta,
                              const float* bn_mean, const float* bn_variance,
                              float epsilon,
                              std::size_t num_filters,
                              std::size_t weights_per_filter,
                              float* fused_w, float* fused_b)
    {
        for (std::size_t f = 0; f < num_filters; ++f)
        {
            const float sigma_eff =
                bn_gamma[f] / std::sqrt(bn_variance[f] + epsilon);

            for (std::size_t k = 0; k < weights_per_filter; ++k)
            {
                fused_w[f * weights_per_filter + k] =
                    conv_w[f * weights_per_filter + k] * sigma_eff;
            }

            const float b_in = (conv_b != nullptr) ? conv_b[f] : 0.0f;
            fused_b[f] = (b_in - bn_mean[f]) * sigma_eff + bn_beta[f];
        }
    }

    /**
     * Per-channel parameters for QBatchNorm (standalone int8 batchnorm).
     *
     * Each output channel carries its own (multiplier, shift, bias_addend)
     * triple. multiplier / shift encode sigma_eff_c * input_scale /
     * output_scale as Q0.31 + shift; bias_addend_c is the integer offset
     * added to the requantized accumulator before the output zero_point
     * bias and saturation pass. Layer-wide output zero_point and qmin/qmax
     * live on the QBatchNorm struct, not here, to match the existing
     * per-channel pattern in QDepthwiseConv2D / QConv2DPerChannel.
     */
    struct QBatchNormChannelParams
    {
        int32_t multiplier;
        int32_t shift;
        int32_t bias_addend;
    };

    /**
     * Build per-channel parameters for QBatchNorm from float BN constants.
     *
     * Decomposes a real-domain affine map
     *   y_real = sigma_eff_c * x_real + offset_c
     *   sigma_eff_c = gamma[c] / sqrt(variance[c] + eps)
     *   offset_c    = beta[c] - sigma_eff_c * mean[c]
     * into integer (multiplier_c, shift_c, bias_addend_c) so the int8 path
     * computes y_q = mult_shift(x_q - zp_in) + bias_addend_c + zp_out, then
     * saturates. bias_addend_c carries the per-channel offset translated
     * into the output scale; the layer's apply step does not see
     * input_zero_point because it is folded into the MAC subtraction.
     */
    inline void buildQBatchNormChannelParams(const float* gamma,
                                             const float* beta,
                                             const float* mean,
                                             const float* variance,
                                             float epsilon,
                                             float input_scale,
                                             float output_scale,
                                             std::size_t num_channels,
                                             QBatchNormChannelParams* out)
    {
        for (std::size_t c = 0; c < num_channels; ++c)
        {
            const float sigma_eff = gamma[c] /
                std::sqrt(variance[c] + epsilon);
            const double ratio = static_cast<double>(sigma_eff) *
                                 static_cast<double>(input_scale) /
                                 static_cast<double>(output_scale);
            int32_t mult = 0;
            int32_t shft = 0;
            quantizeMultiplier(ratio, mult, shft);

            const float offset_real =
                beta[c] - sigma_eff * mean[c];
            const long bias_q = std::lround(
                static_cast<double>(offset_real) /
                static_cast<double>(output_scale));

            out[c].multiplier = mult;
            out[c].shift = shft;
            out[c].bias_addend = static_cast<int32_t>(bias_q);
        }
    }

    /**
     * Build a 256-entry int32 exp lookup table for QSoftmax.
     *
     * Index convention: caller computes diff = q_in - q_max, which is in
     * [-255, 0] for int8. The LUT entry exp[diff + 255] holds
     *   round(exp(diff * input_scale) * (1 << kQSoftmaxExpScaleBits))
     * which is approximately Q1.30 fixed-point in (0, 1]. At runtime
     * QSoftmax sums the table values for each input element, then divides
     * each by the sum and scales to the int8 output grid.
     *
     * input_scale matches the input tensor's affine scale; zero_point
     * never enters this table (subtracted off in (q_in - q_max) which
     * cancels zero_point).
     */
    static constexpr std::size_t kQSoftmaxExpLUTSize = 256;
    static constexpr int kQSoftmaxExpScaleBits = 30;

    inline void buildQSoftmaxExpLUT(float input_scale, int32_t* lut_out)
    {
        const double scale = static_cast<double>(1u << kQSoftmaxExpScaleBits);
        for (std::size_t idx = 0; idx < kQSoftmaxExpLUTSize; ++idx)
        {
            const int32_t diff = static_cast<int32_t>(idx) - 255;
            const double exp_v = std::exp(
                static_cast<double>(diff) * static_cast<double>(input_scale));
            double q = exp_v * scale;
            if (q < 0.0) q = 0.0;
            if (q > 2147483647.0) q = 2147483647.0;
            lut_out[idx] = static_cast<int32_t>(::llround(q));
        }
    }

    /**
     * Phase 12 -- QLSTM calibration constants.
     *
     * Bundles every (multiplier, shift) pair the cell consumes, populated
     * from float scales emitted by host calibration. The sigmoid/tanh LUT
     * output conventions are fixed (sigmoid 1/256 zp -128, tanh 1/128 zp 0)
     * so they are not part of the input.
     *
     * Gate ordering is {i, f, g, o}, mirroring QLSTMCell.
     */
    struct QLSTMScales
    {
        float input_scale;
        float hidden_scale;
        float cell_scale;
        float lut_input_scale;
        float w_input_scale[4];
        float w_recurrent_scale[4];
    };

    struct QLSTMParams
    {
        int32_t input_to_lut_multiplier   [4];
        int32_t input_to_lut_shift        [4];
        int32_t recurrent_to_lut_multiplier[4];
        int32_t recurrent_to_lut_shift    [4];
        int32_t f_times_c_multiplier;
        int32_t f_times_c_shift;
        int32_t i_times_g_multiplier;
        int32_t i_times_g_shift;
        int32_t cell_to_tanh_multiplier;
        int32_t cell_to_tanh_shift;
        int32_t output_multiplier;
        int32_t output_shift;
    };

    /**
     * Sigmoid LUT output scale (TFLite convention).
     */
    static constexpr float kQLstmSigmoidOutputScale = 1.0f / 256.0f;
    static constexpr float kQLstmTanhOutputScale    = 1.0f / 128.0f;

    inline void buildQLSTMParams(const QLSTMScales& s, QLSTMParams& out)
    {
        for (std::size_t g = 0; g < 4; ++g)
        {
            const double r_in = static_cast<double>(s.input_scale) *
                                static_cast<double>(s.w_input_scale[g]) /
                                static_cast<double>(s.lut_input_scale);
            quantizeMultiplier(r_in, out.input_to_lut_multiplier[g],
                                     out.input_to_lut_shift[g]);

            const double r_rec = static_cast<double>(s.hidden_scale) *
                                 static_cast<double>(s.w_recurrent_scale[g]) /
                                 static_cast<double>(s.lut_input_scale);
            quantizeMultiplier(r_rec, out.recurrent_to_lut_multiplier[g],
                                      out.recurrent_to_lut_shift[g]);
        }

        const double sig = static_cast<double>(kQLstmSigmoidOutputScale);
        const double tnh = static_cast<double>(kQLstmTanhOutputScale);
        const double c   = static_cast<double>(s.cell_scale);
        const double h   = static_cast<double>(s.hidden_scale);

        quantizeMultiplier(sig * c / c, out.f_times_c_multiplier,
                                        out.f_times_c_shift);
        quantizeMultiplier(sig * tnh / c, out.i_times_g_multiplier,
                                          out.i_times_g_shift);
        quantizeMultiplier(c / static_cast<double>(s.lut_input_scale),
                           out.cell_to_tanh_multiplier,
                           out.cell_to_tanh_shift);
        quantizeMultiplier(sig * tnh / h, out.output_multiplier,
                                          out.output_shift);
    }

    /**
     * Quantize per-gate biases into the LUT input scale.
     *
     * bias_real    [4 * NumHidden] float biases per cell.
     * bias_int_out [4 * NumHidden] int32 in the LUT input scale, the
     *              shape consumed by QLSTMCell::biases.
     */
    inline void quantizeQLSTMBiases(const float* bias_real,
                                    std::size_t num_hidden,
                                    float lut_input_scale,
                                    int32_t* bias_int_out)
    {
        const std::size_t total = 4 * num_hidden;
        for (std::size_t i = 0; i < total; ++i)
        {
            const long q = std::lround(
                static_cast<double>(bias_real[i]) /
                static_cast<double>(lut_input_scale));
            bias_int_out[i] = static_cast<int32_t>(q);
        }
    }

    /**
     * Phase 12 -- QGRU calibration constants.
     *
     * Gate ordering is {r, z, n}. Same conventions as QLSTM but the cell
     * state collapses into the hidden state so there is no cell_scale.
     */
    struct QGRUScales
    {
        float input_scale;
        float hidden_scale;
        float lut_input_scale;
        float w_input_scale[3];
        float w_recurrent_scale[3];
    };

    struct QGRUParams
    {
        int32_t input_to_lut_multiplier   [3];
        int32_t input_to_lut_shift        [3];
        int32_t recurrent_to_lut_multiplier[3];
        int32_t recurrent_to_lut_shift    [3];
        int32_t r_times_h_multiplier;
        int32_t r_times_h_shift;
        int32_t one_minus_z_times_n_multiplier;
        int32_t one_minus_z_times_n_shift;
        int32_t z_times_h_multiplier;
        int32_t z_times_h_shift;
    };

    inline void buildQGRUParams(const QGRUScales& s, QGRUParams& out)
    {
        for (std::size_t g = 0; g < 3; ++g)
        {
            const double r_in = static_cast<double>(s.input_scale) *
                                static_cast<double>(s.w_input_scale[g]) /
                                static_cast<double>(s.lut_input_scale);
            quantizeMultiplier(r_in, out.input_to_lut_multiplier[g],
                                     out.input_to_lut_shift[g]);

            const double r_rec = static_cast<double>(s.hidden_scale) *
                                 static_cast<double>(s.w_recurrent_scale[g]) /
                                 static_cast<double>(s.lut_input_scale);
            quantizeMultiplier(r_rec, out.recurrent_to_lut_multiplier[g],
                                      out.recurrent_to_lut_shift[g]);
        }

        const double sig = static_cast<double>(kQLstmSigmoidOutputScale);
        const double tnh = static_cast<double>(kQLstmTanhOutputScale);
        const double h   = static_cast<double>(s.hidden_scale);

        quantizeMultiplier(sig, out.r_times_h_multiplier,
                                out.r_times_h_shift);
        quantizeMultiplier(sig * tnh / h, out.one_minus_z_times_n_multiplier,
                                          out.one_minus_z_times_n_shift);
        quantizeMultiplier(sig, out.z_times_h_multiplier,
                                out.z_times_h_shift);
    }

    inline void quantizeQGRUBiases(const float* bias_real,
                                   std::size_t num_hidden,
                                   float lut_input_scale,
                                   int32_t* bias_int_out)
    {
        const std::size_t total = 3 * num_hidden;
        for (std::size_t i = 0; i < total; ++i)
        {
            const long q = std::lround(
                static_cast<double>(bias_real[i]) /
                static_cast<double>(lut_input_scale));
            bias_int_out[i] = static_cast<int32_t>(q);
        }
    }

    /*
     * Phase 18 -- Quantized CfC (Closed-form Continuous-time) cell.
     *
     * The backbone trunk x1 emerges in the tanh LUT output scale (1/128), so the
     * ff1/ff2/time-A/time-B MACs that consume x1 all rescale from that fixed
     * scale rather than a calibrated activation scale. The elapsed time `ts` is
     * a regular-sampling constant: it folds into the time-A requantizer and into
     * the combined time bias (see buildQCfCTimeBias). Sigmoid/tanh LUT output
     * conventions are the TFLite-fixed ones shared with QLSTM / QGRU.
     */
    struct QCfCScales
    {
        float input_scale;            // x tensor
        float hidden_scale;           // h tensor
        float lut_input_scale;        // shared sigmoid/tanh LUT input scale
        float w_backbone_input_scale;
        float w_backbone_hidden_scale;
        float w_ff1_scale;
        float w_ff2_scale;
        float w_time_a_scale;
        float w_time_b_scale;
        double ts;                    // elapsed time, folded into time-A
    };

    struct QCfCParams
    {
        int32_t backbone_input_multiplier;   int32_t backbone_input_shift;
        int32_t backbone_hidden_multiplier;  int32_t backbone_hidden_shift;
        int32_t ff1_multiplier;              int32_t ff1_shift;
        int32_t ff2_multiplier;              int32_t ff2_shift;
        int32_t time_a_multiplier;           int32_t time_a_shift;
        int32_t time_b_multiplier;           int32_t time_b_shift;
        int32_t one_minus_t_times_ff1_multiplier;
        int32_t one_minus_t_times_ff1_shift;
        int32_t t_times_ff2_multiplier;
        int32_t t_times_ff2_shift;
    };

    inline void buildQCfCParams(const QCfCScales& s, QCfCParams& out)
    {
        const double lut = static_cast<double>(s.lut_input_scale);
        const double sig = static_cast<double>(kQLstmSigmoidOutputScale);
        const double tnh = static_cast<double>(kQLstmTanhOutputScale);
        const double h   = static_cast<double>(s.hidden_scale);

        // Backbone: input-MAC (input scale) + hidden-MAC (hidden scale).
        quantizeMultiplier(static_cast<double>(s.input_scale) *
                           static_cast<double>(s.w_backbone_input_scale) / lut,
                           out.backbone_input_multiplier, out.backbone_input_shift);
        quantizeMultiplier(static_cast<double>(s.hidden_scale) *
                           static_cast<double>(s.w_backbone_hidden_scale) / lut,
                           out.backbone_hidden_multiplier, out.backbone_hidden_shift);

        // Heads + time-gate consume x1 in the tanh output scale (1/128). ts
        // folds into the time-A requantizer.
        quantizeMultiplier(tnh * static_cast<double>(s.w_ff1_scale) / lut,
                           out.ff1_multiplier, out.ff1_shift);
        quantizeMultiplier(tnh * static_cast<double>(s.w_ff2_scale) / lut,
                           out.ff2_multiplier, out.ff2_shift);
        quantizeMultiplier(tnh * static_cast<double>(s.w_time_a_scale) * s.ts / lut,
                           out.time_a_multiplier, out.time_a_shift);
        quantizeMultiplier(tnh * static_cast<double>(s.w_time_b_scale) / lut,
                           out.time_b_multiplier, out.time_b_shift);

        // Interpolation back to hidden scale: (1-t)*ff1 and t*ff2 are each a
        // sigmoid-grid value times a tanh-grid value.
        quantizeMultiplier(sig * tnh / h, out.one_minus_t_times_ff1_multiplier,
                                          out.one_minus_t_times_ff1_shift);
        quantizeMultiplier(sig * tnh / h, out.t_times_ff2_multiplier,
                                          out.t_times_ff2_shift);
    }

    // Quantize a real-valued bias vector (backbone, ff1, or ff2) into the LUT
    // input scale.
    inline void quantizeQCfCBias(const float* bias_real, std::size_t count,
                                 float lut_input_scale, int32_t* bias_int_out)
    {
        for (std::size_t i = 0; i < count; ++i)
            bias_int_out[i] = static_cast<int32_t>(std::lround(
                static_cast<double>(bias_real[i]) /
                static_cast<double>(lut_input_scale)));
    }

    // Combined time-gate bias: q( b_A * ts + b_B ) in the LUT input scale. ts is
    // the regular-sampling constant (matches the time-A requantizer fold).
    inline void quantizeQCfCTimeBias(const float* b_a_real, const float* b_b_real,
                                     std::size_t num_hidden, double ts,
                                     float lut_input_scale, int32_t* bias_int_out)
    {
        for (std::size_t i = 0; i < num_hidden; ++i)
        {
            const double real = static_cast<double>(b_a_real[i]) * ts +
                                static_cast<double>(b_b_real[i]);
            bias_int_out[i] = static_cast<int32_t>(std::lround(
                real / static_cast<double>(lut_input_scale)));
        }
    }

    /**
     * Phase 13 -- QFFT1D twiddle table builder.
     *
     * Emits Q1.15 cos/sin tables for k = 0..n/2 - 1 of -2 * pi * k / n.
     * QFFT1D consumes the table verbatim. The largest representable Q1.15
     * value is 32767 (== 1.0 - 2^-15), which is the convention TFLite /
     * CMSIS-DSP follow for symmetric int16 twiddles; the all-positive 1.0
     * case (k = 0) maps to 32767 here for parity with that convention.
     */
    inline void buildQFFTTwiddles(std::size_t n, int16_t* cos_out,
                                  int16_t* sin_out)
    {
        const double two_pi = 6.283185307179586476925286766559;
        const std::size_t half = n / 2;
        for (std::size_t k = 0; k < half; ++k)
        {
            const double phase = -two_pi * static_cast<double>(k) /
                                  static_cast<double>(n);
            double c = std::cos(phase);
            double s = std::sin(phase);
            c *= 32768.0;
            s *= 32768.0;
            long ci = std::lround(c);
            long si = std::lround(s);
            if (ci >  32767) ci =  32767;
            if (ci < -32768) ci = -32768;
            if (si >  32767) si =  32767;
            if (si < -32768) si = -32768;
            cos_out[k] = static_cast<int16_t>(ci);
            sin_out[k] = static_cast<int16_t>(si);
        }
    }

    /**
     * Phase 13 -- QAttention1D (linear, ReLU-kernel) requantizer pack.
     *
     * Bundles the five Requantizers consumed by QAttention1D, populated
     * from float scales emitted by host calibration. ReLU on Q/K is folded
     * by setting q_requantizer.qmin / k_requantizer.qmin to their
     * respective zero_points (matches the clampForRelu helper).
     *
     * scale fields use the same naming as the corresponding tensor:
     *   input_scale     - X tensor
     *   q_scale, k_scale, v_scale - per-projection output scales
     *   kv_scale        - K^T @ V output
     *   output_scale    - Y tensor
     *   w_q_scale, w_k_scale, w_v_scale - symmetric weight scales (per-tensor)
     */
    struct QAttention1DScales
    {
        float input_scale;
        float w_q_scale;
        float w_k_scale;
        float w_v_scale;
        float q_scale;
        float k_scale;
        float v_scale;
        float kv_scale;
        float output_scale;
    };

    /**
     * Phase 13 -- QAttentionSoftmax1D (standard softmax) requantizer pack.
     *
     * Adds a score_scale (Q @ K^T scaled by 1/sqrt(P)) and softmax_scale
     * (TFLite-fixed at 1/256) to the linear-attention scale list. The
     * softmax zero point is also TFLite-fixed (-128) and is wired through
     * the AttnType field on QAttentionSoftmax1D, not through the
     * Requantizer pack.
     */
    struct QAttentionSoftmaxScales
    {
        float input_scale;
        float w_q_scale;
        float w_k_scale;
        float w_v_scale;
        float q_scale;
        float k_scale;
        float v_scale;
        float score_scale;
        float output_scale;
    };

    /**
     * 1 / sqrt(n) as a double. Pulled out so the score-requantizer builder
     * can fold the softmax-attention sqrt(d_k) factor without forcing
     * callers to compute it themselves.
     */
    inline double qAttentionInvSqrt(std::size_t n)
    {
        return 1.0 / std::sqrt(static_cast<double>(n));
    }

    /**
     * Phase 15 -- Percentile-based range observer.
     *
     * Records every sample so that an arbitrary percentile pair can be
     * queried after the dataset has been streamed. Heavy-tail activations
     * (post-softmax, large-receptive-field conv outputs) often have a few
     * extreme samples that pin the naive min/max range and waste most of
     * the int8 grid on outliers. Clipping at e.g. 0.05 / 99.95 percentile
     * trades a few saturating outliers for tighter mid-range resolution.
     *
     * Range query is O(n log n) on the recorded buffer; std::nth_element
     * gives O(n) average per percentile but two calls cost roughly the
     * same as one sort, so the implementation sorts once and indexes.
     *
     * has_data flips on the first sample so callers can tell "all zeros"
     * apart from "no samples yet", same convention as RangeObserver.
     */
    struct PercentileObserver
    {
        std::vector<float> samples;
        bool has_data;

        PercentileObserver()
            : samples(), has_data(false)
        {
        }

        void reset()
        {
            samples.clear();
            has_data = false;
        }

        void reserve(std::size_t n)
        {
            samples.reserve(n);
        }

        void observe(float x)
        {
            samples.push_back(x);
            has_data = true;
        }

        void observe(const float* xs, std::size_t n)
        {
            samples.reserve(samples.size() + n);
            for (std::size_t i = 0; i < n; ++i)
            {
                samples.push_back(xs[i]);
            }
            if (n > 0) has_data = true;
        }

        /**
         * Compute the [lower, upper] percentile range.
         *
         * Percentiles are in [0.0, 100.0]. lower and upper may be equal;
         * the returned range degenerates to a single value (which the
         * downstream AffineParams builder will widen to include zero).
         * Mutates the internal buffer (sorts in place).
         */
        void rangeAtPercentile(float lower_pct, float upper_pct,
                               float& fmin_out, float& fmax_out)
        {
            if (samples.empty())
            {
                fmin_out = 0.0f;
                fmax_out = 0.0f;
                return;
            }
            std::sort(samples.begin(), samples.end());
            const std::size_t n = samples.size();
            const double lo_d = static_cast<double>(lower_pct) / 100.0;
            const double hi_d = static_cast<double>(upper_pct) / 100.0;
            std::size_t lo_idx = static_cast<std::size_t>(
                lo_d * static_cast<double>(n - 1) + 0.5);
            std::size_t hi_idx = static_cast<std::size_t>(
                hi_d * static_cast<double>(n - 1) + 0.5);
            if (lo_idx >= n) lo_idx = n - 1;
            if (hi_idx >= n) hi_idx = n - 1;
            fmin_out = samples[lo_idx];
            fmax_out = samples[hi_idx];
        }
    };

    /**
     * Phase 15 -- KL-divergence (entropy) range observer.
     *
     * TensorRT-style symmetric calibration for activation tensors. The
     * float distribution is histogrammed into kNumBins; sweep candidate
     * clip thresholds T (in bin units) from kTargetBins (== 128, matches
     * int8 symmetric range) up to kNumBins, fold the |x| > T tail into
     * the boundary target bin, and compute the KL divergence between the
     * reference distribution (clipped + tail-folded) and the int8-quantized
     * distribution mapped back through the same threshold. The threshold
     * that minimizes KL is the calibrated absmax.
     *
     * Symmetric output: zero_point is forced to 0, scale = |T| / qmax.
     * Use computeAffineParamsSymmetric on the returned range to lift the
     * threshold into AffineParams when wiring into a weight tensor; for
     * activations whose zero-point need not be zero, prefer
     * PercentileObserver instead.
     */
    struct KLDivergenceObserver
    {
        static constexpr std::size_t kNumBins   = 2048;
        static constexpr std::size_t kTargetBins = 128;

        float absmax;
        std::vector<std::size_t> histogram;
        bool has_data;

        KLDivergenceObserver()
            : absmax(0.0f), histogram(kNumBins, 0u), has_data(false)
        {
        }

        void reset()
        {
            absmax = 0.0f;
            for (std::size_t i = 0; i < kNumBins; ++i) histogram[i] = 0u;
            has_data = false;
        }

        /**
         * First pass: record the absolute-value range. Caller streams the
         * dataset through observeAbsRange before observeHistogram so the
         * histogram bin width is fixed for the second pass.
         */
        void observeAbsRange(float x)
        {
            const float ax = (x < 0.0f) ? -x : x;
            if (ax > absmax) absmax = ax;
            has_data = true;
        }

        void observeAbsRange(const float* xs, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i) observeAbsRange(xs[i]);
        }

        /**
         * Second pass: drop each sample into the |x| histogram.
         */
        void observeHistogram(float x)
        {
            if (absmax <= 0.0f) return;
            const float ax = (x < 0.0f) ? -x : x;
            std::size_t bin = static_cast<std::size_t>(
                (static_cast<double>(ax) / static_cast<double>(absmax)) *
                 static_cast<double>(kNumBins));
            if (bin >= kNumBins) bin = kNumBins - 1;
            histogram[bin] += 1u;
        }

        void observeHistogram(const float* xs, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i) observeHistogram(xs[i]);
        }

        /**
         * Sweep candidate thresholds, return the bin-unit threshold that
         * minimizes the KL divergence. The float clip-threshold returned
         * by computeThreshold() is `(best_bin / kNumBins) * absmax`.
         */
        std::size_t findBestThresholdBin() const
        {
            double best_kl = -1.0;
            std::size_t best_bin = kTargetBins;
            for (std::size_t T = kTargetBins; T <= kNumBins; ++T)
            {
                std::vector<double> P(T, 0.0);
                for (std::size_t i = 0; i < T; ++i)
                {
                    P[i] = static_cast<double>(histogram[i]);
                }
                double outliers = 0.0;
                for (std::size_t i = T; i < kNumBins; ++i)
                {
                    outliers += static_cast<double>(histogram[i]);
                }
                P[T - 1] += outliers;

                std::vector<double> Q(T, 0.0);
                for (std::size_t k = 0; k < kTargetBins; ++k)
                {
                    const double start_d =
                        (static_cast<double>(k) * static_cast<double>(T)) /
                         static_cast<double>(kTargetBins);
                    const double end_d =
                        (static_cast<double>(k + 1) *
                         static_cast<double>(T)) /
                         static_cast<double>(kTargetBins);
                    const std::size_t start =
                        static_cast<std::size_t>(start_d);
                    const std::size_t end =
                        (end_d > static_cast<double>(T))
                        ? T : static_cast<std::size_t>(end_d);
                    double sum = 0.0;
                    std::size_t nonzero = 0;
                    for (std::size_t i = start; i < end; ++i)
                    {
                        sum += P[i];
                        if (histogram[i] > 0u) ++nonzero;
                    }
                    if (nonzero == 0 || sum == 0.0) continue;
                    const double avg = sum / static_cast<double>(nonzero);
                    for (std::size_t i = start; i < end; ++i)
                    {
                        if (histogram[i] > 0u) Q[i] = avg;
                    }
                }
                double P_sum = 0.0;
                double Q_sum = 0.0;
                for (std::size_t i = 0; i < T; ++i)
                {
                    P_sum += P[i];
                    Q_sum += Q[i];
                }
                if (P_sum <= 0.0 || Q_sum <= 0.0) continue;
                double kl = 0.0;
                for (std::size_t i = 0; i < T; ++i)
                {
                    const double p = P[i] / P_sum;
                    const double q = Q[i] / Q_sum;
                    if (p > 0.0 && q > 0.0)
                    {
                        kl += p * std::log(p / q);
                    }
                }
                if (best_kl < 0.0 || kl < best_kl)
                {
                    best_kl = kl;
                    best_bin = T;
                }
            }
            return best_bin;
        }

        /**
         * Calibrated absmax (clip threshold) in the float domain. Feed
         * this into computeAffineParamsSymmetric or buildRequantizer as
         * the weight/activation absmax.
         */
        float computeThreshold() const
        {
            if (!has_data || absmax <= 0.0f) return 0.0f;
            const std::size_t best_bin = findBestThresholdBin();
            return absmax *
                   (static_cast<float>(best_bin) /
                    static_cast<float>(kNumBins));
        }
    };

    /**
     * Phase 15 -- Cross-Layer Equalization (Nagel et al. 2019).
     *
     * Adjacent layers whose per-channel weight ranges are wildly imbalanced
     * suffer disproportionate quantization error: a channel with absmax
     * 0.01 next to one with absmax 1.0 loses 6+ bits of resolution under
     * per-tensor symmetric quantization. CLE rebalances those ranges by
     * pushing a positive scale through the activation; for ReLU / identity
     * (positively homogeneous) activations the model output is unchanged.
     *
     * Two adjacent dense layers (row-major, [out, in]):
     *
     *   y1 = W1 @ x + b1          ; W1 shape [C, In]
     *   y2 = W2 @ act(y1) + b2    ; W2 shape [Out, C]
     *
     * For each channel c of the intermediate tensor:
     *
     *   r1[c] = max_i |W1[c, i]|
     *   r2[c] = max_o |W2[o, c]|
     *   s[c]  = sqrt(r1[c] / r2[c])   (geometric mean equalization)
     *
     *   W1[c, :] /= s[c]
     *   b1[c]   /= s[c]
     *   W2[:, c] *= s[c]
     *
     * Zero / near-zero rows (r1[c] == 0) are skipped (no scaling possible
     * without losing the channel). The transform is fully float-domain
     * and runs before any quantization step.
     */
    inline void crossLayerEqualizeDense(float* w1, float* b1, float* w2,
                                        std::size_t in_dim,
                                        std::size_t mid_dim,
                                        std::size_t out_dim,
                                        float epsilon = 1.0e-8f)
    {
        for (std::size_t c = 0; c < mid_dim; ++c)
        {
            float r1 = 0.0f;
            for (std::size_t i = 0; i < in_dim; ++i)
            {
                const float v = w1[c * in_dim + i];
                const float a = (v < 0.0f) ? -v : v;
                if (a > r1) r1 = a;
            }
            float r2 = 0.0f;
            for (std::size_t o = 0; o < out_dim; ++o)
            {
                const float v = w2[o * mid_dim + c];
                const float a = (v < 0.0f) ? -v : v;
                if (a > r2) r2 = a;
            }
            if (r1 <= epsilon || r2 <= epsilon) continue;
            const float s = std::sqrt(r1 / r2);
            if (s <= epsilon) continue;
            const float inv_s = 1.0f / s;
            for (std::size_t i = 0; i < in_dim; ++i)
            {
                w1[c * in_dim + i] *= inv_s;
            }
            if (b1 != nullptr)
            {
                b1[c] *= inv_s;
            }
            for (std::size_t o = 0; o < out_dim; ++o)
            {
                w2[o * mid_dim + c] *= s;
            }
        }
    }

    /**
     * Conv2D variant of CLE.
     *
     * w1 is the upstream conv weights in OHWI layout
     *   [num_filters_1][kh1][kw1][in_channels_1].
     * w2 is the downstream conv weights in OHWI layout
     *   [num_filters_2][kh2][kw2][in_channels_2 == num_filters_1].
     * b1 is the upstream conv bias [num_filters_1] (may be null).
     *
     * Channel c of the intermediate is filter c of w1 and input channel c
     * of w2. Range queries collapse the spatial / non-channel axes via
     * absmax.
     */
    inline void crossLayerEqualizeConv2D(float* w1, float* b1, float* w2,
                                         std::size_t num_filters_1,
                                         std::size_t weights_per_filter_1,
                                         std::size_t num_filters_2,
                                         std::size_t kh2,
                                         std::size_t kw2,
                                         float epsilon = 1.0e-8f)
    {
        const std::size_t in_channels_2 = num_filters_1;
        const std::size_t per_filter_2 = kh2 * kw2 * in_channels_2;
        for (std::size_t c = 0; c < num_filters_1; ++c)
        {
            float r1 = 0.0f;
            for (std::size_t i = 0; i < weights_per_filter_1; ++i)
            {
                const float v = w1[c * weights_per_filter_1 + i];
                const float a = (v < 0.0f) ? -v : v;
                if (a > r1) r1 = a;
            }
            float r2 = 0.0f;
            for (std::size_t f = 0; f < num_filters_2; ++f)
            {
                for (std::size_t kh = 0; kh < kh2; ++kh)
                {
                    for (std::size_t kw = 0; kw < kw2; ++kw)
                    {
                        const std::size_t idx =
                            f * per_filter_2 +
                            kh * (kw2 * in_channels_2) +
                            kw * in_channels_2 + c;
                        const float v = w2[idx];
                        const float a = (v < 0.0f) ? -v : v;
                        if (a > r2) r2 = a;
                    }
                }
            }
            if (r1 <= epsilon || r2 <= epsilon) continue;
            const float s = std::sqrt(r1 / r2);
            if (s <= epsilon) continue;
            const float inv_s = 1.0f / s;
            for (std::size_t i = 0; i < weights_per_filter_1; ++i)
            {
                w1[c * weights_per_filter_1 + i] *= inv_s;
            }
            if (b1 != nullptr)
            {
                b1[c] *= inv_s;
            }
            for (std::size_t f = 0; f < num_filters_2; ++f)
            {
                for (std::size_t kh = 0; kh < kh2; ++kh)
                {
                    for (std::size_t kw = 0; kw < kw2; ++kw)
                    {
                        const std::size_t idx =
                            f * per_filter_2 +
                            kh * (kw2 * in_channels_2) +
                            kw * in_channels_2 + c;
                        w2[idx] *= s;
                    }
                }
            }
        }
    }

    /**
     * quantizeMultiplier for a signed ratio.
     *
     * quantizeMultiplier expects a positive ratio (it decomposes |mantissa|).
     * Coefficients in the state-space layers (a/b/c/d) are signed, so fit the
     * magnitude and carry the sign on the Q0.31 multiplier -- the runtime
     * saturatingRoundingDoublingHighMul handles a negative multiplier directly.
     */
    inline void signedQuantizeMultiplier(double ratio, int32_t& multiplier, int32_t& shift)
    {
        const double mag = (ratio < 0.0) ? -ratio : ratio;
        quantizeMultiplier(mag, multiplier, shift);
        if (ratio < 0.0)
        {
            multiplier = -multiplier;
        }
    }

    /**
     * Phase 19 -- diagonal state-space (QStateSpace1D) per-channel parameters.
     *
     * Fills the four per-channel (multiplier, shift) coefficient arrays from
     * the real-valued diagonal coefficients and the activation / state scales:
     *
     *   a[c]  decay        : ratio = a[c]                     (dimensionless)
     *   b[c]  input drive  : ratio = b[c] * x_scale / s_scale
     *   c[c]  readout      : ratio = c[c] * s_scale / y_scale
     *   d[c]  skip         : ratio = d[c] * x_scale / y_scale (d may be nullptr)
     *
     * s_scale is the calibrated state scale (state stored int32). Stability
     * needs |a[c]| < 1 so the recurrence -- and the int32 state -- stays bounded.
     * The d_* output arrays may be nullptr together with d_real to skip the
     * skip path.
     */
    inline void buildQSSMParams(const float* a_real, const float* b_real,
                                const float* c_real, const float* d_real,
                                float x_scale, float s_scale, float y_scale,
                                std::size_t num_channels,
                                int32_t* a_mult, int32_t* a_shift,
                                int32_t* b_mult, int32_t* b_shift,
                                int32_t* c_mult, int32_t* c_shift,
                                int32_t* d_mult, int32_t* d_shift)
    {
        for (std::size_t c = 0; c < num_channels; ++c)
        {
            signedQuantizeMultiplier(static_cast<double>(a_real[c]),
                                     a_mult[c], a_shift[c]);
            signedQuantizeMultiplier(static_cast<double>(b_real[c]) *
                                     static_cast<double>(x_scale) /
                                     static_cast<double>(s_scale),
                                     b_mult[c], b_shift[c]);
            signedQuantizeMultiplier(static_cast<double>(c_real[c]) *
                                     static_cast<double>(s_scale) /
                                     static_cast<double>(y_scale),
                                     c_mult[c], c_shift[c]);
            if (d_real != nullptr && d_mult != nullptr)
            {
                signedQuantizeMultiplier(static_cast<double>(d_real[c]) *
                                         static_cast<double>(x_scale) /
                                         static_cast<double>(y_scale),
                                         d_mult[c], d_shift[c]);
            }
        }
    }

    /**
     * Phase 19 -- per-channel hard-sigmoid gate parameters for
     * QSelectiveStateSpace1D.
     *
     * The gate pre-activation is g_pre = wg[c] * x_real + bg[c], emitted in Q15
     * (so the layer can clamp it to [0, 1] == [0, 32767] directly):
     *
     *   gate_mult/shift : ratio = wg[c] * x_scale * 2^15  (applied to xq - x_zp)
     *   gate_bias       : round(bg[c] * 2^15)
     */
    inline void buildQSelectiveGateParams(const float* wg_real, const float* bg_real,
                                          float x_scale, std::size_t num_channels,
                                          int32_t* gate_mult, int32_t* gate_shift,
                                          int32_t* gate_bias)
    {
        const double q15 = 32768.0;
        for (std::size_t c = 0; c < num_channels; ++c)
        {
            signedQuantizeMultiplier(static_cast<double>(wg_real[c]) *
                                     static_cast<double>(x_scale) * q15,
                                     gate_mult[c], gate_shift[c]);
            gate_bias[c] = static_cast<int32_t>(std::lround(
                static_cast<double>(bg_real[c]) * q15));
        }
    }

} // namespace tinymind

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
