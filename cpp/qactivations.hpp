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
 * Quantized activation functions.
 *
 * In the affine quantization model the float zero is represented by the
 * tensor's zero_point, so ReLU's clamp-at-zero becomes a clamp-at-zero_point
 * in the integer domain. Saturation cliffs (ReLU6) become a clamp at the
 * quantized representation of the cliff value.
 *
 * Two ways to apply these are supported:
 *
 *   1. Standalone helpers (qrelu, qrelu6) on a buffer that has already been
 *      requantized to the destination storage type. These are the simplest
 *      drop-in replacements for the float activation functions.
 *
 *   2. Fused-clamp helpers (clampForRelu, clampForRelu6) which return
 *      (qmin, qmax) pairs that the upstream Requantizer can use directly.
 *      Folding the activation into the requantizer is what TFLite/CMSIS-NN
 *      do for runtime efficiency: no second pass over the buffer.
 *
 * Sigmoid/tanh are also supported via 256-entry int8 lookup tables. The
 * LUT is keyed by the unsigned reinterpretation of the int8 input
 * (input + 128), so the runtime apply is a single load with no compare or
 * arithmetic. Host-only builders (buildQSigmoidLUT / buildQTanhLUT) walk
 * every int8 input value, dequantize to float, apply the float reference,
 * and requantize into the destination tensor's grid; the resulting int8
 * table is embedded as a constant for the deployable build.
 *
 * int16 activations use TFLite's 513-entry interpolated table instead of
 * one entry per input: qApplyLUT16 indexes it with the top 9 bits of the
 * input and interpolates linearly with the low 7, exactly as TFLite's
 * LUTLookup does. buildQLUT16 (host-only) fills one for any function, as
 * TFLite's LUTPopulate<int16_t> does but with its samples on the lookup's
 * knots; buildQSigmoidLUT16 and buildQTanhLUT16 are the two this library's
 * importers use.
 */

namespace tinymind {

    /**
     * Pointwise quantized ReLU.
     *
     * Clamps x at zero_point. Storage may be signed (int8) or unsigned
     * (uint8); the comparison is performed in the same type. Pure integer,
     * freestanding-safe.
     */
    template<typename Storage>
    inline Storage qrelu(Storage x, Storage zero_point)
    {
        return (x < zero_point) ? zero_point : x;
    }

    template<typename Storage>
    inline void qreluBuffer(Storage* buf, std::size_t n, Storage zero_point)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            if (buf[i] < zero_point)
            {
                buf[i] = zero_point;
            }
        }
    }

    /**
     * Pointwise quantized ReLU6 (TF Lite Mobile-style activation).
     *
     * q_six is the quantized representation of the float value 6.0 in the
     * destination tensor's affine grid. Caller computes it with
     * computeQuantizedSix() during calibration and embeds it as a constant
     * alongside the layer.
     */
    template<typename Storage>
    inline Storage qrelu6(Storage x, Storage zero_point, Storage q_six)
    {
        if (x < zero_point) return zero_point;
        if (x > q_six) return q_six;
        return x;
    }

    template<typename Storage>
    inline void qrelu6Buffer(Storage* buf, std::size_t n,
                             Storage zero_point, Storage q_six)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            Storage v = buf[i];
            if (v < zero_point) v = zero_point;
            else if (v > q_six) v = q_six;
            buf[i] = v;
        }
    }

    /**
     * Size of the int8 sigmoid/tanh lookup table.
     *
     * One entry per int8 input value; index is the uint8 reinterpretation
     * of the int8 (i.e. input + 128). Storing the table as int8 keeps it
     * 256 bytes regardless of the activation.
     */
    static constexpr std::size_t kQActivationLUTSize = 256;

    /**
     * Convert an int8 input to the uint8 LUT index.
     *
     * The cast pattern is the canonical "shift-by-128" trick: int8 -128 maps
     * to index 0, int8 0 maps to index 128, int8 127 maps to index 255.
     */
    inline std::size_t qActivationLUTIndex(int8_t x)
    {
        return static_cast<std::size_t>(static_cast<uint8_t>(static_cast<int32_t>(x) + 128));
    }

    /**
     * Apply a 256-entry int8 LUT pointwise.
     *
     * The LUT is caller-owned and produced by buildQSigmoidLUT /
     * buildQTanhLUT (or any equivalent host-side table). Pure integer at
     * runtime, freestanding-safe.
     */
    inline int8_t qApplyLUT(int8_t x, const int8_t* lut)
    {
        return lut[qActivationLUTIndex(x)];
    }

    inline void qApplyLUTBuffer(int8_t* buf, std::size_t n, const int8_t* lut)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            buf[i] = lut[qActivationLUTIndex(buf[i])];
        }
    }

    /**
     * Size of an int16 lookup table: 512 base values and one more for the
     * slope of the last interval.
     */
    static constexpr std::size_t kQActivationLUT16Size = 513;

    namespace detail {

        // Two's complement wrap to int16, as TFLite's int16 arithmetic does,
        // without an implementation-defined narrowing conversion.
        inline int32_t wrapToInt16(int32_t v)
        {
            return static_cast<int32_t>((static_cast<uint32_t>(v) + 32768u) & 0xFFFFu) - 32768;
        }

    } // namespace detail

    /**
     * Look up an int16 input in a 513-entry table: the top 9 bits of the
     * input pick an interval, the low 7 interpolate along it, rounding
     * half up. Bit-exact with TFLite's LUTLookup(int16_t, const int16_t*),
     * including its int16 wrap of a slope wider than 32767 (a table that
     * steep is not one buildQLUT16 makes for a monotone function).
     */
    inline int16_t qApplyLUT16(int16_t x, const int16_t* lut)
    {
        const uint32_t biased = static_cast<uint32_t>(static_cast<int32_t>(x) + 32768);
        const std::size_t index = static_cast<std::size_t>(biased >> 7);
        const int32_t offset = static_cast<int32_t>(biased & 0x7Fu);
        const int32_t base = static_cast<int32_t>(lut[index]);
        const int32_t slope = detail::wrapToInt16(static_cast<int32_t>(lut[index + 1]) - base);
        const int32_t delta = (slope * offset + 64) >> 7;
        return static_cast<int16_t>(detail::wrapToInt16(base + delta));
    }

    /**
     * Fused-clamp helper for ReLU.
     *
     * The Requantizer that produces this layer's output already does a
     * saturation pass; folding the activation in just means raising qmin
     * to zero_point (so anything below the float zero is clamped during
     * requantization, not on a separate pass). qmax is left untouched.
     */
    template<typename Storage>
    inline void clampForRelu(Storage zero_point,
                             Storage& qmin_inout, Storage qmax)
    {
        (void)qmax; // qmax does not change for plain ReLU
        if (qmin_inout < zero_point)
        {
            qmin_inout = zero_point;
        }
    }

    /**
     * Fused-clamp helper for ReLU6.
     *
     * Raises qmin to zero_point and lowers qmax to q_six, so the
     * Requantizer's existing saturation step covers the activation.
     */
    template<typename Storage>
    inline void clampForRelu6(Storage zero_point, Storage q_six,
                              Storage& qmin_inout, Storage& qmax_inout)
    {
        if (qmin_inout < zero_point)
        {
            qmin_inout = zero_point;
        }
        if (qmax_inout > q_six)
        {
            qmax_inout = q_six;
        }
    }

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    /**
     * General threshold quantizer: returns round(threshold / scale) +
     * zero_point. Forward-declared so computeQuantizedSix below can call
     * it; the body is provided in the trailing block where <cmath> is in
     * scope.
     */
    inline int32_t computeQuantizedThreshold(float threshold,
                                             float output_scale,
                                             int32_t zero_point);

    /**
     * Compute the quantized representation of 6.0 in a target tensor's
     * affine grid for use as the q_six argument to qrelu6 / clampForRelu6.
     *
     * Host-only because the calibration step needs the float scale; the
     * resulting int32 is embedded as a constant for the deployable build.
     */
    inline int32_t computeQuantizedSix(float output_scale, int32_t zero_point)
    {
        return computeQuantizedThreshold(6.0f, output_scale, zero_point);
    }
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    inline int32_t computeQuantizedThreshold(float threshold,
                                             float output_scale,
                                             int32_t zero_point)
    {
        const long q = std::lround(static_cast<double>(threshold) /
                                   static_cast<double>(output_scale)) +
                       static_cast<long>(zero_point);
        return static_cast<int32_t>(q);
    }

    /**
     * Saturating round-to-nearest float -> int8 used by the LUT builders.
     *
     * Centralized so the sigmoid and tanh tables share identical rounding
     * semantics with std::lround (ties away from zero, matching TFLite).
     */
    inline int8_t qSaturateRoundToInt8(float v)
    {
        long q = std::lround(static_cast<double>(v));
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        return static_cast<int8_t>(q);
    }

    /**
     * Build a 256-entry int8 lookup table for sigmoid.
     *
     * For every int8 input value, dequantize to float using the input
     * tensor's (scale, zero_point), apply 1 / (1 + exp(-x)), then
     * requantize to the output tensor's grid. The output of sigmoid is in
     * (0, 1), so callers typically calibrate the output tensor with
     * output_scale = 1/256 and output_zero_point = -128 to use the full
     * int8 range; any other pair works as long as it covers (0, 1).
     *
     * lut_out must point to a buffer of kQActivationLUTSize int8 entries.
     */
    inline void buildQSigmoidLUT(float input_scale, int32_t input_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 int8_t* lut_out)
    {
        for (std::size_t idx = 0; idx < kQActivationLUTSize; ++idx)
        {
            // idx is the uint8 reinterpretation of the int8 input; recover
            // the signed input by subtracting 128 before dequantizing.
            const int32_t x_i8 = static_cast<int32_t>(idx) - 128;
            const float x_f = input_scale *
                              static_cast<float>(x_i8 - input_zero_point);
            const float y_f = 1.0f / (1.0f + std::exp(-x_f));
            const float y_q = static_cast<float>(output_zero_point) +
                              y_f / output_scale;
            lut_out[idx] = qSaturateRoundToInt8(y_q);
        }
    }

    /**
     * Build a 256-entry int8 lookup table for tanh.
     *
     * Output range is (-1, 1); a symmetric output calibration with
     * output_scale = 1/128 and output_zero_point = 0 yields the full int8
     * range. Otherwise mirrors buildQSigmoidLUT.
     */
    inline void buildQTanhLUT(float input_scale, int32_t input_zero_point,
                              float output_scale, int32_t output_zero_point,
                              int8_t* lut_out)
    {
        for (std::size_t idx = 0; idx < kQActivationLUTSize; ++idx)
        {
            const int32_t x_i8 = static_cast<int32_t>(idx) - 128;
            const float x_f = input_scale *
                              static_cast<float>(x_i8 - input_zero_point);
            const float y_f = std::tanh(x_f);
            const float y_q = static_cast<float>(output_zero_point) +
                              y_f / output_scale;
            lut_out[idx] = qSaturateRoundToInt8(y_q);
        }
    }

    /**
     * Fill a 513-entry int16 table for qApplyLUT16 with transform(x), from
     * the input grid (input_scale, input_zero_point) to the symmetric int16
     * output grid of output_scale (zero point 0, as TFLite requires of int16
     * tensors). lut_out must hold kQActivationLUT16Size entries.
     *
     * Entry i is the function at input -32768 + 128 i, the knot qApplyLUT16
     * interpolates from, less half the linear interpolation's error at the
     * interval's midpoint, as TFLite's LUTPopulateInt16 does, so the error
     * is shared between the knots and the midpoint. Unlike TFLite's, the
     * samples sit exactly on the knots: TFLite spaces them 65535 / 512
     * inputs apart and scales by 65536 / 65535 / output_scale, which skews
     * its table by up to half an input step (tanh(0) reads -3 steps on a
     * 6 / 32768 input grid). The last entry, at input 32768, only sets the
     * slope of the last interval.
     */
    inline void buildQLUT16(double input_scale, int32_t input_zero_point, double output_scale,
                            double (*transform)(double), int16_t* lut_out)
    {
        const int steps = 512;
        const double inv = 1.0 / output_scale;
        for (int i = 0; i <= steps; ++i)
        {
            const double x = input_scale * static_cast<double>(-32768 + 128 * i - input_zero_point);
            const double sample = std::round(transform(x) * inv);
            double v = sample;
            if (i < steps)
            {
                const double x_next = input_scale * static_cast<double>(-32768 + 128 * (i + 1) - input_zero_point);
                const double mid_interp = std::round((transform(x_next) * inv + sample) / 2);
                const double mid = std::round(transform((x + x_next) / 2) * inv);
                v = sample - std::round((mid_interp - mid) / 2);
            }
            lut_out[i] = static_cast<int16_t>((v < -32768.0) ? -32768.0 : ((v > 32767.0) ? 32767.0 : v));
        }
    }

    namespace detail {

        inline double logisticTransform(double x)
        {
            return 1.0 / (1.0 + std::exp(-x));
        }

        inline double tanhTransform(double x)
        {
            return std::tanh(x);
        }

    } // namespace detail

    /**
     * 513-entry int16 sigmoid table for qApplyLUT16 (see buildQLUT16);
     * output_scale = 2^-15 covers (0, 1).
     */
    inline void buildQSigmoidLUT16(double input_scale, int32_t input_zero_point, double output_scale,
                                   int16_t* lut_out)
    {
        buildQLUT16(input_scale, input_zero_point, output_scale, detail::logisticTransform, lut_out);
    }

    /**
     * 513-entry int16 tanh table for qApplyLUT16; output_scale = 2^-15
     * covers (-1, 1).
     */
    inline void buildQTanhLUT16(double input_scale, int32_t input_zero_point, double output_scale,
                                int16_t* lut_out)
    {
        buildQLUT16(input_scale, input_zero_point, output_scale, detail::tanhTransform, lut_out);
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
