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
 * Sigmoid/tanh lookup tables are out of scope for this phase; they require
 * separate table generation similar to lookupTables.cpp and are tracked
 * for a later phase.
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

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
