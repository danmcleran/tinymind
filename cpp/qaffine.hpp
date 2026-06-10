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
 * Affine (per-tensor) integer quantization primitives.
 *
 * Distinct from the QValue Q-format pipeline. QValue is a compile-time
 * fixed/fractional bit split with no runtime metadata; this file adds the
 * TFLite/CMSIS-NN style affine model:
 *
 *     real = scale * (q - zero_point)
 *
 * The inference path is pure integer: an int32 accumulator from a quantized
 * layer is rescaled to the destination storage type via a precomputed
 * (multiplier, shift, zero_point) triple that captures the ratio of input
 * scales to output scale. No floating point is required at runtime; the
 * float scale field on QAffineTensor is host-side calibration metadata only,
 * and is compiled out unless TINYMIND_ENABLE_FLOAT=1.
 *
 * Coexists with everything else in TinyMind: nothing here touches QValue,
 * the NeuralNet template, or any existing layer. Quantized layers added in
 * later phases will use Requantizer between accumulation and storage.
 */

namespace tinymind {

    /**
     * Saturating rounding doubling high multiply.
     *
     * Computes (a * b + 2^30) / 2^31 with INT32_MIN x INT32_MIN saturated
     * to INT32_MAX. This is the canonical Q0.31 fixed-point multiply used
     * by gemmlowp and TFLite reference kernels: given a Q0.31 multiplier
     * and an int32 input, it returns the high 32 bits of the doubled
     * product, with rounding to nearest.
     */
    inline int32_t saturatingRoundingDoublingHighMul(int32_t a, int32_t b)
    {
        const bool overflow = (a == static_cast<int32_t>(0x80000000)) &&
                              (b == static_cast<int32_t>(0x80000000));
        const int64_t a64 = static_cast<int64_t>(a);
        const int64_t b64 = static_cast<int64_t>(b);
        const int64_t ab = a64 * b64;
        const int32_t nudge = (ab >= 0)
            ? static_cast<int32_t>(1 << 30)
            : static_cast<int32_t>(1 - (1 << 30));
        const int32_t high = static_cast<int32_t>((ab + nudge) / (static_cast<int64_t>(1) << 31));
        return overflow ? static_cast<int32_t>(0x7FFFFFFF) : high;
    }

    /**
     * Rounding divide by power of two.
     *
     * Equivalent to `round(x / 2^exponent)` with banker's-rounding semantics
     * matching gemmlowp: ties round away from zero except where the bias
     * adjustment for negative numbers offsets the threshold by one. exponent
     * must be non-negative and <= 31.
     */
    inline int32_t roundingDivideByPOT(int32_t x, int32_t exponent)
    {
        if (exponent <= 0)
        {
            return x;
        }

        // Build the mask through unsigned so exponent == 31 is well-defined:
        // the signed `(1 << 31) - 1` overflowed (1 << 31 is INT32_MIN, then
        // - 1 wraps), yet the doc contract explicitly allows exponent <= 31.
        const int32_t mask = static_cast<int32_t>(
            (static_cast<uint32_t>(1) << exponent) - 1u);
        const int32_t remainder = x & mask;
        const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
        return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
    }

    /**
     * Combine saturatingRoundingDoublingHighMul with roundingDivideByPOT,
     * and handle the left-shift case (shift < 0) up front. This is the
     * gemmlowp / TFLite "MultiplyByQuantizedMultiplier" primitive: takes
     * an int32 accumulator and an integer (multiplier, shift) pair and
     * returns the rescaled int32 without zero-point bias or saturation.
     *
     * Phase 10 callers (QAdd's per-input rescaling, QConcat) use this
     * directly when they need the int32 intermediate before the final
     * Requantizer; the existing Requantizer.apply continues to call the
     * primitives separately for back-compat.
     */
    inline int32_t multiplyByQuantizedMultiplier(int32_t x, int32_t multiplier,
                                                 int32_t shift)
    {
        const int32_t left_shift  = (shift < 0) ? -shift : 0;
        const int32_t right_shift = (shift > 0) ?  shift : 0;

        // Left-shift in 64-bit head room, then saturate back to int32 before
        // the Q0.31 multiply. The previous `x * (1 << left_shift)` was
        // signed-overflow UB: `1 << left_shift` itself overflows at
        // left_shift >= 31, and the int32 product overflows for a large
        // accumulator at moderate left_shift. In int64 the shift is
        // well-defined and the saturation clamps to the int32 rails -- so the
        // result is unchanged for in-contract (small left_shift) inputs and
        // merely saturates instead of invoking UB for out-of-contract ones.
        // |x| << 31 fits int64, and a distance past 31 can only saturate a
        // non-zero x, so cap the distance to keep the shift defined.
        int64_t shifted = static_cast<int64_t>(x);
        if (left_shift > 0)
        {
            // Multiply rather than `<<`: shifting a negative value is itself
            // UB. (1 << sh) with sh <= 31 is a positive int64; |x| * 2^31
            // fits int64, so the product is well-defined for any sign.
            const int32_t sh = (left_shift < 31) ? left_shift : 31;
            shifted *= (static_cast<int64_t>(1) << sh);
        }
        if (shifted > static_cast<int64_t>(INT32_MAX)) { shifted = INT32_MAX; }
        if (shifted < static_cast<int64_t>(INT32_MIN)) { shifted = INT32_MIN; }

        const int32_t mul = saturatingRoundingDoublingHighMul(
            static_cast<int32_t>(shifted), multiplier);
        return roundingDivideByPOT(mul, right_shift);
    }

    /**
     * Per-tensor affine quantization metadata.
     *
     * StorageType is the runtime representation (e.g. int8_t). The scale
     * field is host-only calibration data and is gated on
     * TINYMIND_ENABLE_FLOAT so that freestanding inference builds carry
     * only the integer zero_point.
     */
    template<typename StorageType_>
    struct QAffineTensor
    {
        typedef StorageType_ StorageType;

#if TINYMIND_ENABLE_FLOAT
        float scale;
#endif // TINYMIND_ENABLE_FLOAT
        StorageType zero_point;
    };

    /**
     * Integer requantizer: int32 accumulator -> destination storage.
     *
     * The (multiplier, shift) pair encodes the ratio of effective scales
     * (input_scale * weight_scale / output_scale) as a Q0.31 multiplier
     * combined with a shift. A negative shift left-shifts the accumulator
     * before the multiply (capturing scale ratios > 1.0); a positive shift
     * right-shifts the result of the multiply with rounding.
     *
     * The qmin/qmax fields drive saturation to the destination domain
     * (e.g. -128/127 for symmetric int8, 0/255 for uint8).
     */
    template<typename SrcAccum_, typename DstStorage_>
    struct Requantizer
    {
        typedef SrcAccum_ SrcAccumType;
        typedef DstStorage_ DstStorageType;

        int32_t multiplier;
        int32_t shift;
        DstStorageType zero_point;
        DstStorageType qmin;
        DstStorageType qmax;

        DstStorageType apply(SrcAccumType acc) const
        {
            const int32_t left_shift = (shift < 0) ? -shift : 0;
            const int32_t right_shift = (shift > 0) ? shift : 0;

            int32_t shifted = static_cast<int32_t>(acc);
            if (left_shift > 0)
            {
                shifted = shifted * (static_cast<int32_t>(1) << left_shift);
            }

            const int32_t mul = saturatingRoundingDoublingHighMul(shifted, multiplier);
            const int32_t scaled = roundingDivideByPOT(mul, right_shift);
            int32_t with_zp = scaled + static_cast<int32_t>(zero_point);

            if (with_zp < static_cast<int32_t>(qmin))
            {
                with_zp = static_cast<int32_t>(qmin);
            }
            if (with_zp > static_cast<int32_t>(qmax))
            {
                with_zp = static_cast<int32_t>(qmax);
            }

            return static_cast<DstStorageType>(with_zp);
        }
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    /**
     * Decompose a positive real-valued scale ratio into the Q0.31
     * (multiplier, shift) pair consumed by Requantizer. Host-only: relies
     * on std::frexp from <cmath>.
     *
     * Output convention:
     *   shift < 0: accumulator is left-shifted before multiply (ratio > 1)
     *   shift > 0: result is right-shifted after multiply (ratio < 1)
     *   shift == 0: multiplier alone scales the accumulator
     *
     * Caller passes ratio = (input_scale * weight_scale) / output_scale
     * computed during calibration; the result is stable across runs.
     */
    inline void quantizeMultiplier(double ratio, int32_t& multiplier, int32_t& shift);
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    inline void quantizeMultiplier(double ratio, int32_t& multiplier, int32_t& shift)
    {
        if (ratio == 0.0)
        {
            multiplier = 0;
            shift = 0;
            return;
        }

        int exponent = 0;
        const double mantissa = std::frexp(ratio, &exponent);
        // mantissa is in [0.5, 1.0); scale to Q0.31 by multiplying by 2^31.
        int64_t q = static_cast<int64_t>(::llround(mantissa * static_cast<double>(static_cast<int64_t>(1) << 31)));

        // frexp returns mantissa in [0.5, 1.0), and we round to a fixed
        // 31-bit value, which can yield 2^31 (== mantissa rounded up to 1.0).
        // Renormalize that single edge case.
        if (q == (static_cast<int64_t>(1) << 31))
        {
            q >>= 1;
            ++exponent;
        }

        multiplier = static_cast<int32_t>(q);
        shift = -exponent;
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
