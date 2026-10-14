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

#if defined(__ARM_FEATURE_SAT)
#include <arm_acle.h>
#endif

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
 * Coexists with everything else in tinymind: nothing here touches QValue,
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
     *
     * gemmlowp writes this with a sign-dependent nudge (2^30 for a
     * non-negative product, 1 - 2^30 for a negative one) followed by a
     * division that truncates toward zero. For a negative product,
     * truncation is a floor after adding 2^31 - 1, and the two offsets sum
     * to 2^30 -- so both signs reduce to floor((a * b + 2^30) / 2^31), one
     * add and one arithmetic shift with no branch on the sign. Same result
     * bit for bit (unit_test/quantization locks it against the gemmlowp
     * form, ties included); on Cortex-M4 it drops the negative-product
     * path that cost a branch and a second 64-bit add per requantize.
     * |a * b| <= 2^62, so the add cannot overflow int64, and the result
     * fits int32 except for INT32_MIN * INT32_MIN, which saturates.
     */
    inline int32_t saturatingRoundingDoublingHighMul(int32_t a, int32_t b)
    {
        const bool overflow = (a == static_cast<int32_t>(0x80000000)) &&
                              (b == static_cast<int32_t>(0x80000000));
        const int64_t ab = static_cast<int64_t>(a) * static_cast<int64_t>(b);
        const int32_t high = static_cast<int32_t>(
            (ab + (static_cast<int64_t>(1) << 30)) >> 31);
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
     * roundingDivideByPOT(saturatingRoundingDoublingHighMul(x, multiplier),
     * exponent) in one 64-bit rounding, for multiplier >= 0 and
     * 2 <= exponent <= 31 (every requantizer TFLite emits for a scale
     * ratio below 0.25).
     *
     * The two-step form rounds twice: h = floor((x*m + 2^30) / 2^31), then
     * r = floor((h + 2^(e-1) - [h < 0]) / 2^e), which is gemmlowp's
     * remainder/threshold test written as one floor. Nested floors of
     * divisions by positive integers compose, so
     *
     *   r = floor((x*m + 2^30 + 2^(30+e) - [h < 0] * 2^31) / 2^(31+e))
     *
     * with no intermediate rounding lost. [h < 0] may be replaced by
     * [x < 0]: with m >= 0 they differ only when -2^30 <= x*m < 0, where
     * h == 0 and both choices give r == 0. The shift by 31+e >= 33 is the
     * high word shifted by e-1, so on Cortex-M4 this is an SMLAL and an
     * ASR in place of the mask, threshold and compare of the two-step form.
     * Both steps' results are unchanged bit for bit; the saturating corner
     * of the high multiply needs a negative multiplier, which is excluded.
     */
    inline int32_t roundingHighMulDivideByPOT(int32_t x, int32_t multiplier, int32_t exponent)
    {
        // nudge = 2^(30+e) + 2^30 - [x < 0] * 2^31, assembled from 32-bit
        // halves (high 2^(e-2) - neg, low 2^30 | neg << 31) so the low word
        // is a constant and the sign costs a shift and a subtract. The sum
        // is formed modulo 2^64; its true value fits int64.
        const uint32_t negative = static_cast<uint32_t>(x) >> 31;
        const uint32_t nudgeHigh = (static_cast<uint32_t>(1) << (exponent - 2)) - negative;
        const uint32_t nudgeLow = (negative << 31) | static_cast<uint32_t>(0x40000000u);
        const uint64_t nudge = (static_cast<uint64_t>(nudgeHigh) << 32) | nudgeLow;
        const uint64_t v = nudge + static_cast<uint64_t>(
            static_cast<int64_t>(x) * static_cast<int64_t>(multiplier));
        return static_cast<int32_t>(static_cast<uint32_t>(v >> 32)) >> (exponent - 1);
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
            int32_t scaled;
            // multiplier >= 0 and shift >= 2 in one sign test, so the
            // optimizer can hoist a single loop-invariant branch.
            if ((multiplier | (shift - 2)) >= 0)
            {
                // The common case (scale ratio below 0.25): one rounding
                // that matches the two-step form below bit for bit.
                scaled = roundingHighMulDivideByPOT(static_cast<int32_t>(acc), multiplier, shift);
            }
            else
            {
                const int32_t left_shift = (shift < 0) ? -shift : 0;
                const int32_t right_shift = (shift > 0) ? shift : 0;

                int32_t shifted = static_cast<int32_t>(acc);
                if (left_shift > 0)
                {
                    shifted = shifted * (static_cast<int32_t>(1) << left_shift);
                }

                const int32_t mul = saturatingRoundingDoublingHighMul(shifted, multiplier);
                scaled = roundingDivideByPOT(mul, right_shift);
            }
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

    namespace detail {

        /**
         * Full representable range of a requantizer's destination storage.
         * Types without a specialization report known == false.
         */
        template<typename T>
        struct StorageRange
        {
            static constexpr bool known = false;
            static constexpr int32_t min = 0;
            static constexpr int32_t max = 0;
        };

        template<>
        struct StorageRange<int8_t>
        {
            static constexpr bool known = true;
            static constexpr int32_t min = -128;
            static constexpr int32_t max = 127;

            static int32_t saturate(int32_t v)
            {
#if defined(__ARM_FEATURE_SAT)
                // GCC 13 lowers the compare form to two compare-and-move
                // pairs; the ACLE intrinsic is one SSAT.
                return __ssat(v, 8);
#else
                return (v < min) ? min : ((v > max) ? max : v);
#endif
            }
        };

        template<>
        struct StorageRange<uint8_t>
        {
            static constexpr bool known = true;
            static constexpr int32_t min = 0;
            static constexpr int32_t max = 255;

            static int32_t saturate(int32_t v)
            {
#if defined(__ARM_FEATURE_SAT)
                return static_cast<int32_t>(__usat(v, 8));
#else
                return (v < min) ? min : ((v > max) ? max : v);
#endif
            }
        };

        /**
         * A Requantizer decoded once for a kernel's inner loop, for the
         * common case: multiplier >= 0, shift >= 2, and qmin/qmax spanning
         * the whole storage range (no fused activation clamp). Kernels
         * instantiate their output loop once with this and once with the
         * Requantizer itself, and pick per filter with init(), so the
         * choice never depends on the optimizer unswitching a branch.
         *
         * Two things leave the loop compared to Requantizer::apply. The
         * zero point is folded into the rounding nudge's high word:
         * floor((h + zp * 2^(e-1)) / 2^(e-1)) == floor(h / 2^(e-1)) + zp
         * exactly. And the clamp is against constants, one SSAT/USAT on
         * Arm, so the loop holds three requantizer values instead of six.
         * init() refuses a zero point large enough for the folded high
         * word to leave int32 (|zp| * 2^(e-1) > 2^29).
         */
        template<typename DstStorage_>
        struct FastRequantizer
        {
            typedef DstStorage_ DstStorageType;
            typedef StorageRange<DstStorageType> Range;

            int32_t multiplier;
            uint32_t nudgeHigh;
            int32_t highShift;

            template<typename SrcAccum_>
            bool init(const Requantizer<SrcAccum_, DstStorageType>& rq)
            {
                if (!Range::known || (rq.multiplier < 0) || (rq.shift < 2) || (rq.shift > 31) ||
                    (static_cast<int32_t>(rq.qmin) != Range::min) ||
                    (static_cast<int32_t>(rq.qmax) != Range::max))
                {
                    return false;
                }
                const int64_t zpTerm = static_cast<int64_t>(rq.zero_point) *
                                       (static_cast<int64_t>(1) << (rq.shift - 1));
                const int64_t limit = static_cast<int64_t>(1) << 29;
                if ((zpTerm > limit) || (zpTerm < -limit))
                {
                    return false;
                }
                multiplier = rq.multiplier;
                nudgeHigh = static_cast<uint32_t>((static_cast<int64_t>(1) << (rq.shift - 2)) + zpTerm);
                highShift = rq.shift - 1;
                return true;
            }

            // Same derivation as roundingHighMulDivideByPOT, with the zero
            // point already in nudgeHigh.
            DstStorageType apply(int32_t acc) const
            {
                const uint32_t negative = static_cast<uint32_t>(acc) >> 31;
                const uint32_t high = nudgeHigh - negative;
                const uint32_t low = (negative << 31) | static_cast<uint32_t>(0x40000000u);
                const uint64_t nudge = (static_cast<uint64_t>(high) << 32) | low;
                const uint64_t v = nudge + static_cast<uint64_t>(
                    static_cast<int64_t>(acc) * static_cast<int64_t>(multiplier));
                const int32_t r = static_cast<int32_t>(static_cast<uint32_t>(v >> 32)) >> highShift;
                return static_cast<DstStorageType>(Range::saturate(r));
            }
        };

        // Windows and weight rows up to this many bytes are copied into a
        // local by FilterWeights.
        constexpr std::size_t kLocalWeightsMaxBytes = 64;

        /**
         * One filter's weights as a kernel's output loop reads them. A row of
         * up to kLocalWeightsMaxBytes is copied into a local: every output
         * store is an int8 write that may alias the caller's int8 weight
         * array, so the optimizer must otherwise reload the weights after
         * each store, and a local it can keep in registers. Longer rows are
         * read in place.
         */
        template<typename WeightType, std::size_t N,
                 bool Copy = (N * sizeof(WeightType) <= kLocalWeightsMaxBytes)>
        struct FilterWeights
        {
            WeightType w[N];

            explicit FilterWeights(const WeightType* source)
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    w[i] = source[i];
                }
            }

            const WeightType* data() const
            {
                return w;
            }
        };

        template<typename WeightType, std::size_t N>
        struct FilterWeights<WeightType, N, false>
        {
            const WeightType* p;

            explicit FilterWeights(const WeightType* source) : p(source)
            {
            }

            const WeightType* data() const
            {
                return p;
            }
        };

    } // namespace detail

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
