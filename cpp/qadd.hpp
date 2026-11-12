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
 * Quantized elementwise ADD (TFLite ADD semantics).
 *
 * Adds two int8 affine tensors with potentially different scales and
 * zero_points and produces an int8 affine output. Required for skip
 * connections in ResNet, MobileNet-V2/V3, EfficientNet, U-Net.
 *
 * Math (matches TFLite reference kernel and CMSIS-NN):
 *
 *   twice_max  = 2 * max(scale_A, scale_B)
 *   real_a     = scale_A / twice_max
 *   real_b     = scale_B / twice_max
 *   real_out   = twice_max / ((1 << left_shift) * scale_Y)
 *
 *   shifted_a  = (qA - zp_A) << left_shift
 *   shifted_b  = (qB - zp_B) << left_shift
 *   scaled_a   = MultiplyByQuantizedMultiplier(shifted_a, mult_a, shift_a)
 *   scaled_b   = MultiplyByQuantizedMultiplier(shifted_b, mult_b, shift_b)
 *   sum        = scaled_a + scaled_b
 *   out        = MultiplyByQuantizedMultiplier(sum, mult_out, shift_out)
 *                + zp_Y, saturated to [qmin, qmax]
 *
 * left_shift is normally 20 (TFLite default); it is captured by the
 * (mult_a, shift_a) and (mult_out, shift_out) decompositions so calibration
 * code does the heavy lifting and the runtime only does integer math.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    namespace detail {

        /**
         * One QAdd input's rescaling, decoded once per forward() for the
         * common case, as a single 64-bit rounding:
         *
         *   multiplyByQuantizedMultiplier((q - zp) << left_shift, m, e)
         *
         * for multiplier m >= 0 and 0 <= e <= 31, provided that
         * |q - zp| << (left_shift + 1) stays within 2^30 for every q in the
         * storage range (TFLite's left_shift of 20 leaves eight bits spare
         * for int8). The reference rounds twice: h = floor((x*m + 2^30) /
         * 2^31), then roundingDivideByPOT(h, e). With x2 = 2x, both roundings
         * compose into one floor (the derivation is roundingHighMulDivideByPOT's
         * in qaffine.hpp, with [h < 0] replaced by [x < 0], valid for m >= 0):
         *
         *   e == 0:  floor((x2*m + 2^31) / 2^32)
         *   e >= 1:  floor((x2*m + 2^31 + 2^32 * (2^(e-1) - [x < 0])) / 2^(32+e))
         *
         * The doubling puts the shift at 32 or more for every e, so the
         * result is the high word of an SMLAL shifted right by e, and e == 0
         * needs no case of its own. |x2*m| <= 2^61 and the high word is at
         * most about 2^30, so nothing overflows; the saturating corner of the
         * reference's high multiply needs x == INT32_MIN, which the bound on
         * x excludes. The result equals the reference bit for bit:
         * unit_test/formal/requantizer proves apply() equal to it for every
         * input and parameter in this domain (the addin jobs), and
         * unit_test/quantization compares QAdd with it over every input pair.
         */
        template<typename Storage_>
        struct FastAddInput
        {
            int32_t zero_point;
            int32_t leftScale2;
            int32_t multiplier;
            uint32_t nudgeHigh;
            uint32_t negativeStep;
            int32_t shift;

            bool init(Storage_ zp, int32_t left_shift, int32_t mult, int32_t sh)
            {
                typedef StorageRange<Storage_> Range;
                if (!Range::known || (mult < 0) || (sh < 0) || (sh > 31) ||
                    (left_shift < 0) || (left_shift > 29))
                {
                    return false;
                }
                const int32_t z = static_cast<int32_t>(zp);
                const int32_t below = z - Range::min;
                const int32_t above = Range::max - z;
                const int32_t span = (below > above) ? below : above;
                if (span > (static_cast<int32_t>(1) << (29 - left_shift)))
                {
                    return false;
                }
                zero_point = z;
                leftScale2 = static_cast<int32_t>(1) << (left_shift + 1);
                multiplier = mult;
                nudgeHigh = (sh > 0) ? (static_cast<uint32_t>(1) << (sh - 1)) : 0u;
                negativeStep = (sh > 0) ? 1u : 0u;
                shift = sh;
                return true;
            }

            int32_t apply(int32_t q) const
            {
                const int32_t x2 = (q - zero_point) * leftScale2;
                const uint32_t negative = (static_cast<uint32_t>(x2) >> 31) & negativeStep;
                const uint64_t nudge =
                    (static_cast<uint64_t>(nudgeHigh - negative) << 32) |
                    static_cast<uint32_t>(0x80000000u);
                const uint64_t v = nudge + static_cast<uint64_t>(
                    static_cast<int64_t>(x2) * static_cast<int64_t>(multiplier));
                return static_cast<int32_t>(static_cast<uint32_t>(v >> 32)) >> shift;
            }
        };

        /**
         * QAdd's reference loop: the TFLite arithmetic, for any parameters.
         * Kept out of line so that QAdd::forward stays small enough to be
         * inlined into its caller (see there), and templated on the storage
         * types only, with a run-time length, so that every QAdd of the same
         * storage types shares one copy instead of one per length.
         */
        template<typename InA, typename InB, typename Out>
        TINYMIND_NOINLINE void qaddReference(const InA* a, const InB* b, Out* out, std::size_t n,
                                             InA zeroPointA, InB zeroPointB, int32_t left_shift,
                                             int32_t multA, int32_t shiftA,
                                             int32_t multB, int32_t shiftB,
                                             const Requantizer<int32_t, Out>& requantizer)
        {
            // Copy the requantizer into a local: every output store is an
            // int8_t write, which may alias it, so its fields would otherwise
            // be reloaded once per element.
            const int32_t zpA = static_cast<int32_t>(zeroPointA);
            const int32_t zpB = static_cast<int32_t>(zeroPointB);
            const int32_t leftScale = static_cast<int32_t>(1) << left_shift;
            const Requantizer<int32_t, Out> rq = requantizer;

            for (std::size_t i = 0; i < n; ++i)
            {
                const int32_t a_minus_zp = static_cast<int32_t>(a[i]) - zpA;
                const int32_t b_minus_zp = static_cast<int32_t>(b[i]) - zpB;

                const int32_t shifted_a = a_minus_zp * leftScale;
                const int32_t shifted_b = b_minus_zp * leftScale;

                const int32_t scaled_a = multiplyByQuantizedMultiplier(
                    shifted_a, multA, shiftA);
                const int32_t scaled_b = multiplyByQuantizedMultiplier(
                    shifted_b, multB, shiftB);

                const int32_t sum = scaled_a + scaled_b;
                out[i] = rq.apply(sum);
            }
        }

    } // namespace detail

    template<
        typename InAStorage_,
        typename InBStorage_,
        typename OutStorage_,
        std::size_t N_>
    struct QAdd
    {
        typedef InAStorage_ InAType;
        typedef InBStorage_ InBType;
        typedef OutStorage_ OutType;

        static constexpr std::size_t Size = N_;

        InAType input_a_zero_point;
        InBType input_b_zero_point;
        int32_t left_shift;
        int32_t input_a_multiplier;
        int32_t input_a_shift;
        int32_t input_b_multiplier;
        int32_t input_b_shift;
        Requantizer<int32_t, OutType> output_requantizer;

        void forward(const InAType* a, const InBType* b, OutType* out) const
        {
            // Decode both input rescalings and the output requantizer once.
            // When all three fit the fast forms, which covers TFLite ADD
            // without a fused activation, one loop does the work. Anything
            // else takes the reference loop, kept out of line: with it
            // inlined here, forward() grows past what the compiler inlines
            // into its caller, and a const layer object (as the importers
            // emit) no longer has its parameters folded into the loop as
            // constants -- 41% slower on the F411 than with them folded.
            detail::FastAddInput<InAType> fastA;
            detail::FastAddInput<InBType> fastB;
            detail::FastRequantizer<OutType> fastOut;
            if (fastA.init(input_a_zero_point, left_shift, input_a_multiplier, input_a_shift) &&
                fastB.init(input_b_zero_point, left_shift, input_b_multiplier, input_b_shift) &&
                fastOut.init(output_requantizer))
            {
                for (std::size_t i = 0; i < N_; ++i)
                {
                    const int32_t sum = fastA.apply(static_cast<int32_t>(a[i])) +
                                        fastB.apply(static_cast<int32_t>(b[i]));
                    out[i] = fastOut.apply(sum);
                }
                return;
            }
            detail::qaddReference(a, b, out, N_, input_a_zero_point, input_b_zero_point, left_shift,
                                  input_a_multiplier, input_a_shift,
                                  input_b_multiplier, input_b_shift, output_requantizer);
        }
    };

} // namespace tinymind
