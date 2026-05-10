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

// Phase 1 quantization unit tests.
//
// Exercises the integer requantization primitives in cpp/qaffine.hpp:
//
//   * SaturatingRoundingDoublingHighMul edge cases (INT32_MIN x INT32_MIN
//     saturates, sign of nudge follows sign of product).
//   * RoundingDivideByPOT (no-op for shift<=0, rounding semantics for >0).
//   * quantizeMultiplier round-trips for ratios <1, ==1, >1, and 0.0.
//   * Requantizer.apply for identity, halving, doubling, zero-point bias,
//     and saturation at the destination storage bounds.
//   * QAffineTensor float scale field present under
//     TINYMIND_ENABLE_FLOAT=1.

#include <cstdint>
#include <climits>

#include "qaffine.hpp"
#include "compiler.h"

#define BOOST_TEST_MODULE quantization_unit_test
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

using tinymind::saturatingRoundingDoublingHighMul;
using tinymind::roundingDivideByPOT;
using tinymind::Requantizer;
using tinymind::QAffineTensor;
#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
using tinymind::quantizeMultiplier;
#endif

namespace {

constexpr int32_t kQ31One = static_cast<int32_t>(0x7FFFFFFF); // ~ 1.0 in Q0.31

// Construct a Requantizer for symmetric int8 output (qmin=-128, qmax=127)
// from a Q0.31 multiplier and a shift, with zero_point = 0.
Requantizer<int32_t, int8_t> makeI8Requant(int32_t multiplier, int32_t shift, int8_t zp = 0)
{
    Requantizer<int32_t, int8_t> r;
    r.multiplier = multiplier;
    r.shift = shift;
    r.zero_point = zp;
    r.qmin = static_cast<int8_t>(-128);
    r.qmax = static_cast<int8_t>(127);
    return r;
}

} // namespace

BOOST_AUTO_TEST_SUITE(quantization)

BOOST_AUTO_TEST_CASE(saturating_high_mul_basic)
{
    // Q31 1.0 (kQ31One ~= 2^31 - 1) multiplied by any int32 returns
    // approximately the input (off by ~1 ULP because kQ31One is just
    // under 1.0 and rounding favors nearest).
    BOOST_TEST(saturatingRoundingDoublingHighMul(0, kQ31One) == 0);
    BOOST_TEST(saturatingRoundingDoublingHighMul(1000, kQ31One) == 1000);
    BOOST_TEST(saturatingRoundingDoublingHighMul(-1000, kQ31One) == -1000);

    // Q31 0.5 == 1<<30. Multiplying doubles and shifts: result is input/2.
    const int32_t half = static_cast<int32_t>(1) << 30;
    BOOST_TEST(saturatingRoundingDoublingHighMul(1000, half) == 500);
    BOOST_TEST(saturatingRoundingDoublingHighMul(-1000, half) == -500);
}

BOOST_AUTO_TEST_CASE(saturating_high_mul_int_min_squared_saturates)
{
    // The one overflow case in gemmlowp: INT32_MIN * INT32_MIN would yield
    // 2^62, doubled to 2^63 which doesn't fit in int32; saturate to MAX.
    const int32_t imin = static_cast<int32_t>(0x80000000);
    BOOST_TEST(saturatingRoundingDoublingHighMul(imin, imin) == static_cast<int32_t>(0x7FFFFFFF));
}

BOOST_AUTO_TEST_CASE(rounding_divide_by_pot_noop_for_zero_or_negative)
{
    BOOST_TEST(roundingDivideByPOT(1234, 0) == 1234);
    BOOST_TEST(roundingDivideByPOT(-1234, 0) == -1234);
    BOOST_TEST(roundingDivideByPOT(1234, -3) == 1234);
}

BOOST_AUTO_TEST_CASE(rounding_divide_by_pot_rounds_to_nearest)
{
    // Divide by 8 with rounding. Threshold is mask>>1 = 3, so remainders
    // > 3 round up.
    BOOST_TEST(roundingDivideByPOT(8, 3) == 1);    // exactly 1
    BOOST_TEST(roundingDivideByPOT(11, 3) == 1);   // 11/8=1 rem 3, no round
    BOOST_TEST(roundingDivideByPOT(12, 3) == 2);   // 12/8=1 rem 4, round up
    BOOST_TEST(roundingDivideByPOT(15, 3) == 2);   // 15/8=1 rem 7, round up
    BOOST_TEST(roundingDivideByPOT(-8, 3) == -1);
    // -12/8 = -1.5; gemmlowp rounds negative halves toward -inf -> -2.
    BOOST_TEST(roundingDivideByPOT(-12, 3) == -2);
    BOOST_TEST(roundingDivideByPOT(-11, 3) == -1); // -1.375 -> -1
    BOOST_TEST(roundingDivideByPOT(-13, 3) == -2); // -1.625 -> -2
}

BOOST_AUTO_TEST_CASE(requantizer_identity)
{
    // multiplier == kQ31One, shift == 0: rescale by ~1.0. Output equals
    // input clamped to int8 range.
    auto r = makeI8Requant(kQ31One, 0, 0);
    BOOST_TEST(static_cast<int>(r.apply(0)) == 0);
    BOOST_TEST(static_cast<int>(r.apply(50)) == 50);
    BOOST_TEST(static_cast<int>(r.apply(-50)) == -50);
}

BOOST_AUTO_TEST_CASE(requantizer_halve)
{
    // multiplier == kQ31One, shift == 1: divide-by-two with rounding.
    auto r = makeI8Requant(kQ31One, 1, 0);
    BOOST_TEST(static_cast<int>(r.apply(100)) == 50);
    BOOST_TEST(static_cast<int>(r.apply(-100)) == -50);
    BOOST_TEST(static_cast<int>(r.apply(0)) == 0);
}

BOOST_AUTO_TEST_CASE(requantizer_double_via_left_shift)
{
    // multiplier == kQ31One, shift == -1: left-shift accumulator before
    // multiply, doubling the effective scale.
    auto r = makeI8Requant(kQ31One, -1, 0);
    BOOST_TEST(static_cast<int>(r.apply(30)) == 60);
    BOOST_TEST(static_cast<int>(r.apply(-30)) == -60);
}

BOOST_AUTO_TEST_CASE(requantizer_saturates_at_qmin_qmax)
{
    auto r = makeI8Requant(kQ31One, 0, 0);
    BOOST_TEST(static_cast<int>(r.apply(1000)) == 127);    // saturate high
    BOOST_TEST(static_cast<int>(r.apply(-1000)) == -128);  // saturate low
}

BOOST_AUTO_TEST_CASE(requantizer_applies_zero_point)
{
    // zero_point=10 shifts every output up by 10 before saturation.
    auto r = makeI8Requant(kQ31One, 0, 10);
    BOOST_TEST(static_cast<int>(r.apply(0)) == 10);
    BOOST_TEST(static_cast<int>(r.apply(50)) == 60);
    BOOST_TEST(static_cast<int>(r.apply(-50)) == -40);

    // Saturation still respects qmin/qmax post-zero-point add.
    BOOST_TEST(static_cast<int>(r.apply(200)) == 127);
    BOOST_TEST(static_cast<int>(r.apply(-200)) == -128);
}

BOOST_AUTO_TEST_CASE(qaffine_tensor_holds_zero_point)
{
    QAffineTensor<int8_t> t{};
#if TINYMIND_ENABLE_FLOAT
    t.scale = 0.0625f;
    BOOST_TEST(t.scale == 0.0625f);
#endif
    t.zero_point = static_cast<int8_t>(-3);
    BOOST_TEST(static_cast<int>(t.zero_point) == -3);
}

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
BOOST_AUTO_TEST_CASE(quantize_multiplier_unity)
{
    int32_t m = 0;
    int32_t s = 0;
    quantizeMultiplier(1.0, m, s);
    // 1.0 -> mantissa 0.5, exponent 1; mantissa*2^31 == 2^30; renormalized
    // by the q == 2^31 branch isn't taken (q == 2^30 here). shift = -1
    // means left-shift by 1, then Q31-mul-by-0.5 yields the original.
    BOOST_TEST(m == (static_cast<int32_t>(1) << 30));
    BOOST_TEST(s == -1);

    Requantizer<int32_t, int8_t> r;
    r.multiplier = m;
    r.shift = s;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;
    BOOST_TEST(static_cast<int>(r.apply(42)) == 42);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_half)
{
    int32_t m = 0;
    int32_t s = 0;
    quantizeMultiplier(0.5, m, s);
    // 0.5 -> mantissa 0.5, exponent 0; q == 2^30, shift == 0.
    BOOST_TEST(m == (static_cast<int32_t>(1) << 30));
    BOOST_TEST(s == 0);

    Requantizer<int32_t, int8_t> r;
    r.multiplier = m;
    r.shift = s;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;
    BOOST_TEST(static_cast<int>(r.apply(100)) == 50);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_zero)
{
    int32_t m = 1234;
    int32_t s = 5;
    quantizeMultiplier(0.0, m, s);
    BOOST_TEST(m == 0);
    BOOST_TEST(s == 0);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_round_trip_small)
{
    // Pick an effective scale ratio < 1 (typical of conv outputs feeding
    // requantization). Verify Requantizer reproduces the float scaling
    // within 1 ULP of the int8 range for a small accumulator.
    const double ratio = 0.125; // exact, no rounding error
    int32_t m = 0;
    int32_t s = 0;
    quantizeMultiplier(ratio, m, s);

    Requantizer<int32_t, int8_t> r;
    r.multiplier = m;
    r.shift = s;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;

    BOOST_TEST(static_cast<int>(r.apply(80)) == 10);   // 80 * 0.125
    BOOST_TEST(static_cast<int>(r.apply(800)) == 100); // 800 * 0.125
    BOOST_TEST(static_cast<int>(r.apply(-80)) == -10);
}
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_SUITE_END()
