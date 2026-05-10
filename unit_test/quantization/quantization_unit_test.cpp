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
#include <cmath>

#include "qaffine.hpp"
#include "qcalibration.hpp"
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
using tinymind::AffineParams;
using tinymind::RangeObserver;
using tinymind::computeAffineParamsAsymmetric;
using tinymind::computeAffineParamsSymmetric;
using tinymind::quantize;
using tinymind::dequantize;
using tinymind::quantizeBuffer;
using tinymind::dequantizeBuffer;
using tinymind::computePerChannelSymmetricScales;
using tinymind::buildRequantizer;
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

// ---------------------------------------------------------------------------
// Phase 2: calibration helpers (qcalibration.hpp).
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(range_observer_tracks_min_max)
{
    RangeObserver obs;
    BOOST_TEST(!obs.has_data);

    obs.observe(0.5f);
    BOOST_TEST(obs.has_data);
    BOOST_TEST(obs.min_value == 0.5f);
    BOOST_TEST(obs.max_value == 0.5f);

    const float buf[] = {-2.0f, 3.5f, 1.0f, -2.5f, 0.0f};
    obs.observe(buf, 5);
    BOOST_TEST(obs.min_value == -2.5f);
    BOOST_TEST(obs.max_value == 3.5f);

    obs.reset();
    BOOST_TEST(!obs.has_data);
    BOOST_TEST(obs.min_value == 0.0f);
}

BOOST_AUTO_TEST_CASE(asymmetric_params_symmetric_input)
{
    // Symmetric input range [-1, 1] over destination [-128, 127]:
    //   scale = 2 / 255, zero_point ~ -1/(2/255) - 128 ~ -0.5, lround -> 0.
    AffineParams p = computeAffineParamsAsymmetric(-1.0f, 1.0f, -128, 127);
    BOOST_TEST(std::abs(p.scale - (2.0f / 255.0f)) < 1e-7f);
    // zp = qmin - fmin/scale = -128 - (-1.0)/(2/255) = -128 + 127.5 = -0.5
    // lround(-0.5) is implementation-defined ties; clamp keeps it in range.
    BOOST_TEST((p.zero_point == 0 || p.zero_point == -1));
}

BOOST_AUTO_TEST_CASE(asymmetric_params_relu_range)
{
    // Post-ReLU activation: range [0, 6], extended by the calibrator to
    // ensure 0 is representable. fmin already 0, fmax=6.
    //   scale = 6/255, zero_point = qmin - 0/scale = -128.
    AffineParams p = computeAffineParamsAsymmetric(0.0f, 6.0f, -128, 127);
    BOOST_TEST(std::abs(p.scale - (6.0f / 255.0f)) < 1e-7f);
    BOOST_TEST(p.zero_point == -128);
}

BOOST_AUTO_TEST_CASE(asymmetric_params_extends_strictly_positive_range)
{
    // Range [2, 4] would skip 0; calibrator must extend fmin down to 0.
    AffineParams p = computeAffineParamsAsymmetric(2.0f, 4.0f, -128, 127);
    BOOST_TEST(std::abs(p.scale - (4.0f / 255.0f)) < 1e-7f);
    BOOST_TEST(p.zero_point == -128);
}

BOOST_AUTO_TEST_CASE(asymmetric_params_zero_range_safe)
{
    AffineParams p = computeAffineParamsAsymmetric(0.0f, 0.0f, -128, 127);
    BOOST_TEST(p.scale == 1.0f);
    BOOST_TEST(p.zero_point == -128);
}

BOOST_AUTO_TEST_CASE(symmetric_params_basic)
{
    // Weights with fmin=-2, fmax=1: absmax=2, qmax_signed=127.
    // scale = 2/127, zero_point = 0.
    AffineParams p = computeAffineParamsSymmetric(-2.0f, 1.0f, 127);
    BOOST_TEST(std::abs(p.scale - (2.0f / 127.0f)) < 1e-7f);
    BOOST_TEST(p.zero_point == 0);
}

BOOST_AUTO_TEST_CASE(symmetric_params_zero_range_safe)
{
    AffineParams p = computeAffineParamsSymmetric(0.0f, 0.0f, 127);
    BOOST_TEST(p.scale == 1.0f);
    BOOST_TEST(p.zero_point == 0);
}

BOOST_AUTO_TEST_CASE(quantize_dequantize_round_trip_within_half_lsb)
{
    // For any value in the calibrated range, |x - dequant(quant(x))| should
    // be at most 0.5 * scale (rounding to nearest grid point).
    const float fmin = -3.0f;
    const float fmax = 3.0f;
    AffineParams p = computeAffineParamsAsymmetric(fmin, fmax, -128, 127);

    const float samples[] = {-3.0f, -1.5f, -0.7f, 0.0f, 0.8f, 1.5f, 3.0f};
    for (float x : samples)
    {
        int8_t q = quantize<int8_t>(x, p.scale, p.zero_point, -128, 127);
        float r = dequantize<int8_t>(q, p.scale, p.zero_point);
        BOOST_TEST(std::abs(r - x) <= 0.5f * p.scale + 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(quantize_saturates_outside_range)
{
    AffineParams p = computeAffineParamsAsymmetric(-1.0f, 1.0f, -128, 127);
    int8_t hi = quantize<int8_t>(10.0f, p.scale, p.zero_point, -128, 127);
    int8_t lo = quantize<int8_t>(-10.0f, p.scale, p.zero_point, -128, 127);
    BOOST_TEST(static_cast<int>(hi) == 127);
    BOOST_TEST(static_cast<int>(lo) == -128);
}

BOOST_AUTO_TEST_CASE(quantize_buffer_round_trips)
{
    AffineParams p = computeAffineParamsSymmetric(-1.0f, 1.0f, 127);
    const float src[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int8_t qs[5] = {0};
    float roundtrip[5] = {0};
    quantizeBuffer<int8_t>(src, qs, 5, p.scale, p.zero_point, -127, 127);
    dequantizeBuffer<int8_t>(qs, roundtrip, 5, p.scale, p.zero_point);

    for (std::size_t i = 0; i < 5; ++i)
    {
        BOOST_TEST(std::abs(roundtrip[i] - src[i]) <= 0.5f * p.scale + 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(per_channel_symmetric_scales)
{
    // 3 channels, 4 weights each, with distinct absmaxes 0.5, 1.0, 4.0.
    const float weights[] = {
        0.1f, -0.5f, 0.3f, 0.0f,    // channel 0: absmax 0.5
        -1.0f, 0.5f, 0.0f, 0.9f,    // channel 1: absmax 1.0
        -4.0f, 2.0f, -3.5f, 1.0f,   // channel 2: absmax 4.0
    };
    float scales[3] = {0.0f, 0.0f, 0.0f};
    computePerChannelSymmetricScales(weights, 3, 4, 127, scales);

    BOOST_TEST(std::abs(scales[0] - (0.5f / 127.0f)) < 1e-7f);
    BOOST_TEST(std::abs(scales[1] - (1.0f / 127.0f)) < 1e-7f);
    BOOST_TEST(std::abs(scales[2] - (4.0f / 127.0f)) < 1e-7f);
}

BOOST_AUTO_TEST_CASE(per_channel_zero_channel_falls_back_to_unit)
{
    const float weights[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scales[1] = {0.0f};
    computePerChannelSymmetricScales(weights, 1, 4, 127, scales);
    BOOST_TEST(scales[0] == 1.0f);
}

BOOST_AUTO_TEST_CASE(build_requantizer_matches_manual_decomposition)
{
    // effective_scale = (in * w) / out = (0.25 * 0.5) / 0.25 = 0.5.
    auto r = buildRequantizer<int8_t>(0.25f, 0.5f, 0.25f,
                                      /*output_zp=*/0, -128, 127);
    BOOST_TEST(static_cast<int>(r.apply(100)) == 50);
    BOOST_TEST(static_cast<int>(r.apply(-100)) == -50);

    // Sanity: zero_point passes through.
    auto r2 = buildRequantizer<int8_t>(0.25f, 0.5f, 0.25f,
                                       /*output_zp=*/-10, -128, 127);
    BOOST_TEST(static_cast<int>(r2.apply(100)) == 40);
}

BOOST_AUTO_TEST_CASE(end_to_end_quantized_dot_product)
{
    // Verify the full calibration path: float weights + activations get
    // quantized, an int32 dot product is requantized back to int8, then
    // dequantized to a float that's close to the float reference.
    const float w_floats[] = {0.5f, -0.25f, 0.75f, 1.0f};
    const float a_floats[] = {1.0f, 2.0f, -1.0f, 0.5f};
    float ref = 0.0f;
    for (int i = 0; i < 4; ++i) ref += w_floats[i] * a_floats[i];
    // ref = 0.5 - 0.5 - 0.75 + 0.5 = -0.25

    AffineParams wp = computeAffineParamsSymmetric(-0.25f, 1.0f, 127);
    AffineParams ap = computeAffineParamsAsymmetric(-1.0f, 2.0f, -128, 127);
    AffineParams op = computeAffineParamsAsymmetric(-2.0f, 2.0f, -128, 127);

    int8_t wq[4];
    int8_t aq[4];
    quantizeBuffer<int8_t>(w_floats, wq, 4, wp.scale, 0, -127, 127);
    quantizeBuffer<int8_t>(a_floats, aq, 4, ap.scale, ap.zero_point, -128, 127);

    int32_t acc = 0;
    for (int i = 0; i < 4; ++i)
    {
        acc += static_cast<int32_t>(wq[i]) *
               (static_cast<int32_t>(aq[i]) - ap.zero_point);
    }

    auto r = buildRequantizer<int8_t>(ap.scale, wp.scale, op.scale,
                                      op.zero_point, -128, 127);
    int8_t out_q = r.apply(acc);
    float out = dequantize<int8_t>(out_q, op.scale, op.zero_point);

    // 1 LSB of output scale is op.scale; allow a couple LSBs of tolerance
    // since input/weight quantization each contribute their own rounding.
    BOOST_TEST(std::abs(out - ref) <= 3.0f * op.scale);
}
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_SUITE_END()
