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

// Unit tests for cpp/lookupTable.hpp and the precomputed activation tables
// in cpp/lookupTables.cpp. Three layers:
//   1. Branch coverage on LookupTable<Q>::getValue (clamp lo/hi, knot, flat,
//      interpolate).
//   2. Bit-exact verification that each (function, Q-format) table matches the
//      reference generator output.
//   3. Invariants and end-to-end accuracy of getValue against std:: math.

#include <cmath>
#include <cstdint>
#include <limits>

#include "qformat.hpp"
#include "interpolate.hpp"
#include "lookupTable.hpp"
#include "sigmoid.hpp"
#include "tanh.hpp"
#include "exp.hpp"
#include "log.hpp"
#include "sin.hpp"
#include "cos.hpp"
#include "lookupFunctions.hpp"
#include "error.hpp"
#include "compiler.h"

#define BOOST_TEST_MODULE lookuptable_unit_test
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING_GCC_ONLY("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

using tinymind::QValue;
using tinymind::LookupTable;

namespace {

// Convert a real value to the Q-format raw bit pattern the way the generator
// does (toSaturatedFixed64 in activationTableGenerator.cpp): truncate toward
// zero, saturate to the format's range, then reinterpret as unsigned. Without
// the saturation an out-of-range value wraps, as exp(5) did in Q8.8.
template<typename Q>
typename Q::FullWidthFieldType doubleToRaw(const double x)
{
    using V = typename Q::FullWidthValueType;
    using F = typename Q::FullWidthFieldType;
    const double scale = std::ldexp(1.0, Q::NumberOfFractionalBits);
    const double scaled = x * scale;
    const double limit = std::ldexp(1.0, static_cast<int>(sizeof(V) * 8u) - 1);
    V signedRaw;
    if (scaled >= limit)
    {
        signedRaw = std::numeric_limits<V>::max();
    }
    else if (scaled < -limit)
    {
        signedRaw = std::numeric_limits<V>::min();
    }
    else
    {
        signedRaw = static_cast<V>(scaled);
    }
    return static_cast<F>(signedRaw);
}

template<typename Q>
double rawToDouble(const typename Q::FullWidthFieldType raw)
{
    using V = typename Q::FullWidthValueType;
    const V signedRaw = static_cast<V>(raw);
    return static_cast<double>(signedRaw) / std::ldexp(1.0, Q::NumberOfFractionalBits);
}

template<typename Q>
Q makeQ(const double x)
{
    return Q(doubleToRaw<Q>(x));
}

template<typename Q>
double qToDouble(const Q& q)
{
    return rawToDouble<Q>(static_cast<typename Q::FullWidthFieldType>(q.getValue()));
}

// x_i sample point used by the generator: i in [0, 96), x_i = -6 + i/8.
double sampleX(const std::size_t i)
{
    return static_cast<double>(MIN_X_TABLE_VALUE)
         + static_cast<double>(i) / static_cast<double>(1 << ACTIVATION_DELTA_SHIFT);
}

// Reference functions matching activationTableGenerator.cpp.
double refSigmoid(double x) { const double e = std::exp(x); return e / (e + 1.0); }
double refTanh   (double x) { return std::tanh(x); }
double refExp    (double x) { return std::exp(x); }
double refLog    (double x) { return (x > 0.0) ? std::log(x) : 0.0; }
double refSin    (double x) { return std::sin(x); }
double refCos    (double x) { return std::cos(x); }

template<typename Q, typename Table>
void verifyBitExact(const Table& table, double (*ref)(double))
{
    for (std::size_t i = 0; i < NUMBER_OF_ACTIVATION_TABLE_VALUES; ++i)
    {
        const double x = sampleX(i);
        const auto expected = doubleToRaw<Q>(ref(x));
        BOOST_CHECK_EQUAL(table.values[i], expected);
    }
}

// Sweep getValue over a fine grid and compare to the canonical math function.
// Tolerance covers Q-format quantization (1 LSB) plus piecewise-linear chord
// error (h^2 * max|f''| / 8 with h = 1/8). For Q8.8 the LSB dominates; for
// Q16.16 the chord error dominates.
template<typename Q, typename Table>
void verifyAccuracy(const Table& table, double (*ref)(double),
                    const double xLow, const double xHigh,
                    const double absTol)
{
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    const double step = 1.0 / 64.0; // 8x finer than the table spacing
    for (double x = xLow; x <= xHigh; x += step)
    {
        const Q in = makeQ<Q>(x);
        const Q out = LookupTable<Q>::getValue(in, &table.values[0], MAX_INDEX);
        BOOST_CHECK_SMALL(qToDouble<Q>(out) - ref(x), absTol);
    }
}

// Knots of the log table (the mantissa range [1, 2]) and of the quarter-wave
// sin/cos tables ([0, pi/2]), computed exactly as the generator computes them.
double logKnot(const std::size_t i)
{
    return 1.0 + std::ldexp(static_cast<double>(i), -LOG_TABLE_SHIFT);
}
double trigKnot(const std::size_t i)
{
    return (std::acos(-1.0) / 2.0) * std::ldexp(static_cast<double>(i), -TRIG_TABLE_SHIFT);
}

template<typename Q, typename Table>
void verifyKnots(const Table& table, const std::size_t count, double (*knot)(std::size_t), double (*ref)(double))
{
    for (std::size_t i = 0; i < count; ++i)
    {
        BOOST_CHECK_EQUAL(table.values[i], doubleToRaw<Q>(ref(knot(i))));
    }
}

// Sweep a reduced-range lookup (LogLookup, TrigLookup) against the canonical
// function. Inputs whose true result the format cannot hold are skipped.
template<typename Q>
void verifyFunction(Q (*fn)(const Q&), double (*ref)(double),
                    const double xLow, const double xHigh, const double step,
                    const double absTol)
{
    const double largest = rawToDouble<Q>(static_cast<typename Q::FullWidthFieldType>(std::numeric_limits<typename Q::FullWidthValueType>::max()));
    for (double x = xLow; x <= xHigh; x += step)
    {
        const Q in = makeQ<Q>(x);
        const double xq = qToDouble<Q>(in);
        const double expected = ref(xq);
        if ((expected >= largest) || (expected <= -largest))
        {
            continue;
        }
        BOOST_CHECK_SMALL(qToDouble<Q>(fn(in)) - expected, absTol);
    }
}

} // namespace

BOOST_AUTO_TEST_SUITE(test_suite_lookuptable)

// ----- linearInterpolation (interpolate.hpp) ---------------------------------
// Two branches: x1 == x0 short-circuit, and the linear path.

BOOST_AUTO_TEST_CASE(linear_interpolation_endpoints_double)
{
    using tinymind::linearInterpolation;
    BOOST_CHECK_CLOSE(linearInterpolation<double>(0.0, 0.0, 1.0, 10.0, 20.0), 10.0, 1e-9);
    BOOST_CHECK_CLOSE(linearInterpolation<double>(1.0, 0.0, 1.0, 10.0, 20.0), 20.0, 1e-9);
    BOOST_CHECK_CLOSE(linearInterpolation<double>(0.5, 0.0, 1.0, 10.0, 20.0), 15.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(linear_interpolation_degenerate_returns_y0)
{
    // x1 == x0 hits the divide-by-zero guard at interpolate.hpp:29 and must
    // return y0 unchanged, not divide by zero.
    using tinymind::linearInterpolation;
    BOOST_CHECK_CLOSE(linearInterpolation<double>(5.0, 1.0, 1.0, 7.0, 99.0), 7.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(linear_interpolation_fixed_point)
{
    // Same boundary checks in Q8.8.
    using tinymind::linearInterpolation;
    using Q = QValue<8, 8, true>;
    const Q result = linearInterpolation<Q>(Q(0, 128),  // x = 0.5
                                            Q(0, 0),    // x0 = 0
                                            Q(1, 0),    // x1 = 1
                                            Q(0, 0),    // y0 = 0
                                            Q(1, 0));   // y1 = 1
    // Expect 0.5 = raw 128.
    BOOST_CHECK_EQUAL(result.getValue(), Q(0, 128).getValue());
}

// ----- Branch coverage on LookupTable<Q>::getValue ---------------------------
//
// All five reachable branches in cpp/lookupTable.hpp:35-89 are exercised on a
// single representative type (signed Q8.8 over the sigmoid table). Other
// (function, Q-format) pairs share the exact same control-flow.

using BranchQ = QValue<8, 8, true>;
using SigmoidTable8_8 =
    tinymind::SigmoidValuesTableSelector<8, 8, true>::SigmoidTableType;

BOOST_AUTO_TEST_CASE(branch_clamp_below_min)
{
    static const SigmoidTable8_8 table;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    const BranchQ in = makeQ<BranchQ>(-7.0);
    const BranchQ out = LookupTable<BranchQ>::getValue(in, &table.values[0], MAX_INDEX);
    BOOST_CHECK_EQUAL(static_cast<BranchQ::FullWidthFieldType>(out.getValue()),
                      table.values[0]);
}

BOOST_AUTO_TEST_CASE(branch_clamp_above_max)
{
    static const SigmoidTable8_8 table;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    const BranchQ in = makeQ<BranchQ>(7.0);
    const BranchQ out = LookupTable<BranchQ>::getValue(in, &table.values[0], MAX_INDEX);
    BOOST_CHECK_EQUAL(static_cast<BranchQ::FullWidthFieldType>(out.getValue()),
                      table.values[MAX_INDEX]);
}

BOOST_AUTO_TEST_CASE(branch_exact_knot)
{
    // x = 0 lands exactly on knot index 48 (since x_i = -6 + i/8). The
    // interpolation should short-circuit and return the knot value.
    static const SigmoidTable8_8 table;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    const BranchQ in = makeQ<BranchQ>(0.0);
    const BranchQ out = LookupTable<BranchQ>::getValue(in, &table.values[0], MAX_INDEX);
    BOOST_CHECK_EQUAL(static_cast<BranchQ::FullWidthFieldType>(out.getValue()),
                      table.values[48]);
}

BOOST_AUTO_TEST_CASE(branch_equal_neighbours_in_saturated_tail)
{
    // In Q8.8, sigmoid is so close to 1 by x = +5 that adjacent table entries
    // stored at LSB resolution are identical. Hits the
    // tableValues[lower]==tableValues[upper] short-circuit at line 64.
    static const SigmoidTable8_8 table;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    bool foundFlat = false;
    for (std::size_t i = 60; i < NUMBER_OF_ACTIVATION_TABLE_VALUES - 1; ++i)
    {
        if (table.values[i] == table.values[i + 1])
        {
            const double xMid = sampleX(i) + 1.0 / 16.0; // halfway between knots
            const BranchQ in = makeQ<BranchQ>(xMid);
            const BranchQ out = LookupTable<BranchQ>::getValue(in, &table.values[0], MAX_INDEX);
            BOOST_CHECK_EQUAL(static_cast<BranchQ::FullWidthFieldType>(out.getValue()),
                              table.values[i]);
            foundFlat = true;
            break;
        }
    }
    BOOST_CHECK(foundFlat);
}

BOOST_AUTO_TEST_CASE(branch_interpolation_interior)
{
    // x = 1.0625 is strictly between knots i=56 (x=1.0) and i=57 (x=1.125), in
    // a region where the sigmoid is steeply rising and adjacent table entries
    // differ. Forces the linearInterpolation fallback.
    static const SigmoidTable8_8 table;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    const BranchQ in = makeQ<BranchQ>(1.0625);
    const BranchQ out = LookupTable<BranchQ>::getValue(in, &table.values[0], MAX_INDEX);
    // Result must lie strictly between the two knot values.
    const auto lo = std::min(table.values[56], table.values[57]);
    const auto hi = std::max(table.values[56], table.values[57]);
    const auto raw = static_cast<BranchQ::FullWidthFieldType>(out.getValue());
    BOOST_CHECK_GE(raw, lo);
    BOOST_CHECK_LE(raw, hi);
    // And within ~2 LSB of the canonical sigmoid.
    BOOST_CHECK_SMALL(qToDouble<BranchQ>(out) - refSigmoid(1.0625), 2.0 / 256.0);
}

BOOST_AUTO_TEST_CASE(saturating_policy_matches_wrapping_policy_every_input)
{
    // Inside the table's range nothing saturates, so a saturating Q8.8 must
    // return exactly what the wrapping Q8.8 does, on every input -- knots,
    // interior points and both clamped tails.
    using SatQ = QValue<8, 8, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy>;
    static const SigmoidTable8_8 table;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    for (int32_t raw = -32768; raw <= 32767; ++raw)
    {
        const BranchQ in(static_cast<BranchQ::FullWidthValueType>(raw));
        const SatQ satIn(static_cast<SatQ::FullWidthValueType>(raw));
        const BranchQ out = LookupTable<BranchQ>::getValue(in, &table.values[0], MAX_INDEX);
        const SatQ satOut = LookupTable<SatQ>::getValue(satIn, &table.values[0], MAX_INDEX);
        BOOST_REQUIRE_EQUAL(static_cast<int32_t>(out.getValue()), static_cast<int32_t>(satOut.getValue()));
    }
}

// ----- Bit-exact table verification ------------------------------------------
//
// Reproduces the generator's cast chain and asserts every entry of every
// enabled table matches. Detects silent corruption of lookupTables.cpp.

BOOST_AUTO_TEST_CASE(table_bit_exact_sigmoid_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::SigmoidValuesTableSelector<8, 8, true>::SigmoidTableType t;
    verifyBitExact<Q>(t, refSigmoid);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_sigmoid_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SigmoidValuesTableSelector<16, 16, true>::SigmoidTableType t;
    verifyBitExact<Q>(t, refSigmoid);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_tanh_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::TanhValuesTableSelector<8, 8, true>::TanhTableType t;
    verifyBitExact<Q>(t, refTanh);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_tanh_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::TanhValuesTableSelector<16, 16, true>::TanhTableType t;
    verifyBitExact<Q>(t, refTanh);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_exp_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::ExpValuesTableSelector<8, 8, true>::ExpTableType t;
    verifyBitExact<Q>(t, refExp);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_exp_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::ExpValuesTableSelector<16, 16, true>::ExpTableType t;
    verifyBitExact<Q>(t, refExp);
}
// exp(x) passes Q8.8's largest value (127.996) at x = 4.85, so the last nine
// entries (x = 4.875 .. 5.875) must saturate. A raw double-to-integer cast
// wrapped them: x = 4.875 came out as -124.9 and x = 5.625 as +21.3.
BOOST_AUTO_TEST_CASE(table_exp_q8_8_saturates_instead_of_wrapping)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::ExpValuesTableSelector<8, 8, true>::ExpTableType t;
    constexpr ptrdiff_t MAX_INDEX = NUMBER_OF_ACTIVATION_TABLE_VALUES - 1;
    const int16_t maxRaw = std::numeric_limits<int16_t>::max();

    for (std::size_t i = 1; i < NUMBER_OF_ACTIVATION_TABLE_VALUES; ++i)
    {
        BOOST_REQUIRE_GE(static_cast<int16_t>(t.values[i]), static_cast<int16_t>(t.values[i - 1]));
    }
    BOOST_CHECK_EQUAL(static_cast<int16_t>(t.values[MAX_INDEX]), maxRaw);

    for (double x = 4.0; x <= 7.0; x += 1.0 / 64.0)
    {
        const Q out = LookupTable<Q>::getValue(makeQ<Q>(x), &t.values[0], MAX_INDEX);
        BOOST_REQUIRE_GT(qToDouble<Q>(out), 50.0);
    }
}

// cos(0) = 1 does not fit Q1.15, whose largest value is 1 - 2^-15; the
// wrapped entry was 0x8000, i.e. -1. The quarter-wave cos table starts at 0.
BOOST_AUTO_TEST_CASE(table_cos_q1_15_saturates_at_zero)
{
    static const tinymind::CosValuesTableSelector<1, 15, true>::CosTableType t;
    BOOST_CHECK_EQUAL(static_cast<int16_t>(t.values[0]), std::numeric_limits<int16_t>::max());
}

BOOST_AUTO_TEST_CASE(table_bit_exact_log_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::LogValuesTableSelector<8, 8, true>::LogTableType t;
    verifyKnots<Q>(t, NUMBER_OF_LOG_TABLE_VALUES, logKnot, refLog);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_log_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::LogValuesTableSelector<16, 16, true>::LogTableType t;
    verifyKnots<Q>(t, NUMBER_OF_LOG_TABLE_VALUES, logKnot, refLog);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_sin_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::SinValuesTableSelector<8, 8, true>::SinTableType t;
    verifyKnots<Q>(t, NUMBER_OF_TRIG_TABLE_VALUES, trigKnot, refSin);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_sin_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SinValuesTableSelector<16, 16, true>::SinTableType t;
    verifyKnots<Q>(t, NUMBER_OF_TRIG_TABLE_VALUES, trigKnot, refSin);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_cos_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::CosValuesTableSelector<8, 8, true>::CosTableType t;
    verifyKnots<Q>(t, NUMBER_OF_TRIG_TABLE_VALUES, trigKnot, refCos);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_cos_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::CosValuesTableSelector<16, 16, true>::CosTableType t;
    verifyKnots<Q>(t, NUMBER_OF_TRIG_TABLE_VALUES, trigKnot, refCos);
}

// ----- Invariants ------------------------------------------------------------
//
// Run on Q16.16 where quantization noise won't mask structural bugs.

BOOST_AUTO_TEST_CASE(invariant_sigmoid_monotonic_and_bounded)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SigmoidValuesTableSelector<16, 16, true>::SigmoidTableType t;
    double prev = -1.0;
    for (std::size_t i = 0; i < NUMBER_OF_ACTIVATION_TABLE_VALUES; ++i)
    {
        const double v = rawToDouble<Q>(t.values[i]);
        BOOST_CHECK_GE(v, 0.0);
        BOOST_CHECK_LE(v, 1.0);
        BOOST_CHECK_GE(v, prev);
        prev = v;
    }
}

BOOST_AUTO_TEST_CASE(invariant_tanh_monotonic_and_bounded)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::TanhValuesTableSelector<16, 16, true>::TanhTableType t;
    double prev = -2.0;
    for (std::size_t i = 0; i < NUMBER_OF_ACTIVATION_TABLE_VALUES; ++i)
    {
        const double v = rawToDouble<Q>(t.values[i]);
        BOOST_CHECK_GE(v, -1.0);
        BOOST_CHECK_LE(v, 1.0);
        BOOST_CHECK_GE(v, prev);
        prev = v;
    }
}

BOOST_AUTO_TEST_CASE(invariant_tanh_antisymmetric)
{
    // tanh(-x) == -tanh(x). Check pairs that straddle x=0 (knot index 48).
    using Q = QValue<16, 16, true>;
    static const tinymind::TanhValuesTableSelector<16, 16, true>::TanhTableType t;
    for (std::size_t k = 1; k <= 47; ++k)
    {
        const double minus = rawToDouble<Q>(t.values[48 - k]);
        const double plus  = rawToDouble<Q>(t.values[48 + k]);
        BOOST_CHECK_SMALL(minus + plus, 2.0 / 65536.0); // 2 LSB
    }
}

BOOST_AUTO_TEST_CASE(invariant_sigmoid_complement)
{
    // sigmoid(-x) + sigmoid(x) == 1 for any x.
    using Q = QValue<16, 16, true>;
    static const tinymind::SigmoidValuesTableSelector<16, 16, true>::SigmoidTableType t;
    for (std::size_t k = 1; k <= 47; ++k)
    {
        const double minus = rawToDouble<Q>(t.values[48 - k]);
        const double plus  = rawToDouble<Q>(t.values[48 + k]);
        BOOST_CHECK_SMALL(minus + plus - 1.0, 2.0 / 65536.0);
    }
}

BOOST_AUTO_TEST_CASE(invariant_cos_even)
{
    // cos(-x) == cos(x), through the range reduction on both sides.
    using Q = QValue<16, 16, true>;
    for (double x = 0.0; x <= 200.0; x += 0.0371)
    {
        const double plus = qToDouble<Q>(tinymind::TrigLookup<Q>::cos(makeQ<Q>(x)));
        const double minus = qToDouble<Q>(tinymind::TrigLookup<Q>::cos(makeQ<Q>(-x)));
        BOOST_CHECK_SMALL(minus - plus, 2.0 / 65536.0);
    }
}

BOOST_AUTO_TEST_CASE(invariant_sin_odd)
{
    // sin(-x) == -sin(x), through the range reduction on both sides.
    using Q = QValue<16, 16, true>;
    for (double x = 0.0; x <= 200.0; x += 0.0371)
    {
        const double plus = qToDouble<Q>(tinymind::TrigLookup<Q>::sin(makeQ<Q>(x)));
        const double minus = qToDouble<Q>(tinymind::TrigLookup<Q>::sin(makeQ<Q>(-x)));
        BOOST_CHECK_SMALL(minus + plus, 2.0 / 65536.0);
    }
}

BOOST_AUTO_TEST_CASE(invariant_endpoints)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SigmoidValuesTableSelector<16, 16, true>::SigmoidTableType sig;
    BOOST_CHECK_SMALL(rawToDouble<Q>(sig.values[0]),
                      0.005);
    BOOST_CHECK_SMALL(rawToDouble<Q>(sig.values[NUMBER_OF_ACTIVATION_TABLE_VALUES - 1]) - 1.0,
                      0.005);

    static const tinymind::TanhValuesTableSelector<16, 16, true>::TanhTableType th;
    BOOST_CHECK_SMALL(rawToDouble<Q>(th.values[0]) + 1.0, 1e-4);
    BOOST_CHECK_SMALL(rawToDouble<Q>(th.values[NUMBER_OF_ACTIVATION_TABLE_VALUES - 1]) - 1.0, 1e-4);
}

// ----- End-to-end accuracy of LookupTable<Q>::getValue -----------------------
//
// Tolerance: 1 LSB (Q-format quantization) + h^2 * max|f''| / 8 (chord error,
// h = 1/8). Slack added for tables built via truncate-toward-zero rather than
// round-to-nearest. For Q8.8 the LSB dominates; for Q16.16 the chord error
// dominates.

BOOST_AUTO_TEST_CASE(accuracy_sigmoid_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::SigmoidValuesTableSelector<8, 8, true>::SigmoidTableType t;
    verifyAccuracy<Q>(t, refSigmoid, -5.5, 5.5, 0.012);
}
BOOST_AUTO_TEST_CASE(accuracy_sigmoid_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SigmoidValuesTableSelector<16, 16, true>::SigmoidTableType t;
    verifyAccuracy<Q>(t, refSigmoid, -5.5, 5.5, 5e-4);
}
BOOST_AUTO_TEST_CASE(accuracy_tanh_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::TanhValuesTableSelector<8, 8, true>::TanhTableType t;
    verifyAccuracy<Q>(t, refTanh, -5.5, 5.5, 0.012);
}
BOOST_AUTO_TEST_CASE(accuracy_tanh_q16_16)
{
    // Chord-error floor at h=1/8: h^2 * max|tanh''| / 8 ~ 1.5e-3 (tanh'' peaks
    // near ~0.77 around x=+/-0.66). LSB at Q16.16 is negligible by comparison.
    using Q = QValue<16, 16, true>;
    static const tinymind::TanhValuesTableSelector<16, 16, true>::TanhTableType t;
    verifyAccuracy<Q>(t, refTanh, -5.5, 5.5, 2.5e-3);
}
// ----- LogLookup and TrigLookup ---------------------------------------------
//
// Tolerances: log interpolates log(m) on [1, 2] with knots 1/64 apart, chord
// error (1/64)^2 / 8 = 3.1e-5, plus table truncation and rounding (a few LSB
// in Q16.16). sin/cos interpolate a quarter wave with knots (pi/2)/64 apart,
// chord error (pi/128)^2 / 8 = 7.5e-5, plus about one LSB from the range
// reduction. In Q8.8 one LSB (3.9e-3) dominates.

BOOST_AUTO_TEST_CASE(accuracy_log_q16_16)
{
    using Q = QValue<16, 16, true>;
    verifyFunction<Q>(&tinymind::LogLookup<Q>::log, refLog, 1.0 / 65536.0, 0.25, 1.0 / 65536.0, 8e-5);
    verifyFunction<Q>(&tinymind::LogLookup<Q>::log, refLog, 0.25, 32767.0, 0.0731, 8e-5);
}
BOOST_AUTO_TEST_CASE(accuracy_log_q8_8)
{
    using Q = QValue<8, 8, true>;
    verifyFunction<Q>(&tinymind::LogLookup<Q>::log, refLog, 1.0 / 256.0, 127.99, 1.0 / 256.0, 2.0 * 4.0 / 256.0);
}
BOOST_AUTO_TEST_CASE(log_is_monotonic_over_every_positive_q8_8_input)
{
    using Q = QValue<8, 8, true>;
    Q previous = tinymind::LogLookup<Q>::log(Q(static_cast<Q::FullWidthFieldType>(1)));
    for (int raw = 2; raw <= std::numeric_limits<int16_t>::max(); ++raw)
    {
        const Q current = tinymind::LogLookup<Q>::log(Q(static_cast<Q::FullWidthFieldType>(raw)));
        BOOST_REQUIRE_GE(current.getValue(), previous.getValue());
        previous = current;
    }
}
BOOST_AUTO_TEST_CASE(log_of_zero_and_negative_inputs)
{
    // log(x <= 0) is minus the largest Q16.16 value, so negating it (as the
    // cross-entropy does) cannot wrap.
    using Q = QValue<16, 16, true>;
    const int32_t expected = -std::numeric_limits<int32_t>::max();
    BOOST_CHECK_EQUAL(tinymind::LogLookup<Q>::log(makeQ<Q>(0.0)).getValue(), expected);
    BOOST_CHECK_EQUAL(tinymind::LogLookup<Q>::log(makeQ<Q>(-3.5)).getValue(), expected);
    BOOST_CHECK_EQUAL(tinymind::LogLookup<Q>::log(makeQ<Q>(1.0)).getValue(), 0);
}
BOOST_AUTO_TEST_CASE(log_below_one_eighth_regression)
{
    // The [-6, 6) table interpolated toward log(0) = 0 below 0.125, so
    // log(0.01) came out as -0.166 and log(0.001) as -0.017 (#289).
    using Q = QValue<16, 16, true>;
    BOOST_CHECK_SMALL(qToDouble<Q>(tinymind::LogLookup<Q>::log(makeQ<Q>(0.01))) - std::log(qToDouble<Q>(makeQ<Q>(0.01))), 8e-5);
    BOOST_CHECK_SMALL(qToDouble<Q>(tinymind::LogLookup<Q>::log(makeQ<Q>(0.001))) - std::log(qToDouble<Q>(makeQ<Q>(0.001))), 8e-5);
}
BOOST_AUTO_TEST_CASE(cross_entropy_grows_with_confidence_in_the_wrong_class)
{
    // Target class 0; the more probability the output puts on class 1, the
    // larger the loss must be. The old log table made it shrink below 0.125.
    using Q = QValue<16, 16, true>;
    typedef tinymind::CrossEntropyLossCalculator<Q, 2> Loss;
    const Q target[2] = {makeQ<Q>(1.0), makeQ<Q>(0.0)};
    double previous = -1.0;
    for (double p = 0.999; p >= 0.0005; p *= 0.8)
    {
        const Q output[2] = {makeQ<Q>(p), makeQ<Q>(1.0 - p)};
        const double loss = qToDouble<Q>(Loss::calculateError(&target[0], &output[0]));
        BOOST_CHECK_SMALL(loss + std::log(qToDouble<Q>(output[0])), 8e-5);
        BOOST_REQUIRE_GT(loss, previous);
        previous = loss;
    }
}
BOOST_AUTO_TEST_CASE(accuracy_sin_q16_16)
{
    using Q = QValue<16, 16, true>;
    verifyFunction<Q>(&tinymind::TrigLookup<Q>::sin, refSin, -32767.0, 32767.0, 1.0371, 1.2e-4);
    verifyFunction<Q>(&tinymind::TrigLookup<Q>::sin, refSin, -7.0, 7.0, 1.0 / 1024.0, 1.2e-4);
}
BOOST_AUTO_TEST_CASE(accuracy_cos_q16_16)
{
    using Q = QValue<16, 16, true>;
    verifyFunction<Q>(&tinymind::TrigLookup<Q>::cos, refCos, -32767.0, 32767.0, 1.0371, 1.2e-4);
    verifyFunction<Q>(&tinymind::TrigLookup<Q>::cos, refCos, -7.0, 7.0, 1.0 / 1024.0, 1.2e-4);
}
BOOST_AUTO_TEST_CASE(accuracy_sin_cos_q8_8)
{
    using Q = QValue<8, 8, true>;
    verifyFunction<Q>(&tinymind::TrigLookup<Q>::sin, refSin, -127.99, 127.99, 1.0 / 256.0, 2.0 / 256.0);
    verifyFunction<Q>(&tinymind::TrigLookup<Q>::cos, refCos, -127.99, 127.99, 1.0 / 256.0, 2.0 / 256.0);
}
BOOST_AUTO_TEST_CASE(sin_cos_beyond_six_regression)
{
    // The [-6, 6) table clamped: sin(6.2) and sin(10) both came out as
    // sin(5.875) = -0.397 (#290).
    using Q = QValue<16, 16, true>;
    BOOST_CHECK_SMALL(qToDouble<Q>(tinymind::TrigLookup<Q>::sin(makeQ<Q>(6.2))) - std::sin(qToDouble<Q>(makeQ<Q>(6.2))), 1.2e-4);
    BOOST_CHECK_SMALL(qToDouble<Q>(tinymind::TrigLookup<Q>::sin(makeQ<Q>(10.0))) - std::sin(10.0), 1.2e-4);
    BOOST_CHECK_SMALL(qToDouble<Q>(tinymind::TrigLookup<Q>::cos(makeQ<Q>(-50.0))) - std::cos(-50.0), 1.2e-4);
}

BOOST_AUTO_TEST_SUITE_END()
