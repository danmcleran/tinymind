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
#include "lookupTable.hpp"
#include "sigmoid.hpp"
#include "tanh.hpp"
#include "exp.hpp"
#include "log.hpp"
#include "sin.hpp"
#include "cos.hpp"
#include "compiler.h"

#define BOOST_TEST_MODULE lookuptable_unit_test
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

using tinymind::QValue;
using tinymind::LookupTable;

namespace {

// Convert a real value to the Q-format raw bit pattern using the same cast
// chain the generator uses (truncate toward zero, then reinterpret as
// unsigned). Matches activationTableGenerator.cpp:320 num = activate*scale,
// then cast-to-unsigned.
template<typename Q>
typename Q::FullWidthFieldType doubleToRaw(const double x)
{
    using V = typename Q::FullWidthValueType;
    using F = typename Q::FullWidthFieldType;
    const double scale = std::ldexp(1.0, Q::NumberOfFractionalBits);
    const V signedRaw = static_cast<V>(x * scale);
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

} // namespace

BOOST_AUTO_TEST_SUITE(test_suite_lookuptable)

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
BOOST_AUTO_TEST_CASE(table_bit_exact_log_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::LogValuesTableSelector<8, 8, true>::LogTableType t;
    verifyBitExact<Q>(t, refLog);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_log_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::LogValuesTableSelector<16, 16, true>::LogTableType t;
    verifyBitExact<Q>(t, refLog);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_sin_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::SinValuesTableSelector<8, 8, true>::SinTableType t;
    verifyBitExact<Q>(t, refSin);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_sin_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SinValuesTableSelector<16, 16, true>::SinTableType t;
    verifyBitExact<Q>(t, refSin);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_cos_q8_8)
{
    using Q = QValue<8, 8, true>;
    static const tinymind::CosValuesTableSelector<8, 8, true>::CosTableType t;
    verifyBitExact<Q>(t, refCos);
}
BOOST_AUTO_TEST_CASE(table_bit_exact_cos_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::CosValuesTableSelector<16, 16, true>::CosTableType t;
    verifyBitExact<Q>(t, refCos);
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
    // cos(-x) == cos(x).
    using Q = QValue<16, 16, true>;
    static const tinymind::CosValuesTableSelector<16, 16, true>::CosTableType t;
    for (std::size_t k = 1; k <= 47; ++k)
    {
        const double minus = rawToDouble<Q>(t.values[48 - k]);
        const double plus  = rawToDouble<Q>(t.values[48 + k]);
        BOOST_CHECK_SMALL(minus - plus, 2.0 / 65536.0);
    }
}

BOOST_AUTO_TEST_CASE(invariant_sin_odd)
{
    // sin(-x) == -sin(x).
    using Q = QValue<16, 16, true>;
    static const tinymind::SinValuesTableSelector<16, 16, true>::SinTableType t;
    for (std::size_t k = 1; k <= 47; ++k)
    {
        const double minus = rawToDouble<Q>(t.values[48 - k]);
        const double plus  = rawToDouble<Q>(t.values[48 + k]);
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
BOOST_AUTO_TEST_CASE(accuracy_log_q16_16)
{
    // Chord error scales with |log''(x)| = 1/x^2, which is ~4 at x=0.5. Use a
    // looser tolerance covering [0.5, 5.5]; tighten the lower bound to shrink it.
    using Q = QValue<16, 16, true>;
    static const tinymind::LogValuesTableSelector<16, 16, true>::LogTableType t;
    // log table is 0 for x <= 0; only meaningful for x > 0.
    verifyAccuracy<Q>(t, refLog, 0.5, 5.5, 1e-2);
}
BOOST_AUTO_TEST_CASE(accuracy_sin_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::SinValuesTableSelector<16, 16, true>::SinTableType t;
    verifyAccuracy<Q>(t, refSin, -5.5, 5.5, 2e-3);
}
BOOST_AUTO_TEST_CASE(accuracy_cos_q16_16)
{
    using Q = QValue<16, 16, true>;
    static const tinymind::CosValuesTableSelector<16, 16, true>::CosTableType t;
    verifyAccuracy<Q>(t, refCos, -5.5, 5.5, 2e-3);
}

BOOST_AUTO_TEST_SUITE_END()
