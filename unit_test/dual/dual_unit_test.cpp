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

// Forward-mode autodiff (Dual<ValueType>) correctness tests.
//
// The point is that the chain-rule mechanics are value-type-agnostic: the same
// dual expressions produce the same derivatives for IEEE double and for
// TinyMind's fixed-point QValue. QValue only differs by rounding (the precision
// question), not by mechanics. Tolerances reflect that: double is exact, Q16.16
// is checked to ~1e-3.

#include <cmath>

#include "qformat.hpp"
#include "nnproperties.hpp"   // ValueConverter
#include "dual.hpp"
#include "compiler.h"

#define BOOST_TEST_MODULE dual_unit_test
#include <boost/test/included/unit_test.hpp>

using tinymind::Dual;
using tinymind::chainRule;

typedef tinymind::QValue<16, 16, true> Q16;

namespace {

// Value type -> double for comparison.
double toDouble(double v) { return v; }
double toDouble(const Q16& v)
{
    return static_cast<double>(v.getValue()) / 65536.0;
}

// Run the identical dual expressions on a given value type and check against
// the analytic derivatives. zero/one/two/x0 are supplied in the value type so
// the body is fully generic (QValue(int) means raw bits, so constants must be
// built by the caller via the (fixedPart, fractionalPart) constructor).
template<typename V>
void checkArithmetic(const V& one, const V& x0, double tol)
{
    // Seed: value x0, derivative 1 (differentiate w.r.t. this input).
    Dual<V> x(x0, one);

    // f(x) = x*x + x  ->  f(2) = 6, f'(2) = 2x+1 = 5.
    Dual<V> poly = (x * x) + x;
    BOOST_TEST(toDouble(poly.value) == 6.0, boost::test_tools::tolerance(tol));
    BOOST_TEST(toDouble(poly.deriv) == 5.0, boost::test_tools::tolerance(tol));

    // g(x) = 1 / x  ->  g(2) = 0.5, g'(2) = -1/x^2 = -0.25. Exercises the
    // quotient rule (the only operator needing a squared denominator).
    Dual<V> recip = Dual<V>(one) / x;
    BOOST_TEST(toDouble(recip.value) == 0.5, boost::test_tools::tolerance(tol));
    BOOST_TEST(toDouble(recip.deriv) == -0.25, boost::test_tools::tolerance(tol));

    // A constant (Dual built from a single value) has zero derivative.
    Dual<V> c(x0);
    BOOST_TEST(toDouble(c.deriv) == 0.0, boost::test_tools::tolerance(tol));
}

// Chain-rule lift: given scalar f = h(x0) and fp = h'(x0), chainRule must
// produce (f, fp * x.deriv). Uses tanh at x0=2 as the worked example.
template<typename V>
void checkChainRule(const V& one, const V& x0, double tol)
{
    const double x0d = toDouble(x0);
    const double t = std::tanh(x0d);
    const double tp = 1.0 - t * t;

    Dual<V> x(x0, one);
    const V f  = tinymind::ValueConverter<double, V>::convertToDestinationType(t);
    const V fp = tinymind::ValueConverter<double, V>::convertToDestinationType(tp);

    Dual<V> act = chainRule(x, f, fp);
    BOOST_TEST(toDouble(act.value) == t, boost::test_tools::tolerance(tol));
    BOOST_TEST(toDouble(act.deriv) == tp, boost::test_tools::tolerance(tol));
}

} // namespace

BOOST_AUTO_TEST_SUITE(dual_tests)

BOOST_AUTO_TEST_CASE(arithmetic_double)
{
    checkArithmetic<double>(1.0, 2.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(arithmetic_qformat)
{
    checkArithmetic<Q16>(Q16(1, 0), Q16(2, 0), 1e-3);
}

BOOST_AUTO_TEST_CASE(chain_rule_double)
{
    checkChainRule<double>(1.0, 2.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(chain_rule_qformat)
{
    checkChainRule<Q16>(Q16(1, 0), Q16(2, 0), 1e-3);
}

BOOST_AUTO_TEST_CASE(higher_input_derivative_via_nesting)
{
    // Nesting Dual<Dual<double>> gives a second derivative. f(x)=x*x*x:
    // f'(x)=3x^2, f''(x)=6x. At x=2: f'=12, f''=12.
    typedef Dual<double> D1;
    typedef Dual<D1> D2;

    // Outer deriv seed = 1 (d/dx); inner deriv seed = 1 too, so the inner
    // dual tracks the first derivative and the outer tracks its derivative.
    D2 x(D1(2.0, 1.0), D1(1.0, 0.0));
    D2 f = (x * x) * x;

    BOOST_TEST(f.value.value == 8.0, boost::test_tools::tolerance(1e-12)); // x^3
    BOOST_TEST(f.value.deriv == 12.0, boost::test_tools::tolerance(1e-12)); // 3x^2
    BOOST_TEST(f.deriv.deriv == 12.0, boost::test_tools::tolerance(1e-12)); // 6x
}

BOOST_AUTO_TEST_SUITE_END()
