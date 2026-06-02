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
#include "dualActivations.hpp"
#include "dualmath.hpp"
#include "multidual.hpp"
#include "taylor.hpp"
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

// Activation overloads: tanh(Dual)/sigmoid(Dual) must produce the activation
// value and its analytic derivative coefficient (chain-ruled by x.deriv == 1).
template<typename V>
void checkActivations(const V& one, const V& x0, double tol)
{
    const double x0d = toDouble(x0);
    const double t = std::tanh(x0d);
    const double s = 1.0 / (1.0 + std::exp(-x0d));

    Dual<V> x(x0, one);

    Dual<V> th = tinymind::tanh(x);
    BOOST_TEST(toDouble(th.value) == t, boost::test_tools::tolerance(tol));
    BOOST_TEST(toDouble(th.deriv) == (1.0 - t * t), boost::test_tools::tolerance(tol));

    Dual<V> sg = tinymind::sigmoid(x);
    BOOST_TEST(toDouble(sg.value) == s, boost::test_tools::tolerance(tol));
    BOOST_TEST(toDouble(sg.deriv) == (s * (1.0 - s)), boost::test_tools::tolerance(tol));
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

BOOST_AUTO_TEST_CASE(activations_double)
{
    checkActivations<double>(1.0, 0.5, 1e-12);
}

BOOST_AUTO_TEST_CASE(activations_qformat)
{
    // QValue routes tanh/sigmoid through the int LUTs, so tolerance reflects
    // LUT resolution, not a mechanics error.
    checkActivations<Q16>(Q16(1, 0), Q16(0, 32768) /* 0.5 */, 2e-2);
}

BOOST_AUTO_TEST_CASE(pinn_style_first_derivative)
{
    // u(x) = tanh(w*x + b); du/dx = w * (1 - tanh(w*x+b)^2). This is the shape
    // of a single-neuron PINN field; forward-mode gives du/dx in one pass.
    const double w = 1.5, b = -0.25, x0 = 0.7;
    Dual<double> x(x0, 1.0);
    Dual<double> wD(w), bD(b);                 // constants (deriv 0)
    Dual<double> u = tinymind::tanh((wD * x) + bD);

    const double pre = w * x0 + b;
    const double expect = w * (1.0 - std::tanh(pre) * std::tanh(pre));
    BOOST_TEST(u.deriv == expect, boost::test_tools::tolerance(1e-12));
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

BOOST_AUTO_TEST_CASE(second_derivative_of_tanh_via_nesting)
{
    // tanh through nested duals: g=tanh(x), g'=1-g^2, g''=-2g(1-g^2). The
    // recursive DualScalarActivation<Dual<W>> specialization makes the
    // activation differentiable at both levels.
    typedef Dual<double> D1;
    typedef Dual<D1> D2;

    const double x0 = 0.6;
    D2 x(D1(x0, 1.0), D1(1.0, 0.0));
    D2 f = tinymind::tanh(x);

    const double g = std::tanh(x0);
    BOOST_TEST(f.value.value == g, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.value.deriv == (1.0 - g * g), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.deriv.deriv == (-2.0 * g * (1.0 - g * g)),
               boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(math_overloads_double)
{
    const double x0 = 0.7;
    Dual<double> x(x0, 1.0);

    Dual<double> e = tinymind::exp(x);
    BOOST_TEST(e.value == std::exp(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(e.deriv == std::exp(x0), boost::test_tools::tolerance(1e-12));

    Dual<double> s = tinymind::sin(x);
    BOOST_TEST(s.value == std::sin(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(s.deriv == std::cos(x0), boost::test_tools::tolerance(1e-12));

    Dual<double> c = tinymind::cos(x);
    BOOST_TEST(c.value == std::cos(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(c.deriv == -std::sin(x0), boost::test_tools::tolerance(1e-12));

    Dual<double> r = tinymind::sqrt(Dual<double>(2.0, 1.0));
    BOOST_TEST(r.value == std::sqrt(2.0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(r.deriv == 1.0 / (2.0 * std::sqrt(2.0)), boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(sin_second_derivative_nested)
{
    // sin''(x) = -sin(x), via nested duals (recursive DualScalarMath).
    typedef Dual<double> D1;
    typedef Dual<D1> D2;
    const double x0 = 0.9;
    D2 x(D1(x0, 1.0), D1(1.0, 0.0));
    D2 f = tinymind::sin(x);

    BOOST_TEST(f.value.value == std::sin(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.value.deriv == std::cos(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.deriv.deriv == -std::sin(x0), boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(sqrt_qformat)
{
    // sqrt works for fixed-point too (SquareRootApproximation); rounding only.
    Dual<Q16> r = tinymind::sqrt(Dual<Q16>(Q16(2, 0), Q16(1, 0)));
    BOOST_TEST(toDouble(r.value) == std::sqrt(2.0), boost::test_tools::tolerance(1e-3));
    BOOST_TEST(toDouble(r.deriv) == 1.0 / (2.0 * std::sqrt(2.0)),
               boost::test_tools::tolerance(1e-2));
}

BOOST_AUTO_TEST_CASE(trig_exp_duals_qformat)
{
    // QValue exp/sin/cos duals via the lookup tables (LUT tolerance, not a
    // mechanics error). sin/cos LUTs are accurate to ~2e-3 over [-5.5, 5.5].
    const double x0 = 0.5;

    Dual<Q16> s = tinymind::sin(Dual<Q16>(Q16(0, 32768) /*0.5*/, Q16(1, 0)));
    BOOST_TEST(toDouble(s.value) == std::sin(x0), boost::test_tools::tolerance(1e-2));
    BOOST_TEST(toDouble(s.deriv) == std::cos(x0), boost::test_tools::tolerance(1e-2));

    Dual<Q16> c = tinymind::cos(Dual<Q16>(Q16(0, 32768), Q16(1, 0)));
    BOOST_TEST(toDouble(c.value) == std::cos(x0), boost::test_tools::tolerance(1e-2));
    BOOST_TEST(toDouble(c.deriv) == -std::sin(x0), boost::test_tools::tolerance(1e-2));

    Dual<Q16> e = tinymind::exp(Dual<Q16>(Q16(0, 32768), Q16(1, 0)));
    BOOST_TEST(toDouble(e.value) == std::exp(x0), boost::test_tools::tolerance(3e-2));
    BOOST_TEST(toDouble(e.deriv) == std::exp(x0), boost::test_tools::tolerance(3e-2));
}

BOOST_AUTO_TEST_CASE(mixed_partial_derivative)
{
    // f(x,y) = sin(x) * y^2 ;  d^2f/dx dy = cos(x) * 2y.
    // Nested dual with DIFFERENT seed directions per level: inner = d/dx,
    // outer = d/dy. f.deriv.deriv is the mixed partial.
    typedef Dual<double> D1;
    typedef Dual<D1> D2;
    const double x0 = 0.6, y0 = 1.3;

    D2 x(D1(x0, 1.0), D1(0.0, 0.0));   // seed x at inner (d/dx); x does not vary with y
    D2 y(D1(y0, 0.0), D1(1.0, 0.0));   // seed y at outer (d/dy)

    D2 f = tinymind::sin(x) * (y * y);

    const double expect = std::cos(x0) * 2.0 * y0;
    BOOST_TEST(f.value.value == std::sin(x0) * y0 * y0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.deriv.deriv == expect, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(multidual_gradient_one_pass)
{
    // f(w0,w1) = w0^2 * w1.  grad = (2 w0 w1, w0^2).  One forward pass gives
    // value and the full gradient.
    typedef tinymind::MultiDual<double, 2> M;
    M w0 = M::seed(3.0, 0, 1.0);
    M w1 = M::seed(2.0, 1, 1.0);

    M f = (w0 * w0) * w1;
    BOOST_TEST(f.value == 18.0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.grad[0] == 12.0, boost::test_tools::tolerance(1e-12)); // 2 w0 w1
    BOOST_TEST(f.grad[1] == 9.0,  boost::test_tools::tolerance(1e-12)); // w0^2
}

BOOST_AUTO_TEST_CASE(multidual_under_dual_weight_grad_of_derivative)
{
    // g(x;w) = w * x^2.  dg/dx = 2 w x ;  d/dw (dg/dx) = 2 x.
    // MultiDual carries the weight gradient; Dual carries d/dx -- so a single
    // evaluation in Dual<MultiDual> yields the weight gradient of a derivative,
    // which is exactly what residual-loss PINN training needs.
    typedef tinymind::MultiDual<double, 1> M;
    typedef Dual<M> DM;

    const double w0 = 2.0, x0 = 3.0;
    DM w(M::seed(w0, 0, 1.0));                 // weight: seeded in M, constant in x
    DM x(M(x0), M(1.0));                        // input:  d/dx seed at the Dual level

    DM g = (w * x) * x;

    BOOST_TEST(g.deriv.value == 2.0 * w0 * x0, boost::test_tools::tolerance(1e-12)); // dg/dx
    BOOST_TEST(g.deriv.grad[0] == 2.0 * x0,    boost::test_tools::tolerance(1e-12)); // d/dw (dg/dx)
}

BOOST_AUTO_TEST_CASE(taylor_jet_polynomial_high_order)
{
    // f(x) = x^3 : f'=3x^2, f''=6x, f'''=6, f''''=0.  One sweep, all orders.
    typedef tinymind::Jet<double, 4> J;
    const double x0 = 1.7;
    J x = J::variable(x0);
    J f = (x * x) * x;
    BOOST_TEST(f.derivative(0) == x0 * x0 * x0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.derivative(1) == 3.0 * x0 * x0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.derivative(2) == 6.0 * x0,      boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.derivative(3) == 6.0,           boost::test_tools::tolerance(1e-12));
    BOOST_TEST(f.derivative(4) == 0.0,           boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(taylor_jet_sin_exp_sqrt)
{
    typedef tinymind::Jet<double, 4> J;
    const double x0 = 0.6;

    J s = tinymind::sin(J::variable(x0));
    BOOST_TEST(s.derivative(1) ==  std::cos(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(s.derivative(2) == -std::sin(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(s.derivative(3) == -std::cos(x0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(s.derivative(4) ==  std::sin(x0), boost::test_tools::tolerance(1e-12));

    J e = tinymind::exp(J::variable(x0));
    for (int k = 0; k <= 4; ++k)
        BOOST_TEST(e.derivative(k) == std::exp(x0), boost::test_tools::tolerance(1e-12));

    J r = tinymind::sqrt(J::variable(2.0));
    BOOST_TEST(r.derivative(1) ==  1.0 / (2.0 * std::sqrt(2.0)), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(r.derivative(2) == -1.0 / (4.0 * std::pow(2.0, 1.5)), boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(taylor_jet_tanh_third_order)
{
    // tanh': 1-t^2 ; tanh'': -2t(1-t^2) ; tanh''': 2(1-t^2)(3t^2-1).
    typedef tinymind::Jet<double, 3> J;
    const double x0 = 0.4;
    const double t = std::tanh(x0);
    J th = tinymind::tanh(J::variable(x0));
    BOOST_TEST(th.derivative(1) == (1.0 - t * t), boost::test_tools::tolerance(1e-10));
    BOOST_TEST(th.derivative(2) == (-2.0 * t * (1.0 - t * t)), boost::test_tools::tolerance(1e-10));
    BOOST_TEST(th.derivative(3) == (2.0 * (1.0 - t * t) * (3.0 * t * t - 1.0)),
               boost::test_tools::tolerance(1e-10));
}

BOOST_AUTO_TEST_SUITE_END()
