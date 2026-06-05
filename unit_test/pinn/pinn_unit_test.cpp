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

// Unit tests for the PINN building blocks (cpp/pinn.hpp + cpp/revdual.hpp) that
// the dual / multidual suites cover only at the scalar-arithmetic level:
//   1. PinnMlp::forward<double> reproduces a hand-computed tanh-MLP forward,
//   2. PinnMlp::forward<Dual> input derivative matches central finite diff,
//   3. PinnMlp::forward<MultiDual> weight gradient matches central finite diff,
//   4. pinn::sgdStep (forward-mode trainer) drives a regression loss down,
//   5. pinn::sgdStep and pinn::sgdStepReverse agree (gradient + converged loss),
//   6. an actual 1-D heat-equation PINN residual converges below a fixed
//      threshold -- the automated counterpart of examples/pinn_heat1d --train.

#include <cstddef>
#include <cstdint>
#include <cmath>

#include "qformat.hpp"
#include "nnproperties.hpp"
#include "dual.hpp"
#include "dualActivations.hpp"
#include "dualmath.hpp"
#include "multidual.hpp"
#include "pinn.hpp"
#include "revdual.hpp"

#define BOOST_TEST_MODULE pinn_unit_test
#include <boost/test/included/unit_test.hpp>

using tinymind::Dual;
using tinymind::MultiDual;
using tinymind::RevVar;

namespace {

// Lift a double literal into any scalar type (double / QValue / dual types).
template<typename S> struct Lit
{
    static S make(double d)
    { return tinymind::ValueConverter<double, S>::convertToDestinationType(d); }
};
template<typename V> struct Lit<Dual<V> >
{
    static Dual<V> make(double d) { return Dual<V>(Lit<V>::make(d)); }
};
template<typename V, std::size_t N> struct Lit<MultiDual<V, N> >
{
    static MultiDual<V, N> make(double d) { return MultiDual<V, N>(Lit<V>::make(d)); }
};
template<> struct Lit<RevVar>
{
    static RevVar make(double d) { return RevVar::constant(d); }
};

void lcgInit(double* p, std::size_t n, unsigned seed, double scale)
{
    unsigned s = seed;
    for (std::size_t i = 0; i < n; ++i)
    {
        s = s * 1103515245u + 12345u;
        p[i] = scale * ((static_cast<double>((s >> 16) & 0x7fff) / 32767.0) - 0.5);
    }
}

} // namespace

// 1. PinnMlp::forward<double> == hand-computed tanh-hidden / linear-output MLP.
BOOST_AUTO_TEST_CASE(pinnmlp_forward_matches_hand_computed)
{
    typedef tinymind::pinn::PinnMlp<2, 2, 1> Net;   // 4 + 2 + 2 + 1 = 9 params
    const std::size_t np = Net::NumParams;
    BOOST_TEST(np == 9u);

    double p[Net::NumParams] = {
        // W1 (2 hidden x 2 inputs)
        0.5, -0.3,  0.8, 0.2,
        // b1
        0.1, -0.4,
        // W2 (1 output x 2 hidden)
        0.7, -0.6,
        // b2
        0.05
    };
    double in[2] = { 0.6, -0.2 };

    double out[1];
    Net::forward<double>(p, in, out);

    const double h0 = std::tanh(0.1 + 0.5 * 0.6 + (-0.3) * (-0.2));
    const double h1 = std::tanh(-0.4 + 0.8 * 0.6 + 0.2 * (-0.2));
    const double expect = 0.05 + 0.7 * h0 + (-0.6) * h1;
    BOOST_TEST(out[0] == expect, boost::test_tools::tolerance(1e-12));
}

// 2. PinnMlp::forward<Dual> du/dx matches central finite differences.
BOOST_AUTO_TEST_CASE(pinnmlp_input_derivative_matches_fd)
{
    typedef tinymind::pinn::PinnMlp<2, 4, 1> Net;
    double p[Net::NumParams];
    lcgInit(p, Net::NumParams, 1234u, 0.6);

    const double x0 = 0.35, x1 = -0.8;
    Dual<double> dp[Net::NumParams];
    for (std::size_t i = 0; i < Net::NumParams; ++i) dp[i] = Dual<double>(p[i]); // constant
    Dual<double> din[2] = { Dual<double>(x0, 1.0), Dual<double>(x1, 0.0) }; // d/dx0
    Dual<double> dout[1];
    Net::forward<Dual<double> >(dp, din, dout);

    const double h = 1e-6;
    double a[2] = { x0 + h, x1 }, b[2] = { x0 - h, x1 }, fa[1], fb[1];
    Net::forward<double>(p, a, fa);
    Net::forward<double>(p, b, fb);
    const double fd = (fa[0] - fb[0]) / (2.0 * h);
    BOOST_TEST(dout[0].deriv == fd, boost::test_tools::tolerance(1e-5));
}

// 3. PinnMlp::forward<MultiDual> weight gradient of a scalar loss vs central FD.
BOOST_AUTO_TEST_CASE(pinnmlp_weight_gradient_matches_fd)
{
    typedef tinymind::pinn::PinnMlp<2, 3, 1> Net;
    const std::size_t NP = Net::NumParams;
    double p[NP];
    lcgInit(p, NP, 99u, 0.5);

    const double x[2] = { 0.4, 0.25 };

    // loss(p) = u(p; x)^2.
    typedef MultiDual<double, Net::NumParams> M;
    M mp[Net::NumParams];
    for (std::size_t i = 0; i < NP; ++i) mp[i] = M::seed(p[i], i, 1.0);
    M min[2] = { M(x[0]), M(x[1]) };
    M mout[1];
    Net::forward<M>(mp, min, mout);
    const M loss = mout[0] * mout[0];

    const double hh = 1e-6;
    for (std::size_t i = 0; i < NP; ++i)
    {
        double pp[NP], pm[NP];
        for (std::size_t k = 0; k < NP; ++k) { pp[k] = p[k]; pm[k] = p[k]; }
        pp[i] += hh; pm[i] -= hh;
        double op[1], om[1];
        Net::forward<double>(pp, x, op);
        Net::forward<double>(pm, x, om);
        const double fd = (op[0] * op[0] - om[0] * om[0]) / (2.0 * hh);
        BOOST_CHECK_SMALL(loss.grad[i] - fd, 1e-4);
    }
}

namespace {

// Fit PinnMlp 1->H->1 to a smooth target g(x) = sin(pi x) on [-1, 1].
template<std::size_t H>
struct RegressionLoss
{
    template<typename S>
    S operator()(const S* p) const
    {
        typedef tinymind::pinn::PinnMlp<1, H, 1> Net;
        const int N = 9;
        S acc = Lit<S>::make(0.0);
        for (int i = 0; i < N; ++i)
        {
            const double x = -1.0 + 2.0 * i / (N - 1);
            const double g = std::sin(3.14159265358979323846 * x);
            S in[1] = { Lit<S>::make(x) };
            S out[1];
            Net::template forward<S>(p, in, out);
            const S e = out[0] - Lit<S>::make(g);
            acc = acc + e * e;
        }
        return acc * Lit<S>::make(1.0 / N);
    }
};

} // namespace

// 4. pinn::sgdStep (forward-mode MultiDual trainer) drives the loss down >=10x.
BOOST_AUTO_TEST_CASE(sgdstep_forward_mode_reduces_loss)
{
    typedef tinymind::pinn::PinnMlp<1, 8, 1> Net;
    const std::size_t NP = Net::NumParams;
    RegressionLoss<8> loss;

    double p[NP], vel[NP];
    lcgInit(p, NP, 2024u, 0.4);
    for (std::size_t i = 0; i < NP; ++i) vel[i] = 0.0;

    const double loss0 = loss(p);
    for (int e = 0; e < 4000; ++e)
        tinymind::pinn::sgdStep<NP>(p, vel, 0.05, 0.9, loss);
    const double lossN = loss(p);
    BOOST_CHECK_LT(lossN, loss0 * 0.1);
}

// 5. sgdStep (forward-mode) and sgdStepReverse (reverse-mode) agree: identical
//    gradient on the first step, and convergence to the same loss.
BOOST_AUTO_TEST_CASE(sgdstep_and_sgdstepreverse_agree)
{
    typedef tinymind::pinn::PinnMlp<1, 6, 1> Net;
    const std::size_t NP = Net::NumParams;
    RegressionLoss<6> loss;

    double p0[NP];
    lcgInit(p0, NP, 555u, 0.4);

    // Forward-mode gradient (MultiDual seeds) vs reverse-mode adjoints.
    typedef MultiDual<double, Net::NumParams> M;
    M mp[Net::NumParams];
    for (std::size_t i = 0; i < NP; ++i) mp[i] = M::seed(p0[i], i, 1.0);
    const M fwd = loss(&mp[0]);

    tinymind::revReset();
    RevVar rp[Net::NumParams];
    for (std::size_t i = 0; i < NP; ++i) rp[i] = RevVar::leaf(p0[i]);
    const RevVar rev = loss(&rp[0]);
    tinymind::revBackward(rev);

    BOOST_TEST(fwd.value == rev.value(), boost::test_tools::tolerance(1e-10));
    for (std::size_t i = 0; i < NP; ++i)
        BOOST_CHECK_SMALL(fwd.grad[i] - rp[i].adjoint(), 1e-9);

    // Both trainers from the same init reach the same loss.
    struct RevLoss { const RegressionLoss<6>* L; RevVar operator()(const RevVar* p) const { return (*L)(p); } };
    RevLoss revLoss; revLoss.L = &loss;

    double pa[NP], va[NP], pb[NP], vb[NP];
    for (std::size_t i = 0; i < NP; ++i) { pa[i] = pb[i] = p0[i]; va[i] = vb[i] = 0.0; }
    for (int e = 0; e < 1500; ++e)
    {
        tinymind::pinn::sgdStep<NP>(pa, va, 0.05, 0.9, loss);
        tinymind::pinn::sgdStepReverse<NP>(pb, vb, 0.05, 0.9, revLoss);
    }
    BOOST_TEST(loss(pa) == loss(pb), boost::test_tools::tolerance(1e-6));
}

namespace {

// 1-D heat-equation PINN, mirroring examples/pinn_heat1d --train but smaller.
// u(x,t) fit to u_t = nu * u_xx on x in [0,1], t in [0,T], reference
// u*(x,t) = exp(-nu*pi^2*t) * sin(pi*x).
namespace heat {

const double PI  = 3.14159265358979323846;
const double NU  = 0.3;
const double TMAX = 0.30;
const std::size_t H = 8;
typedef tinymind::pinn::PinnMlp<2, H, 1> Net;
const std::size_t NP = Net::NumParams;   // 4H + 1 = 33

double analytic(double x, double t) { return std::exp(-NU * PI * PI * t) * std::sin(PI * x); }

template<typename S>
S uValue(const S* p, double x, double t)
{
    S in[2] = { Lit<S>::make(x), Lit<S>::make(t) };
    S out[1];
    Net::forward<S>(p, in, out);
    return out[0];
}

template<typename S>
S residual(const S* p, double x, double t)
{
    typedef Dual<S>  DS;
    typedef Dual<DS> DDS;
    const S oneS  = Lit<S>::make(1.0);
    const S zeroS = S();
    const S nuS   = Lit<S>::make(NU);

    DS dp[NP];
    for (std::size_t i = 0; i < NP; ++i) dp[i] = DS(p[i]);
    DS in_t[2] = { DS(Lit<S>::make(x)), DS(Lit<S>::make(t), oneS) };  // seed d/dt
    DS o1[1];
    Net::forward<DS>(dp, in_t, o1);
    const S ut = o1[0].deriv;

    DDS ddp[NP];
    for (std::size_t i = 0; i < NP; ++i) ddp[i] = DDS(DS(p[i]));
    DDS in_x[2] = { DDS(DS(Lit<S>::make(x), oneS), DS(oneS, zeroS)),  // seed d/dx twice
                    DDS(DS(Lit<S>::make(t))) };
    DDS o2[1];
    Net::forward<DDS>(ddp, in_x, o2);
    const S uxx = o2[0].deriv.deriv;

    return ut - (nuS * uxx);
}

struct Loss
{
    template<typename S>
    S operator()(const S* p) const
    {
        const int CX = 7, CT = 5;
        S pde = Lit<S>::make(0.0); int npde = 0;
        for (int i = 1; i < CX - 1; ++i)
            for (int k = 1; k < CT; ++k)
            {
                const double x = static_cast<double>(i) / (CX - 1);
                const double t = TMAX * k / (CT - 1);
                const S r = residual<S>(p, x, t);
                pde = pde + r * r; ++npde;
            }
        S bc = Lit<S>::make(0.0); int nbc = 0;
        for (int k = 0; k < CT; ++k)
        {
            const double t = TMAX * k / (CT - 1);
            const S u0 = uValue<S>(p, 0.0, t);
            const S u1 = uValue<S>(p, 1.0, t);
            bc = bc + u0 * u0 + u1 * u1; nbc += 2;
        }
        S ic = Lit<S>::make(0.0); int nic = 0;
        for (int i = 0; i < CX; ++i)
        {
            const double x = static_cast<double>(i) / (CX - 1);
            const S e = uValue<S>(p, x, 0.0) - Lit<S>::make(std::sin(PI * x));
            ic = ic + e * e; ++nic;
        }
        return pde / Lit<S>::make(npde)
             + Lit<S>::make(10.0) * (bc / Lit<S>::make(nbc))
             + Lit<S>::make(10.0) * (ic / Lit<S>::make(nic));
    }
};

double solutionL2(const double* p)
{
    double se = 0.0; int n = 0;
    for (int i = 0; i <= 10; ++i)
        for (int k = 0; k <= 6; ++k)
        {
            const double x = i / 10.0, t = TMAX * k / 6.0;
            const double e = uValue<double>(p, x, t) - analytic(x, t);
            se += e * e; ++n;
        }
    return std::sqrt(se / n);
}

} // namespace heat
} // namespace

// 6. The heat-equation PINN residual trains below a fixed threshold via the
//    exact-autodiff residual + MultiDual one-pass weight gradient. This is the
//    automated, pass/fail counterpart of examples/pinn_heat1d --train.
BOOST_AUTO_TEST_CASE(pinn_heat_residual_converges)
{
    heat::Loss loss;
    double p[heat::NP], vel[heat::NP];
    lcgInit(p, heat::NP, 12345u, 0.2);
    for (std::size_t i = 0; i < heat::NP; ++i) vel[i] = 0.0;

    const double loss0 = loss(p);
    for (int e = 0; e < 3000; ++e)
        tinymind::pinn::sgdStep<heat::NP>(p, vel, 0.004, 0.85, loss);
    const double lossN = loss(p);
    const double solL2 = heat::solutionL2(p);

    BOOST_CHECK_LT(lossN, loss0 * 0.25);   // residual loss falls >= 4x
    BOOST_CHECK_LT(solL2, 0.05);           // field within 5% of the analytic solution
}
