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

// Unit tests for the Liquid Time-Constant cell (cpp/ltc.hpp):
//   1. fused-solver steady state matches the closed-form fixed point,
//   2. reverse-mode (RevVar) gradient through the cell matches finite differences,
//   3. forward-mode (MultiDual) and reverse-mode agree on value and gradient,
//   4. the cell trains (loss falls) via the existing pinn::sgdStepReverse path.

#include <cstddef>
#include <cstdint>
#include <cmath>

#define TINYMIND_LTC_REVERSE_TRAINING 1   // RevVar Constant<> lift

#include "qformat.hpp"
#include "nnproperties.hpp"
#include "ltc.hpp"
#include "revdual.hpp"
#include "multidual.hpp"

#define BOOST_TEST_MODULE ltc_unit_test
#include <boost/test/included/unit_test.hpp>

using tinymind::RevVar;
using tinymind::MultiDual;
using tinymind::pinn::Constant;

namespace {

// A small fixed sequence task: unit drive in, leaky-integrator step response out.
struct Task
{
    static const std::size_t L   = 8;
    static const std::size_t NIN = 1;
    static const std::size_t NST = 3;

    typedef tinymind::ltc::LtcCell<NIN, NST> Cell;
    static const std::size_t NReadout = NST + 1;
    static const std::size_t NP = Cell::NumParams + NReadout;

    double in[L];
    double tgt[L];

    Task()
    {
        for (std::size_t t = 0; t < L; ++t)
        {
            in[t]  = 1.0;
            tgt[t] = 1.0 - std::exp(-0.30 * static_cast<double>(t));
        }
    }

    template<typename S>
    S loss(const S* p) const
    {
        const S* cellP = p;
        const S* wout  = p + Cell::NumParams;
        const S  bout  = wout[NST];

        S state[NST];
        for (std::size_t i = 0; i < NST; ++i) state[i] = Constant<S>::of(0.0);

        S acc = Constant<S>::of(0.0);
        for (std::size_t t = 0; t < L; ++t)
        {
            S x = Constant<S>::of(in[t]);
            S ns[NST];
            Cell::template step<S>(cellP, &x, state, ns, 1.0, 1);
            for (std::size_t i = 0; i < NST; ++i) state[i] = ns[i];

            S y = bout;
            for (std::size_t i = 0; i < NST; ++i) y = y + wout[i] * state[i];

            const S e = y - Constant<S>::of(tgt[t]);
            acc = acc + e * e;
        }
        return acc * Constant<S>::of(1.0 / static_cast<double>(L));
    }
};

// Deterministic small init; tau slice set positive.
void initParams(double* p, std::size_t n, std::size_t tauBase, std::size_t nst)
{
    unsigned s = 777u;
    for (std::size_t i = 0; i < n; ++i)
    {
        s = s * 1103515245u + 12345u;
        const double r = (static_cast<double>((s >> 16) & 0x7fff) / 32767.0) - 0.5;
        p[i] = 0.20 * r;
    }
    for (std::size_t i = 0; i < nst; ++i) p[tauBase + i] = 1.0;
}

} // namespace

// 1. Fused semi-implicit Euler step converges to the analytic fixed point.
//    With W_in = W_rec = b = 0, the synapse is f = sigmoid(0) = 0.5 (constant),
//    so x* = 0.5*A / (1/tau + 0.5), independent of dt.
BOOST_AUTO_TEST_CASE(fused_solver_fixed_point)
{
    typedef tinymind::ltc::LtcCell<1, 2> Cell;
    double p[Cell::NumParams];
    for (std::size_t i = 0; i < Cell::NumParams; ++i) p[i] = 0.0;

    const double tau0 = 2.0, tau1 = 4.0;
    const double A0   = 3.0, A1   = -1.0;
    p[Cell::OffTau + 0] = tau0;  p[Cell::OffTau + 1] = tau1;
    p[Cell::OffA   + 0] = A0;    p[Cell::OffA   + 1] = A1;

    const double f = 0.5; // sigmoid(0)
    const double xs0 = f * A0 / (1.0 / tau0 + f);
    const double xs1 = f * A1 / (1.0 / tau1 + f);

    double in = 1.0, st[2] = {0.0, 0.0}, out[2];
    for (int n = 0; n < 500; ++n)            // iterate to steady state
    {
        Cell::step<double>(p, &in, st, out, 1.0, 1);
        st[0] = out[0]; st[1] = out[1];
    }
    BOOST_CHECK_CLOSE(st[0], xs0, 1e-3);
    BOOST_CHECK_CLOSE(st[1], xs1, 1e-3);

    // Fixed point is dt-invariant: a smaller step reaches the same place.
    double st2[2] = {0.0, 0.0}, out2[2];
    for (int n = 0; n < 4000; ++n)
    {
        Cell::step<double>(p, &in, st2, out2, 0.25, 1);
        st2[0] = out2[0]; st2[1] = out2[1];
    }
    BOOST_CHECK_CLOSE(st2[0], xs0, 1e-3);
    BOOST_CHECK_CLOSE(st2[1], xs1, 1e-3);
}

// 1b. unfolds > 1 (multiple fused steps per input sample) reaches the SAME
//     analytic fixed point -- exercises the inner unfold loop.
BOOST_AUTO_TEST_CASE(unfolds_reaches_same_fixed_point)
{
    typedef tinymind::ltc::LtcCell<1, 2> Cell;
    double p[Cell::NumParams];
    for (std::size_t i = 0; i < Cell::NumParams; ++i) p[i] = 0.0;
    const double tau0 = 2.0, tau1 = 4.0, A0 = 3.0, A1 = -1.0;
    p[Cell::OffTau + 0] = tau0;  p[Cell::OffTau + 1] = tau1;
    p[Cell::OffA   + 0] = A0;    p[Cell::OffA   + 1] = A1;

    const double f = 0.5;
    const double xs0 = f * A0 / (1.0 / tau0 + f);
    const double xs1 = f * A1 / (1.0 / tau1 + f);

    double in = 1.0, st[2] = {0.0, 0.0}, out[2];
    for (int n = 0; n < 200; ++n)        // 200 calls x 4 unfolds = 800 fused steps
    {
        Cell::step<double>(p, &in, st, out, 0.5, /*unfolds=*/4);
        st[0] = out[0]; st[1] = out[1];
    }
    BOOST_CHECK_CLOSE(st[0], xs0, 1e-3);
    BOOST_CHECK_CLOSE(st[1], xs1, 1e-3);
}

// 1c. Fixed-point (Q16.16) inference matches the double reference -- backs the
//     "step<S> infers in QValue" claim (sigmoid LUT + Q-format divide).
BOOST_AUTO_TEST_CASE(qformat_inference_parity)
{
    typedef tinymind::ltc::LtcCell<2, 3> Cell;
    typedef tinymind::QValue<16, 16, true> Q;

    double p[Cell::NumParams];
    initParams(p, Cell::NumParams, Cell::OffTau, 3);   // tau slice = 1.0, small weights

    Q pq[Cell::NumParams];
    for (std::size_t i = 0; i < Cell::NumParams; ++i)
        pq[i] = tinymind::ValueConverter<double, Q>::convertToDestinationType(p[i]);

    double in_d[2] = { 0.6, -0.4 };
    Q in_q[2] = { tinymind::ValueConverter<double, Q>::convertToDestinationType(in_d[0]),
                  tinymind::ValueConverter<double, Q>::convertToDestinationType(in_d[1]) };

    double sd[3] = {0, 0, 0}, od[3];
    Q sq[3] = { Q(0), Q(0), Q(0) }, oq[3];
    for (int t = 0; t < 12; ++t)
    {
        Cell::step<double>(p, in_d, sd, od, 1.0, 1);
        Cell::step<Q>(pq, in_q, sq, oq, 1.0, 1);
        for (std::size_t i = 0; i < 3; ++i) { sd[i] = od[i]; sq[i] = oq[i]; }
    }
    for (std::size_t i = 0; i < 3; ++i)
    {
        const double q_back = tinymind::ValueConverter<Q, double>::convertToDestinationType(sq[i]);
        BOOST_CHECK_SMALL(q_back - sd[i], 0.02);   // Q16.16 + sigmoid LUT error
    }
}

// 2. Reverse-mode adjoint through the LTC cell matches central finite differences.
BOOST_AUTO_TEST_CASE(reverse_gradient_matches_finite_difference)
{
    Task task;
    double p[Task::NP];
    initParams(p, Task::NP, Task::Cell::OffTau, Task::NST);

    // Reverse-mode: one backward pass yields d(loss)/dp for all params.
    tinymind::revReset();
    RevVar rp[Task::NP];
    for (std::size_t i = 0; i < Task::NP; ++i) rp[i] = RevVar::leaf(p[i]);
    const RevVar L = task.loss<RevVar>(rp);
    tinymind::revBackward(L);

    // Central finite differences on the double loss.
    const double h = 1e-6;
    for (std::size_t i = 0; i < Task::NP; ++i)
    {
        double pp[Task::NP], pm[Task::NP];
        for (std::size_t k = 0; k < Task::NP; ++k) { pp[k] = p[k]; pm[k] = p[k]; }
        pp[i] += h; pm[i] -= h;
        const double fd = (task.loss<double>(pp) - task.loss<double>(pm)) / (2.0 * h);
        const double ad = rp[i].adjoint();
        BOOST_CHECK_SMALL(ad - fd, 1e-4);
    }
}

// 3. Forward-mode (MultiDual) and reverse-mode (RevVar) agree on value + gradient.
BOOST_AUTO_TEST_CASE(forward_and_reverse_mode_agree)
{
    Task task;
    double p[Task::NP];
    initParams(p, Task::NP, Task::Cell::OffTau, Task::NST);

    typedef MultiDual<double, Task::NP> M;
    M mp[Task::NP];
    for (std::size_t i = 0; i < Task::NP; ++i) mp[i] = M::seed(p[i], i, 1.0);
    const M ML = task.loss<M>(mp);

    tinymind::revReset();
    RevVar rp[Task::NP];
    for (std::size_t i = 0; i < Task::NP; ++i) rp[i] = RevVar::leaf(p[i]);
    const RevVar RL = task.loss<RevVar>(rp);
    tinymind::revBackward(RL);

    const double dl = task.loss<double>(p);
    BOOST_CHECK_CLOSE(ML.value, dl, 1e-9);
    BOOST_CHECK_CLOSE(RL.value(), dl, 1e-9);
    for (std::size_t i = 0; i < Task::NP; ++i)
        BOOST_CHECK_SMALL(ML.grad[i] - rp[i].adjoint(), 1e-9);
}

// 4. The cell trains: loss drops by >=10x through pinn::sgdStepReverse.
BOOST_AUTO_TEST_CASE(trains_via_reverse_mode)
{
    Task task;
    struct LossFn { const Task* t; RevVar operator()(const RevVar* p) const { return t->loss<RevVar>(p); } };
    LossFn lossFn; lossFn.t = &task;

    double p[Task::NP], v[Task::NP];
    initParams(p, Task::NP, Task::Cell::OffTau, Task::NST);
    for (std::size_t i = 0; i < Task::NP; ++i) v[i] = 0.0;

    const double loss0 = task.loss<double>(p);
    const std::size_t tauBase = Task::Cell::OffTau;
    for (int e = 0; e < 600; ++e)
    {
        tinymind::pinn::sgdStepReverse<Task::NP>(p, v, 0.05, 0.9, lossFn);
        for (std::size_t i = 0; i < Task::NST; ++i)
            if (p[tauBase + i] < 0.1) p[tauBase + i] = 0.1;
    }
    const double lossN = task.loss<double>(p);
    BOOST_CHECK_LT(lossN, loss0 * 0.1);
}
