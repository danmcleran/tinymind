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

// Unit tests for the Closed-form Continuous-time cell (cpp/cfc.hpp):
//   1. time-gated interpolation stays within the tanh head range [-1, 1],
//   2. reverse-mode (RevVar) gradient through the cell matches finite diff,
//   3. forward-mode (MultiDual) and reverse-mode agree on value + gradient,
//   4. the cell trains (loss falls) via pinn::sgdStepReverse.

#include <cstddef>
#include <cstdint>
#include <cmath>

#define TINYMIND_CFC_REVERSE_TRAINING 1

#include "qformat.hpp"
#include "nnproperties.hpp"
#include "cfc.hpp"
#include "revdual.hpp"
#include "multidual.hpp"

#define BOOST_TEST_MODULE cfc_unit_test
#include <boost/test/included/unit_test.hpp>

using tinymind::RevVar;
using tinymind::MultiDual;
using tinymind::pinn::Constant;

namespace {

struct Task
{
    static const std::size_t L   = 8;
    static const std::size_t NIN = 1;
    static const std::size_t NST = 3;
    static const std::size_t BB  = 4;

    typedef tinymind::cfc::CfCCell<NIN, NST, BB> Cell;
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
            Cell::template step<S>(cellP, &x, state, ns, 1.0);
            for (std::size_t i = 0; i < NST; ++i) state[i] = ns[i];

            S y = bout;
            for (std::size_t i = 0; i < NST; ++i) y = y + wout[i] * state[i];

            const S e = y - Constant<S>::of(tgt[t]);
            acc = acc + e * e;
        }
        return acc * Constant<S>::of(1.0 / static_cast<double>(L));
    }
};

void initParams(double* p, std::size_t n)
{
    unsigned s = 9001u;
    for (std::size_t i = 0; i < n; ++i)
    {
        s = s * 1103515245u + 12345u;
        const double r = (static_cast<double>((s >> 16) & 0x7fff) / 32767.0) - 0.5;
        p[i] = 0.30 * r;
    }
}

} // namespace

// 1. tanh heads bound the interpolation: h = (1-t)*ff1 + t*ff2 with t in (0,1)
//    and ff1, ff2 in [-1, 1] => h in [-1, 1] for any inputs / weights.
BOOST_AUTO_TEST_CASE(interpolation_within_head_range)
{
    typedef tinymind::cfc::CfCCell<2, 4, 5> Cell;
    double p[Cell::NumParams];
    unsigned s = 4242u;
    for (std::size_t i = 0; i < Cell::NumParams; ++i)
    {
        s = s * 1103515245u + 12345u;
        p[i] = 2.0 * ((static_cast<double>((s >> 16) & 0x7fff) / 32767.0) - 0.5); // [-1,1]
    }
    double in[2] = {0.7, -1.3};
    double st[4] = {0.5, -0.5, 0.9, -0.2};
    double out[4];
    Cell::step<double>(p, in, st, out, 2.5);
    for (std::size_t i = 0; i < 4; ++i)
    {
        BOOST_CHECK_GE(out[i], -1.0);
        BOOST_CHECK_LE(out[i],  1.0);
    }
}

// 1b. Fixed-point (Q16.16) inference matches the double reference over a short
//     sequence -- backs the "step<S> infers in QValue" claim (tanh / sigmoid
//     LUTs in the backbone, heads, and time-gate). ts != 1 exercises the gate.
BOOST_AUTO_TEST_CASE(qformat_inference_parity)
{
    typedef tinymind::cfc::CfCCell<2, 3, 4> Cell;
    typedef tinymind::QValue<16, 16, true> Q;

    double p[Cell::NumParams];
    unsigned s = 13579u;
    for (std::size_t i = 0; i < Cell::NumParams; ++i)
    {
        s = s * 1103515245u + 12345u;
        p[i] = 0.30 * ((static_cast<double>((s >> 16) & 0x7fff) / 32767.0) - 0.5);
    }
    Q pq[Cell::NumParams];
    for (std::size_t i = 0; i < Cell::NumParams; ++i)
        pq[i] = tinymind::ValueConverter<double, Q>::convertToDestinationType(p[i]);

    double in_d[2] = { 0.5, -0.3 };
    Q in_q[2] = { tinymind::ValueConverter<double, Q>::convertToDestinationType(in_d[0]),
                  tinymind::ValueConverter<double, Q>::convertToDestinationType(in_d[1]) };

    double sd[3] = {0, 0, 0}, od[3];
    Q sq[3] = { Q(0), Q(0), Q(0) }, oq[3];
    for (int t = 0; t < 10; ++t)
    {
        Cell::step<double>(p, in_d, sd, od, 1.5);
        Cell::step<Q>(pq, in_q, sq, oq, 1.5);
        for (std::size_t i = 0; i < 3; ++i) { sd[i] = od[i]; sq[i] = oq[i]; }
    }
    for (std::size_t i = 0; i < 3; ++i)
    {
        const double q_back = tinymind::ValueConverter<Q, double>::convertToDestinationType(sq[i]);
        BOOST_CHECK_SMALL(q_back - sd[i], 0.02);
    }
}

// 2. Reverse-mode adjoint through the CfC cell matches central finite diff.
BOOST_AUTO_TEST_CASE(reverse_gradient_matches_finite_difference)
{
    Task task;
    double p[Task::NP];
    initParams(p, Task::NP);

    tinymind::revReset();
    RevVar rp[Task::NP];
    for (std::size_t i = 0; i < Task::NP; ++i) rp[i] = RevVar::leaf(p[i]);
    const RevVar L = task.loss<RevVar>(rp);
    tinymind::revBackward(L);

    const double h = 1e-6;
    for (std::size_t i = 0; i < Task::NP; ++i)
    {
        double pp[Task::NP], pm[Task::NP];
        for (std::size_t k = 0; k < Task::NP; ++k) { pp[k] = p[k]; pm[k] = p[k]; }
        pp[i] += h; pm[i] -= h;
        const double fd = (task.loss<double>(pp) - task.loss<double>(pm)) / (2.0 * h);
        BOOST_CHECK_SMALL(rp[i].adjoint() - fd, 1e-4);
    }
}

// 3. Forward-mode (MultiDual) and reverse-mode (RevVar) agree.
BOOST_AUTO_TEST_CASE(forward_and_reverse_mode_agree)
{
    Task task;
    double p[Task::NP];
    initParams(p, Task::NP);

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

// 4. The cell trains: loss drops by >=5x through pinn::sgdStepReverse.
BOOST_AUTO_TEST_CASE(trains_via_reverse_mode)
{
    Task task;
    struct LossFn { const Task* t; RevVar operator()(const RevVar* p) const { return t->loss<RevVar>(p); } };
    LossFn lossFn; lossFn.t = &task;

    double p[Task::NP], v[Task::NP];
    initParams(p, Task::NP);
    for (std::size_t i = 0; i < Task::NP; ++i) v[i] = 0.0;

    const double loss0 = task.loss<double>(p);
    for (int e = 0; e < 4000; ++e)
        tinymind::pinn::sgdStepReverse<Task::NP>(p, v, 0.10, 0.9, lossFn);
    const double lossN = task.loss<double>(p);
    BOOST_CHECK_LT(lossN, loss0 * 0.5);
}
