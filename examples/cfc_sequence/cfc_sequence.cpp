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

// Closed-form Continuous-time (CfC) cell demo: train a CfC cell + linear
// readout to track a target with IRREGULAR time steps, using only the existing
// reverse-mode autodiff trainer (pinn::sgdStepReverse). The per-step elapsed
// time `ts` feeds the CfC time-gate -- the headline feature LTC/CfC have over a
// plain RNN. The same scalar-templated step<S> serves both the RevVar gradient
// and the double inference pass; no hand-written CfC backprop.

#define TINYMIND_CFC_REVERSE_TRAINING 1

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "cfc.hpp"
#include "revdual.hpp"

using tinymind::RevVar;
using tinymind::pinn::Constant;

struct Task
{
    static const std::size_t L   = 20;
    static const std::size_t NIN = 1;
    static const std::size_t NST = 6;
    static const std::size_t BB  = 8;

    typedef tinymind::cfc::CfCCell<NIN, NST, BB> Cell;
    static const std::size_t NReadout = NST + 1;
    static const std::size_t NP = Cell::NumParams + NReadout;

    double in[L];
    double ts[L];   // irregular elapsed-time per step
    double tgt[L];

    Task()
    {
        // Irregular sampling: cumulative time t_cum advances by a varying dt,
        // target is a continuous decaying sinusoid sampled at those instants.
        double t_cum = 0.0;
        for (std::size_t k = 0; k < L; ++k)
        {
            const double dt = 0.5 + 0.5 * ((k * 7) % 5) / 4.0; // 0.5 .. 1.0, varying
            ts[k]  = dt;
            t_cum += dt;
            in[k]  = 1.0;
            tgt[k] = std::exp(-0.20 * t_cum) * std::sin(0.8 * t_cum);
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
            Cell::template step<S>(cellP, &x, state, ns, ts[t]);
            for (std::size_t i = 0; i < NST; ++i) state[i] = ns[i];

            S y = bout;
            for (std::size_t i = 0; i < NST; ++i) y = y + wout[i] * state[i];

            const S e = y - Constant<S>::of(tgt[t]);
            acc = acc + e * e;
        }
        return acc * Constant<S>::of(1.0 / static_cast<double>(L));
    }

    void predict(const double* p, double* out) const
    {
        const double* cellP = p;
        const double* wout  = p + Cell::NumParams;
        const double  bout  = wout[NST];
        double state[NST];
        for (std::size_t i = 0; i < NST; ++i) state[i] = 0.0;
        for (std::size_t t = 0; t < L; ++t)
        {
            double x = in[t], ns[NST];
            Cell::step<double>(cellP, &x, state, ns, ts[t]);
            for (std::size_t i = 0; i < NST; ++i) state[i] = ns[i];
            double y = bout;
            for (std::size_t i = 0; i < NST; ++i) y += wout[i] * state[i];
            out[t] = y;
        }
    }
};

struct LossFn { const Task* task; RevVar operator()(const RevVar* p) const { return task->loss<RevVar>(p); } };

int main()
{
    Task task;
    LossFn lossFn; lossFn.task = &task;

    double params[Task::NP], velocity[Task::NP];
    for (std::size_t i = 0; i < Task::NP; ++i) velocity[i] = 0.0;

    unsigned s = 2718u;
    for (std::size_t i = 0; i < Task::NP; ++i)
    {
        s = s * 1103515245u + 12345u;
        params[i] = 0.30 * ((static_cast<double>((s >> 16) & 0x7fff) / 32767.0) - 0.5);
    }

    const double loss0 = task.loss<double>(params);
    std::printf("CfC cell: NumParams=%zu (cell=%zu + readout=%zu), seq=%zu, irregular ts\n",
                Task::NP, Task::Cell::NumParams, Task::NReadout, Task::L);
    std::printf("epoch %4d   loss %.6e\n", 0, loss0);

    std::FILE* lossCsv = std::fopen("cfc_loss.csv", "w");
    std::fprintf(lossCsv, "epoch,loss\n");
    std::fprintf(lossCsv, "0,%.8e\n", loss0);

    const int epochs = 4000;
    for (int e = 1; e <= epochs; ++e)
    {
        tinymind::pinn::sgdStepReverse<Task::NP>(params, velocity, 0.08, 0.9, lossFn);
        const double le = task.loss<double>(params);
        std::fprintf(lossCsv, "%d,%.8e\n", e, le);
        if (e % 800 == 0)
            std::printf("epoch %4d   loss %.6e\n", e, le);
    }
    std::fclose(lossCsv);

    double pred[Task::L];
    task.predict(params, pred);

    // Fit CSV: per-step elapsed time, target, predicted (irregular sampling).
    std::FILE* fitCsv = std::fopen("cfc_fit.csv", "w");
    std::fprintf(fitCsv, "t,ts,target,predicted\n");
    for (std::size_t t = 0; t < Task::L; ++t)
        std::fprintf(fitCsv, "%zu,%.4f,%.6f,%.6f\n", t, task.ts[t], task.tgt[t], pred[t]);
    std::fclose(fitCsv);

    std::printf("\n  t      ts       target     predicted\n");
    for (std::size_t t = 0; t < Task::L; t += 2)
        std::printf("%3zu  %5.3f    %8.5f    %8.5f\n", t, task.ts[t], task.tgt[t], pred[t]);

    const double lossN = task.loss<double>(params);
    std::printf("\nfinal loss %.6e  (%.1fx reduction)\n", lossN, loss0 / lossN);
    return (lossN < loss0 * 0.5) ? 0 : 1;
}
