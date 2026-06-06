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

// Liquid Time-Constant cell demo: train an LTC cell + linear readout to fit the
// step response of a leaky integrator, using ONLY the existing reverse-mode
// autodiff trainer (pinn::sgdStepReverse). The exact same scalar-templated
// forward (LtcCell::step<S>) provides both the RevVar training gradient and the
// double inference pass -- no hand-written backprop for the LTC dynamics.

#define TINYMIND_LTC_REVERSE_TRAINING 1   // enable the RevVar Constant<> lift

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "ltc.hpp"
#include "revdual.hpp"     // pinn::sgdStepReverse, RevVar

using tinymind::RevVar;
using tinymind::pinn::Constant;

// ---- task definition -------------------------------------------------------
struct Task
{
    static const std::size_t L   = 24;   // sequence length
    static const std::size_t NIN = 1;    // inputs per step
    static const std::size_t NST = 6;    // LTC neurons (state width)

    typedef tinymind::ltc::LtcCell<NIN, NST> Cell;   // sigmoid synapse (LTC default)

    static const std::size_t NReadout = NST + 1;            // weights + bias
    static const std::size_t NP = Cell::NumParams + NReadout;

    double in[L];
    double tgt[L];

    Task()
    {
        // Constant unit drive; target = leaky-integrator step response.
        for (std::size_t t = 0; t < L; ++t)
        {
            in[t]  = 1.0;
            tgt[t] = 1.0 - std::exp(-0.30 * static_cast<double>(t));
        }
    }

    // Scalar-templated forward + MSE. S = double for inference, S = RevVar for
    // the reverse-mode weight gradient. Identical arithmetic either way.
    template<typename S>
    S loss(const S* p) const
    {
        const S* cellP = p;
        const S* wout  = p + Cell::NumParams;   // NST readout weights
        const S  bout  = wout[NST];             // readout bias

        S state[NST];
        for (std::size_t i = 0; i < NST; ++i) state[i] = Constant<S>::of(0.0);

        S acc = Constant<S>::of(0.0);
        for (std::size_t t = 0; t < L; ++t)
        {
            S x = Constant<S>::of(in[t]);
            S ns[NST];
            Cell::template step<S>(cellP, &x, state, ns, /*dt=*/1.0, /*unfolds=*/1);
            for (std::size_t i = 0; i < NST; ++i) state[i] = ns[i];

            S y = bout;
            for (std::size_t i = 0; i < NST; ++i) y = y + wout[i] * state[i];

            const S e = y - Constant<S>::of(tgt[t]);
            acc = acc + e * e;
        }
        return acc * Constant<S>::of(1.0 / static_cast<double>(L));
    }

    // Inference: write the predicted output sequence (double path).
    void predict(const double* p, double* out) const
    {
        const double* cellP = p;
        const double* wout  = p + Cell::NumParams;
        const double  bout  = wout[NST];

        double state[NST];
        for (std::size_t i = 0; i < NST; ++i) state[i] = 0.0;

        for (std::size_t t = 0; t < L; ++t)
        {
            double x = in[t];
            double ns[NST];
            Cell::step<double>(cellP, &x, state, ns, 1.0, 1);
            for (std::size_t i = 0; i < NST; ++i) state[i] = ns[i];

            double y = bout;
            for (std::size_t i = 0; i < NST; ++i) y += wout[i] * state[i];
            out[t] = y;
        }
    }
};

// RevVar loss functor for pinn::sgdStepReverse.
struct LossFn
{
    const Task* task;
    RevVar operator()(const RevVar* p) const { return task->loss<RevVar>(p); }
};

int main()
{
    Task task;
    LossFn lossFn; lossFn.task = &task;

    double params[Task::NP];
    double velocity[Task::NP];
    for (std::size_t i = 0; i < Task::NP; ++i) velocity[i] = 0.0;

    // Deterministic small init via a tiny LCG; tau initialized to 1 (> 0).
    unsigned state = 12345u;
    for (std::size_t i = 0; i < Task::NP; ++i)
    {
        state = state * 1103515245u + 12345u;
        const double r = (static_cast<double>((state >> 16) & 0x7fff) / 32767.0) - 0.5; // [-0.5,0.5]
        params[i] = 0.20 * r;
    }
    const std::size_t tauBase = Task::Cell::OffTau;   // cellP is params + 0
    for (std::size_t i = 0; i < Task::NST; ++i) params[tauBase + i] = 1.0;

    // Reverse-mode training: gradient w.r.t. all NP params in one backward pass.
    const double lr = 0.05, momentum = 0.9;
    const int epochs = 600;
    double loss0 = task.loss<double>(params);
    std::printf("LTC cell: NumParams=%zu (cell=%zu + readout=%zu), seq=%zu\n",
                Task::NP, Task::Cell::NumParams, Task::NReadout, Task::L);
    std::printf("epoch %4d   loss %.6e\n", 0, loss0);

    // Learning-curve CSV (header + one row per epoch) for plot.py.
    std::FILE* lossCsv = std::fopen("ltc_loss.csv", "w");
    std::fprintf(lossCsv, "epoch,loss\n");
    std::fprintf(lossCsv, "0,%.8e\n", loss0);

    for (int e = 1; e <= epochs; ++e)
    {
        tinymind::pinn::sgdStepReverse<Task::NP>(params, velocity, lr, momentum, lossFn);
        // Keep per-neuron time constants positive (LTC requires tau > 0).
        for (std::size_t i = 0; i < Task::NST; ++i)
            if (params[tauBase + i] < 0.1) params[tauBase + i] = 0.1;

        const double le = task.loss<double>(params);
        std::fprintf(lossCsv, "%d,%.8e\n", e, le);
        if (e % 100 == 0)
            std::printf("epoch %4d   loss %.6e\n", e, le);
    }
    std::fclose(lossCsv);

    double pred[Task::L];
    task.predict(params, pred);

    // Fit CSV: target vs predicted across the whole sequence.
    std::FILE* fitCsv = std::fopen("ltc_fit.csv", "w");
    std::fprintf(fitCsv, "t,target,predicted\n");
    for (std::size_t t = 0; t < Task::L; ++t)
        std::fprintf(fitCsv, "%zu,%.6f,%.6f\n", t, task.tgt[t], pred[t]);
    std::fclose(fitCsv);

    std::printf("\n  t      target     predicted\n");
    for (std::size_t t = 0; t < Task::L; t += 3)
        std::printf("%3zu    %8.5f    %8.5f\n", t, task.tgt[t], pred[t]);

    const double lossN = task.loss<double>(params);
    std::printf("\nfinal loss %.6e  (%.1fx reduction)\n", lossN, loss0 / lossN);
    return (lossN < loss0 * 0.1) ? 0 : 1;   // expect >=10x reduction
}
