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

// Closed-form Continuous-time (CfC) cell demo: irregular sampling, where the
// elapsed time between samples carries information the cell cannot get any
// other way.
//
// Task: a first-order system  dy/dt = (u - y) / tau  driven by a piecewise-
// constant input u, observed at IRREGULAR instants. Between two samples the
// input holds the new sample's value, so the exact discrete update is
//
//     y_k = y_{k-1} * exp(-ts_k / tau) + (1 - exp(-ts_k / tau)) * u_k
//
// with ts_k the elapsed time since the previous sample. The cell sees only u_k
// as its input; how far y moves toward u_k depends on ts_k, which reaches the
// cell only through the CfC time-gate. A model that ignores ts can only learn
// one average decay per step.
//
// The example trains two models with the same seed and budget:
//   * the full CfC cell (ts feeds the time-gate);
//   * the same cell with the time-gate's ts coefficient (W_A, b_A) held at 0,
//     which makes it blind to ts;
// and scores both on held-out sequences with fresh inputs and fresh time
// steps. It exits non-zero unless the full cell beats the ts-blind cell by at
// least 2x on held-out MSE, so the demonstration checks itself.
//
// Training uses only the existing reverse-mode autodiff trainer
// (pinn::sgdStepReverse); the same scalar-templated step<S> serves both the
// RevVar gradient and the double inference pass.

#define TINYMIND_CFC_REVERSE_TRAINING 1

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "cfc.hpp"
#include "revdual.hpp"

using tinymind::RevVar;
using tinymind::pinn::Constant;

static const std::size_t NIN = 1;
static const std::size_t NST = 8;
static const std::size_t BB  = 16;
typedef tinymind::cfc::CfCCell<NIN, NST, BB> Cell;
static const std::size_t NReadout = NST + 1;
static const std::size_t NP = Cell::NumParams + NReadout;

static const std::size_t L    = 24;  // steps per sequence
static const std::size_t NSEQ = 10;  // sequences per set
static const double TAU = 1.0;

struct Lcg
{
    unsigned s;
    explicit Lcg(unsigned seed) : s(seed) {}
    double uniform()
    {
        s = s * 1103515245u + 12345u;
        return static_cast<double>((s >> 16) & 0x7fff) / 32767.0;
    }
};

struct SequenceSet
{
    double u[NSEQ][L];
    double ts[NSEQ][L];
    double y[NSEQ][L];

    explicit SequenceSet(unsigned seed)
    {
        Lcg rng(seed);
        for (std::size_t n = 0; n < NSEQ; ++n)
        {
            double level = 0.0;
            double state = 0.0;
            for (std::size_t k = 0; k < L; ++k)
            {
                // The input level switches at random among {-1, -0.5, 0, 0.5, 1}.
                if ((k == 0) || (rng.uniform() < 0.3))
                {
                    level = -1.0 + 0.5 * static_cast<double>(static_cast<int>(rng.uniform() * 4.999));
                }
                // Elapsed time log-uniform on [0.1, 2.5]: per-step decay 0.90 .. 0.08.
                const double dt = 0.1 * std::exp(rng.uniform() * std::log(25.0));
                const double a = std::exp(-dt / TAU);
                state = a * state + (1.0 - a) * level;
                u[n][k] = level;
                ts[n][k] = dt;
                y[n][k] = state;
            }
        }
    }
};

static double toDouble(const double v) { return v; }
static double toDouble(const RevVar& v) { return v.value(); }

template<typename S>
static S runSequence(const S* p, const SequenceSet& set, const std::size_t n, double* out)
{
    const S* cellP = p;
    const S* wout  = p + Cell::NumParams;
    const S  bout  = wout[NST];

    S state[NST];
    for (std::size_t i = 0; i < NST; ++i)
    {
        state[i] = Constant<S>::of(0.0);
    }

    S acc = Constant<S>::of(0.0);
    for (std::size_t k = 0; k < L; ++k)
    {
        S x = Constant<S>::of(set.u[n][k]);
        S ns[NST];
        Cell::template step<S>(cellP, &x, state, ns, set.ts[n][k]);
        for (std::size_t i = 0; i < NST; ++i)
        {
            state[i] = ns[i];
        }

        S yhat = bout;
        for (std::size_t i = 0; i < NST; ++i)
        {
            yhat = yhat + wout[i] * state[i];
        }
        if (out != nullptr)
        {
            out[k] = toDouble(yhat);
        }

        const S e = yhat - Constant<S>::of(set.y[n][k]);
        acc = acc + e * e;
    }
    return acc;
}

template<typename S>
static S setLoss(const S* p, const SequenceSet& set)
{
    S acc = Constant<S>::of(0.0);
    for (std::size_t n = 0; n < NSEQ; ++n)
    {
        acc = acc + runSequence<S>(p, set, n, nullptr);
    }
    return acc * Constant<S>::of(1.0 / static_cast<double>(NSEQ * L));
}

struct LossFn
{
    const SequenceSet* set;
    RevVar operator()(const RevVar* p) const { return setLoss<RevVar>(p, *set); }
};

static void initParams(double* params, double* velocity, const bool tsBlind)
{
    Lcg rng(2718u);
    for (std::size_t i = 0; i < NP; ++i)
    {
        params[i] = 0.30 * (rng.uniform() - 0.5);
        velocity[i] = 0.0;
    }
    if (tsBlind)
    {
        for (std::size_t i = Cell::OffWA; i < Cell::OffWB; ++i)
        {
            params[i] = 0.0;
        }
    }
}

// Train for `epochs`. With tsBlind the time-gate's ts coefficient (W_A, b_A)
// is pinned at 0 after every step, so ts never reaches the output.
static void train(double* params, double* velocity, const SequenceSet& set, const int epochs,
                  const bool tsBlind, std::FILE* lossCsv, const char* label)
{
    LossFn lossFn;
    lossFn.set = &set;
    for (int e = 1; e <= epochs; ++e)
    {
        tinymind::pinn::sgdStepReverse<NP>(params, velocity, 0.05, 0.9, lossFn);
        if (tsBlind)
        {
            for (std::size_t i = Cell::OffWA; i < Cell::OffWB; ++i)
            {
                params[i] = 0.0;
                velocity[i] = 0.0;
            }
        }
        if (lossCsv != nullptr)
        {
            std::fprintf(lossCsv, "%d,%.8e\n", e, setLoss<double>(params, set));
        }
        if (e % 500 == 0)
        {
            std::printf("  [%s] epoch %4d   train loss %.6e\n", label, e, setLoss<double>(params, set));
        }
    }
}

int main()
{
    const SequenceSet trainSet(11u);
    const SequenceSet testSet(9001u);   // fresh inputs AND fresh time steps
    const int epochs = 2500;

    std::printf("CfC cell: NumParams=%zu (cell=%zu + readout=%zu); %zu train + %zu held-out sequences of %zu irregular steps\n",
                NP, Cell::NumParams, NReadout, NSEQ, NSEQ, L);

    // Trivial baseline: predict the held-out set's own mean (generous to it).
    double mean = 0.0;
    for (std::size_t n = 0; n < NSEQ; ++n)
    {
        for (std::size_t k = 0; k < L; ++k)
        {
            mean += testSet.y[n][k];
        }
    }
    mean /= static_cast<double>(NSEQ * L);
    double meanMse = 0.0;
    for (std::size_t n = 0; n < NSEQ; ++n)
    {
        for (std::size_t k = 0; k < L; ++k)
        {
            const double d = testSet.y[n][k] - mean;
            meanMse += d * d;
        }
    }
    meanMse /= static_cast<double>(NSEQ * L);

    static double full[NP], fullV[NP], blind[NP], blindV[NP];
    initParams(full, fullV, false);
    initParams(blind, blindV, true);
    const double loss0 = setLoss<double>(full, trainSet);

    std::FILE* lossCsv = std::fopen("cfc_loss.csv", "w");
    std::fprintf(lossCsv, "epoch,loss\n0,%.8e\n", loss0);
    train(full, fullV, trainSet, epochs, false, lossCsv, "full    ");
    std::fclose(lossCsv);
    train(blind, blindV, trainSet, epochs, true, nullptr, "ts-blind");

    const double fullTrain = setLoss<double>(full, trainSet);
    const double fullTest = setLoss<double>(full, testSet);
    const double blindTrain = setLoss<double>(blind, trainSet);
    const double blindTest = setLoss<double>(blind, testSet);

    // Fit CSV: the first held-out sequence, both models.
    double predFull[L], predBlind[L];
    runSequence<double>(full, testSet, 0, predFull);
    runSequence<double>(blind, testSet, 0, predBlind);
    std::FILE* fitCsv = std::fopen("cfc_fit.csv", "w");
    std::fprintf(fitCsv, "t,ts,input,target,predicted,predicted_ts_blind\n");
    double tCum = 0.0;
    for (std::size_t k = 0; k < L; ++k)
    {
        tCum += testSet.ts[0][k];
        std::fprintf(fitCsv, "%.4f,%.4f,%.4f,%.6f,%.6f,%.6f\n", tCum, testSet.ts[0][k], testSet.u[0][k],
                     testSet.y[0][k], predFull[k], predBlind[k]);
    }
    std::fclose(fitCsv);

    std::printf("\n  held-out sequence 0\n     ts    input    target       CfC  ts-blind\n");
    for (std::size_t k = 0; k < L; k += 3)
    {
        std::printf("  %5.3f  %6.2f  %8.5f  %8.5f  %8.5f\n", testSet.ts[0][k], testSet.u[0][k],
                    testSet.y[0][k], predFull[k], predBlind[k]);
    }

    std::printf("\n                          train MSE   held-out MSE\n");
    std::printf("  CfC (ts -> time-gate)   %.3e   %.3e\n", fullTrain, fullTest);
    std::printf("  CfC, ts-blind           %.3e   %.3e\n", blindTrain, blindTest);
    std::printf("  predict held-out mean   -           %.3e\n", meanMse);
    std::printf("\nheld-out MSE, ts-blind / full: %.1fx\n", blindTest / fullTest);

    const bool ok = (fullTrain < loss0 * 0.5) && (fullTest * 2.0 < blindTest) && (fullTest < meanMse);
    std::printf("%s\n", ok ? "OK: the time-gate uses ts" : "FAIL: ts did not help on held-out data");
    return ok ? 0 : 1;
}
