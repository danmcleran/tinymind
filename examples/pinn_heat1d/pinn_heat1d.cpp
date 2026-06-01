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

// PINN building block: the 1-D heat-equation residual r = u_t - nu * u_xx,
// computed by forward-mode automatic differentiation of a field u(x, t)
// w.r.t. its INPUT coordinates -- the derivative a PDE residual needs and the
// one TinyMind's weight-gradient backprop cannot provide.
//
// This exemplar proves the differentiation machinery end-to-end, not a trained
// solver:
//   (1) On an exact heat solution u = x^2/2 + nu*t, the residual evaluates to
//       ~0 to machine precision -- validating the residual operator and the
//       autodiff derivatives (u_t via a first-order dual, u_xx via a nested
//       second-order dual).
//   (2) On a nonlinear fixed-weight tanh field, forward-mode u_x / u_xx / u_t
//       match high-accuracy central finite differences -- showing the autodiff
//       is exact (no step-size/truncation error) on a realistic PINN field.
//
// Training (optimizing weights so the residual vanishes) is the next milestone;
// it additionally needs derivatives of the residual w.r.t. the WEIGHTS.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "dual.hpp"
#include "dualActivations.hpp"

using tinymind::Dual;
// Unqualified tanh() below resolves by ADL: tinymind::tanh for Dual arguments,
// std::tanh for plain double. Lets the field template serve both.
using std::tanh;

typedef Dual<double>  D1;  // first-order: value + d/ds
typedef Dual<D1>      D2;  // second-order: nest once more

static const double NU = 0.3; // thermal diffusivity

// ---- generic constant lifting: a double literal into any (nested) dual -----
template<typename S> struct Lit;
template<> struct Lit<double> { static double make(double d) { return d; } };
template<typename V> struct Lit<Dual<V> >
{
    static Dual<V> make(double d) { return Dual<V>(Lit<V>::make(d)); }
};

// ---------------------------------------------------------------------------
// Field A: an exact solution of u_t = nu * u_xx, namely u = x^2/2 + nu*t
// (u_t = nu, u_xx = 1, so nu*u_xx = nu). Pure arithmetic -- the residual must
// be ~0 once the autodiff derivatives are correct.
// ---------------------------------------------------------------------------
template<typename S>
S fieldExact(const S& x, const S& t)
{
    const S half = Lit<S>::make(0.5);
    const S nu   = Lit<S>::make(NU);
    return (x * x) * half + nu * t;
}

// ---------------------------------------------------------------------------
// Field B: a nonlinear field, u = sum_j wout[j] * tanh(w0[j]*x + w1[j]*t + b[j])
// with fixed (hand-chosen) weights -- the shape of a single-hidden-layer PINN.
// Untrained, so its heat residual is nonzero; that residual is exactly the
// signal a PINN trainer would minimize.
// ---------------------------------------------------------------------------
template<typename S>
S fieldMLP(const S& x, const S& t)
{
    static const int H = 4;
    static const double w0[H]   = { 1.2, -0.8,  0.5, -1.5 };
    static const double w1[H]   = {-0.4,  0.9, -1.1,  0.3 };
    static const double bh[H]   = { 0.1, -0.2,  0.05, 0.4 };
    static const double wout[H] = { 0.7,  0.5, -0.6,  0.9 };
    static const double bout    = -0.15;

    S acc = Lit<S>::make(bout);
    for (int j = 0; j < H; ++j)
    {
        const S z = Lit<S>::make(w0[j]) * x
                  + Lit<S>::make(w1[j]) * t
                  + Lit<S>::make(bh[j]);
        acc = acc + Lit<S>::make(wout[j]) * tanh(z);
    }
    return acc;
}

int main(int argc, char** argv)
{
    const bool bench = (argc > 1) && (std::strcmp(argv[1], "--bench") == 0);
    (void)bench;

    std::printf("1-D heat equation residual via forward-mode autodiff\n");
    std::printf("  u_t = nu * u_xx,  nu = %.3f\n\n", NU);

    int failures = 0;
    const int GX = 5, GT = 4;

    // --- Check 1: exact solution -> residual ~ 0 -----------------------------
    double maxAbsResExact = 0.0;
    for (int i = 0; i < GX; ++i)
    {
        for (int k = 0; k < GT; ++k)
        {
            const double x = -1.0 + 2.0 * i / (GX - 1);
            const double t = 0.05 + 0.9 * k / (GT - 1);
            const double ut  = fieldExact<D1>(D1(x), D1(t, 1.0)).deriv;
            const D2 xv(D1(x, 1.0), D1(1.0, 0.0));
            const D2 tv(D1(t), D1(0.0));
            const double uxx = fieldExact<D2>(xv, tv).deriv.deriv;
            const double res = ut - NU * uxx;
            if (std::fabs(res) > maxAbsResExact) maxAbsResExact = std::fabs(res);
        }
    }
    std::printf("[exact solution u = x^2/2 + nu*t]\n");
    std::printf("  max |residual| over %dx%d grid = %.3e  (expect ~0)\n",
                GX, GT, maxAbsResExact);
    if (maxAbsResExact > 1e-9) { ++failures; std::printf("  FAIL\n"); }
    else                       { std::printf("  OK\n"); }

    // --- Check 2: nonlinear field, autodiff vs central finite differences ----
    const double h = 1e-5;
    double maxErrDx = 0.0, maxErrDt = 0.0, maxErrDxx = 0.0;
    double maxAbsResMLP = 0.0;
    for (int i = 0; i < GX; ++i)
    {
        for (int k = 0; k < GT; ++k)
        {
            const double x = -1.0 + 2.0 * i / (GX - 1);
            const double t = 0.05 + 0.9 * k / (GT - 1);

            const double ux  = fieldMLP<D1>(D1(x, 1.0), D1(t)).deriv;
            const double ut  = fieldMLP<D1>(D1(x), D1(t, 1.0)).deriv;
            const D2 xv(D1(x, 1.0), D1(1.0, 0.0));
            const D2 tv(D1(t), D1(0.0));
            const double uxx = fieldMLP<D2>(xv, tv).deriv.deriv;

            // Central finite-difference references (double evaluation).
            const double fd_ux  = (fieldMLP<double>(x + h, t) - fieldMLP<double>(x - h, t)) / (2 * h);
            const double fd_ut  = (fieldMLP<double>(x, t + h) - fieldMLP<double>(x, t - h)) / (2 * h);
            const double fd_uxx = (fieldMLP<double>(x + h, t) - 2 * fieldMLP<double>(x, t)
                                   + fieldMLP<double>(x - h, t)) / (h * h);

            maxErrDx  = std::fmax(maxErrDx,  std::fabs(ux  - fd_ux));
            maxErrDt  = std::fmax(maxErrDt,  std::fabs(ut  - fd_ut));
            maxErrDxx = std::fmax(maxErrDxx, std::fabs(uxx - fd_uxx));

            const double res = ut - NU * uxx;
            maxAbsResMLP = std::fmax(maxAbsResMLP, std::fabs(res));
        }
    }
    std::printf("\n[nonlinear tanh field: autodiff vs central finite difference]\n");
    std::printf("  max |u_x  - FD| = %.3e\n", maxErrDx);
    std::printf("  max |u_t  - FD| = %.3e\n", maxErrDt);
    std::printf("  max |u_xx - FD| = %.3e   (gap is FD round-off ~eps/h^2, not AD error)\n",
                maxErrDxx);
    // u_x / u_t: FD central truncation ~ h^2 ~ 1e-10 bounds the gap. u_xx: the
    // second-difference is round-off dominated (~eps/h^2 ~ 1e-6); the autodiff
    // value is the exact one, so allow a looser bound on the comparison.
    if (maxErrDx  > 1e-7) { ++failures; std::printf("  u_x  FAIL\n"); }
    if (maxErrDt  > 1e-7) { ++failures; std::printf("  u_t  FAIL\n"); }
    if (maxErrDxx > 1e-3) { ++failures; std::printf("  u_xx FAIL\n"); }
    if (failures == 0)    { std::printf("  OK (autodiff matches finite differences)\n"); }

    std::printf("\n  untrained-field heat residual max |r| = %.3e"
                "  (nonzero: this is the PINN training signal)\n", maxAbsResMLP);

    std::printf("\n%s\n", failures == 0 ? "PINN RESIDUAL MACHINERY VERIFIED"
                                        : "FAILURES DETECTED");
    return failures == 0 ? 0 : 1;
}
