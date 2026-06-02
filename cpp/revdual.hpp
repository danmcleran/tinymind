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
#pragma once

// Reverse-mode (adjoint) automatic differentiation via a tape: record every
// operation on the forward pass, then sweep the tape once in reverse to get the
// gradient of a single scalar output w.r.t. ALL leaf inputs in O(operations) --
// independent of the number of inputs. This is the asymptotically cheaper path
// for the loss gradient w.r.t. many weights (MultiDual is one pass but O(N) work
// per op; reverse-mode is one backward pass total).
//
// RevVar composes under Dual<> (the policy specializations below), so a PINN
// residual's input derivatives (forward-mode) and the weight gradient
// (reverse-mode) are obtained together -- reverse-over-forward: build the
// residual/loss in Dual<RevVar> (or Dual<Dual<RevVar>>), call revBackward on the
// resulting RevVar, and read each weight's adjoint().
//
// HOST-ONLY: the tape uses std::vector (heap). Gated on TINYMIND_ENABLE_FLOAT &&
// TINYMIND_ENABLE_STD -- this is training-side tooling, not the deployable
// freestanding inference path. Not thread-safe (single global tape).

#include "include/tinymind_platform.hpp"

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

#include "dual.hpp"
#include "dualActivations.hpp"
#include "dualmath.hpp"

#include <cmath>
#include <vector>
#include <cstddef>

namespace tinymind {

    struct RevTape
    {
        struct Node { double p0; double p1; int i0; int i1; };
        std::vector<Node> nodes;
        std::vector<double> adj;

        int push(double p0, int i0, double p1, int i1)
        {
            Node n; n.p0 = p0; n.i0 = i0; n.p1 = p1; n.i1 = i1;
            nodes.push_back(n);
            return static_cast<int>(nodes.size()) - 1;
        }
        void reset() { nodes.clear(); adj.clear(); }
    };

    inline RevTape& revTape() { static RevTape t; return t; }
    inline void revReset() { revTape().reset(); }

    struct RevVar
    {
        double v;
        int idx;   // index into the tape; -1 for a non-differentiable constant

        RevVar() : v(0.0), idx(-1) {}
        explicit RevVar(double value) : v(value), idx(-1) {}  // constant
        RevVar(double value, int index) : v(value), idx(index) {}

        static RevVar constant(double value) { return RevVar(value); }
        static RevVar leaf(double value) { return RevVar(value, revTape().push(0.0, -1, 0.0, -1)); }

        double value() const { return v; }
        double adjoint() const { return (idx >= 0) ? revTape().adj[idx] : 0.0; }
    };

    inline RevVar operator+(const RevVar& a, const RevVar& b)
    { return RevVar(a.v + b.v, revTape().push(1.0, a.idx, 1.0, b.idx)); }
    inline RevVar operator-(const RevVar& a, const RevVar& b)
    { return RevVar(a.v - b.v, revTape().push(1.0, a.idx, -1.0, b.idx)); }
    inline RevVar operator-(const RevVar& a)
    { return RevVar(-a.v, revTape().push(-1.0, a.idx, 0.0, -1)); }
    inline RevVar operator*(const RevVar& a, const RevVar& b)
    { return RevVar(a.v * b.v, revTape().push(b.v, a.idx, a.v, b.idx)); }
    inline RevVar operator/(const RevVar& a, const RevVar& b)
    {
        const double inv = 1.0 / b.v;
        return RevVar(a.v * inv, revTape().push(inv, a.idx, -a.v * inv * inv, b.idx));
    }

    inline RevVar revTanh(const RevVar& a) { const double c = std::tanh(a.v); return RevVar(c, revTape().push(1.0 - c * c, a.idx, 0.0, -1)); }
    inline RevVar revSigmoid(const RevVar& a) { const double s = 1.0 / (1.0 + std::exp(-a.v)); return RevVar(s, revTape().push(s * (1.0 - s), a.idx, 0.0, -1)); }
    inline RevVar revExp(const RevVar& a) { const double e = std::exp(a.v); return RevVar(e, revTape().push(e, a.idx, 0.0, -1)); }
    inline RevVar revSin(const RevVar& a) { return RevVar(std::sin(a.v), revTape().push(std::cos(a.v), a.idx, 0.0, -1)); }
    inline RevVar revCos(const RevVar& a) { return RevVar(std::cos(a.v), revTape().push(-std::sin(a.v), a.idx, 0.0, -1)); }
    inline RevVar revSqrt(const RevVar& a) { const double r = std::sqrt(a.v); return RevVar(r, revTape().push(0.5 / r, a.idx, 0.0, -1)); }

    // Reverse sweep: seed the output adjoint to 1 and accumulate to all leaves.
    inline void revBackward(const RevVar& out)
    {
        RevTape& t = revTape();
        t.adj.assign(t.nodes.size(), 0.0);
        if (out.idx < 0) return;
        t.adj[out.idx] = 1.0;
        for (int i = static_cast<int>(t.nodes.size()) - 1; i >= 0; --i)
        {
            const double a = t.adj[i];
            if (a == 0.0) continue;
            const RevTape::Node& n = t.nodes[i];
            if (n.i0 >= 0) t.adj[n.i0] += a * n.p0;
            if (n.i1 >= 0) t.adj[n.i1] += a * n.p1;
        }
    }

    // Let RevVar sit beneath Dual<> (reverse-over-forward): the scalar
    // activation/math of a RevVar records on the tape.
    template<>
    struct DualScalarActivation<RevVar>
    {
        static RevVar tanhValue(const RevVar& x)    { return revTanh(x); }
        static RevVar sigmoidValue(const RevVar& x) { return revSigmoid(x); }
        static RevVar one() { return RevVar::constant(1.0); }
    };

    template<>
    struct DualScalarMath<RevVar>
    {
        static RevVar expVal(const RevVar& x)  { return revExp(x); }
        static RevVar sinVal(const RevVar& x)  { return revSin(x); }
        static RevVar cosVal(const RevVar& x)  { return revCos(x); }
        static RevVar sqrtVal(const RevVar& x) { return revSqrt(x); }
        static RevVar one() { return RevVar::constant(1.0); }
        static RevVar two() { return RevVar::constant(2.0); }
    };

namespace pinn {

    /**
     * Reverse-mode counterpart to pinn::sgdStep: one momentum-SGD step whose
     * weight gradient comes from a single reverse-mode backward pass (cost
     * independent of the parameter count), rather than the O(N) forward-mode
     * MultiDual sweep. `loss` is an object with
     * `RevVar operator()(const RevVar* params) const` -- evaluate the loss with
     * the parameters as RevVar leaves (for reverse-over-forward, build the
     * residual internally in Dual<RevVar>). Returns the pre-update loss value.
     */
    template<std::size_t NumParams, typename LossFn>
    double sgdStepReverse(double* params, double* velocity,
                          double learningRate, double momentum, const LossFn& loss)
    {
        revReset();
        RevVar rp[NumParams];
        for (std::size_t i = 0; i < NumParams; ++i) rp[i] = RevVar::leaf(params[i]);

        const RevVar L = loss(&rp[0]);
        revBackward(L);

        for (std::size_t i = 0; i < NumParams; ++i)
        {
            velocity[i] = momentum * velocity[i] - learningRate * rp[i].adjoint();
            params[i] += velocity[i];
        }
        return L.value();
    }

} // namespace pinn

} // namespace tinymind

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
