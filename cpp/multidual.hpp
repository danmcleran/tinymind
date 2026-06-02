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

// Vector forward-mode automatic differentiation: a value carrying N tangent
// (partial-derivative) directions at once. One forward evaluation of a scalar
// function f: R^N -> R in MultiDual<V, N> yields f and its full gradient
// (df/ds_0 .. df/ds_{N-1}) exactly -- no finite-difference step-size error and a
// single traversal instead of N+1 separate passes.
//
// For PINNs this gives the loss gradient w.r.t. ALL network weights in one pass:
// seed each weight as a distinct tangent. MultiDual composes under Dual<> (the
// nested-policy specializations below), so the residual's input derivatives
// (u_t, u_xx) and the weight gradient are obtained together. Cost is O(N) per
// op -- the same FLOPs as N forward passes, but exact and one sweep; reverse
// mode would be asymptotically cheaper for very large nets and remains future
// work. Built from the value type's arithmetic, so it works for float/double
// and fixed-point QValue alike.

#include "include/tinymind_platform.hpp"
#include "dual.hpp"
#include "dualActivations.hpp"
#include "dualmath.hpp"

#include <cstddef>

namespace tinymind {

    template<typename ValueType, std::size_t N>
    struct MultiDual
    {
        ValueType value;
        ValueType grad[N];

        MultiDual() : value() { for (std::size_t i = 0; i < N; ++i) grad[i] = ValueType(); }
        explicit MultiDual(const ValueType& v) : value(v)
        {
            for (std::size_t i = 0; i < N; ++i) grad[i] = ValueType();
        }

        // Seed tangent direction k with `one` (the value type's unit).
        static MultiDual seed(const ValueType& v, std::size_t k, const ValueType& one)
        {
            MultiDual d(v);
            d.grad[k] = one;
            return d;
        }
    };

    template<typename V, std::size_t N>
    inline MultiDual<V, N> operator+(const MultiDual<V, N>& a, const MultiDual<V, N>& b)
    {
        MultiDual<V, N> r; r.value = a.value + b.value;
        for (std::size_t i = 0; i < N; ++i) r.grad[i] = a.grad[i] + b.grad[i];
        return r;
    }

    template<typename V, std::size_t N>
    inline MultiDual<V, N> operator-(const MultiDual<V, N>& a, const MultiDual<V, N>& b)
    {
        MultiDual<V, N> r; r.value = a.value - b.value;
        for (std::size_t i = 0; i < N; ++i) r.grad[i] = a.grad[i] - b.grad[i];
        return r;
    }

    template<typename V, std::size_t N>
    inline MultiDual<V, N> operator-(const MultiDual<V, N>& a)
    {
        MultiDual<V, N> r; r.value = V() - a.value;
        for (std::size_t i = 0; i < N; ++i) r.grad[i] = V() - a.grad[i];
        return r;
    }

    template<typename V, std::size_t N>
    inline MultiDual<V, N> operator*(const MultiDual<V, N>& a, const MultiDual<V, N>& b)
    {
        MultiDual<V, N> r; r.value = a.value * b.value;
        for (std::size_t i = 0; i < N; ++i)
            r.grad[i] = (a.grad[i] * b.value) + (a.value * b.grad[i]);
        return r;
    }

    template<typename V, std::size_t N>
    inline MultiDual<V, N> operator/(const MultiDual<V, N>& a, const MultiDual<V, N>& b)
    {
        const V denom = b.value * b.value;
        MultiDual<V, N> r; r.value = a.value / b.value;
        for (std::size_t i = 0; i < N; ++i)
            r.grad[i] = ((a.grad[i] * b.value) - (a.value * b.grad[i])) / denom;
        return r;
    }

    // Chain rule for a scalar function with value f and derivative fPrime,
    // both evaluated at a.value.
    template<typename V, std::size_t N>
    inline MultiDual<V, N> chainRule(const MultiDual<V, N>& a, const V& f, const V& fPrime)
    {
        MultiDual<V, N> r; r.value = f;
        for (std::size_t i = 0; i < N; ++i) r.grad[i] = fPrime * a.grad[i];
        return r;
    }

    // Elementary functions: reuse the scalar evaluators from the Dual policies.
    template<typename V, std::size_t N>
    inline MultiDual<V, N> tanh(const MultiDual<V, N>& a)
    {
        const V t = DualScalarActivation<V>::tanhValue(a.value);
        return chainRule(a, t, DualScalarActivation<V>::one() - (t * t));
    }
    template<typename V, std::size_t N>
    inline MultiDual<V, N> sigmoid(const MultiDual<V, N>& a)
    {
        const V s = DualScalarActivation<V>::sigmoidValue(a.value);
        return chainRule(a, s, s * (DualScalarActivation<V>::one() - s));
    }
    template<typename V, std::size_t N>
    inline MultiDual<V, N> exp(const MultiDual<V, N>& a)
    {
        const V f = DualScalarMath<V>::expVal(a.value);
        return chainRule(a, f, f);
    }
    template<typename V, std::size_t N>
    inline MultiDual<V, N> sin(const MultiDual<V, N>& a)
    {
        return chainRule(a, DualScalarMath<V>::sinVal(a.value), DualScalarMath<V>::cosVal(a.value));
    }
    template<typename V, std::size_t N>
    inline MultiDual<V, N> cos(const MultiDual<V, N>& a)
    {
        return chainRule(a, DualScalarMath<V>::cosVal(a.value), V() - DualScalarMath<V>::sinVal(a.value));
    }
    template<typename V, std::size_t N>
    inline MultiDual<V, N> sqrt(const MultiDual<V, N>& a)
    {
        const V f = DualScalarMath<V>::sqrtVal(a.value);
        return chainRule(a, f, DualScalarMath<V>::one() / (DualScalarMath<V>::two() * f));
    }

    // Let MultiDual sit beneath Dual<> (so input-derivatives and the weight
    // gradient compose): the scalar activation/math of a MultiDual is the same
    // op on that MultiDual.
    template<typename V, std::size_t N>
    struct DualScalarActivation<MultiDual<V, N> >
    {
        typedef MultiDual<V, N> M;
        static M tanhValue(const M& x)    { return tinymind::tanh(x); }
        static M sigmoidValue(const M& x) { return tinymind::sigmoid(x); }
        static M one() { return M(DualScalarActivation<V>::one()); }
    };

    template<typename V, std::size_t N>
    struct DualScalarMath<MultiDual<V, N> >
    {
        typedef MultiDual<V, N> M;
        static M expVal(const M& x)  { return tinymind::exp(x); }
        static M sinVal(const M& x)  { return tinymind::sin(x); }
        static M cosVal(const M& x)  { return tinymind::cos(x); }
        static M sqrtVal(const M& x) { return tinymind::sqrt(x); }
        static M one() { return M(DualScalarMath<V>::one()); }
        static M two() { return M(DualScalarMath<V>::two()); }
    };

} // namespace tinymind
