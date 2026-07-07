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
#include <cstddef>
#include <cstdint>

// Taylor-mode automatic differentiation: a truncated Taylor polynomial (a
// "jet") in one variable, carrying coefficients c_k = f^(k)(x0)/k! up to a
// compile-time Order. Propagating arithmetic and elementary functions through
// the jet yields all derivatives up to Order in a single sweep, without the
// exponential blow-up of nesting first-order duals (Dual<Dual<...>>). This is
// the ergonomic path for high-order single-variable derivatives (e.g. u_xxx in
// a PDE residual).
//
// Seed an input with Jet::variable(x0) (c_1 = 1); read the k-th derivative as
// derivative(k) = c_k * k!. Built from the value type's arithmetic plus the
// scalar elementary evaluators in DualScalarMath<V>, so it works for float,
// double, and fixed-point QValue. tanh/sigmoid are composed from exp.

#include "include/tinymind_platform.hpp"
#include "dual.hpp"
#include "dualActivations.hpp"
#include "dualmath.hpp"
#include "constants.hpp"
#include "include/nnproperties.hpp"   // ValueConverter

#include <cstddef>

namespace tinymind {

    template<typename ValueType, std::size_t Order>
    struct Jet
    {
        ValueType c[Order + 1];

        Jet() { for (std::size_t k = 0; k <= Order; ++k) c[k] = ValueType(); }

        static Jet constant(const ValueType& v) { Jet j; j.c[0] = v; return j; }
        static Jet variable(const ValueType& v)
        {
            Jet j; j.c[0] = v;
            if (Order >= 1) j.c[1] = DualScalarMath<ValueType>::one();
            return j;
        }

        // k-th derivative = c_k * k!.
        ValueType derivative(std::size_t k) const
        {
            ValueType fact = DualScalarMath<ValueType>::one();
            for (std::size_t i = 2; i <= k; ++i) fact = fact * vInt(i);
            return c[k] * fact;
        }

        // Integer -> ValueType (k as a value; reused by the recurrences).
        static ValueType vInt(std::size_t n)
        {
            return ValueConverter<double, ValueType>::convertToDestinationType(static_cast<double>(n));
        }
    };

    template<typename V, std::size_t O>
    inline Jet<V, O> operator+(const Jet<V, O>& a, const Jet<V, O>& b)
    {
        Jet<V, O> r; for (std::size_t k = 0; k <= O; ++k) r.c[k] = a.c[k] + b.c[k]; return r;
    }
    template<typename V, std::size_t O>
    inline Jet<V, O> operator-(const Jet<V, O>& a, const Jet<V, O>& b)
    {
        Jet<V, O> r; for (std::size_t k = 0; k <= O; ++k) r.c[k] = a.c[k] - b.c[k]; return r;
    }
    template<typename V, std::size_t O>
    inline Jet<V, O> operator*(const Jet<V, O>& a, const Jet<V, O>& b)
    {
        Jet<V, O> r;
        for (std::size_t k = 0; k <= O; ++k)
        {
            V s = V();
            for (std::size_t i = 0; i <= k; ++i) s = s + a.c[i] * b.c[k - i];
            r.c[k] = s;
        }
        return r;
    }
    template<typename V, std::size_t O>
    inline Jet<V, O> operator/(const Jet<V, O>& a, const Jet<V, O>& b)
    {
        Jet<V, O> r;
        r.c[0] = a.c[0] / b.c[0];
        for (std::size_t k = 1; k <= O; ++k)
        {
            V s = a.c[k];
            for (std::size_t i = 0; i < k; ++i) s = s - r.c[i] * b.c[k - i];
            r.c[k] = s / b.c[0];
        }
        return r;
    }

    // Add / subtract a scalar constant (affects the 0th coefficient only).
    template<typename V, std::size_t O>
    inline Jet<V, O> addScalar(const Jet<V, O>& a, const V& s)
    {
        Jet<V, O> r = a; r.c[0] = r.c[0] + s; return r;
    }

    template<typename V, std::size_t O>
    inline Jet<V, O> exp(const Jet<V, O>& a)
    {
        Jet<V, O> r;
        r.c[0] = DualScalarMath<V>::expVal(a.c[0]);
        for (std::size_t k = 1; k <= O; ++k)
        {
            V s = V();
            for (std::size_t i = 1; i <= k; ++i)
                s = s + Jet<V, O>::vInt(i) * a.c[i] * r.c[k - i];
            r.c[k] = s / Jet<V, O>::vInt(k);
        }
        return r;
    }

    template<typename V, std::size_t O>
    inline void sinCos(const Jet<V, O>& a, Jet<V, O>& sn, Jet<V, O>& cs)
    {
        sn.c[0] = DualScalarMath<V>::sinVal(a.c[0]);
        cs.c[0] = DualScalarMath<V>::cosVal(a.c[0]);
        for (std::size_t k = 1; k <= O; ++k)
        {
            V ss = V(), cc = V();
            for (std::size_t i = 1; i <= k; ++i)
            {
                const V w = Jet<V, O>::vInt(i) * a.c[i];
                ss = ss + w * cs.c[k - i];
                cc = cc + w * sn.c[k - i];
            }
            sn.c[k] = ss / Jet<V, O>::vInt(k);
            cs.c[k] = (V() - cc) / Jet<V, O>::vInt(k);
        }
    }
    template<typename V, std::size_t O>
    inline Jet<V, O> sin(const Jet<V, O>& a) { Jet<V, O> s, c; sinCos(a, s, c); return s; }
    template<typename V, std::size_t O>
    inline Jet<V, O> cos(const Jet<V, O>& a) { Jet<V, O> s, c; sinCos(a, s, c); return c; }

    template<typename V, std::size_t O>
    inline Jet<V, O> sqrt(const Jet<V, O>& a)
    {
        Jet<V, O> r;
        r.c[0] = DualScalarMath<V>::sqrtVal(a.c[0]);
        const V two = DualScalarMath<V>::two();
        for (std::size_t k = 1; k <= O; ++k)
        {
            V s = a.c[k];
            for (std::size_t i = 1; i < k; ++i) s = s - r.c[i] * r.c[k - i];
            r.c[k] = s / (two * r.c[0]);
        }
        return r;
    }

    // tanh(x) = (e^{2x} - 1) / (e^{2x} + 1);  sigmoid(x) = e^x / (e^x + 1).
    template<typename V, std::size_t O>
    inline Jet<V, O> tanh(const Jet<V, O>& a)
    {
        const V one = DualScalarMath<V>::one();
        const V two = DualScalarMath<V>::two();
        Jet<V, O> two_a = a; for (std::size_t k = 0; k <= O; ++k) two_a.c[k] = two * a.c[k];
        const Jet<V, O> e = exp(two_a);
        return addScalar(e, V() - one) / addScalar(e, one);
    }
    template<typename V, std::size_t O>
    inline Jet<V, O> sigmoid(const Jet<V, O>& a)
    {
        const V one = DualScalarMath<V>::one();
        const Jet<V, O> e = exp(a);
        return e / addScalar(e, one);
    }

} // namespace tinymind
