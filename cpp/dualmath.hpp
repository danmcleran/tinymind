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

// Elementary math on forward-mode dual numbers: exp, sin, cos, sqrt. Together
// with the arithmetic in dual.hpp and the activations in dualActivations.hpp,
// these let a PDE field / residual be written directly over Dual<> -- including
// sinusoidal (SIREN-style) fields and PDEs with exp/trig source terms.
//
// Each overload evaluates the scalar function and its analytic derivative, then
// threads the derivative through the chain rule:
//   exp:  f = e^x,      f' = e^x
//   sin:  f = sin x,    f' = cos x
//   cos:  f = cos x,    f' = -sin x
//   sqrt: f = sqrt x,   f' = 1 / (2 sqrt x)
//
// SCOPE: exp/sin/cos are provided for float/double (the primary PINN regime) and
// recurse through nested duals for higher-order derivatives. sqrt is provided
// for every value type, including fixed-point QValue (via SquareRootApproximation).
// QValue exp/sin/cos duals are NOT yet wired -- they need scalar lookup-table
// policies (the tables exist in exp.hpp / sin.hpp / cos.hpp but there is no
// standalone scalar accessor); add a DualScalarMath<QValue> specialization to
// enable them. Using exp/sin/cos on a Dual<QValue> is a compile error until then.

#include "include/tinymind_platform.hpp"
#include "dual.hpp"
#include "constants.hpp"
#include "adam.hpp"   // SquareRootApproximation<ValueType>

#if TINYMIND_ENABLE_STD
#include <cmath>
#endif

namespace tinymind {

    // Forward declarations for the nested-dual recursion below.
    template<typename ValueType> Dual<ValueType> exp(const Dual<ValueType>& a);
    template<typename ValueType> Dual<ValueType> sin(const Dual<ValueType>& a);
    template<typename ValueType> Dual<ValueType> cos(const Dual<ValueType>& a);
    template<typename ValueType> Dual<ValueType> sqrt(const Dual<ValueType>& a);

    /**
     * Scalar elementary functions used by the Dual<> math overloads. The
     * primary template provides only sqrt (valid for fixed-point QValue via
     * SquareRootApproximation) plus the unit/two constants; float/double
     * specializations add exp/sin/cos via std::. A Dual<W> specialization
     * recurses for higher-order derivatives.
     */
    template<typename ValueType>
    struct DualScalarMath
    {
        static ValueType sqrtVal(const ValueType& x)
        {
            return SquareRootApproximation<ValueType>::sqrt(x);
        }
        static ValueType one() { return Constants<ValueType>::one(); }
        static ValueType two() { return Constants<ValueType>::one() + Constants<ValueType>::one(); }
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    template<>
    struct DualScalarMath<float>
    {
        static float expVal(float x)  { return std::exp(x); }
        static float sinVal(float x)  { return std::sin(x); }
        static float cosVal(float x)  { return std::cos(x); }
        static float sqrtVal(float x) { return std::sqrt(x); }
        static float one() { return 1.0f; }
        static float two() { return 2.0f; }
    };

    template<>
    struct DualScalarMath<double>
    {
        static double expVal(double x)  { return std::exp(x); }
        static double sinVal(double x)  { return std::sin(x); }
        static double cosVal(double x)  { return std::cos(x); }
        static double sqrtVal(double x) { return std::sqrt(x); }
        static double one() { return 1.0; }
        static double two() { return 2.0; }
    };
#endif

    // Nested dual: the "scalar" math of a Dual is the same op on that Dual,
    // enabling higher-order derivatives (e.g. d^2/dx^2 of a sin-activated field).
    template<typename W>
    struct DualScalarMath<Dual<W> >
    {
        static Dual<W> expVal(const Dual<W>& x)  { return tinymind::exp(x); }
        static Dual<W> sinVal(const Dual<W>& x)  { return tinymind::sin(x); }
        static Dual<W> cosVal(const Dual<W>& x)  { return tinymind::cos(x); }
        static Dual<W> sqrtVal(const Dual<W>& x) { return tinymind::sqrt(x); }
        static Dual<W> one() { return Dual<W>(DualScalarMath<W>::one()); }
        static Dual<W> two() { return Dual<W>(DualScalarMath<W>::two()); }
    };

    template<typename ValueType>
    inline Dual<ValueType> exp(const Dual<ValueType>& a)
    {
        const ValueType f = DualScalarMath<ValueType>::expVal(a.value);
        return chainRule(a, f, f);                       // (e^x)' = e^x
    }

    template<typename ValueType>
    inline Dual<ValueType> sin(const Dual<ValueType>& a)
    {
        const ValueType s = DualScalarMath<ValueType>::sinVal(a.value);
        const ValueType c = DualScalarMath<ValueType>::cosVal(a.value);
        return chainRule(a, s, c);                       // (sin x)' = cos x
    }

    template<typename ValueType>
    inline Dual<ValueType> cos(const Dual<ValueType>& a)
    {
        const ValueType c = DualScalarMath<ValueType>::cosVal(a.value);
        const ValueType s = DualScalarMath<ValueType>::sinVal(a.value);
        return chainRule(a, c, ValueType() - s);         // (cos x)' = -sin x
    }

    template<typename ValueType>
    inline Dual<ValueType> sqrt(const Dual<ValueType>& a)
    {
        const ValueType f = DualScalarMath<ValueType>::sqrtVal(a.value);
        const ValueType fPrime =
            DualScalarMath<ValueType>::one() / (DualScalarMath<ValueType>::two() * f);
        return chainRule(a, f, fPrime);                  // (sqrt x)' = 1/(2 sqrt x)
    }

} // namespace tinymind
