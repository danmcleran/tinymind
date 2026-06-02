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

#include "include/tinymind_platform.hpp"

namespace tinymind {

    /**
     * Forward-mode automatic-differentiation dual number.
     *
     * Carries a value and its derivative with respect to one seeded input
     * direction. f(a + b*eps) = f(a) + b*f'(a)*eps, with eps^2 = 0. Built
     * purely from the value type's +, -, *, / operators, so it works for any
     * arithmetic ValueType -- IEEE float/double AND TinyMind's fixed-point
     * QValue. Whether a given type carries enough precision for a particular
     * use (e.g. a PINN PDE residual) is a separate question from whether the
     * mechanics compile and compute the chain rule correctly; both hold for
     * every ValueType that models the arithmetic interface.
     *
     * Default-constructed derivative is the type's zero (a constant); seed an
     * input direction with deriv == the type's one.
     */
    template<typename ValueType>
    struct Dual
    {
        ValueType value;
        ValueType deriv;

        Dual() : value(), deriv() {}
        explicit Dual(const ValueType& v) : value(v), deriv() {} // constant: deriv = 0
        Dual(const ValueType& v, const ValueType& d) : value(v), deriv(d) {}
    };

    template<typename ValueType>
    inline Dual<ValueType> operator+(const Dual<ValueType>& a, const Dual<ValueType>& b)
    {
        return Dual<ValueType>(a.value + b.value, a.deriv + b.deriv);
    }

    template<typename ValueType>
    inline Dual<ValueType> operator-(const Dual<ValueType>& a, const Dual<ValueType>& b)
    {
        return Dual<ValueType>(a.value - b.value, a.deriv - b.deriv);
    }

    // Unary negation. ValueType() is the type's zero (QValue and float/double).
    template<typename ValueType>
    inline Dual<ValueType> operator-(const Dual<ValueType>& a)
    {
        return Dual<ValueType>(ValueType() - a.value, ValueType() - a.deriv);
    }

    // Product rule: (a*b)' = a'b + ab'.
    template<typename ValueType>
    inline Dual<ValueType> operator*(const Dual<ValueType>& a, const Dual<ValueType>& b)
    {
        return Dual<ValueType>(a.value * b.value,
                               (a.deriv * b.value) + (a.value * b.deriv));
    }

    // Quotient rule: (a/b)' = (a'b - ab') / b^2.
    template<typename ValueType>
    inline Dual<ValueType> operator/(const Dual<ValueType>& a, const Dual<ValueType>& b)
    {
        const ValueType denom = b.value * b.value;
        return Dual<ValueType>(a.value / b.value,
                               ((a.deriv * b.value) - (a.value * b.deriv)) / denom);
    }

    /**
     * Lift a scalar function onto a dual via the chain rule, given the
     * function value f(x) and its scalar derivative fPrime = f'(x), both
     * already evaluated at a.value. This is the single hook every activation
     * needs: e.g. for tanh, f = tanh(x) and fPrime = 1 - tanh(x)^2. The scalar
     * tanh/exp/etc. evaluation is type-specific (LUT for QValue, std:: for
     * float) and lives outside this header.
     */
    template<typename ValueType>
    inline Dual<ValueType> chainRule(const Dual<ValueType>& a,
                                     const ValueType& f,
                                     const ValueType& fPrime)
    {
        return Dual<ValueType>(f, fPrime * a.deriv);
    }

} // namespace tinymind
