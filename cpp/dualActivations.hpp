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
#include "dual.hpp"
#include "constants.hpp"
#include "activationFunctions.hpp"

#if TINYMIND_ENABLE_STD
#include <cmath>
#endif

namespace tinymind {

    // Forward declarations: the nested-Dual scalar policy below calls these.
    template<typename ValueType> Dual<ValueType> tanh(const Dual<ValueType>& a);
    template<typename ValueType> Dual<ValueType> sigmoid(const Dual<ValueType>& a);

    /**
     * Scalar-activation provider for the Dual<> activation overloads -- the one
     * type-specific hook in forward-mode AD. The primary template targets
     * TinyMind's fixed-point QValue: it sources tanh/sigmoid from the existing
     * lookup-table activation policies and the unit "one" from Constants.
     *
     * IEEE float/double get explicit specializations (host-only) that use
     * std::tanh / std::exp. A Dual<W> specialization recurses, so a nested
     * Dual<Dual<...>> (used for 2nd- and higher-order input derivatives)
     * differentiates the activation at every level. The dual activation
     * overloads close over this policy, so tanh(Dual<V>) works for every V.
     */
    template<typename ValueType>
    struct DualScalarActivation
    {
        static ValueType tanhValue(const ValueType& x)
        {
            return TanhActivationPolicy<ValueType>::activationFunction(x);
        }
        static ValueType sigmoidValue(const ValueType& x)
        {
            return SigmoidActivationPolicy<ValueType>::activationFunction(x);
        }
        static ValueType one() { return Constants<ValueType>::one(); }
    };

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    template<>
    struct DualScalarActivation<float>
    {
        static float tanhValue(float x)    { return std::tanh(x); }
        static float sigmoidValue(float x) { return 1.0f / (1.0f + std::exp(-x)); }
        static float one()                 { return 1.0f; }
    };

    template<>
    struct DualScalarActivation<double>
    {
        static double tanhValue(double x)    { return std::tanh(x); }
        static double sigmoidValue(double x) { return 1.0 / (1.0 + std::exp(-x)); }
        static double one()                  { return 1.0; }
    };
#endif

    // Nested dual: differentiating an activation of a Dual recurses one level,
    // enabling higher-order input derivatives (e.g. d^2u/dx^2 via
    // Dual<Dual<...>>). The "scalar" tanh of a Dual is just tanh on that Dual.
    template<typename W>
    struct DualScalarActivation<Dual<W> >
    {
        static Dual<W> tanhValue(const Dual<W>& x)    { return tinymind::tanh(x); }
        static Dual<W> sigmoidValue(const Dual<W>& x) { return tinymind::sigmoid(x); }
        static Dual<W> one() { return Dual<W>(DualScalarActivation<W>::one()); }
    };

    /**
     * tanh on a dual: value = tanh(x), derivative coefficient = 1 - tanh(x)^2,
     * threaded through the chain rule. d/dx tanh(x) = 1 - tanh(x)^2.
     */
    template<typename ValueType>
    inline Dual<ValueType> tanh(const Dual<ValueType>& a)
    {
        const ValueType t = DualScalarActivation<ValueType>::tanhValue(a.value);
        const ValueType fPrime = DualScalarActivation<ValueType>::one() - (t * t);
        return chainRule(a, t, fPrime);
    }

    /**
     * sigmoid on a dual: value = s = sigmoid(x), derivative coefficient
     * = s * (1 - s). d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
     */
    template<typename ValueType>
    inline Dual<ValueType> sigmoid(const Dual<ValueType>& a)
    {
        const ValueType s = DualScalarActivation<ValueType>::sigmoidValue(a.value);
        const ValueType fPrime = s * (DualScalarActivation<ValueType>::one() - s);
        return chainRule(a, s, fPrime);
    }

} // namespace tinymind
