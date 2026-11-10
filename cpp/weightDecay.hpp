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

#include "include/tinymind_traits.hpp"

namespace tinymind {
    /**
     * No-op weight decay policy. No regularization applied.
     */
    template<typename ValueType>
    struct NullWeightDecayPolicy
    {
        static ValueType applyDecay(const ValueType& weight, const ValueType& learningRate)
        {
            (void)learningRate;
            return weight;
        }
    };

    /**
     * L2 weight decay (ridge regularization) for fixed-point types.
     * Modifies the weight update to: w_new = w_old * (1 - lr * lambda) + delta_w
     * This prevents weights from growing unboundedly, which is especially
     * important for fixed-point where large weights cause overflow.
     *
     * The network applies it to connection weights, not biases, under every
     * optimizer. With SGD this is exactly an L2 penalty (lambda/2)*w^2 in the
     * loss. Under Adam and RMSprop the same shrink is applied to the weight
     * before the adaptive step, so it is decoupled weight decay (AdamW,
     * Loshchilov & Hutter), not an L2 term added to the gradient.
     *
     * IntegerPart and FractionalPart are passed to the QValue(int, fraction)
     * constructor; FractionalPart is a raw count of LSBs in ValueType's own
     * format. The default is lambda = 0.001 computed for ValueType's format
     * by QFractionLsbs and kept at 1 LSB or more: 1/256 in Q8.8 (0.001 is
     * below one LSB there), 66/65536 in Q16.16. For floating-point types,
     * provide your own decay policy.
     */
    template<typename ValueType, int IntegerPart = 0, unsigned long long FractionalPart = QFractionLsbs<ValueType, 1, 1000>::value>
    struct L2WeightDecay
    {
        static ValueType applyDecay(const ValueType& weight, const ValueType& learningRate)
        {
            static const ValueType lambda(static_cast<typename ValueType::FixedPartFieldType>(IntegerPart), static_cast<typename ValueType::FractionalPartFieldType>(FractionalPart));

            return weight - (learningRate * lambda * weight);
        }
    };
}
