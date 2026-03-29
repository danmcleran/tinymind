/**
* Copyright (c) 2020 Intel Corporation
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
     * IntegerPart and FractionalPart are passed to the QValue(int, unsigned)
     * constructor. For floating-point types, provide your own decay policy.
     *
     * Example for Q8.8: L2WeightDecay<ValueType, 0, 1> gives lambda ≈ 1/256 ≈ 0.004
     */
    template<typename ValueType, int IntegerPart = 0, unsigned FractionalPart = 1>
    struct L2WeightDecay
    {
        static ValueType applyDecay(const ValueType& weight, const ValueType& learningRate)
        {
            static const ValueType lambda(IntegerPart, FractionalPart);

            return weight - (learningRate * lambda * weight);
        }
    };
}
