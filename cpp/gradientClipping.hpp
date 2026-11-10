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

namespace tinymind {
    /**
     * No-op gradient clipping policy. Passes gradients through unchanged.
     */
    template<typename ValueType>
    struct NullGradientClippingPolicy
    {
        static ValueType clip(const ValueType& gradient)
        {
            return gradient;
        }
    };

    /**
     * Clips gradients to the range [-maxValue, maxValue] for fixed-point types.
     * Prevents exploding gradients, which is especially critical for
     * fixed-point arithmetic where overflow is catastrophic.
     *
     * IntegerPart and FractionalPart are passed to the QValue(int, fraction)
     * constructor (FractionalPart is a raw count of LSBs in ValueType's own
     * format) to build the bound maxValue >= 0; the range is the symmetric
     * [-maxValue, maxValue]. The network clips every weight and bias
     * gradient with it. For floating-point types, provide your own clip
     * policy.
     *
     * Usage:
     *   typedef GradientClipByValue<ValueType> MyClipPolicy; // clips to [-1.0, 1.0]
     */
    template<typename ValueType, int IntegerPart = 1, unsigned long long FractionalPart = 0>
    struct GradientClipByValue
    {
        static_assert(IntegerPart >= 0, "The clip bound is a magnitude; the range is [-bound, bound].");

        static ValueType clip(const ValueType& gradient)
        {
            static const ValueType maxValue(static_cast<typename ValueType::FixedPartFieldType>(IntegerPart), static_cast<typename ValueType::FractionalPartFieldType>(FractionalPart));
            // Negate the whole bound. Building it as QValue(-IntegerPart,
            // FractionalPart) gave -IntegerPart + fraction, so a bound of 1.5
            // clipped to [-0.5, 1.5].
            static const ValueType minValue = ValueType(static_cast<typename ValueType::FullWidthValueType>(0)) - maxValue;

            if (gradient > maxValue)
            {
                return maxValue;
            }
            if (gradient < minValue)
            {
                return minValue;
            }
            return gradient;
        }
    };
}
