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

#include <cstddef>

namespace tinymind {
    /**
     * Fixed learning rate policy. No scheduling — the learning rate
     * remains constant throughout training. This is the default.
     */
    template<typename ValueType>
    struct FixedLearningRatePolicy
    {
        void initialize(const ValueType& initialRate)
        {
            (void)initialRate;
        }

        ValueType step(const ValueType& currentRate)
        {
            return currentRate;
        }
    };

    /**
     * Step decay learning rate schedule for fixed-point types.
     * Multiplies the learning rate by a decay factor every StepInterval
     * training steps.
     *
     * DecayIntegerPart and DecayFractionalPart are passed to the
     * QValue(int, unsigned) constructor to create the decay factor.
     * For floating-point types, provide your own schedule policy.
     *
     * Example for Q8.8: StepDecaySchedule<ValueType, 1000, 0, 230>
     *   gives factor ≈ 230/256 ≈ 0.9, applied every 1000 steps.
     */
    template<typename ValueType, size_t StepInterval = 1000,
             int DecayIntegerPart = 0, unsigned DecayFractionalPart = 230>
    struct StepDecaySchedule
    {
        StepDecaySchedule() : mStepCount(0)
        {
        }

        void initialize(const ValueType& initialRate)
        {
            (void)initialRate;
            mStepCount = 0;
        }

        ValueType step(const ValueType& currentRate)
        {
            static const ValueType decayFactor(DecayIntegerPart, DecayFractionalPart);

            ++mStepCount;
            if (mStepCount >= StepInterval)
            {
                mStepCount = 0;
                return currentRate * decayFactor;
            }
            return currentRate;
        }

    private:
        size_t mStepCount;
    };
}
