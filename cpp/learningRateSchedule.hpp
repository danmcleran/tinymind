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

#include "include/tinymind_traits.hpp"

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
     * QValue(int, fraction) constructor to create the decay factor;
     * DecayFractionalPart is a raw count of LSBs in ValueType's own format.
     * The default factor is 0.9 computed for ValueType's format by
     * QFractionLsbs (230/256 in Q8.8, 58982/65536 in Q16.16). For
     * floating-point types, provide your own schedule policy.
     *
     * Example: StepDecaySchedule<ValueType, 1000> multiplies the rate by
     * 0.9 every 1000 steps in any Q format. StepDecaySchedule<Q8_8, 1000, 0, 192>
     * spells a factor of 192/256 = 0.75 explicitly.
     */
    template<typename ValueType, size_t StepInterval = 1000,
             int DecayIntegerPart = 0, unsigned long long DecayFractionalPart = QFractionLsbs<ValueType, 9, 10>::value>
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
            static const ValueType decayFactor(static_cast<typename ValueType::FixedPartFieldType>(DecayIntegerPart), static_cast<typename ValueType::FractionalPartFieldType>(DecayFractionalPart));

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
