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

#include <cstddef>
#include <cstdint>

#include "include/tinymind_traits.hpp"

namespace tinymind {
    namespace detail
    {
        // 0 in ValueType: QValue(0, 0) for fixed point, a cast for float.
        template<typename ValueType, bool IsFloat = tinymind::is_floating_point<ValueType>::value>
        struct PrototypeZero
        {
            static ValueType value()
            {
                return ValueType(0, 0);
            }
        };

        template<typename ValueType>
        struct PrototypeZero<ValueType, true>
        {
            static ValueType value()
            {
                return static_cast<ValueType>(0);
            }
        };
    }

    /**
     * Distance policies for PrototypeHead.
     */

    /**
     * sum_i (a[i] - b[i])^2. The usual nearest-class-mean distance. In a
     * fixed-point format the sum must stay inside the format's range (QValue
     * wraps by default): with Q16.16, NumberOfInputs x (largest difference)^2
     * below 32767.
     */
    template<typename ValueType>
    struct SquaredEuclideanDistance
    {
        static ValueType distance(ValueType const* const a, ValueType const* const b, const size_t count)
        {
            ValueType sum = detail::PrototypeZero<ValueType>::value();
            for (size_t i = 0; i < count; ++i)
            {
                const ValueType difference = a[i] - b[i];
                sum += difference * difference;
            }

            return sum;
        }
    };

    /**
     * sum_i |a[i] - b[i]|. No multiplies, so it is cheaper on a core without
     * a fast multiplier and has far more headroom before it overflows.
     */
    template<typename ValueType>
    struct ManhattanDistance
    {
        static ValueType distance(ValueType const* const a, ValueType const* const b, const size_t count)
        {
            const ValueType zero = detail::PrototypeZero<ValueType>::value();
            ValueType sum = zero;
            for (size_t i = 0; i < count; ++i)
            {
                const ValueType difference = a[i] - b[i];
                sum += (difference < zero) ? (zero - difference) : difference;
            }

            return sum;
        }
    };

    /**
     * A nearest-class-mean (prototype) classifier to put on top of a frozen
     * model: head-only adaptation without gradients.
     *
     * The frozen model turns a sample into an embedding of NumberOfInputs
     * values. The head keeps one prototype per class, the running mean of
     * the embeddings it was shown for that class, and predicts the class
     * whose prototype is nearest. Learning is one pass with no learning rate
     * and no epochs; a class can be added on the device at any time; a class
     * never shown is never predicted. RAM: NumberOfClasses x NumberOfInputs
     * values plus one count per class.
     *
     * Each learn() moves the prototype by (input - prototype) / count. Once a
     * class's count reaches the cap set by setMaximumCount(), the count stops
     * growing and the prototype becomes an exponential moving average with
     * weight 1 / cap, so it keeps tracking a slowly drifting input instead of
     * freezing on old samples. The default cap is effectively unlimited.
     *
     * Usage:
     *   PrototypeHead<QValue<16, 16, true>, 32, 6> head;
     *   backbone(sample, embedding);
     *   head.learn(embedding, label);                   // on device
     *   const size_t predicted = head.predict(embedding);
     *
     * @tparam ValueType       QValue or float/double
     * @tparam NumberOfInputs  Size of the embedding
     * @tparam NumberOfClasses Number of classes
     * @tparam DistancePolicy  SquaredEuclideanDistance (default) or ManhattanDistance
     */
    template<
        typename ValueType,
        size_t NumberOfInputs,
        size_t NumberOfClasses,
        typename DistancePolicy = SquaredEuclideanDistance<ValueType> >
    class PrototypeHead
    {
    public:
        PrototypeHead() : mMaximumCount(UINT16_MAX)
        {
            clear();
        }

        /**
         * Forget every class.
         */
        void clear()
        {
            for (size_t c = 0; c < NumberOfClasses; ++c)
            {
                forget(c);
            }
        }

        /**
         * Forget one class: it is not predicted again until it is learned.
         */
        void forget(const size_t classIndex)
        {
            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                mPrototypes[classIndex * NumberOfInputs + i] = detail::PrototypeZero<ValueType>::value();
            }
            mCounts[classIndex] = 0;
        }

        /**
         * Fold one labelled embedding into its class's prototype.
         */
        void learn(ValueType const* const input, const size_t classIndex)
        {
            ValueType* const prototype = &mPrototypes[classIndex * NumberOfInputs];
            if (mCounts[classIndex] < mMaximumCount)
            {
                ++mCounts[classIndex];
            }
            const int count = static_cast<int>(mCounts[classIndex]);
            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                const ValueType difference = input[i] - prototype[i];
                prototype[i] += difference / count;
            }
        }

        /**
         * The class whose prototype is nearest (the first one on a tie), or
         * NumberOfClasses when no class has been learned.
         */
        size_t predict(ValueType const* const input) const
        {
            size_t best = NumberOfClasses;
            ValueType bestDistance = detail::PrototypeZero<ValueType>::value();
            for (size_t c = 0; c < NumberOfClasses; ++c)
            {
                if (mCounts[c] == 0)
                {
                    continue;
                }
                const ValueType d = distance(input, c);
                if ((best == NumberOfClasses) || (d < bestDistance))
                {
                    best = c;
                    bestDistance = d;
                }
            }

            return best;
        }

        ValueType distance(ValueType const* const input, const size_t classIndex) const
        {
            return DistancePolicy::distance(input, &mPrototypes[classIndex * NumberOfInputs], NumberOfInputs);
        }

        /**
         * Cap on a class's count (at least 1); see the class comment.
         */
        void setMaximumCount(const uint16_t maximumCount)
        {
            mMaximumCount = (maximumCount == 0) ? static_cast<uint16_t>(1) : maximumCount;
            for (size_t c = 0; c < NumberOfClasses; ++c)
            {
                if (mCounts[c] > mMaximumCount)
                {
                    mCounts[c] = mMaximumCount;
                }
            }
        }

        uint16_t getMaximumCount() const
        {
            return mMaximumCount;
        }

        uint16_t getCount(const size_t classIndex) const
        {
            return mCounts[classIndex];
        }

        ValueType getPrototype(const size_t classIndex, const size_t input) const
        {
            return mPrototypes[classIndex * NumberOfInputs + input];
        }

        /**
         * Load one class's prototype, for example a shipped one, with the
         * weight it carries against later samples (count 0 forgets it).
         */
        void setPrototype(const size_t classIndex, ValueType const* const prototype, const uint16_t count)
        {
            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                mPrototypes[classIndex * NumberOfInputs + i] = prototype[i];
            }
            mCounts[classIndex] = (count > mMaximumCount) ? mMaximumCount : count;
        }

    private:
        ValueType mPrototypes[NumberOfClasses * NumberOfInputs];
        uint16_t mCounts[NumberOfClasses];
        uint16_t mMaximumCount;

        static_assert(NumberOfInputs > 0, "A head needs at least one input.");
        static_assert(NumberOfClasses > 0, "A head needs at least one class.");
    };
}
