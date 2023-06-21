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

#include "constants.hpp"

namespace tinymind {
    template<unsigned NumberOfFractionalBits>
    struct ZeroToleranceShiftValue
    {
        static constexpr unsigned result = (NumberOfFractionalBits > 7) ? (NumberOfFractionalBits - 7) : 1;
    };

    template<typename ValueType, bool IsSigned>
    struct ZeroToleranceResolution
    {
    };

    template<typename ValueType>
    struct ZeroToleranceResolution<ValueType, true>
    {
        static ValueType getMaxValue()
        {
            static const ValueType zeroTolerance(1 << ZeroToleranceShiftValue<ValueType::NumberOfFractionalBits>::result);
            return zeroTolerance;
        }

        static ValueType getMinValue()
        {
            static const ValueType zeroTolerance(1 << ZeroToleranceShiftValue<ValueType::NumberOfFractionalBits>::result);
            static const ValueType negativeTolerance = (Constants<ValueType>::negativeOne() * zeroTolerance);
            return negativeTolerance;
        }
    };

    template<typename ValueType>
    struct ZeroToleranceResolution<ValueType, false>
    {
        static ValueType getMaxValue()
        {
            static const ValueType zeroTolerance(1 << ZeroToleranceShiftValue<ValueType::NumberOfFractionalBits>::result);
            return zeroTolerance;
        }

        static ValueType getMinValue()
        {
            return ValueType(0);
        }
    };

    template<typename ValueType>
    struct ZeroToleranceCalculator
    {
        static bool isWithinZeroTolerance(const ValueType& value)
        {
            static const ValueType maxValue = ZeroToleranceResolution<ValueType, ValueType::IsSigned>::getMaxValue();
            static const ValueType minValue = ZeroToleranceResolution<ValueType, ValueType::IsSigned>::getMinValue();

            return ((Constants<ValueType>::zero() == value) || ((value < maxValue) && (value > minValue)));
        }
    };
}
