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

#include "typeChooser.hpp"

#ifndef UINT64_MAX
#define UINT64_MAX 0xFFFFFFFFFFFFFFFFULL
#endif // UINT64_MAX

namespace tinymind {

    typedef enum
    {
        AdditionOp,
        SubtractionOp,
        MultiplicationOp,
        DivisionOp,
    } OperatorType_e;

    template<typename T, unsigned NumberOfFractionalBits>
    class RoundUpPolicy
    {
    public:
        static T round(const T& value)
        {
            T result(value);

            //Round up
            result = result + ((result & (1 << (NumberOfFractionalBits - 1))) << 1);
            result >>= NumberOfFractionalBits;

            return result;
        }
    };

    template<typename T, unsigned NumberOfFractionalBits>
    class TruncatePolicy
    {
    public:
        static T round(const T& value)
        {
            T result(value);

            result >>= NumberOfFractionalBits;

            return result;
        }
    };

    template<typename T>
    class WrapPolicy // No saturation - just return the value (wrap-around behavior)
    {
    public:
        static T saturate(const T& origValue, const T& target, const T& minValue, const T& maxValue, const OperatorType_e opType)
        {
            (void)minValue; // Suppress unused parameter warnings
            (void)maxValue;

            T result;

            switch (opType) {
                case AdditionOp:
                    result = origValue + target;
                    break;
                case SubtractionOp:
                    result = origValue - target;
                    break;
                case MultiplicationOp:
                    result = origValue * target;
                    break;
                case DivisionOp:
                    result = origValue / target;
                    break;
            }

            return result;
        }
    };

    template<typename T>
    class MinMaxSaturatePolicy // Clamp value to [minValue, maxValue] range
    {
    public:
        static T saturate(const T& origValue, const T& target, const T& minValue, const T& maxValue, const OperatorType_e opType)
        {
            T result;
            switch (opType) {
                case AdditionOp:
                    if (target > 0) {
                        if (origValue > (maxValue - target)) {
                            result = maxValue; // Saturate to maximum
                        } else {
                            result = origValue + target;  // Safe addition
                        }
                    } else if (target < 0) {
                        if (origValue < (minValue - target)) {
                            result = minValue; // Saturate to minimum
                        } else {
                            result = origValue + target;  // Safe addition
                        }
                    } else {
                        result = origValue; // Adding zero: no overflow possible
                    }
                    break;
                case SubtractionOp:
                    if (target > 0) {
                        if (origValue < (minValue + target)) {
                            result = minValue;  // Saturate to minimum
                        } else {
                            result = origValue - target; // Safe subtraction
                        }
                    } else if (target < 0) {
                        if (origValue > (maxValue + target)) {
                            result = maxValue; // Saturate to maximum
                        } else {
                            result = origValue - target; // Safe subtraction
                        }
                    } else {
                        result = origValue; // Subtracting zero: no overflow possible
                    }
                    break;
                case MultiplicationOp:
                    // Check for multiplication overflow
                    if ((origValue * target) > maxValue) {
                        result = maxValue;
                    // Check for multiplication underflow as well    
                    } else if ((origValue * target) < minValue) {
                        result = minValue;
                    } else {
                        result = origValue * target; // Safe multiplication
                    }
                    break;
                case DivisionOp:
                    if (target == 0) { // Handle division by zero first
                        result = (origValue > 0) ? maxValue : minValue;
                    } else if ((origValue / target) > maxValue) {
                        result = maxValue;
                    } else if ((origValue / target) < minValue) {
                        result = minValue;
                    } else {
                        result = origValue / target; // Safe division
                    }
                    break;
            }

            return result;
        }
    };

    template<typename T, unsigned NumberOfFixedBits, unsigned NumberOfFractionalBits, bool IsSigned>
    struct SignExtender
    {
        static const T SignBitMask = (static_cast<T>(1) << (NumberOfFixedBits + NumberOfFractionalBits - 1));
        static const T SignExtensionBits = static_cast<T>((~(static_cast<size_t>(SignBitMask) - 1)) ^ static_cast<size_t>(SignBitMask));

        static void signExtend(T& value)
        {
            if (value & SignBitMask)
            {
                value |= SignExtensionBits;
            }
        }        
    };

    template<typename T, unsigned NumberOfFixedBits, unsigned NumberOfFractionalBits>
    struct SignExtender<T, NumberOfFixedBits, NumberOfFractionalBits, false>
    {
        static void signExtend(T& value)
        {
            (void)value; // Suppress unused parameter warning
        }
    };

    template<typename SourceType, typename ResultType>
    struct NoShiftPolicy
    {
        static ResultType shift(const SourceType& value)
        {
            return static_cast<ResultType>(value);
        }
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftRightPolicy
    {
        static ResultType shift(const SourceType& value)
        {
            return static_cast<ResultType>(value >> (SourceNumberOfFractionalBits - ResultNumberOfFractionalBits));
        }
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftLeftPolicy
    {
        static ResultType shift(const SourceType& value)
        {
            return (static_cast<ResultType>(value) << (ResultNumberOfFractionalBits - SourceNumberOfFractionalBits));
        }
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits, bool IsSourceBigger>
    struct ShiftPolicySelectorShiftRequired
    {
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftPolicySelectorShiftRequired<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits, true>
    {
        typedef ShiftRightPolicy<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits> ShiftPolicyType;
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftPolicySelectorShiftRequired<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits, false>
    {
        typedef ShiftLeftPolicy<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits> ShiftPolicyType;
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits, bool NumberOfBitsEqual>
    struct ShiftPolicySelectorCheckForEqual
    {
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftPolicySelectorCheckForEqual<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits, true>
    {
        typedef NoShiftPolicy<SourceType, ResultType> ShiftPolicyType;
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftPolicySelectorCheckForEqual<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits, false>
    {
        typedef typename ShiftPolicySelectorShiftRequired<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits, (SourceNumberOfFractionalBits > ResultNumberOfFractionalBits)>::ShiftPolicyType ShiftPolicyType;
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftPolicySelector
    {
        typedef typename ShiftPolicySelectorCheckForEqual<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits, (SourceNumberOfFractionalBits == ResultNumberOfFractionalBits)>::ShiftPolicyType ShiftPolicyType;
    };

    template<typename SourceType, typename ResultType, unsigned SourceNumberOfFractionalBits, unsigned ResultNumberOfFractionalBits>
    struct ShiftPolicy
    {
        typedef typename ShiftPolicySelector<SourceType, ResultType, SourceNumberOfFractionalBits, ResultNumberOfFractionalBits>::ShiftPolicyType ShiftPolicyType;

        static ResultType shift(const SourceType& value)
        {
            const ResultType result = ShiftPolicyType::shift(value);

            return result;
        }
    };

    template<unsigned NumBits, bool IsSigned>
    struct MultiplicationResultFullWidthFieldTypeChooser
    {
        static const unsigned FullWidthResult = FullWidthFieldTypeChooser<NumBits, IsSigned>::Result;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FullWidthValueType MultiplicationResultFullWidthValueType;
    };

    template<unsigned NumBits, bool IsSigned>
    struct DivisionResultFullWidthValueTypeChooser
    {
        static const unsigned FullWidthResult = FullWidthFieldTypeChooser<NumBits, IsSigned>::Result;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FullWidthValueType DivisionResultFullWidthValueType;
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits, bool IsSigned>
    struct QTypeChooser
    {
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FixedPartFieldType                                         FixedPartFieldType;
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FractionalPartFieldType                                    FractionalPartFieldType;
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FullWidthFieldType                                         FullWidthFieldType;
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FullWidthValueType                                         FullWidthValueType;
        typedef typename MultiplicationResultFullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::MultiplicationResultFullWidthValueType MultiplicationResultFullWidthValueType;
        typedef typename DivisionResultFullWidthValueTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::DivisionResultFullWidthValueType             DivisionResultFullWidthValueType;
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits, bool IsSigned>
    struct QValueMaxCalculator
    {
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits>
    struct QValueMaxCalculator<NumFixedBits, NumFractionalBits, false>
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FixedPartFieldType      FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FractionalPartFieldType FractionalPartFieldType;

        static const FixedPartFieldType MaxFixedPartValue      = (static_cast<FixedPartFieldType>((1ULL << NumFixedBits) - 1));
        static const FixedPartFieldType MaxFractionalPartValue = (static_cast<FractionalPartFieldType>((1ULL << NumFractionalBits) - 1));
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits>
    struct QValueMaxCalculator<NumFixedBits, NumFractionalBits, true>
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, true>::FixedPartFieldType      FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, true>::FractionalPartFieldType FractionalPartFieldType;

        static const FixedPartFieldType MaxFixedPartValue      = (static_cast<FixedPartFieldType>((1ULL << (NumFixedBits - 1)) - 1));
        static const FixedPartFieldType MaxFractionalPartValue = (static_cast<FractionalPartFieldType>((1ULL << NumFractionalBits) - 1));
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits, bool IsSigned>
    struct QValueMinCalculator
    {
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits>
    struct QValueMinCalculator<NumFixedBits, NumFractionalBits, false>
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FixedPartFieldType      FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FractionalPartFieldType FractionalPartFieldType;

        static const FixedPartFieldType MinFixedPartValue      = 0;
        static const FixedPartFieldType MinFractionalPartValue = 0;
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits>
    struct QValueMinCalculator<NumFixedBits, NumFractionalBits, true>
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, true>::FixedPartFieldType      FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, true>::FractionalPartFieldType FractionalPartFieldType;

        static const FixedPartFieldType MinFixedPartValue      = -static_cast<FixedPartFieldType>(1ULL << (NumFixedBits - 1));
        static const FixedPartFieldType MinFractionalPartValue = 0;
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits, bool IsSigned>
    struct QValueBounds
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, IsSigned>::FullWidthFieldType FullWidthFieldType;
        
        // Get min value for fixed-point Q-format
        static FullWidthFieldType getMinValue()
        {
            FullWidthFieldType minFixedPart = static_cast<FullWidthFieldType>(QValueMinCalculator<NumFixedBits, NumFractionalBits, IsSigned>::MinFixedPartValue);
            return minFixedPart << NumFractionalBits;
        }

        // Get max value for fixed-point Q-format
        static FullWidthFieldType getMaxValue()
        {
            FullWidthFieldType maxFixedPart = QValueMaxCalculator<NumFixedBits, NumFractionalBits, IsSigned>::MaxFixedPartValue;
            return maxFixedPart << NumFractionalBits;
        }
    };
    
    template<unsigned NumFixedBits, unsigned NumFractionalBits, bool QValueIsSigned,
        template<typename, unsigned> class QValueRoundingPolicy = TruncatePolicy,
        template<typename> class QValueSaturatePolicy = WrapPolicy>

    struct QValue
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FullWidthValueType                     FullWidthValueType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FixedPartFieldType                     FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FractionalPartFieldType                FractionalPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FullWidthFieldType                     FullWidthFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::MultiplicationResultFullWidthValueType MultiplicationResultFullWidthValueType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::DivisionResultFullWidthValueType       DivisionResultFullWidthValueType;
        typedef QValueRoundingPolicy<MultiplicationResultFullWidthValueType, NumFractionalBits>                                RoundingPolicy;
        typedef QValueSaturatePolicy<FullWidthValueType>                                                                       SaturatePolicy;               // For addition and subtraction
        typedef QValueSaturatePolicy<MultiplicationResultFullWidthValueType>                                                   MultiplicationSaturatePolicy; // For multiplication
        typedef QValueSaturatePolicy<DivisionResultFullWidthValueType>                                                         DivisionSaturatePolicy;       // For division

        static FullWidthValueType QFormatMinValue() { return static_cast<FullWidthValueType>(QValueBounds<NumFixedBits, NumFractionalBits, QValueIsSigned>::getMinValue()); }
        static FullWidthValueType QFormatMaxValue() { return static_cast<FullWidthValueType>(QValueBounds<NumFixedBits, NumFractionalBits, QValueIsSigned>::getMaxValue()); }

        static const unsigned                NumberOfFixedBits      = NumFixedBits;
        static const unsigned                NumberOfFractionalBits = NumFractionalBits;
        static const bool                    IsSigned               = QValueIsSigned;
        static const FixedPartFieldType      MaxFixedPartValue      = QValueMaxCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MaxFixedPartValue;
        static const FractionalPartFieldType MaxFractionalPartValue = QValueMaxCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MaxFractionalPartValue;
        static const FixedPartFieldType      MinFixedPartValue      = QValueMinCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MinFixedPartValue;
        static const FractionalPartFieldType MinFractionalPartValue = QValueMinCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MinFractionalPartValue;

        QValue() : mValue(0)
        {
        }

        QValue(const QValue& other) : mValue(other.mValue)
        {
        }

        QValue(const FullWidthFieldType& value) : mValue(static_cast<FullWidthValueType>(value))
        {
        }

        QValue(const FixedPartFieldType& fixedPart, const FractionalPartFieldType& fractionalPart)
        {
            FixedPartFieldType fixedResult(fixedPart);

            SignExtender<FixedPartFieldType, NumberOfFixedBits, NumberOfFractionalBits, IsSigned>::signExtend(fixedResult);

            mFixedPart = fixedResult;
            mFractionalPart = fractionalPart;
        }

        QValue& operator=(const QValue& other)
        {
            mValue = other.mValue;

            return *this;
        }

        QValue& operator=(const FullWidthValueType& value)
        {
            mValue = value;

            return *this;
        }

        QValue& operator+=(const QValue& other)
        {
            mValue = SaturatePolicy::saturate(mValue, other.mValue, QFormatMinValue(), QFormatMaxValue(), AdditionOp);
            return *this;
        }

        QValue& operator+=(const FullWidthValueType& value)
        {
            mValue = SaturatePolicy::saturate(mValue, value, QFormatMinValue(), QFormatMaxValue(), AdditionOp);
            return *this;
        }

        QValue& operator++()
        {
            FullWidthFieldType oneUnit = (FullWidthFieldType(1) << NumberOfFractionalBits);
            mValue = SaturatePolicy::saturate(mValue, oneUnit, QFormatMinValue(), QFormatMaxValue(), AdditionOp);
            return *this;
        }

        QValue operator++(int)
        {
            QValue temp = *this; // Save copy of current value
            ++(*this); // Use pre-increment logic to increment
            return temp;
        }

        QValue& operator-=(const QValue& other)
        {
            mValue = SaturatePolicy::saturate(mValue, other.mValue, QFormatMinValue(), QFormatMaxValue(), SubtractionOp);
            return *this;
        }

        QValue& operator-=(const FullWidthValueType& value)
        {
            mValue = SaturatePolicy::saturate(mValue, value, QFormatMinValue(), QFormatMaxValue(), SubtractionOp);
            return *this;
        }

        QValue& operator--()
        {
            FullWidthFieldType oneUnit = (FullWidthFieldType(1) << NumberOfFractionalBits);
            mValue = SaturatePolicy::saturate(mValue, oneUnit, QFormatMinValue(), QFormatMaxValue(), SubtractionOp);
            return *this;
        }

        QValue operator--(int)
        {
            QValue temp = *this; // Save copy of current value
            --(*this); // Use pre-decrement logic to decrement
            return temp;
        }

        QValue& operator*=(const QValue& other)
        {
            MultiplicationResultFullWidthValueType left;
            MultiplicationResultFullWidthValueType right;
            MultiplicationResultFullWidthValueType result;

            MultiplicationResultFullWidthValueType min = static_cast<MultiplicationResultFullWidthValueType>(QFormatMinValue()) << NumberOfFractionalBits;
            MultiplicationResultFullWidthValueType max = static_cast<MultiplicationResultFullWidthValueType>(QFormatMaxValue()) << NumberOfFractionalBits;

            left = static_cast<MultiplicationResultFullWidthValueType>(mValue);
            right = static_cast<MultiplicationResultFullWidthValueType>(other.mValue);
            SignExtender<MultiplicationResultFullWidthValueType, NumberOfFixedBits, NumberOfFractionalBits, IsSigned>::signExtend(left);
            SignExtender<MultiplicationResultFullWidthValueType, NumberOfFixedBits, NumberOfFractionalBits, IsSigned>::signExtend(right);

            result = MultiplicationSaturatePolicy::saturate(left, right, min, max, MultiplicationOp);
            result = RoundingPolicy::round(result);

            mValue = static_cast<FullWidthValueType>(result);
            return *this;
        }

        QValue& operator*=(const int multiplicand)
        {
            const QValue other(static_cast<FixedPartFieldType>(multiplicand), 0);

            (*this) *= other;

            return *this;
        }

        QValue& operator/=(const QValue& other)
        {
            DivisionResultFullWidthValueType left;
            FullWidthValueType right;
            DivisionResultFullWidthValueType result;

            FullWidthValueType min = QFormatMinValue();
            FullWidthValueType max = QFormatMaxValue();

            left = static_cast<DivisionResultFullWidthValueType>(mValue);
            right = other.mValue;

            SignExtender<DivisionResultFullWidthValueType, NumberOfFixedBits, NumberOfFractionalBits, IsSigned>::signExtend(left);
            left <<= NumberOfFractionalBits;

            result = DivisionSaturatePolicy::saturate(left, right, min, max, DivisionOp);

            mValue = static_cast<FullWidthValueType>(result);
            return *this;
        }

        QValue& operator/=(const int divisor)
        {
            const QValue other(static_cast<FixedPartFieldType>(divisor), 0);

            (*this) /= other;

            return *this;
        }

        bool operator==(const QValue& other) const
        {
            return (mValue == other.mValue);
        }

        bool operator==(const FullWidthValueType& value) const
        {
            return (mValue == value);
        }

        bool operator!=(const QValue& other) const
        {
            return (mValue != other.mValue);
        }

        bool operator!=(const FullWidthValueType& value) const
        {
            return (mValue != value);
        }

        void * operator new(size_t, void *p)
        {
            return p;
        }

#if TINYMIND_ENABLE_OSTREAMS
        friend std::ostream& operator<<(std::ostream& os, const QValue& value)
        {
            os << value.getValue();
            return os;
        }
#endif // TINYMIND_ENABLE_OSTREAMS
        template<typename OtherQValueType>
        void convertFromOtherQValueType(const OtherQValueType& other)
        {
            typedef typename ShiftPolicy<
                                        typename OtherQValueType::FractionalPartFieldType,
                                        FractionalPartFieldType,
                                        OtherQValueType::NumberOfFractionalBits,
                                        NumberOfFractionalBits>::ShiftPolicyType ShiftPolicyType;
            mFixedPart = static_cast<FixedPartFieldType>(other.getFixedPart());
            mFractionalPart = ShiftPolicyType::shift(other.getFractionalPart());
        }

        FixedPartFieldType getFixedPart() const
        {
            return mFixedPart;
        }

        FractionalPartFieldType getFractionalPart() const
        {
            return mFractionalPart;
        }

        FullWidthValueType getValue() const
        {
            return static_cast<FullWidthValueType>(mValue);
        }

        void setValue(const FullWidthValueType& value)
        {
            mValue = value;
        }

        void setValue(const FixedPartFieldType& fixedPart, const FractionalPartFieldType& fractionalPart)
        {
            mFixedPart = fixedPart;
            mFractionalPart = fractionalPart;
        }

    private:
        union
        {
            struct
            {
                FractionalPartFieldType mFractionalPart : NumberOfFractionalBits;
                FixedPartFieldType mFixedPart : NumberOfFixedBits;
            };
            FullWidthValueType mValue;
        };

        static_assert(((8 * sizeof(FullWidthFieldType)) >= (NumberOfFixedBits + NumberOfFractionalBits)), "Incorrect type choice for ValueType.");
#ifdef __SIZEOF_INT128__
        static_assert((NumberOfFixedBits + NumberOfFractionalBits) <= 64, "Capped at 64 bits to support 128 bits in division operation.");
#else // __SIZEOF_INT128__
        static_assert((NumberOfFixedBits + NumberOfFractionalBits) <= 32, "Capped at 32 bits to support 64 bits in division operation.");
#endif // __SIZEOF_INT128__
    };

    template<typename QValueType>
    QValueType operator+(const QValueType& left, const QValueType& right)
    {
        QValueType result(left);

        result += right;
        return result;
    }

    template<typename QValueType>
    QValueType operator+(const QValueType& left, const int right)
    {
        QValueType result(left);
        QValueType other(right, 0);

        result += other;
        return result;
    }

    template<typename QValueType>
    QValueType operator-(const QValueType& left, const QValueType& right)
    {
        QValueType result(left);

        result -= right;
        return result;
    }

    template<typename QValueType>
    QValueType operator-(const QValueType& left, const int right)
    {
        QValueType result(left);
        QValueType other(right, 0);

        result -= other;
        return result;
    }

    template<typename QValueType>
    QValueType operator*(const QValueType& left, const QValueType& right)
    {
        QValueType result(left);

        result *= right;
        return result;
    }

    template<typename QValueType>
    QValueType operator*(const QValueType& left, const int right)
    {
        QValueType result(left);
        QValueType other(static_cast<typename QValueType::FixedPartFieldType>(right), 0);

        result *= other;
        return result;
    }

    template<typename QValueType>
    QValueType operator/(const QValueType& left, const QValueType& right)
    {
        QValueType result(left);

        result /= right;
        return result;
    }

    template<typename QValueType>
    QValueType operator/(const QValueType& left, const int right)
    {
        const QValueType other(static_cast<typename QValueType::FixedPartFieldType>(right), 0);
        QValueType result(left);

        result /= other;

        return result;
    }

    template<typename QValueType>
    bool operator>(const QValueType& left, const QValueType& right)
    {
        return (left.getValue() > right.getValue());
    }

    template<typename QValueType>
    bool operator>(const QValueType& left, const int right)
    {
        const QValueType other(static_cast<typename QValueType::FixedPartFieldType>(right), 0);

        return (left.getValue() > other.getValue());
    }

    template<typename QValueType>
    bool operator>=(const QValueType& left, const QValueType& right)
    {
        return (left.getValue() >= right.getValue());
    }

    template<typename QValueType>
    bool operator>=(const QValueType& left, const int right)
    {
        const QValueType other(static_cast<typename QValueType::FixedPartFieldType>(right), 0);

        return (left.getValue() >= other.getValue());
    }

    template<typename QValueType>
    bool operator<(const QValueType& left, const QValueType& right)
    {
        return (left.getValue() < right.getValue());
    }

    template<typename QValueType>
    bool operator<(const QValueType& left, const int right)
    {
        const QValueType other(static_cast<typename QValueType::FixedPartFieldType>(right), 0);

        return (left.getValue() < other.getValue());
    }

    template<typename QValueType>
    bool operator<=(const QValueType& left, const QValueType& right)
    {
        return (left.getValue() <= right.getValue());
    }

    template<typename QValueType>
    bool operator<=(const QValueType& left, const int right)
    {
        const QValueType other(static_cast<typename QValueType::FixedPartFieldType>(right), 0);

        return (left.getValue() <= other.getValue());
    }
}