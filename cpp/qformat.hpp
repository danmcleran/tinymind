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

#include "typeChooser.hpp"

#ifndef UINT64_MAX
#define UINT64_MAX 0xFFFFFFFFFFFFFFFFULL
#endif // UINT64_MAX

namespace tinymind {

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

    template<typename T, unsigned NumberOfFixedBits, unsigned NumberOfFractionalBits, bool IsSigned>
    struct SignExtender
    {
        static constexpr T SignBitMask = (static_cast<T>(1) << (NumberOfFixedBits + NumberOfFractionalBits - 1));
        static constexpr T SignExtensionBits = static_cast<T>((~(static_cast<size_t>(SignBitMask) - 1)) ^ SignBitMask);

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
    struct SaturateCheckResultTypeChooser
    {
        static constexpr unsigned FullWidthResult = FullWidthFieldTypeChooser<NumBits, IsSigned>::Result;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FixedPartFieldType SaturateCheckFixedPartFieldType;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FixedPartFieldType SaturateCheckFractionalPartFieldType;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FullWidthValueType SaturateCheckFullWidthValueType;
    };

    template<unsigned NumBits, bool IsSigned>
    struct MultiplicationResultTypeChooser
    {
        static constexpr unsigned FullWidthResult = FullWidthFieldTypeChooser<NumBits, IsSigned>::Result;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FullWidthFieldType MultiplicationResultFullWidthFieldType;
    };

    template<unsigned NumBits, bool IsSigned>
    struct DivisionResultTypeChooser
    {
        static constexpr unsigned FullWidthResult = FullWidthFieldTypeChooser<NumBits, IsSigned>::Result;
        typedef typename FullWidthFieldTypeChooser<(FullWidthResult << 1), IsSigned>::FullWidthValueType DivisionResultFullWidthValueType;
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits, bool IsSigned>
    struct QTypeChooser
    {
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FixedPartFieldType                           FixedPartFieldType;
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FractionalPartFieldType                      FractionalPartFieldType;
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FullWidthFieldType                           FullWidthFieldType;
        typedef typename FullWidthFieldTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::FullWidthValueType                           FullWidthValueType;
        typedef typename MultiplicationResultTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::MultiplicationResultFullWidthFieldType MultiplicationResultFullWidthFieldType;
        typedef typename DivisionResultTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::DivisionResultFullWidthValueType             DivisionResultFullWidthValueType;
        typedef typename SaturateCheckResultTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::SaturateCheckFixedPartFieldType         SaturateCheckFixedPartFieldType;
        typedef typename SaturateCheckResultTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::SaturateCheckFractionalPartFieldType    SaturateCheckFractionalPartFieldType;
        typedef typename SaturateCheckResultTypeChooser<NumFixedBits + NumFractionalBits, IsSigned>::SaturateCheckFullWidthValueType         SaturateCheckFullWidthValueType;
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
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FullWidthFieldType      FullWidthFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FullWidthValueType      FullWidthValueType;

        static constexpr FixedPartFieldType MaxFixedPartValue           = (static_cast<FixedPartFieldType>((static_cast<FractionalPartFieldType>(1) << NumFixedBits) - 1));
        static constexpr FractionalPartFieldType MaxFractionalPartValue = (static_cast<FractionalPartFieldType>((static_cast<FractionalPartFieldType>(1) << NumFractionalBits) - 1));
        static constexpr FullWidthFieldType MinFullWidthValue           = 0;
        static constexpr FullWidthFieldType MaxFullWidthField           = ((MaxFixedPartValue << NumFractionalBits) | MaxFractionalPartValue);
        static constexpr FullWidthValueType MaxFullWidthValue           = (static_cast<FullWidthValueType>(MaxFullWidthField));
    };

    template<unsigned NumFixedBits, unsigned NumFractionalBits>
    struct QValueMaxCalculator<NumFixedBits, NumFractionalBits, true>
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, true>::FixedPartFieldType      FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, true>::FractionalPartFieldType FractionalPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FullWidthFieldType     FullWidthFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, false>::FullWidthValueType     FullWidthValueType;

        static constexpr FixedPartFieldType MaxFixedPartValue           = (static_cast<FixedPartFieldType>((static_cast<FixedPartFieldType>(1) << (NumFixedBits - 1)) - 1));
        static constexpr FixedPartFieldType MinFixedPartValue           = (static_cast<FixedPartFieldType>((static_cast<FixedPartFieldType>(1) << NumFixedBits) - 1));
        static constexpr FractionalPartFieldType MaxFractionalPartValue = (static_cast<FractionalPartFieldType>((static_cast<FractionalPartFieldType>(1) << NumFractionalBits) - 1));
        static constexpr FullWidthFieldType MinFullWidthField           = ((MinFixedPartValue << NumFractionalBits) | MaxFractionalPartValue);
        static constexpr FullWidthValueType MinFullWidthValue           = (static_cast<FullWidthValueType>(MinFullWidthField));
        static constexpr FullWidthFieldType MaxFullWidthField           = ((MaxFixedPartValue << NumFractionalBits) | MaxFractionalPartValue);
        static constexpr FullWidthValueType MaxFullWidthValue           = (static_cast<FullWidthValueType>(MaxFullWidthField));
    };

    class NoSaturatePolicy
    {
    public:
        template<typename QValueType, typename OtherQValueType>
        static void convertFromOtherQValueType(QValueType& value, const OtherQValueType& otherValue)
        {
            typedef typename QValueType::FixedPartFieldType FixedPartFieldType;
            typedef typename QValueType::FractionalPartFieldType FractionalPartFieldType;
            typedef typename OtherQValueType::FixedPartFieldType OtherFixedPartFieldType;
            typedef typename ShiftPolicy<   OtherFixedPartFieldType,
                                            FractionalPartFieldType,
                                            OtherQValueType::NumberOfFractionalBits,
                                            QValueType::NumberOfFractionalBits>::ShiftPolicyType ShiftPolicyType;

            const FixedPartFieldType fixedPart = static_cast<FixedPartFieldType>(otherValue.getFixedPart());
            const FractionalPartFieldType fractionalPart = ShiftPolicyType::shift(otherValue.getFractionalPart());

            value.setValue(fixedPart, fractionalPart);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void add(QValueType& value, const QValueType& other)
        {
            typedef typename QValueType::FullWidthValueType FullWidthValueType;
            FullWidthValueType left(value.getValue());
            FullWidthValueType right(other.getValue());
            FullWidthValueType sum;

            sum = left + right;

            value.setValue(sum);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void decrement(QValueType& value)
        {
            typedef typename QValueType::FixedPartFieldType FixedPartFieldType;
            typedef typename QValueType::FractionalPartFieldType FractionalPartFieldType;
            FixedPartFieldType fixedPart(value.getFixedPart());
            const FractionalPartFieldType fractionalPart(value.getFractionalPart());

            --fixedPart;

            value.setValue(fixedPart, fractionalPart);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void divide(QValueType& value, const QValueType& other)
        {
            typedef typename QValueType::DivisionResultFullWidthValueType DivisionResultFullWidthValueType;
            typedef typename QValueType::FullWidthValueType FullWidthValueType ;
            DivisionResultFullWidthValueType left;
            FullWidthValueType right;
            DivisionResultFullWidthValueType result;

            left = static_cast<DivisionResultFullWidthValueType>(value.getValue());
            right = other.getValue();

            SignExtender<DivisionResultFullWidthValueType, QValueType::NumberOfFixedBits, QValueType::NumberOfFractionalBits, QValueType::IsSigned>::signExtend(left);

            left <<= QValueType::NumberOfFractionalBits;
            result = (left / right);

            value.setValue(result);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void increment(QValueType& value)
        {
            typedef typename QValueType::FixedPartFieldType FixedPartFieldType;
            typedef typename QValueType::FractionalPartFieldType FractionalPartFieldType;
            FixedPartFieldType fixedPart(value.getFixedPart());
            const FractionalPartFieldType fractionalPart(value.getFractionalPart());

            ++fixedPart;

            value.setValue(fixedPart, fractionalPart);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void multiply(QValueType& value, const QValueType& other)
        {
            typedef typename QValueType::MultiplicationResultFullWidthFieldType MultiplicationResultFullWidthFieldType;
            typedef typename QValueType::RoundingPolicy RoundingPolicy;
            MultiplicationResultFullWidthFieldType left = static_cast<MultiplicationResultFullWidthFieldType>(value.getValue());
            MultiplicationResultFullWidthFieldType right = static_cast<MultiplicationResultFullWidthFieldType>(other.getValue());
            MultiplicationResultFullWidthFieldType result;

            SignExtender<MultiplicationResultFullWidthFieldType, QValueType::NumberOfFixedBits, QValueType::NumberOfFractionalBits, QValueType::IsSigned>::signExtend(left);
            SignExtender<MultiplicationResultFullWidthFieldType, QValueType::NumberOfFixedBits, QValueType::NumberOfFractionalBits, QValueType::IsSigned>::signExtend(right);

            result = left * right;

            result = RoundingPolicy::round(result);

            value.setValue(result);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void subtract(QValueType& value, const QValueType& other)
        {
            typedef typename QValueType::FullWidthValueType FullWidthValueType;
            FullWidthValueType left(value.getValue());
            FullWidthValueType right(other.getValue());
            FullWidthValueType sum;

            sum = left - right;

            value.setValue(sum);
        }
    };

    template<typename QValueType, bool IsSigned>
    struct SaturateMinChecker
    {
    };

    template<typename QValueType>
    struct SaturateMinChecker<QValueType, true>
    {
        typedef typename QValueType::FullWidthValueType FullWidthValueType;
        static bool isLessThanMin(const FullWidthValueType value)
        {
            constexpr FullWidthValueType MIN_VALUE = QValueType::MinFullWidthValue;
            return (value < MIN_VALUE);
        }
    };

    template<typename QValueType>
    struct SaturateMinChecker<QValueType, false>
    {
        typedef typename QValueType::FullWidthValueType FullWidthValueType;
        static bool isLessThanMin(const FullWidthValueType value)
        {
            return false;
        }
    };

    class SaturateMinMaxPolicy
    {
    public:
        template<typename QValueType, typename OtherQValueType>
        static void convertFromOtherQValueType(QValueType& value, const OtherQValueType& otherValue)
        {
            typedef typename QValueType::FixedPartFieldType FixedPartFieldType;
            typedef typename QValueType::FractionalPartFieldType FractionalPartFieldType;
            typedef typename QValueType::FullWidthValueType FullWidthValueType;
            typedef typename OtherQValueType::FixedPartFieldType OtherFixedPartFieldType;
            typedef typename ShiftPolicy<   OtherFixedPartFieldType,
                                            FractionalPartFieldType,
                                            OtherQValueType::NumberOfFractionalBits,
                                            QValueType::NumberOfFractionalBits>::ShiftPolicyType ShiftPolicyType;
            typedef SaturateMinChecker<QValueType, QValueType::IsSigned> SaturateMinCheckerType;
            constexpr FullWidthValueType MAX_VALUE = QValueType::MaxFullWidthValue;
            constexpr FullWidthValueType MIN_VALUE = QValueType::MinFullWidthValue;
            if (otherValue.getValue() > MAX_VALUE)
            {
                value.setValue(MAX_VALUE);
            }
            else if (SaturateMinCheckerType::isLessThanMin(otherValue.getValue()))
            {
                value.setValue(MIN_VALUE);
            }
            else
            {
                FixedPartFieldType fixedPart;
                FractionalPartFieldType fractionalPart;

                fixedPart = static_cast<FixedPartFieldType>(otherValue.getFixedPart());
                fractionalPart = ShiftPolicyType::shift(otherValue.getFractionalPart());
                value.setValue(fixedPart, fractionalPart);
            }
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void add(QValueType& value, const QValueType& other)
        {
            typedef typename SaturateCheckValueType::FullWidthValueType SaturateCheckFullWidthValueType;
            SaturateCheckFullWidthValueType sum;
            SaturateCheckValueType satLeft;
            SaturateCheckValueType satRight;
            SaturateCheckValueType satCheck;

            convertFromOtherQValueType(satLeft, value);
            convertFromOtherQValueType(satRight, other);

            SaturateCheckFullWidthValueType left(satLeft.getValue());
            SaturateCheckFullWidthValueType right(satRight.getValue());

            sum = left + right;

            satCheck.setValue(sum);

            convertFromOtherQValueType(value, satCheck);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void decrement(QValueType& value)
        {
            typedef typename SaturateCheckValueType::FixedPartFieldType SaturateCheckFixedPartFieldType;
            typedef typename SaturateCheckValueType::FractionalPartFieldType SaturateCheckFractionalPartFieldType;
            SaturateCheckValueType satCheck;
            SaturateCheckFixedPartFieldType fixedPart(value.getFixedPart());
            const SaturateCheckFractionalPartFieldType fractionalPart(value.getFractionalPart());

            --fixedPart;

            satCheck.setValue(fixedPart, fractionalPart);

            convertFromOtherQValueType(value, satCheck);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void divide(QValueType& value, const QValueType& other)
        {
            typedef typename QValueType::DivisionResultFullWidthValueType DivisionResultFullWidthValueType;
            typedef typename QValueType::FullWidthValueType FullWidthValueType ;
            DivisionResultFullWidthValueType left;
            FullWidthValueType right;
            DivisionResultFullWidthValueType result;
            SaturateCheckValueType satCheck;

            left = static_cast<DivisionResultFullWidthValueType>(value.getValue());
            right = other.getValue();

            SignExtender<DivisionResultFullWidthValueType, QValueType::NumberOfFixedBits, QValueType::NumberOfFractionalBits, QValueType::IsSigned>::signExtend(left);

            left <<= QValueType::NumberOfFractionalBits;
            result = (left / right);

            satCheck.setValue(result);

            convertFromOtherQValueType(value, satCheck);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void increment(QValueType& value)
        {
            typedef typename SaturateCheckValueType::FixedPartFieldType SaturateCheckFixedPartFieldType;
            typedef typename SaturateCheckValueType::FractionalPartFieldType SaturateCheckFractionalPartFieldType;
            SaturateCheckValueType satCheck;
            SaturateCheckFixedPartFieldType fixedPart(value.getFixedPart());
            const SaturateCheckFractionalPartFieldType fractionalPart(value.getFractionalPart());

            ++fixedPart;

            satCheck.setValue(fixedPart, fractionalPart);

            convertFromOtherQValueType(value, satCheck);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void multiply(QValueType& value, const QValueType& other)
        {
            typedef typename QValueType::MultiplicationResultFullWidthFieldType MultiplicationResultFullWidthFieldType;
            typedef typename QValueType::RoundingPolicy RoundingPolicy;
            MultiplicationResultFullWidthFieldType left = static_cast<MultiplicationResultFullWidthFieldType>(value.getValue());
            MultiplicationResultFullWidthFieldType right = static_cast<MultiplicationResultFullWidthFieldType>(other.getValue());
            MultiplicationResultFullWidthFieldType result;
            SaturateCheckValueType satCheck;

            SignExtender<MultiplicationResultFullWidthFieldType, QValueType::NumberOfFixedBits, QValueType::NumberOfFractionalBits, QValueType::IsSigned>::signExtend(left);
            SignExtender<MultiplicationResultFullWidthFieldType, QValueType::NumberOfFixedBits, QValueType::NumberOfFractionalBits, QValueType::IsSigned>::signExtend(right);

            result = left * right;

            result = RoundingPolicy::round(result);

            satCheck.setValue(result);

            convertFromOtherQValueType(value, satCheck);
        }

        template<typename SaturateCheckValueType, typename QValueType>
        static void subtract(QValueType& value, const QValueType& other)
        {
            typedef typename SaturateCheckValueType::FullWidthValueType SaturateCheckFullWidthValueType;
            SaturateCheckFullWidthValueType sum;
            SaturateCheckValueType satLeft;
            SaturateCheckValueType satRight;
            SaturateCheckValueType satCheck;

            convertFromOtherQValueType(satLeft, value);
            convertFromOtherQValueType(satRight, other);

            SaturateCheckFullWidthValueType left(satLeft.getValue());
            SaturateCheckFullWidthValueType right(satRight.getValue());

            sum = left - right;

            satCheck.setValue(sum);

            convertFromOtherQValueType(value, satCheck);
        }
    };

    template<   unsigned NumFixedBits,
                unsigned NumFractionalBits,
                bool QValueIsSigned,
                template<typename, unsigned> class QValueRoundingPolicy = TruncatePolicy,
                class QValueSaturatePolicy = NoSaturatePolicy>
    struct QValue
    {
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FullWidthValueType                     FullWidthValueType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FixedPartFieldType                     FixedPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FractionalPartFieldType                FractionalPartFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::FullWidthFieldType                     FullWidthFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::MultiplicationResultFullWidthFieldType MultiplicationResultFullWidthFieldType;
        typedef typename QTypeChooser<NumFixedBits, NumFractionalBits, QValueIsSigned>::DivisionResultFullWidthValueType       DivisionResultFullWidthValueType;
        typedef QValueRoundingPolicy<MultiplicationResultFullWidthFieldType, NumFractionalBits>                                RoundingPolicy;
        typedef QValueSaturatePolicy                                                                                           SaturatePolicy;
        typedef QValue<NumFixedBits, NumFractionalBits, QValueIsSigned, QValueRoundingPolicy, QValueSaturatePolicy>            QValueType;
        typedef QValue<(NumFixedBits << 1), (NumFractionalBits << 1), QValueIsSigned, QValueRoundingPolicy, QValueSaturatePolicy> SaturateCheckQValueType;

        static constexpr unsigned                NumberOfFixedBits      = NumFixedBits;
        static constexpr unsigned                NumberOfFractionalBits = NumFractionalBits;
        static constexpr bool                    IsSigned               = QValueIsSigned;
        static constexpr FixedPartFieldType      MaxFixedPartValue      = QValueMaxCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MaxFixedPartValue;
        static constexpr FractionalPartFieldType MaxFractionalPartValue = QValueMaxCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MaxFractionalPartValue;
        static constexpr FullWidthValueType      MaxFullWidthValue      = QValueMaxCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MaxFullWidthValue;
        static constexpr FullWidthValueType      MinFullWidthValue      = QValueMaxCalculator<NumFixedBits, NumFractionalBits, QValueIsSigned>::MinFullWidthValue;

        QValue() : mValue(0)
        {
        }

        QValue(const QValue& other) : mValue(other.mValue)
        {
        }

        QValue(const FullWidthFieldType& value) : mValue(value)
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

        QValue& operator=(const FullWidthFieldType& value)
        {
            mValue = value;

            return *this;
        }

        QValue& operator+=(const QValue& other)
        {
            QValueSaturatePolicy::template add<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator+=(const FullWidthFieldType& value)
        {
            const QValue other(value);

            QValueSaturatePolicy::template add<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator++()
        {
            QValueSaturatePolicy::template increment<SaturateCheckQValueType>(*this);

            return *this;
        }

        QValue operator++(int)
        {
            QValueSaturatePolicy::template increment<SaturateCheckQValueType>(*this);

            return *this;
        }

        QValue& operator-=(const QValue& other)
        {
            QValueSaturatePolicy::template subtract<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator-=(const FullWidthFieldType& value)
        {
            const QValue other(value);

            QValueSaturatePolicy::template subtract<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator--()
        {
            QValueSaturatePolicy::template decrement<SaturateCheckQValueType>(*this);

            return *this;
        }

        QValue operator--(int)
        {
            QValueSaturatePolicy::template decrement<SaturateCheckQValueType>(*this);

            return *this;
        }

        QValue& operator*=(const QValue& other)
        {
            QValueSaturatePolicy::template multiply<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator*=(const int multiplicand)
        {
            const QValue other(static_cast<FixedPartFieldType>(multiplicand), 0);

            QValueSaturatePolicy::template multiply<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator/=(const QValue& other)
        {
            QValueSaturatePolicy::template divide<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        QValue& operator/=(const int divisor)
        {
            const QValue other(static_cast<FixedPartFieldType>(divisor), 0);

            QValueSaturatePolicy::template divide<SaturateCheckQValueType>(*this, other);

            return *this;
        }

        bool operator==(const QValue& other) const
        {
            return (mValue == other.mValue);
        }

        bool operator==(const FullWidthValueType& value) const
        {
            return (mValue.getValue() == value);
        }

        bool operator!=(const QValue& other) const
        {
            return (mValue != other.mValue);
        }

        bool operator!=(const FullWidthValueType& value) const
        {
            return (mValue.getValue() != value);
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
            QValueSaturatePolicy::convertFromOtherQValueType(*this, other);
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
            mValue = static_cast<FullWidthFieldType>(value);
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
            FullWidthFieldType mValue;
        };

        static_assert(((8 * sizeof(FullWidthFieldType)) >= (NumberOfFixedBits + NumberOfFractionalBits)), "Incorrect type choice for ValueType.");
#ifdef __SIZEOF_INT128__
        static_assert((NumberOfFixedBits + NumberOfFractionalBits) <= 128, "Capped at 64 bits to support 128 bits in division operation.");
#else // __SIZEOF_INT128__
        static_assert((NumberOfFixedBits + NumberOfFractionalBits) <= 64, "Capped at 32 bits to support 64 bits in division operation.");
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