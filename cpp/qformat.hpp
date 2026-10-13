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

#include "include/tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

#if TINYMIND_ENABLE_OSTREAMS
#include <ostream>
#endif // TINYMIND_ENABLE_OSTREAMS

#include "typeChooser.hpp"

#ifndef UINT64_MAX
#define UINT64_MAX 0xFFFFFFFFFFFFFFFFULL
#endif // UINT64_MAX

namespace tinymind {

    // The rounding/saturation policies below are bounded-model-checked for
    // overflow/UB freedom and range-correctness by formal/cbmc_qformat.cpp
    // (CBMC cannot type-check the templated form, so that harness proves a
    // concrete transcription of this exact arithmetic).
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

            //Round up. The half-LSB mask is built in T: as an int it was
            // undefined for NumberOfFractionalBits > 32 and, at exactly 32, the
            // sign-extended INT_MIN mask picked up every bit above bit 31.
            result = result + ((result & (static_cast<T>(1) << (NumberOfFractionalBits - 1))) << 1);
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

    // The unsigned type of the same width as T. Wrap-around addition and
    // subtraction are done in it because signed overflow is undefined while
    // unsigned arithmetic is defined modulo 2^N.
    template<typename T>
    struct UnsignedCounterpart
    {
        typedef T Type;
    };

    template<>
    struct UnsignedCounterpart<int8_t>
    {
        typedef uint8_t Type;
    };

    template<>
    struct UnsignedCounterpart<int16_t>
    {
        typedef uint16_t Type;
    };

    template<>
    struct UnsignedCounterpart<int32_t>
    {
        typedef uint32_t Type;
    };

    template<>
    struct UnsignedCounterpart<int64_t>
    {
        typedef uint64_t Type;
    };

#ifdef __SIZEOF_INT128__
    template<>
    struct UnsignedCounterpart<__int128_t>
    {
        typedef __uint128_t Type;
    };
#endif // __SIZEOF_INT128__

    template<typename T>
    class WrapPolicy // No saturation - just return the value (wrap-around behavior)
    {
    public:
        static T saturate(const T& origValue, const T& target, const T& minValue, const T& maxValue, const OperatorType_e opType)
        {
            (void)minValue; // Suppress unused parameter warnings
            (void)maxValue;

            typedef typename UnsignedCounterpart<T>::Type UnsignedType;

            T result{};

            switch (opType)
            {
                case AdditionOp:
                    // Wrap in the unsigned counterpart: for a word at least as
                    // wide as int, origValue + target is signed overflow (UB).
                    result = static_cast<T>(static_cast<UnsignedType>(origValue) + static_cast<UnsignedType>(target));
                    break;
                case SubtractionOp:
                    result = static_cast<T>(static_cast<UnsignedType>(origValue) - static_cast<UnsignedType>(target));
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
            T result{};
            switch (opType)
            {
                case AdditionOp:
                    if (target > 0)
                    {
                        if (origValue > (maxValue - target))
                        {
                            result = maxValue; // Saturate to maximum
                        }
                        else
                        {
                            result = origValue + target;  // Safe addition
                        }
                    }
                    else if (target < 0)
                    {
                        if (origValue < (minValue - target))
                        {
                            result = minValue; // Saturate to minimum
                        }
                        else
                        {
                            result = origValue + target;  // Safe addition
                        }
                    }
                    else
                    {
                        result = origValue; // Adding zero: no overflow possible
                    }
                    break;
                case SubtractionOp:
                    if (target > 0)
                    {
                        if (origValue < (minValue + target))
                        {
                            result = minValue;  // Saturate to minimum
                        }
                        else
                        {
                            result = origValue - target; // Safe subtraction
                        }
                    }
                    else if (target < 0)
                    {
                        if (origValue > (maxValue + target))
                        {
                            result = maxValue; // Saturate to maximum
                        }
                        else
                        {
                            result = origValue - target; // Safe subtraction
                        }
                    }
                    else
                    {
                        result = origValue; // Subtracting zero: no overflow possible
                    }
                    break;
                case MultiplicationOp:
                    // Check for multiplication overflow
                    if ((origValue * target) > maxValue)
                    {
                        result = maxValue;
                    // Check for multiplication underflow as well    
                    }
                    else if ((origValue * target) < minValue)
                    {
                        result = minValue;
                    }
                    else
                    {
                        result = origValue * target; // Safe multiplication
                    }
                    break;
                case DivisionOp:
                    if (target == 0) { // Handle division by zero first
                        result = (origValue > 0) ? maxValue : minValue;
                    }
                    else if ((origValue / target) > maxValue)
                    {
                        result = maxValue;
                    }
                    else if ((origValue / target) < minValue)
                    {
                        result = minValue;
                    }
                    else
                    {
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
        // Every bit above the sign bit. Computed in T: going through size_t
        // truncated the mask to 32 bits on targets with a 32-bit size_t, so a
        // 64-bit intermediate was only sign-extended up to bit 31.
        static const T SignExtensionBits = static_cast<T>(~(SignBitMask | (SignBitMask - 1)));

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

    // Reduce a storage word to the value it represents at NumBits wide:
    // sign-extended from bit NumBits-1 for signed formats, masked for unsigned.
    // A format narrower than its word (e.g. Q8.16 in an int32_t) must keep the
    // bits above its width consistent with its sign, or a negative value reads
    // back as a large positive one. ValueType is the storage type, FieldType
    // its unsigned counterpart. A format that fills its word is already
    // canonical, so that case is a no-op.
    template<typename ValueType, typename FieldType, unsigned NumBits, bool IsSigned, bool FillsWord = (NumBits == (8 * sizeof(ValueType)))>
    struct FormatWidthNormalizer
    {
        static ValueType normalize(const ValueType value)
        {
            const FieldType mask = static_cast<FieldType>((static_cast<FieldType>(1) << NumBits) - 1);
            const FieldType signBit = static_cast<FieldType>(static_cast<FieldType>(1) << (NumBits - 1));
            const FieldType low = static_cast<FieldType>(static_cast<FieldType>(value) & mask);

            // (low ^ signBit) lies in [0, 2^NumBits), which ValueType holds
            // because NumBits is narrower than the word; subtracting signBit
            // then yields the signed value with no shift of a negative number.
            return static_cast<ValueType>(static_cast<ValueType>(low ^ signBit) - static_cast<ValueType>(signBit));
        }
    };

    template<typename ValueType, typename FieldType, unsigned NumBits>
    struct FormatWidthNormalizer<ValueType, FieldType, NumBits, false, false>
    {
        static ValueType normalize(const ValueType value)
        {
            const FieldType mask = static_cast<FieldType>((static_cast<FieldType>(1) << NumBits) - 1);

            return static_cast<ValueType>(static_cast<FieldType>(value) & mask);
        }
    };

    template<typename ValueType, typename FieldType, unsigned NumBits, bool IsSigned>
    struct FormatWidthNormalizer<ValueType, FieldType, NumBits, IsSigned, true>
    {
        static ValueType normalize(const ValueType value)
        {
            return value;
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
    public:
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

    private:
        typedef FormatWidthNormalizer<FullWidthValueType, FullWidthFieldType, NumberOfFixedBits + NumberOfFractionalBits, IsSigned> ValueNormalizer;
        typedef FormatWidthNormalizer<FullWidthValueType, FullWidthFieldType, NumberOfFixedBits, IsSigned>                          FixedPartNormalizer;

        // The whole value, always kept canonical for the format's width (see
        // FormatWidthNormalizer). The fixed and fractional parts are derived
        // from it arithmetically rather than through a bit-field union, so a
        // part-wise write can never leave the bits above the format's width
        // stale, and nothing reads an inactive union member.
        FullWidthValueType mValue;

        static FullWidthValueType normalize(const FullWidthValueType value)
        {
            return ValueNormalizer::normalize(value);
        }

        static FullWidthFieldType fractionalPartMask()
        {
            return static_cast<FullWidthFieldType>((static_cast<FullWidthFieldType>(1) << NumberOfFractionalBits) - 1);
        }

        // fixedPart * 2^frac + fractionalPart. Each part is first reduced to
        // its own width (the truncation the old bit-field store performed),
        // then the two are joined in the unsigned word: 2^frac itself is not
        // representable in the signed word when the format has one fixed bit
        // and fills it (Q1.15, Q1.31), so a signed multiply there overflows.
        static FullWidthValueType fromParts(const FixedPartFieldType& fixedPart, const FractionalPartFieldType& fractionalPart)
        {
            const FullWidthFieldType fixedBits = static_cast<FullWidthFieldType>(FixedPartNormalizer::normalize(static_cast<FullWidthValueType>(fixedPart)));
            const FullWidthFieldType fractionalBits = static_cast<FullWidthFieldType>(static_cast<FullWidthFieldType>(fractionalPart) & fractionalPartMask());
            const FullWidthFieldType word = static_cast<FullWidthFieldType>(static_cast<FullWidthFieldType>(fixedBits << NumberOfFractionalBits) | fractionalBits);

            return normalize(static_cast<FullWidthValueType>(word));
        }

    public:
        QValue() : mValue(0)
        {
        }

        QValue(const QValue& other)
        {
            this->mValue = other.mValue;
        }

        QValue(const FullWidthFieldType& value)
        {
            this->mValue = normalize(static_cast<FullWidthValueType>(value));
        }

        QValue(const FixedPartFieldType& fixedPart, const FractionalPartFieldType& fractionalPart)
        {
            this->mValue = fromParts(fixedPart, fractionalPart);
        }

        QValue& operator=(const QValue& other)
        {
            mValue = other.mValue;

            return *this;
        }

        QValue& operator=(const FullWidthValueType& value)
        {
            mValue = normalize(value);

            return *this;
        }

        QValue& operator+=(const QValue& other)
        {
            mValue = normalize(SaturatePolicy::saturate(mValue, other.mValue, QFormatMinValue(), QFormatMaxValue(), AdditionOp));
            return *this;
        }

        QValue& operator+=(const FullWidthValueType& value)
        {
            mValue = normalize(SaturatePolicy::saturate(mValue, value, QFormatMinValue(), QFormatMaxValue(), AdditionOp));
            return *this;
        }

        QValue& operator++()
        {
            FullWidthFieldType oneUnit = (FullWidthFieldType(1) << NumberOfFractionalBits);
            mValue = normalize(SaturatePolicy::saturate(mValue, oneUnit, QFormatMinValue(), QFormatMaxValue(), AdditionOp));
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
            mValue = normalize(SaturatePolicy::saturate(mValue, other.mValue, QFormatMinValue(), QFormatMaxValue(), SubtractionOp));
            return *this;
        }

        QValue& operator-=(const FullWidthValueType& value)
        {
            mValue = normalize(SaturatePolicy::saturate(mValue, value, QFormatMinValue(), QFormatMaxValue(), SubtractionOp));
            return *this;
        }

        QValue& operator--()
        {
            FullWidthFieldType oneUnit = (FullWidthFieldType(1) << NumberOfFractionalBits);
            mValue = normalize(SaturatePolicy::saturate(mValue, oneUnit, QFormatMinValue(), QFormatMaxValue(), SubtractionOp));
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

            // Scale the saturation bounds up by 2^frac. Expressed as a multiply
            // rather than a left shift: QFormatMinValue() is negative for signed
            // formats, and left-shifting a negative value is undefined behavior
            // before C++20. The wide result type holds the product without
            // overflow, so the multiply is well-defined and bit-identical.
            const MultiplicationResultFullWidthValueType scale = static_cast<MultiplicationResultFullWidthValueType>(1) << NumberOfFractionalBits;
            MultiplicationResultFullWidthValueType min = static_cast<MultiplicationResultFullWidthValueType>(QFormatMinValue()) * scale;
            MultiplicationResultFullWidthValueType max = static_cast<MultiplicationResultFullWidthValueType>(QFormatMaxValue()) * scale;

            left = static_cast<MultiplicationResultFullWidthValueType>(mValue);
            right = static_cast<MultiplicationResultFullWidthValueType>(other.mValue);
            SignExtender<MultiplicationResultFullWidthValueType, NumberOfFixedBits, NumberOfFractionalBits, IsSigned>::signExtend(left);
            SignExtender<MultiplicationResultFullWidthValueType, NumberOfFixedBits, NumberOfFractionalBits, IsSigned>::signExtend(right);

            result = MultiplicationSaturatePolicy::saturate(left, right, min, max, MultiplicationOp);
            result = RoundingPolicy::round(result);

            mValue = normalize(static_cast<FullWidthValueType>(result));
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
            // Scale the dividend up by 2^frac. Multiply, not left shift: `left`
            // is sign-extended and may be negative, and left-shifting a negative
            // value is undefined behavior before C++20. The division-width type
            // holds the product without overflow, so this is bit-identical.
            left *= (static_cast<DivisionResultFullWidthValueType>(1) << NumberOfFractionalBits);

            result = DivisionSaturatePolicy::saturate(left, right, min, max, DivisionOp);

            mValue = normalize(static_cast<FullWidthValueType>(result));
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
            mValue = fromParts(static_cast<FixedPartFieldType>(other.getFixedPart()), ShiftPolicyType::shift(other.getFractionalPart()));
        }

        // Floor of the value: the word shifted right by frac (as unsigned, so
        // no shift of a negative number), sign-extended at the fixed width.
        FixedPartFieldType getFixedPart() const
        {
            const FullWidthValueType shifted = static_cast<FullWidthValueType>(static_cast<FullWidthFieldType>(mValue) >> NumberOfFractionalBits);

            return static_cast<FixedPartFieldType>(FixedPartNormalizer::normalize(shifted));
        }

        FractionalPartFieldType getFractionalPart() const
        {
            return static_cast<FractionalPartFieldType>(static_cast<FullWidthFieldType>(mValue) & fractionalPartMask());
        }

        FullWidthValueType getValue() const
        {
            return static_cast<FullWidthValueType>(mValue);
        }

        void setValue(const FullWidthValueType& value)
        {
            mValue = normalize(value);
        }

        void setValue(const FixedPartFieldType& fixedPart, const FractionalPartFieldType& fractionalPart)
        {
            mValue = fromParts(fixedPart, fractionalPart);
        }

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