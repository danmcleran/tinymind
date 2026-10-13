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

// qformat_unit_test.cpp : Defines the entry point for the Qformat template unit tests.
//
#include <iostream>
#include <cstdint>
#include <limits>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "qformat.hpp"
#include "compiler.h"

typedef tinymind::QValue<1, 7, false, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> UnsignedQ1_7SatPolicyType;
typedef tinymind::QValue<8, 8, false, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> UnsignedQ8_8SatPolicyType;
typedef tinymind::QValue<16, 16, false, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> UnsignedQ16_16SatPolicyType;
typedef tinymind::QValue<8, 8, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ8_8SatPolicyType;
typedef tinymind::QValue<24, 8, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ24_8SatPolicyType;
typedef tinymind::QValue<7, 9, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ7_9SatPolicyType;
#ifdef __SIZEOF_INT128__
typedef tinymind::QValue<32, 32, false, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> UnsignedQ32_32SatPolicyType;
typedef tinymind::QValue<32, 32, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ32_32SatPolicyType;
typedef tinymind::QValue<24, 40, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ24_40SatPolicyType;
#endif // __SIZEOF_INT128__

typedef tinymind::QValue<8, 8, false, tinymind::RoundUpPolicy> UnsignedQ8_8Type;
typedef tinymind::QValue<8, 8, true> SignedQ8_8Type;
typedef tinymind::QValue<7, 9, true> SignedQ7_9Type;
typedef tinymind::QValue<1, 15, true> SignedQ1_15Type;
typedef tinymind::QValue<24, 8, true> SignedQ24_8Type;
typedef tinymind::QValue<16, 16, true> SignedQ16_16Type;
typedef tinymind::QValue<24, 8, false> UnsignedQ24_8Type;
typedef tinymind::QValue<8, 24, true> SignedQ8_24Type;
typedef tinymind::QValue<8, 24, false> UnsignedQ8_24Type;
typedef tinymind::QValue<24, 8, false> UnsignedQ24_8Type;
typedef tinymind::QValue<1, 7, false> UnsignedQ1_7Type;
typedef tinymind::QValue<2, 6, true> SignedQ2_6Type;
#ifdef __SIZEOF_INT128__
typedef tinymind::QValue<32, 32, false> UnsignedQ32_32Type;
typedef tinymind::QValue<32, 32, true> SignedQ32_32Type;
typedef tinymind::QValue<24, 40, true> SignedQ24_40Type;
#endif // __SIZEOF_INT128__

typedef tinymind::QValue<8, 8, false> UnsignedTruncatingQType;

static_assert((std::numeric_limits<uint8_t>::max() == UnsignedQ8_8Type::MaxFixedPartValue), "Incorrect max fixed value.");
static_assert((std::numeric_limits<uint8_t>::max() == UnsignedQ8_8Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((std::numeric_limits<uint8_t>::min() == UnsignedQ8_8Type::MinFixedPartValue), "Incorrect min fixed value.");
static_assert((std::numeric_limits<uint8_t>::min() == UnsignedQ8_8Type::MinFractionalPartValue), "Incorrect min fractional value.");
static_assert((std::numeric_limits<int8_t>::min() == SignedQ8_8Type::MinFixedPartValue), "Incorrect min fixed value.");
static_assert((0 == SignedQ8_8Type::MinFractionalPartValue), "Incorrect min fractional value.");
static_assert((((1ULL << 24) - 1) == UnsignedQ24_8Type::MaxFixedPartValue), "Incorrect max value.");
static_assert((std::numeric_limits<uint8_t>::max() == UnsignedQ24_8Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((std::numeric_limits<int8_t>::max() == SignedQ8_8Type::MaxFixedPartValue), "Incorrect fixed max value.");
static_assert((std::numeric_limits<uint8_t>::max() == SignedQ8_8Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((((1ULL << 23) - 1) == SignedQ24_8Type::MaxFixedPartValue), "Incorrect max fixed value.");
static_assert((std::numeric_limits<uint8_t>::max() == SignedQ24_8Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert(((std::numeric_limits<int8_t>::max() >> 1) == SignedQ7_9Type::MaxFixedPartValue), "Incorrect max fixed value.");
static_assert((((1ULL << 9) - 1) == SignedQ7_9Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((1U == UnsignedQ1_7Type::MaxFixedPartValue), "Incorrect max fixed value.");
static_assert((((1ULL << 7) - 1) == UnsignedQ1_7Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((1U == SignedQ2_6Type::MaxFixedPartValue), "Incorrect max fixed value.");
static_assert((((1ULL << 6) - 1) == SignedQ2_6Type::MaxFractionalPartValue), "Incorrect max fractional value.");
#ifdef __SIZEOF_INT128__
static_assert((std::numeric_limits<uint32_t>::max() == UnsignedQ32_32Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((std::numeric_limits<uint32_t>::max() == SignedQ32_32Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((((1ULL << 40) - 1) == SignedQ24_40Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((std::numeric_limits<uint32_t>::max() == UnsignedQ32_32Type::MaxFractionalPartValue), "Incorrect max fractional value.");
static_assert((std::numeric_limits<int32_t>::max() == SignedQ32_32Type::MaxFixedPartValue), "Incorrect max fixed value.");
static_assert((((1ULL << 23) - 1) == SignedQ24_40Type::MaxFixedPartValue), "Incorrect max fixed value.");
#endif // __SIZEOF_INT128__

static_assert((std::numeric_limits<uint16_t>::max() == std::numeric_limits<typename UnsignedQ8_8Type::FixedPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint16_t>::max() == std::numeric_limits<typename UnsignedQ8_8Type::FractionalPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<int16_t>::max() == std::numeric_limits<typename SignedQ8_8Type::FixedPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint16_t>::max() == std::numeric_limits<typename SignedQ8_8Type::FractionalPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<int16_t>::max() == std::numeric_limits<typename SignedQ7_9Type::FixedPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint16_t>::max() == std::numeric_limits<typename SignedQ7_9Type::FractionalPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<int16_t>::max() == std::numeric_limits<typename SignedQ1_15Type::FixedPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint16_t>::max() == std::numeric_limits<typename SignedQ1_15Type::FractionalPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint32_t>::max() == std::numeric_limits<typename UnsignedQ24_8Type::FixedPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint32_t>::max() == std::numeric_limits<typename UnsignedQ24_8Type::FractionalPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<int32_t>::max() == std::numeric_limits<typename SignedQ24_8Type::FixedPartFieldType>::max()), "Invalid type.");
static_assert((std::numeric_limits<uint32_t>::max() == std::numeric_limits<typename SignedQ24_8Type::FractionalPartFieldType>::max()), "Invalid type.");

// A raw value of the same QValue type as `like`. An int operand to += and -=
// is deleted (it was raw LSBs there but a whole number for q + n), so the
// raw-LSB arithmetic these tests exercise says so explicitly.
template<typename QValueType>
QValueType rawLike(const QValueType&, const long long raw)
{
    return QValueType(static_cast<typename QValueType::FullWidthFieldType>(raw));
}

// Compile-time detectors for the deleted integer overloads.
template<typename Q, typename T, typename = void>
struct CanAddAssign : std::false_type
{
};

template<typename Q, typename T>
struct CanAddAssign<Q, T, decltype(void(std::declval<Q&>() += std::declval<T>()))> : std::true_type
{
};

template<typename Q, typename T, typename = void>
struct CanSubtractAssign : std::false_type
{
};

template<typename Q, typename T>
struct CanSubtractAssign<Q, T, decltype(void(std::declval<Q&>() -= std::declval<T>()))> : std::true_type
{
};

template<typename Q, typename T, typename = void>
struct CanCompareEqual : std::false_type
{
};

template<typename Q, typename T>
struct CanCompareEqual<Q, T, decltype(void(std::declval<const Q&>() == std::declval<T>()))> : std::true_type
{
};

template<typename Q, typename T, typename = void>
struct CanCompareNotEqual : std::false_type
{
};

template<typename Q, typename T>
struct CanCompareNotEqual<Q, T, decltype(void(std::declval<const Q&>() != std::declval<T>()))> : std::true_type
{
};

#define BOOST_TEST_MODULE test module name
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING_GCC_ONLY("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

BOOST_AUTO_TEST_SUITE(test_suite_qformat)

BOOST_AUTO_TEST_CASE(test_case_construction)
{
    SignedQ8_8Type Q0(0, 0);
    SignedQ8_8Type Q1(0, 0);
    SignedQ8_8Type Q2(-1, 0);
    SignedQ8_8Type Q3(-1, 1);
    SignedQ1_15Type Q4(-1, 0);
    SignedQ1_15Type Q5(0, 1);
    SignedQ7_9Type Q6(-1, 0);
    SignedQ7_9Type Q7(1, 0);
    SignedQ24_8Type Q8(-1, 0);
    UnsignedQ8_8Type uQ0;
    UnsignedQ8_8Type uQ1(0, 0);
    UnsignedQ8_8Type uQ2(1, 0);
    SignedQ8_24Type Q9(-1, 0);
    SignedQ8_24Type Q10(1, 0);
    UnsignedQ1_7Type uQ4(1, 0);
    SignedQ2_6Type Q14(-1, 0);
#ifdef __SIZEOF_INT128__
    UnsignedQ32_32Type uQ3(0, 0);
    SignedQ32_32Type Q11(-1, 0);
    SignedQ24_40Type Q12(-1, 0);

    BOOST_TEST(static_cast<UnsignedQ32_32Type::FixedPartFieldType>(0) == uQ3.getFixedPart());
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FractionalPartFieldType>(0) == uQ3.getFractionalPart());
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0) == uQ3.getValue());

    BOOST_TEST(static_cast<SignedQ32_32Type::FixedPartFieldType>(-1) == Q11.getFixedPart());
    BOOST_TEST(static_cast<SignedQ32_32Type::FractionalPartFieldType>(0) == Q11.getFractionalPart());
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0xFFFFFFFF00000000ULL) == Q11.getValue());

    BOOST_TEST(static_cast<SignedQ24_40Type::FixedPartFieldType>(-1) == Q12.getFixedPart());
    BOOST_TEST(static_cast<SignedQ24_40Type::FractionalPartFieldType>(0) == Q12.getFractionalPart());
    BOOST_TEST(static_cast<SignedQ24_40Type::FullWidthValueType>(0xFFFFFF0000000000ULL) == Q12.getValue());
#endif // __SIZEOF_INT128__

    BOOST_TEST(static_cast<SignedQ8_8Type::FixedPartFieldType>(0) == Q0.getFixedPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FractionalPartFieldType>(0) == Q0.getFractionalPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0) == Q0.getValue());

    BOOST_TEST(static_cast<SignedQ8_8Type::FixedPartFieldType>(0) == Q1.getFixedPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FractionalPartFieldType>(0) == Q1.getFractionalPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0) == Q1.getValue());

    BOOST_TEST(static_cast<SignedQ8_8Type::FixedPartFieldType>(-1) == Q2.getFixedPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FractionalPartFieldType>(0) == Q2.getFractionalPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xFF00) == Q2.getValue());

    BOOST_TEST(static_cast<SignedQ8_8Type::FixedPartFieldType>(-1) == Q3.getFixedPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FractionalPartFieldType>(1) == Q3.getFractionalPart());
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xFF01) == Q3.getValue());

    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0x8000) == Q4.getValue());
    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0x0001) == Q5.getValue());

    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0xfe00) == Q6.getValue());
    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0x0200) == Q7.getValue());

    BOOST_TEST(static_cast<UnsignedQ8_8Type::FixedPartFieldType>(0) == uQ0.getFixedPart());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FractionalPartFieldType>(0) == uQ0.getFractionalPart());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ0.getValue());

    BOOST_TEST(static_cast<UnsignedQ8_8Type::FixedPartFieldType>(0) == uQ1.getFixedPart());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FractionalPartFieldType>(0) == uQ1.getFractionalPart());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ1.getValue());

    BOOST_TEST(static_cast<UnsignedQ8_8Type::FixedPartFieldType>(1) == uQ2.getFixedPart());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FractionalPartFieldType>(0) == uQ2.getFractionalPart());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x100) == uQ2.getValue());

    BOOST_TEST(static_cast<SignedQ24_8Type::FullWidthValueType>(0xFFFFFF00) == Q8.getValue());

    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFF000000) == Q9.getValue());
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x1000000) == Q10.getValue());

    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x80) == uQ4.getValue());
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xC0) == Q14.getValue());

    Q0.setValue(1, 0);
    BOOST_TEST(static_cast<SignedQ8_8Type ::FullWidthValueType>(0x100) == Q0.getValue());

    Q0.setValue(-1, 0);
    BOOST_TEST(static_cast<SignedQ8_8Type ::FullWidthValueType>(0xFF00) == Q0.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_addition)
{
    UnsignedQ8_8Type uQ0(0, 0);
    UnsignedQ8_8Type uQ1(0, 1);
    UnsignedQ8_8Type uQ2(1, 1);
    UnsignedQ8_8Type uQ3;
    UnsignedQ8_8Type uQ4(-1, 0);
    SignedQ8_8Type Q0;
    SignedQ8_8Type Q1(-1, 0);
    SignedQ8_8Type Q2(1, 0);
    SignedQ8_8Type Q3;
    SignedQ1_15Type Q4(-1, 0);
    SignedQ1_15Type Q5(0, 1);
    SignedQ24_8Type Q6(-1, 0);
    SignedQ24_8Type Q7(1, 0);
    SignedQ24_8Type Q8;
    SignedQ24_8Type Q9(0,128);
    SignedQ8_24Type Q10(-1, 0);
    SignedQ8_24Type Q11(1, 0);
    SignedQ8_24Type Q12(0x800000);
    SignedQ8_24Type Q13;
    UnsignedQ1_7Type uQ8(0, 0);
    UnsignedQ1_7Type uQ9(0, 64);
    SignedQ2_6Type Q22(1, 0);
    SignedQ2_6Type Q23(-1, 0);
#ifdef __SIZEOF_INT128__
    UnsignedQ32_32Type uQ5(0, 0);
    UnsignedQ32_32Type uQ6(0, 1);
    UnsignedQ32_32Type uQ7(1, 1);
    SignedQ32_32Type Q14(0, 0);
    SignedQ32_32Type Q15(-1, 0);
    SignedQ32_32Type Q16(1, 0);
    SignedQ24_40Type Q17(0, 0);
    SignedQ24_40Type Q18(-1, 0);
    SignedQ24_40Type Q19(1, 0);

    uQ5 += rawLike(uQ5, 0);
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0) == uQ5.getValue());

    uQ5 += uQ5;
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0) == uQ5.getValue());

    uQ5 += uQ6;
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(1) == uQ5.getValue());

    uQ5 += rawLike(uQ5, 1);
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(2) == uQ5.getValue());

    uQ5 += uQ7;
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0x100000003ULL) == uQ5.getValue());

    Q14 += rawLike(Q14, 0);
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0) == Q14.getValue());

    Q14 += Q14;
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0) == Q14.getValue());

    Q14 += Q15;
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0xFFFFFFFF00000000) == Q14.getValue());

    Q14 += Q15;
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0xFFFFFFFE00000000) == Q14.getValue());

    Q14 += Q16;
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0xFFFFFFFF00000000) == Q14.getValue());

    Q14 = Q15 + Q16;
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0) == Q14.getValue());

    Q17 += rawLike(Q17, 0);
    BOOST_TEST(static_cast<SignedQ24_40Type::FullWidthValueType>(0) == Q17.getValue());

    Q17 += Q17;
    BOOST_TEST(static_cast<SignedQ24_40Type::FullWidthValueType>(0) == Q17.getValue());

    Q17 += Q18;
    BOOST_TEST(Q17.getValue() == Q18.getValue());

    Q17 += Q19;
    BOOST_TEST(static_cast<SignedQ24_40Type::FullWidthValueType>(0) == Q17.getValue());

    Q17 += Q19;
    BOOST_TEST(Q17.getValue() == Q19.getValue());
#endif // __SIZEOF_INT128__

    uQ0 += rawLike(uQ0, 0);
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ0.getValue());

    uQ0 += uQ0;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ0.getValue());

    uQ0 += uQ1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(1) == uQ0.getValue());

    uQ0 += rawLike(uQ0, 1);
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(2) == uQ0.getValue());

    uQ0 += uQ2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x103) == uQ0.getValue());

    uQ3 = uQ1 + uQ2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x102) == uQ3.getValue());

    Q0 += rawLike(Q0, 0);
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0) == Q0.getValue());

    Q0 += Q0;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0) == Q0.getValue());

    Q0 += Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(-256) == Q0.getValue());

    Q0 += Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(-512) == Q0.getValue());

    Q0 += Q2;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(-256) == Q0.getValue());

    Q0 += Q2;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0) == Q0.getValue());

    Q3 = Q1 + Q2;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0) == Q3.getValue());

    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0x8000) == Q4.getValue());

    Q4 += Q5;

    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0x8001) == Q4.getValue());

    Q4 = static_cast<SignedQ1_15Type::FullWidthValueType>(0x8000);

    for (uint32_t i = 0; i < 4194303; ++i)
    {
        Q4 += Q5;
    }

    BOOST_TEST(static_cast<SignedQ1_15Type::FullWidthValueType>(0x7FFF) == Q4.getValue());

    Q8 = Q6 + Q7;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x0) == Q8.getValue());

    Q8 = Q7 + Q9;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x180) == Q8.getValue());

    Q13 = Q10 + Q11;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x0) == Q13.getValue());

    Q13 = Q11 + Q11;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x2000000) == Q13.getValue());

    Q13 = Q10 + Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFE000000) == Q13.getValue());

    Q13 = Q10 + Q12;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFF800000) == Q13.getValue());

    Q13 = Q12 + Q12;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x1000000) == Q13.getValue());

    Q13 = Q11 + 1;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x2000000) == Q13.getValue());

    uQ8 = uQ8 + uQ9;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x40) == uQ8.getValue());

    uQ8 = uQ8 + uQ9;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x80) == uQ8.getValue());

    Q22 = Q22 + Q23;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0x0) == Q22.getValue());

    Q22 = Q22 + Q23;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xC0) == Q22.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_subtraction)
{
    UnsignedQ8_8Type uQ0(1, 1);
    UnsignedQ8_8Type uQ1(0, 1);
    UnsignedQ8_8Type uQ2(0, 0xFF);
    UnsignedQ8_8Type uQ3;
    SignedQ8_8Type Q0(1, 0);
    SignedQ8_8Type Q1(-1, 0);
    SignedQ8_8Type Q2(1, 0);
    SignedQ8_8Type Q3;
    SignedQ1_15Type Q4(0x7F, 0xFF);
    SignedQ8_24Type Q10(-1, 0);
    SignedQ8_24Type Q11(1, 0);
    SignedQ8_24Type Q12(0x800000);
    SignedQ8_24Type Q13;
    UnsignedQ1_7Type uQ7(1, 0);
    UnsignedQ1_7Type uQ8(0, 64);
    SignedQ2_6Type Q23(1, 0);
    SignedQ2_6Type Q24(0, 32);
#ifdef __SIZEOF_INT128__
    UnsignedQ32_32Type uQ4(1, 1);
    UnsignedQ32_32Type uQ5(0, 1);
    UnsignedQ32_32Type uQ6;
    SignedQ32_32Type Q14(0, 0);
    SignedQ32_32Type Q15(-1, 0);
    SignedQ32_32Type Q16(1, 0);
    SignedQ32_32Type Q17;
    SignedQ24_40Type Q18(0, 0);
    SignedQ24_40Type Q19(-1, 0);
    SignedQ24_40Type Q20(1, 0);

    uQ6 = uQ5 - uQ4;
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0xFFFFFFFF00000000ULL) == uQ6.getValue());

    uQ6 = uQ4 - uQ5;
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0x100000000ULL) == uQ6.getValue());

    uQ6 -= rawLike(uQ6, 1);
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0xFFFFFFFFULL) == uQ6.getValue());

    uQ6 -= rawLike(uQ6, 1);
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0xFFFFFFFEULL) == uQ6.getValue());

    uQ6 -= rawLike(uQ6, 0);
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0xFFFFFFFEULL) == uQ6.getValue());

    Q15 -= rawLike(Q15, 0);
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0xFFFFFFFF00000000ULL) == Q15.getValue());

    Q14 -= Q15;
    BOOST_TEST(Q16.getValue() == Q14.getValue());

    Q18 -= rawLike(Q18, 0);
    BOOST_TEST(static_cast<SignedQ24_40Type::FullWidthValueType>(0) == Q18.getValue());

    Q18 -= Q19;
    BOOST_TEST(Q20.getValue() == Q18.getValue());
#endif // __SIZEOF_INT128__

    uQ3 = uQ0 - uQ1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x100) == uQ3.getValue());

    uQ3 = uQ3 - uQ2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x1) == uQ3.getValue());

    uQ0 -= rawLike(uQ0, 0);
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ0.getValue());

    uQ0 -= uQ1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x100) == uQ0.getValue());

    uQ0 -= rawLike(uQ0, 1);
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0xFF) == uQ0.getValue());

    uQ0 -= uQ2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ0.getValue());

    Q0 -= rawLike(Q0, 0);
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x100) == Q0.getValue());

    Q0 -= Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x200) == Q0.getValue());

    Q0 -= Q2;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x100) == Q0.getValue());

    Q0 -= Q2;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x0) == Q0.getValue());

    Q0 -= Q2;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xFF00) == Q0.getValue());

    Q13 = Q10 - Q11;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFE000000) == Q13.getValue());

    Q13 = Q10 - Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x0) == Q13.getValue());

    Q13 = Q11 - Q12;
    BOOST_TEST(Q12.getValue() == Q13.getValue());

    Q13 = Q12 - Q11;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFF800000) == Q13.getValue());

    Q13 = Q11 - 1;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x0) == Q13.getValue());

    uQ7 = uQ7 - uQ8;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x40) == uQ7.getValue());

    uQ7 = uQ7 - uQ8;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x0) == uQ7.getValue());

    Q23 = Q23 - Q24;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0x20) == Q23.getValue());

    Q23 = Q23 - Q24;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0x0) == Q23.getValue());

    Q23 = Q23 - Q24;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xE0) == Q23.getValue());

    Q23 = Q23 - Q24;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xC0) == Q23.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_increment_decrement)
{
    UnsignedQ8_8Type uTempQ0;
    UnsignedQ8_8Type uTempQ0_Zero(0, 0);
    UnsignedQ8_8Type uTempQ0_NegOne(-1, 0);

    UnsignedQ8_8Type uQ0(0, 0);
    UnsignedQ8_8Type uQ2(-1, 0);
    SignedQ8_8Type Q0;
    SignedQ8_8Type Q1(-1, 0);
    SignedQ8_24Type Q10(-1, 0);
    SignedQ8_24Type Q11(1, 0);
    SignedQ8_24Type Q12(0x800000);
    SignedQ8_24Type Q13;
    UnsignedQ1_7Type uQ5(0, 0);
    SignedQ2_6Type Q21(1, 0);
#ifdef __SIZEOF_INT128__
    UnsignedQ32_32Type uQ3(1, 0);
    UnsignedQ32_32Type uQ4(0, 0);
    SignedQ32_32Type Q14(0, 0);
    SignedQ32_32Type Q15(-1, 0);
    SignedQ32_32Type Q16(1, 0);
    SignedQ24_40Type Q17(0, 0);
    SignedQ24_40Type Q18(-1, 0);
    SignedQ24_40Type Q19(1, 0);

    ++uQ4;
    BOOST_TEST(uQ4.getValue() == uQ3.getValue());

    --uQ4;
    BOOST_TEST(static_cast<UnsignedQ32_32Type::FullWidthValueType>(0) == uQ4.getValue());

    --Q14;
    BOOST_TEST(Q14.getValue() == Q15.getValue());

    ++Q14;
    BOOST_TEST(static_cast<SignedQ32_32Type::FullWidthValueType>(0) == Q14.getValue());

    ++Q14;
    BOOST_TEST(Q14.getValue() == Q16.getValue());

    ++Q17;
    BOOST_TEST(Q17.getValue() == Q19.getValue());

    --Q17;
    BOOST_TEST(static_cast<SignedQ24_40Type::FullWidthValueType>(0) == Q17.getValue());

    --Q17;
    BOOST_TEST(Q17.getValue() == Q18.getValue());
#endif // __SIZEOF_INT128__

    --uQ0;
    BOOST_TEST(uQ2.getValue() == uQ0.getValue());

    ++uQ0;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ0.getValue());

    uTempQ0 = uQ0--;
    BOOST_TEST(uTempQ0.getValue() == uTempQ0_Zero.getValue());
    BOOST_TEST(uQ2.getValue() == uQ0.getValue());

    uTempQ0 = uQ0++;
    BOOST_TEST(uTempQ0.getValue() == uTempQ0_NegOne.getValue());
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == uQ0.getValue());

    --Q0;
    BOOST_TEST(Q1.getValue() == Q0.getValue());

    ++Q0;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == Q0.getValue());

    Q0--;
    BOOST_TEST(Q1.getValue() == Q0.getValue());

    Q0++;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0) == Q0.getValue());

    Q10++;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0) == Q10.getValue());

    ++Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x1000000) == Q10.getValue());

    Q10--;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0) == Q10.getValue());

    --Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFF000000) == Q10.getValue());

    ++uQ5;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x80) == uQ5.getValue());

    --uQ5;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x0) == uQ5.getValue());

    --Q21;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0x0) == Q21.getValue());

    --Q21;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xC0) == Q21.getValue());

    ++Q21;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0x0) == Q21.getValue());

    ++Q21;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0x40) == Q21.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_multiplication)
{
    UnsignedQ8_8Type uQ0(1, 1);
    UnsignedQ8_8Type uQ1(1, 0);
    UnsignedQ8_8Type uQ2;
    UnsignedQ8_8Type uQ3(2, 0x9F);
    UnsignedQ8_8Type uQ4(0, 0x7F);
    UnsignedQ8_8Type uQ5;
    SignedQ8_8Type Q0(1, 0);
    SignedQ8_8Type Q1(-1, 0);
    SignedQ8_8Type Q2;
    SignedQ8_8Type Q3(0, 0x80);
    SignedQ8_8Type Q4(-1, 0);
    SignedQ8_8Type Q5;
    SignedQ8_8Type Q6(0, 0x80);
    SignedQ8_8Type Q7(-1, 0x80);
    SignedQ8_8Type Q8;
    SignedQ7_9Type Q9(-1, 0);
    SignedQ7_9Type Q10(0, 0x100);
    SignedQ7_9Type Q11;
    SignedQ24_8Type Q12(1, 0);
    SignedQ24_8Type Q13;
    SignedQ24_8Type Q14(0, 128);
    SignedQ24_8Type Q15(-1, 0);
    SignedQ8_24Type Q16(-1, 0);
    SignedQ8_24Type Q17(1, 0);
    SignedQ8_24Type Q18(0x800000);
    SignedQ8_24Type Q19;
    SignedQ8_8Type Q20(-2, 0x94);
    SignedQ8_8Type Q21(0, 0x7F);
    UnsignedQ1_7Type uQ12(1, 0);
    UnsignedQ1_7Type uQ13(0, 64);
    SignedQ2_6Type Q34(-1, 0);
    SignedQ2_6Type Q35(0, 32);
#ifdef __SIZEOF_INT128__
    UnsignedQ32_32Type uQ6(1, 0);
    UnsignedQ32_32Type uQ7(2, 0);
    UnsignedQ32_32Type uQ8(0, ((UINT32_MAX >> 1) + 1));
    UnsignedQ32_32Type uQ9(0, 0);
    UnsignedQ32_32Type uQ10;
    UnsignedQ32_32Type uQ11(0, ((UINT32_MAX >> 2) + 1));
    SignedQ32_32Type Q22(0, 0);
    SignedQ32_32Type Q23(0, ((UINT32_MAX >> 1) + 1));
    SignedQ32_32Type Q24(-1, 0);
    SignedQ32_32Type Q25(1, 0);
    SignedQ32_32Type Q26;
    SignedQ24_40Type Q27(0, 0);
    SignedQ24_40Type Q28(-1, 0);
    SignedQ24_40Type Q29(1, 0);
    SignedQ24_40Type Q30(0, (SignedQ24_40Type::MaxFractionalPartValue) >> 1);
    SignedQ24_40Type Q31;

    uQ10 = uQ6 * uQ6;
    BOOST_TEST(uQ10.getValue() == uQ6.getValue());

    uQ10 *= uQ7;
    BOOST_TEST(uQ10.getValue() == uQ7.getValue());

    uQ10 = uQ7 * uQ6;
    BOOST_TEST(uQ10.getValue() == uQ7.getValue());

    uQ10 = uQ6 * uQ8;
    BOOST_TEST(uQ10.getValue() == uQ8.getValue());

    uQ10 = uQ6 * uQ11;
    BOOST_TEST(uQ10.getValue() == uQ11.getValue());

    uQ10 = uQ8 * uQ8;
    BOOST_TEST(uQ10.getValue() == uQ11.getValue());

    uQ10 = uQ7 * uQ8;
    BOOST_TEST(uQ10.getValue() == uQ6.getValue());

    uQ10 = uQ7 * uQ9;
    BOOST_TEST(uQ10.getValue() == uQ9.getValue());

    Q26 = Q22 * Q22;
    BOOST_TEST(Q26.getValue() == Q22.getValue());

    Q26 = Q25 * Q23;
    BOOST_TEST(Q26.getValue() == Q23.getValue());

    Q26 = Q24 * Q25;
    BOOST_TEST(Q26.getValue() == Q24.getValue());

    Q26 *= Q24;
    BOOST_TEST(Q26.getValue() == Q25.getValue());

    Q31 = Q27 * Q27;
    BOOST_TEST(Q31.getValue() == Q27.getValue());

    Q31 = Q29 * Q30;
    BOOST_TEST(Q31.getValue() == Q30.getValue());

    Q31 = Q28 * Q29;
    BOOST_TEST(Q31.getValue() == Q28.getValue());

    Q31 = Q28 * Q28;
    BOOST_TEST(Q31.getValue() == Q29.getValue());
#endif // __SIZEOF_INT128__

    uQ2 = uQ0 * uQ1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ2.getValue());

    uQ2 = uQ1 * 2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x200) == uQ2.getValue());

    uQ2 = uQ0 * 1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ2.getValue());

    uQ5 = uQ3 * uQ4;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x14D) == uQ5.getValue());

    uQ2 = uQ1;
    uQ2 *= 2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x200) == uQ2.getValue());

    Q2 = Q0 * Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xff00) == Q2.getValue());

    Q5 = Q3 * Q4;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xff80) == Q5.getValue());

    Q8 = Q6 * Q7;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xffC0) == Q8.getValue());

    Q11 = Q9 * Q10;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xff00) == Q11.getValue());

    Q13 = Q12 * Q14;
    BOOST_TEST(static_cast<SignedQ24_8Type::FullWidthValueType>(0x80) == Q13.getValue());

    Q13 = Q14 * Q15;
    BOOST_TEST(static_cast<SignedQ24_8Type::FullWidthValueType>(0xFFFFFF80) == Q13.getValue());

    Q19 = Q17 * Q18;
    BOOST_TEST(Q18.getValue() == Q19.getValue());

    Q19 = Q16 * Q18;
    BOOST_TEST(static_cast<SignedQ24_8Type::FullWidthValueType>(0xFF800000) == Q19.getValue());

    Q19 = Q16;
    Q19 *= Q18;
    BOOST_TEST(static_cast<SignedQ24_8Type::FullWidthValueType>(0xFF800000) == Q19.getValue());

    Q2 = Q20 * Q21;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xff4B) == Q2.getValue());

    uQ12 = uQ12 * uQ13;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x40) == uQ12.getValue());

    uQ12 = uQ12 * uQ13;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x20) == uQ12.getValue());

    Q34 = Q34 * Q35;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xE0) == Q34.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_multiplication_trunc)
{
    UnsignedTruncatingQType uQ0(1, 1);
    UnsignedTruncatingQType uQ1(1, 0);
    UnsignedTruncatingQType uQ2;
    UnsignedTruncatingQType uQ3(2, 0x9F);
    UnsignedTruncatingQType uQ4(0, 0x7F);
    UnsignedTruncatingQType uQ5;

    uQ2 = uQ0 * uQ1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ2.getValue());

    uQ2 = uQ1 * 2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x200) == uQ2.getValue());

    uQ2 = uQ0 * 1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ2.getValue());

    uQ5 = uQ3 * uQ4;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x14C) == uQ5.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_division)
{
    UnsignedQ8_8Type uQ0(1, 1);
    UnsignedQ8_8Type uQ1(1, 0);
    UnsignedQ8_8Type uQ2;
    UnsignedQ8_8Type uQ3(2, 0);
    UnsignedQ8_8Type uQ4(0, 0x80);
    UnsignedQ8_8Type uQ5(128, 0);
    UnsignedQ8_8Type uQ6(64, 0);
    SignedQ8_8Type Q0(1, 0);
    SignedQ8_8Type Q1(-1, 0);
    SignedQ8_8Type Q2;
    SignedQ8_8Type Q3(0, 0x80);
    SignedQ8_8Type Q4(-0, 0x80);
    SignedQ7_9Type Q5(-1, 0);
    SignedQ7_9Type Q6(0, 0x100);
    SignedQ7_9Type Q7;
    SignedQ8_24Type Q8(-1, 0);
    SignedQ8_24Type Q9(1, 0);
    SignedQ8_24Type Q10(0x800000);
    SignedQ8_24Type Q11;
    UnsignedQ1_7Type uQ13(1, 0);
    UnsignedQ1_7Type uQ14(0, 64);
    SignedQ2_6Type Q24(-1, 0);
    SignedQ2_6Type Q25(0, 32);
#ifdef __SIZEOF_INT128__
    UnsignedQ32_32Type uQ7(1, 0);
    UnsignedQ32_32Type uQ8(2, 0);
    UnsignedQ32_32Type uQ9(0, ((UINT32_MAX >> 1) + 1));
    UnsignedQ32_32Type uQ10(0, 0);
    UnsignedQ32_32Type uQ11;
    UnsignedQ32_32Type uQ12(0, ((UINT32_MAX >> 2) + 1));
    SignedQ32_32Type Q12(0, 0);
    SignedQ32_32Type Q13(0, ((UINT32_MAX >> 1) + 1));
    SignedQ32_32Type Q14(-1, 0);
    SignedQ32_32Type Q15(1, 0);
    SignedQ32_32Type Q16;
    SignedQ32_32Type Q17(2, 0);
    SignedQ24_40Type Q18(0, 0);
    SignedQ24_40Type Q19(-1, 0);
    SignedQ24_40Type Q20(1, 0);
    SignedQ24_40Type Q21(0, ((SignedQ24_40Type::MaxFractionalPartValue >> 1) + 1));
    SignedQ24_40Type Q22(2,0);
    SignedQ24_40Type Q23;

    uQ11 = uQ10 / uQ8;
    BOOST_TEST(uQ11.getValue() == uQ10.getValue());

    uQ11 = uQ7 / uQ8;
    BOOST_TEST(uQ11.getValue() == uQ9.getValue());

    uQ11 = uQ8 / uQ7;
    BOOST_TEST(uQ11.getValue() == uQ8.getValue());

    uQ11 = uQ12 / uQ9;
    BOOST_TEST(uQ11.getValue() == uQ9.getValue());

    uQ11 /= uQ9;
    BOOST_TEST(uQ11.getValue() == uQ7.getValue());

    Q16 = Q12 / Q15;
    BOOST_TEST(Q16.getValue() == Q12.getValue());

    Q16 = Q14 / Q15;
    BOOST_TEST(Q16.getValue() == Q14.getValue());

    Q16 = Q15 / Q13;
    BOOST_TEST(Q16.getValue() == Q17.getValue());

    Q16 = Q14 / Q14;
    BOOST_TEST(Q16.getValue() == Q15.getValue());

    Q16 = Q15 / Q14;
    BOOST_TEST(Q16.getValue() == Q14.getValue());

    Q23 = Q20 / Q19;
    BOOST_TEST(Q23.getValue() == Q19.getValue());

    Q23 = Q20 / Q21;
    BOOST_TEST(Q23.getValue() == Q22.getValue());
#endif // __SIZEOF_INT128__

    uQ2 = uQ0 / uQ1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ2.getValue());

    uQ2 = uQ0 / 1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x101) == uQ2.getValue());

    uQ2 = uQ1 / 2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x80) == uQ2.getValue());

    uQ2 = uQ1 / uQ3;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x80) == uQ2.getValue());

    uQ2 = uQ1 / uQ4;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x200) == uQ2.getValue());

    Q2 = Q1 / Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x100) == Q2.getValue());

    Q2 = Q0 / Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xff00) == Q2.getValue());

    Q2 = Q2 / Q1;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x100) == Q2.getValue());

    Q2 = Q0 / Q3;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0x200) == Q2.getValue());

    Q2 = Q1 / Q3;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xfe00) == Q2.getValue());

    Q7 = Q5 / Q6;
    BOOST_TEST(static_cast<SignedQ8_8Type::FullWidthValueType>(0xfc00) == Q7.getValue());

    Q11 = Q9 / Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x2000000) == Q11.getValue());

    Q11 = Q9;
    Q11 /= Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0x2000000) == Q11.getValue());

    Q11 = Q8 / Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFE000000) == Q11.getValue());

    Q11 = Q8;
    Q11 /= Q10;
    BOOST_TEST(static_cast<SignedQ8_24Type::FullWidthValueType>(0xFE000000) == Q11.getValue());

    uQ2 = (uQ6 / uQ5);
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x0080) == uQ2.getValue());

    uQ2 = uQ1;
    uQ2 /= 1;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x0100) == uQ2.getValue());

    uQ2 /= 2;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x0080) == uQ2.getValue());

    typename UnsignedQ8_8Type::FixedPartFieldType divisor = 1;
    uQ2 = uQ1;
    uQ2 /= divisor;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x0100) == uQ2.getValue());

    divisor = 2;
    uQ2 /= divisor;
    BOOST_TEST(static_cast<UnsignedQ8_8Type::FullWidthValueType>(0x0080) == uQ2.getValue());

    uQ13 /= 1;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x80) == uQ13.getValue());

    uQ14 /= uQ13;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x40) == uQ14.getValue());

    uQ14 /= uQ14;
    BOOST_TEST(static_cast<UnsignedQ1_7Type::FullWidthValueType>(0x80) == uQ13.getValue());

    Q24 /= 1;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xC0) == Q24.getValue());

    Q25 /= Q24;
    BOOST_TEST(static_cast<SignedQ2_6Type::FullWidthValueType>(0xE0) == Q25.getValue());
}

BOOST_AUTO_TEST_CASE(test_case_comparators)
{
    UnsignedQ8_8Type uQ0(1, 1);
    UnsignedQ8_8Type uQ1(1, 0);
    UnsignedQ8_8Type uQ2(1, 0);
    SignedQ8_8Type Q0(1, 0);
    SignedQ8_8Type Q1(-1, 0);
    SignedQ8_8Type Q2(-1, 0);
    SignedQ8_8Type Q3(-2, 0);
    int one = 1;
    int negativeOne = -1;
    int bigValue = 100;
    int smallValue = -100;

    BOOST_TEST(uQ0 > uQ1);
    BOOST_TEST(uQ1 < uQ0);
    BOOST_TEST(uQ2 <= uQ1);
    BOOST_TEST(uQ2 >= uQ1);
    BOOST_TEST((uQ2 <= one));
    BOOST_TEST((uQ2 <= bigValue));
    BOOST_TEST((uQ2 < bigValue));
    BOOST_TEST((Q1 >= negativeOne));
    BOOST_TEST((Q1 >= smallValue));
    BOOST_TEST((Q1 > smallValue));
    BOOST_TEST(Q0 > Q1);
    BOOST_TEST(Q1 < Q0);
    BOOST_TEST(Q2 <= Q1);
    BOOST_TEST(Q2 >= Q1);
    BOOST_TEST(Q3 < Q2);
    BOOST_TEST(Q3 <= Q2);
}

BOOST_AUTO_TEST_CASE(test_case_assignment)
{
    SignedQ8_8Type Q0;
    SignedQ8_8Type Q1(2, 0);

    BOOST_TEST(Q0.getValue() == 0x0);
    Q0 = 0x100;
    BOOST_TEST(Q0.getValue() == 0x100);

    Q0 = 0;
    BOOST_TEST(Q0.getValue() == 0x0);
    Q0.setValue(0x100);
    BOOST_TEST(Q0.getValue() == 0x100);

    Q0 = Q1;
    BOOST_TEST(Q0.getValue() == 0x200);
}

BOOST_AUTO_TEST_CASE(test_case_conversion)
{
    SignedQ8_8Type Q0;
    SignedQ8_24Type Q1(1, 0);
    SignedQ8_24Type Q2(1, (1 << 23));
    SignedQ16_16Type Q3(1, (1 << 15));
    SignedQ16_16Type Q4(-1, (1 << 15));
    SignedQ24_8Type Q5(1, (1 << 7));
    SignedQ24_8Type Q6(-1, (1 << 7));
    UnsignedQ24_8Type UQ0(1, (1 << 7));

    BOOST_TEST(Q0.getValue() == 0x0);
    Q0.convertFromOtherQValueType(Q1);
    BOOST_TEST(Q0.getValue() == 0x100);

    Q0.convertFromOtherQValueType(Q2);
    BOOST_TEST(Q0.getValue() == 0x180);

    Q0.convertFromOtherQValueType(Q3);
    BOOST_TEST(Q0.getValue() == 0x180);

    Q0.convertFromOtherQValueType(Q4);
    BOOST_TEST(Q0.getValue() == static_cast<typename SignedQ8_8Type::FullWidthValueType>(0xFF80));

    BOOST_TEST(Q3.getValue() == static_cast<typename SignedQ16_16Type::FullWidthValueType>(0x18000));
    Q3.convertFromOtherQValueType(Q0);
    BOOST_TEST(Q3.getValue() == static_cast<typename SignedQ16_16Type::FullWidthValueType>(0xFFFF8000));

    Q0 = 0;
    BOOST_TEST(Q0.getValue() == 0x0);
    Q0.convertFromOtherQValueType(UQ0);
    BOOST_TEST(Q0.getValue() == static_cast<typename SignedQ8_8Type::FullWidthValueType>(0x180));

    Q0 = 0;
    BOOST_TEST(Q0.getValue() == 0x0);
    Q0.convertFromOtherQValueType(Q5);
    BOOST_TEST(Q0.getValue() == static_cast<typename SignedQ8_8Type::FullWidthValueType>(0x180));

    Q0 = 0;
    BOOST_TEST(Q0.getValue() == 0x0);
    Q0.convertFromOtherQValueType(Q6);
    BOOST_TEST(Q0.getValue() == static_cast<typename SignedQ8_8Type::FullWidthValueType>(0xFF80));

    Q0 = 0x180;
    UQ0 = 0;
    BOOST_TEST(UQ0.getValue() == 0x0UL);
    UQ0.convertFromOtherQValueType(Q0);
    BOOST_TEST(UQ0.getValue() == static_cast<typename UnsignedQ24_8Type::FullWidthValueType>(0x180));
}

BOOST_AUTO_TEST_CASE(test_case_addition_saturation)
{
    UnsignedQ1_7SatPolicyType Q0_1_7(0, 32); // 0.25
    UnsignedQ1_7SatPolicyType Q1_1_7(0, 64); // 0.50
    UnsignedQ1_7SatPolicyType Q2_1_7 = Q0_1_7 + Q1_1_7; // Q1.7(0, 96) = 0.75
    BOOST_TEST(static_cast<UnsignedQ1_7SatPolicyType::FullWidthValueType>(0x60) == Q2_1_7.getValue());

    UnsignedQ8_8SatPolicyType Q0_8_8(200, 0); // Fixed number portion is 200
    UnsignedQ8_8SatPolicyType Q1_8_8(134, 0); // Fixed number portion is 134
    UnsignedQ8_8SatPolicyType Q2_8_8 = Q0_8_8 + Q1_8_8; // Saturated to the max, 255.99609375
    BOOST_TEST(static_cast<UnsignedQ8_8SatPolicyType::FullWidthValueType>(0xFFFF) == Q2_8_8.getValue());

    UnsignedQ16_16SatPolicyType Q0_16_16(64780, 0); // Fixed number portion is 64780
    UnsignedQ16_16SatPolicyType Q1_16_16(13869, 0); // Fixed number portion is 13869
    UnsignedQ16_16SatPolicyType Q2_16_16 = Q0_16_16 + Q1_16_16; // Saturated to the max, 65535.99998
    BOOST_TEST(static_cast<UnsignedQ16_16SatPolicyType::FullWidthValueType>(0xFFFFFFFF) == Q2_16_16.getValue());

    SignedQ24_8SatPolicyType Q0_24_8(-257, 0); // Fixed number portion is -257 (0xFFFEFF)
    SignedQ24_8SatPolicyType Q1_24_8(512, 0); // Fixed number portion is 512 (0x200)
    SignedQ24_8SatPolicyType Q2_24_8 = Q0_24_8 + Q1_24_8; // Q24.8(255, 0) = Fixed number portion is 255
    BOOST_TEST(static_cast<SignedQ24_8SatPolicyType::FullWidthValueType>(0xFF00) == Q2_24_8.getValue());

    SignedQ8_8SatPolicyType Q3_8_8(-115, 0); // Fixed number portion is -115 (0x8D)
    SignedQ8_8SatPolicyType Q4_8_8(-56, 0); // Fixed number portion is -56 (0xC8)
    SignedQ8_8SatPolicyType Q5_8_8 = Q3_8_8 + Q4_8_8; // Fixed number portion is -128 (saturated)
    BOOST_TEST(static_cast<SignedQ8_8SatPolicyType::FullWidthValueType>(0x8000) == Q5_8_8.getValue());

#ifdef __SIZEOF_INT128__
    UnsignedQ32_32SatPolicyType Q0_32_32(2694967296, 0); // 2,694,967,296
    UnsignedQ32_32SatPolicyType Q1_32_32(3293867987, 0); // 3,293,867,987
    UnsignedQ32_32SatPolicyType Q2_32_32 = Q0_32_32 + Q1_32_32; // Saturated to the max, 4,294,967,295.99999
    BOOST_TEST(static_cast<UnsignedQ32_32SatPolicyType::FullWidthValueType>(0xFFFFFFFFFFFFFFFF) == Q2_32_32.getValue());

    SignedQ24_40SatPolicyType Q0_24_40(-8150371, 0); // -8,150,371
    SignedQ24_40SatPolicyType Q1_24_40(-4284353, 0); // -4,284,353
    SignedQ24_40SatPolicyType Q2_24_40 = Q0_24_40 + Q1_24_40; // Fixed number portion is -8,388,608 (saturated)
    BOOST_TEST(static_cast<SignedQ24_40SatPolicyType::FullWidthValueType>(0x8000000000000000) == Q2_24_40.getValue());
#endif // __SIZEOF_INT128__
}

BOOST_AUTO_TEST_CASE(test_case_subtraction_saturation)
{
    UnsignedQ1_7SatPolicyType Q0_1_7(0, 32); // 0.25
    UnsignedQ1_7SatPolicyType Q1_1_7(0, 64); // 0.50
    UnsignedQ1_7SatPolicyType Q2_1_7 = Q0_1_7 - Q1_1_7; // Should be 0 because of saturation
    BOOST_TEST(static_cast<UnsignedQ1_7SatPolicyType::FullWidthValueType>(0) == Q2_1_7.getValue());

    UnsignedQ8_8SatPolicyType Q0_8_8(214, 0); // 214
    UnsignedQ8_8SatPolicyType Q1_8_8(134, 0); // 134
    UnsignedQ8_8SatPolicyType Q2_8_8 = Q0_8_8 - Q1_8_8; // Fixed number portion is 80
    BOOST_TEST(static_cast<UnsignedQ8_8SatPolicyType::FullWidthValueType>(0x5000) == Q2_8_8.getValue());

    UnsignedQ16_16SatPolicyType Q0_16_16(15, 0); // 15
    UnsignedQ16_16SatPolicyType Q1_16_16(0, 16384); // 0.25
    UnsignedQ16_16SatPolicyType Q2_16_16 = Q0_16_16 - Q1_16_16; // 15 - 0.25 = 14.75
    BOOST_TEST(static_cast<UnsignedQ16_16SatPolicyType::FullWidthValueType>(0xEC000) == Q2_16_16.getValue());

    SignedQ24_8SatPolicyType Q0_24_8(-7930000, 0); // -7,930,000
    SignedQ24_8SatPolicyType Q1_24_8(8003533, 0); // 8,003,533
    SignedQ24_8SatPolicyType Q2_24_8 = Q0_24_8 - Q1_24_8; // Should be -15,933,533 (saturated to -8,388608)
    BOOST_TEST(static_cast<SignedQ24_8SatPolicyType::FullWidthValueType>(0x80000000) == Q2_24_8.getValue());

    SignedQ7_9SatPolicyType Q0_7_9(0, 64); // 0.125
    SignedQ7_9SatPolicyType Q1_7_9(0, 256); // 0.5
    SignedQ7_9SatPolicyType Q2_7_9 = Q0_7_9 - Q1_7_9; // Should be -0.375 *
    BOOST_TEST(static_cast<SignedQ7_9SatPolicyType::FullWidthValueType>(0xFF40) == Q2_7_9.getValue());

#ifdef __SIZEOF_INT128__
    UnsignedQ32_32SatPolicyType Q0_32_32(2694967296, 0); // 2,694,967,296
    UnsignedQ32_32SatPolicyType Q1_32_32(3293867987, 0); // 3,293,867,987
    UnsignedQ32_32SatPolicyType Q2_32_32 = Q0_32_32 - Q1_32_32; // Should be -598,900,691 (saturated to 0)
    BOOST_TEST(static_cast<UnsignedQ32_32SatPolicyType::FullWidthValueType>(0) == Q2_32_32.getValue());

    SignedQ24_40SatPolicyType Q0_24_40(8150371, 0); // 8,150,371
    SignedQ24_40SatPolicyType Q1_24_40(-4284353, 0); // -4,284,353
    SignedQ24_40SatPolicyType Q2_24_40 = Q0_24_40 - Q1_24_40; // Should be 12,434,724 (saturated to the max, 8,388,607.99999)
    BOOST_TEST(static_cast<SignedQ24_40SatPolicyType::FullWidthValueType>(0x7FFFFFFFFFFFFFFF) == Q2_24_40.getValue());
#endif // __SIZEOF_INT128__
}

BOOST_AUTO_TEST_CASE(test_case_multiplication_saturation)
{
    UnsignedQ1_7SatPolicyType Q0_1_7(0, 64); // 0.5
    UnsignedQ1_7SatPolicyType Q1_1_7(0, 64); // 0.5
    UnsignedQ1_7SatPolicyType Q2_1_7 = Q0_1_7 * Q1_1_7; // Should be 0.25
    BOOST_TEST(static_cast<UnsignedQ1_7SatPolicyType::FullWidthValueType>(0x20) == Q2_1_7.getValue());

    UnsignedQ8_8SatPolicyType Q0_8_8(127, 0); // 127
    UnsignedQ8_8SatPolicyType Q1_8_8(5, 64); // 5.5
    UnsignedQ8_8SatPolicyType Q2_8_8 = Q0_8_8 * Q1_8_8; // Should be 698.5 (saturated to the max, 255.99609375)
    BOOST_TEST(static_cast<UnsignedQ8_8SatPolicyType::FullWidthValueType>(0xFFFF) == Q2_8_8.getValue());
    
    UnsignedQ16_16SatPolicyType Q0_16_16(294, 16384); // 294.25
    UnsignedQ16_16SatPolicyType Q1_16_16(0, 32768); // 0.5
    UnsignedQ16_16SatPolicyType Q2_16_16 = Q0_16_16 * Q1_16_16; // Should be 147.125
    BOOST_TEST(static_cast<UnsignedQ16_16SatPolicyType::FullWidthValueType>(0x932000) == Q2_16_16.getValue());

    SignedQ7_9SatPolicyType Q0_7_9(62, 0); // 62
    SignedQ7_9SatPolicyType Q1_7_9(-30, 128); // -30.25
    SignedQ7_9SatPolicyType Q2_7_9 = Q0_7_9 * Q1_7_9; // Should be -1875.5 (saturated to -64)
    BOOST_TEST(static_cast<SignedQ7_9SatPolicyType::FullWidthValueType>(0x8000) == Q2_7_9.getValue());

#ifdef __SIZEOF_INT128__
    UnsignedQ32_32SatPolicyType Q0_32_32(523466, 0); // 523,466
    UnsignedQ32_32SatPolicyType Q1_32_32(130339, 0); // 130,339
    UnsignedQ32_32SatPolicyType Q2_32_32 = Q0_32_32 * Q1_32_32; // 523,466 * 130,339 = 68177459894 (saturated to the max, 4294967295.99999)
    BOOST_TEST(static_cast<UnsignedQ32_32SatPolicyType::FullWidthValueType>(0xFFFFFFFFFFFFFFFF) == Q2_32_32.getValue());

    SignedQ24_40SatPolicyType Q0_24_40(-8150371, 0); // -8,150,371
    SignedQ24_40SatPolicyType Q1_24_40(90186, 0); // 90,186
    SignedQ24_40SatPolicyType Q2_24_40 = Q0_24_40 * Q1_24_40; // -8,150,371 * 90,186 = -735,000,000 (saturated to -8,388,608)
    BOOST_TEST(static_cast<SignedQ24_40SatPolicyType::FullWidthValueType>(0x8000000000000000) == Q2_24_40.getValue());
#endif // __SIZEOF_INT128__
}

BOOST_AUTO_TEST_CASE(test_case_division_saturation)
{
    UnsignedQ1_7SatPolicyType Q0_1_7(1, 64); // 1.5
    UnsignedQ1_7SatPolicyType Q1_1_7(1, 96); // 1.75
    UnsignedQ1_7SatPolicyType Q2_1_7 = Q0_1_7 / Q1_1_7; // 1.5 / 1.75 should be 0.857142857 (close to Q1.7(0, 109))
    BOOST_TEST(static_cast<UnsignedQ1_7SatPolicyType::FullWidthValueType>(0x6D) == Q2_1_7.getValue());

    UnsignedQ16_16SatPolicyType Q0_16_16(39569, 0); // 39,569
    UnsignedQ16_16SatPolicyType Q1_16_16(0, 32768); // 0.5
    UnsignedQ16_16SatPolicyType Q2_16_16 = Q0_16_16 / Q1_16_16; // 39,569 / 0.5 should be 79,138 (saturated to the max, 65535.99998)
    BOOST_TEST(static_cast<UnsignedQ16_16SatPolicyType::FullWidthValueType>(0xFFFFFFFF) == Q2_16_16.getValue());

    SignedQ7_9SatPolicyType Q0_7_9(-58, 0); // -58
    SignedQ7_9SatPolicyType Q1_7_9(0, 128); // 0.25
    SignedQ7_9SatPolicyType Q2_7_9 = Q0_7_9 / Q1_7_9; // -58 / 0.25 should be -232 (saturated to -64)
    BOOST_TEST(static_cast<SignedQ7_9SatPolicyType::FullWidthValueType>(0x8000) == Q2_7_9.getValue());

#ifdef __SIZEOF_INT128__
    UnsignedQ32_32SatPolicyType Q0_32_32(3886724233, 0); // 3,886,724,233
    UnsignedQ32_32SatPolicyType Q1_32_32(0, 536870912); // 0.125
    UnsignedQ32_32SatPolicyType Q2_32_32 = Q0_32_32 / Q1_32_32; // 3,886,724,233 / 0.125 should be 31,093,793,864 (saturated to the max, 4,294,967,295.99999)
    BOOST_TEST(static_cast<UnsignedQ32_32SatPolicyType::FullWidthValueType>(0xFFFFFFFFFFFFFFFF) == Q2_32_32.getValue());

    SignedQ24_40SatPolicyType Q0_24_40(-8150371, 0); // -8,150,371
    SignedQ24_40SatPolicyType Q1_24_40(0, 137438953472); // 0.125
    SignedQ24_40SatPolicyType Q2_24_40 = Q0_24_40 / Q1_24_40; // -8,150,371 / 0.125 should be -65,202,968 (saturated to -8,388,608)
    BOOST_TEST(static_cast<SignedQ24_40SatPolicyType::FullWidthValueType>(0x8000000000000000) == Q2_24_40.getValue());
#endif // __SIZEOF_INT128__
}

BOOST_AUTO_TEST_CASE(test_case_saturation_max_includes_fraction)
{
    // The saturation bound is the format's true maximum, every fractional bit
    // set. It used to be the largest whole number (127.0 for Q8.8), so values
    // between 127.0 and 127.996 were clamped down to 127.0.
    BOOST_TEST(SignedQ8_8SatPolicyType::QFormatMaxValue() == 0x7FFF);
    BOOST_TEST(UnsignedQ8_8SatPolicyType::QFormatMaxValue() == 0xFFFF);
    BOOST_TEST(SignedQ8_8SatPolicyType::QFormatMinValue() == static_cast<SignedQ8_8SatPolicyType::FullWidthValueType>(-32768));

    const SignedQ8_8SatPolicyType a(127, 0x80); // 127.5
    const SignedQ8_8SatPolicyType b(0, 0x10);   // 0.0625
    BOOST_TEST((a + b).getValue() == 0x7F90);   // 127.5625, in range: not clamped

    const SignedQ8_8SatPolicyType c(127, 0xF0); // 127.9375
    BOOST_TEST((c + b).getValue() == 0x7FFF);   // 128.0 does not fit: clamped to the max
}

BOOST_AUTO_TEST_CASE(test_case_saturating_unit_step_without_representable_one)
{
    // Q1.15 cannot hold 1.0, and 2^15 converts to -1.0 in its int16_t word,
    // so a saturating ++ used to move the value down and -- move it up.
    typedef tinymind::QValue<1, 15, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SatQ1_15Type;

    SatQ1_15Type half(0, 0x4000); // 0.5
    ++half;                       // 1.5 does not fit: clamp to the max
    BOOST_TEST(half.getValue() == 0x7FFF);

    SatQ1_15Type minusHalf(static_cast<SatQ1_15Type::FullWidthValueType>(-0x4000)); // -0.5
    --minusHalf;                                                                    // -1.5: clamp to the min
    BOOST_TEST(minusHalf.getValue() == static_cast<SatQ1_15Type::FullWidthValueType>(-0x8000));

    SatQ1_15Type almostZero(static_cast<SatQ1_15Type::FullWidthValueType>(-2)); // -2 LSB
    almostZero++;                                                              // 1.0 - 2 LSB fits exactly
    BOOST_TEST(almostZero.getValue() == 0x7FFE);
}

BOOST_AUTO_TEST_CASE(test_case_unsigned_saturating_add_with_addend_above_max)
{
    // MinMaxSaturatePolicy's guard computed maxValue - target in T, which
    // wraps for an unsigned word at least as wide as int when the addend
    // exceeds the bound, so the sum wrapped instead of clamping. QValue no
    // longer takes a raw integer addend, but the policy is public, so it is
    // checked directly with UQ8.16's bounds.
    typedef tinymind::MinMaxSaturatePolicy<uint32_t> PolicyType;

    BOOST_TEST(PolicyType::saturate(0u, 0xFFFFFFFEu, 0u, 0xFFFFFFu, tinymind::AdditionOp) == 0xFFFFFFu);
    BOOST_TEST(PolicyType::saturate(5u, 0xFFFFFFu, 0u, 0xFFFFFFu, tinymind::AdditionOp) == 0xFFFFFFu);
    BOOST_TEST(PolicyType::saturate(5u, 7u, 0u, 0xFFFFFFu, tinymind::AdditionOp) == 12u);
}

BOOST_AUTO_TEST_CASE(test_case_int_operand_is_an_exact_whole_number)
{
    // An int operand used to be built as QValue(n, 0), wrapping an int outside
    // the fixed part's range first: for Q8.8, 200 became -56.
    const SignedQ8_8Type one(1, 0);
    BOOST_TEST((one < 200));
    BOOST_TEST(!(one > 200));
    BOOST_TEST((one > -200));
    BOOST_TEST(((one <= 1) && (one >= 1) && !(one < 1) && !(one > 1)));

    const SignedQ8_8SatPolicyType satOne(1, 0);
    BOOST_TEST((satOne + 200).getValue() == SignedQ8_8SatPolicyType::QFormatMaxValue()); // 201.0 clamps
    BOOST_TEST((satOne - 200).getValue() == SignedQ8_8SatPolicyType::QFormatMinValue()); // -199.0 clamps
    BOOST_TEST((satOne * 300).getValue() == SignedQ8_8SatPolicyType::QFormatMaxValue()); // 300.0 clamps

    SignedQ8_8SatPolicyType ten(10, 0);
    ten /= 300; // 0.0333 truncates to 8 LSBs (0.03125); the old path divided by 300 wrapped to 44
    BOOST_TEST(ten.getValue() == 8);
    BOOST_TEST((SignedQ8_8SatPolicyType(100, 0) / 256).getValue() == 100); // 0.390625, not 100 / 0

    // Wrap arithmetic on an exact operand: 1.0 + 256.0 wraps to 1.0 in Q8.8.
    BOOST_TEST((one + 256).getValue() == 256);
}

BOOST_AUTO_TEST_CASE(test_case_ambiguous_integer_operands_are_deleted)
{
    // q += n, q -= n, q == n and q != n read n as raw LSBs while q + n and
    // q < n read it as a whole number. The members are deleted for any
    // non-QValue operand, so callers must say which they mean; QValue operands
    // are unaffected.
    static_assert(!CanAddAssign<SignedQ8_8Type, int>::value, "q += int must not compile");
    static_assert(!CanSubtractAssign<SignedQ8_8Type, int>::value, "q -= int must not compile");
    static_assert(!CanCompareEqual<SignedQ8_8Type, int>::value, "q == int must not compile");
    static_assert(!CanCompareNotEqual<SignedQ8_8Type, int>::value, "q != int must not compile");
    static_assert(!CanAddAssign<SignedQ16_16Type, SignedQ16_16Type::FullWidthValueType>::value, "q += raw word must not compile");
    static_assert(!CanCompareEqual<SignedQ16_16Type, SignedQ16_16Type::FullWidthValueType>::value, "q == raw word must not compile");
    static_assert(!CanAddAssign<SignedQ8_8Type, double>::value, "q += double must not compile");

    static_assert(CanAddAssign<SignedQ8_8Type, SignedQ8_8Type>::value, "q += QValue must compile");
    static_assert(CanSubtractAssign<SignedQ8_8Type, SignedQ8_8Type>::value, "q -= QValue must compile");
    static_assert(CanCompareEqual<SignedQ8_8Type, SignedQ8_8Type>::value, "q == QValue must compile");
    static_assert(CanCompareNotEqual<SignedQ8_8Type, SignedQ8_8Type>::value, "q != QValue must compile");

    SignedQ8_8Type q(1, 0);
    q.addInteger(2);
    BOOST_TEST(q.getValue() == 3 * 256);
    q.subtractInteger(5);
    BOOST_TEST(q.getValue() == -2 * 256);
    BOOST_TEST(q.compareInteger(-2) == 0);
    BOOST_TEST(q.compareInteger(-1) < 0);
    BOOST_TEST(q.compareInteger(-3) > 0);
}

BOOST_AUTO_TEST_CASE(test_case_division_by_zero_saturates)
{
    // MinMaxSaturatePolicy must clamp division by zero rather than trapping.
    // Positive numerator -> max; negative numerator -> min; zero numerator -> min.
    SignedQ8_8SatPolicyType pos(2, 0); // +2.0
    SignedQ8_8SatPolicyType neg(-3, 0); // -3.0
    SignedQ8_8SatPolicyType zero(0, 0);
    SignedQ8_8SatPolicyType denom(0, 0);

    SignedQ8_8SatPolicyType posDiv = pos / denom;
    BOOST_TEST(posDiv.getValue() == SignedQ8_8SatPolicyType::QFormatMaxValue());

    SignedQ8_8SatPolicyType negDiv = neg / denom;
    BOOST_TEST(negDiv.getValue() == SignedQ8_8SatPolicyType::QFormatMinValue());

    SignedQ8_8SatPolicyType zeroDiv = zero / denom;
    BOOST_TEST(zeroDiv.getValue() == SignedQ8_8SatPolicyType::QFormatMinValue());

    // Same behavior for an unsigned type (no negative numerators possible).
    UnsignedQ8_8SatPolicyType uPos(5, 0);
    UnsignedQ8_8SatPolicyType uZero(0, 0);
    UnsignedQ8_8SatPolicyType uDenom(0, 0);

    UnsignedQ8_8SatPolicyType uPosDiv = uPos / uDenom;
    BOOST_TEST(uPosDiv.getValue() == UnsignedQ8_8SatPolicyType::QFormatMaxValue());

    // Zero / zero on unsigned type: numerator is not > 0, so falls into the
    // minValue branch, which for unsigned is 0.
    UnsignedQ8_8SatPolicyType uZeroDiv = uZero / uDenom;
    BOOST_TEST(uZeroDiv.getValue() == UnsignedQ8_8SatPolicyType::QFormatMinValue());
}

BOOST_AUTO_TEST_CASE(test_case_wrap_division_by_zero_is_defined)
{
    // WrapPolicy division by zero was undefined behaviour (SIGFPE on x86).
    // It now gives the same bound as MinMaxSaturatePolicy: the maximum for a
    // positive numerator, else the minimum. Every divide path is covered: a
    // QValue divisor, an int divisor, compound and free forms, and a 64-bit
    // word, whose quotient is computed in __int128.
    const SignedQ8_8Type pos(2, 0);
    const SignedQ8_8Type neg(-3, 0);
    const SignedQ8_8Type zero(0, 0);

    BOOST_TEST((pos / zero).getValue() == SignedQ8_8Type::QFormatMaxValue());
    BOOST_TEST((neg / zero).getValue() == SignedQ8_8Type::QFormatMinValue());
    BOOST_TEST((zero / zero).getValue() == SignedQ8_8Type::QFormatMinValue());
    BOOST_TEST((pos / 0).getValue() == SignedQ8_8Type::QFormatMaxValue());
    BOOST_TEST((neg / 0).getValue() == SignedQ8_8Type::QFormatMinValue());

    SignedQ8_8Type q(pos);
    q /= zero;
    BOOST_TEST(q.getValue() == SignedQ8_8Type::QFormatMaxValue());
    q = neg;
    q /= 0;
    BOOST_TEST(q.getValue() == SignedQ8_8Type::QFormatMinValue());

    const UnsignedQ8_8Type uPos(5, 0);
    const UnsignedQ8_8Type uZero(0, 0);
    BOOST_TEST((uPos / uZero).getValue() == UnsignedQ8_8Type::QFormatMaxValue());
    BOOST_TEST((uZero / uZero).getValue() == 0);

    const SignedQ32_32Type widePos(7, 0);
    const SignedQ32_32Type wideNeg(-7, 0);
    const SignedQ32_32Type wideZero(0, 0);
    BOOST_TEST((widePos / wideZero).getValue() == SignedQ32_32Type::QFormatMaxValue());
    BOOST_TEST((wideNeg / wideZero).getValue() == SignedQ32_32Type::QFormatMinValue());
    BOOST_TEST((widePos / 0).getValue() == SignedQ32_32Type::QFormatMaxValue());

    // The policy itself, which is public.
    typedef tinymind::WrapPolicy<int32_t> PolicyType;
    BOOST_TEST(PolicyType::saturate(5, 0, -100, 100, tinymind::DivisionOp) == 100);
    BOOST_TEST(PolicyType::saturate(-5, 0, -100, 100, tinymind::DivisionOp) == -100);
    BOOST_TEST(PolicyType::saturate(9, 2, -100, 100, tinymind::DivisionOp) == 4);
}

// Formats narrower than their storage word (issue #187): Q8.16 lives in an
// int32_t, Q4.8 in an int16_t. Every write path must leave the bits above the
// format's width consistent with its sign, whatever the storage held before.
// Each value is therefore placement-constructed over storage pre-filled with
// 0x00 and again with 0xFF: before the fix the 0x00 fill read negatives back
// as large positives, and the 0xFF fill happened to hide it.
typedef tinymind::QValue<8, 16, true> SignedQ8_16Type;
typedef tinymind::QValue<4, 8, true> SignedQ4_8Type;
typedef tinymind::QValue<4, 8, false> UnsignedQ4_8Type;
typedef tinymind::QValue<4, 8, true, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> SignedQ4_8SatPolicyType;

namespace
{
    const unsigned char StorageFills[] = {0x00, 0xFF};

    template<typename QValueType>
    struct PoisonedStorage
    {
        alignas(QValueType) unsigned char bytes[sizeof(QValueType)];

        explicit PoisonedStorage(const unsigned char fill)
        {
            std::memset(bytes, fill, sizeof(bytes));
        }

        QValueType* fromParts(const typename QValueType::FixedPartFieldType fixedPart, const typename QValueType::FractionalPartFieldType fractionalPart)
        {
            return new (bytes) QValueType(fixedPart, fractionalPart);
        }

        QValueType* defaultConstructed()
        {
            return new (bytes) QValueType;
        }
    };
}

BOOST_AUTO_TEST_CASE(test_case_narrow_format_parts_constructor)
{
    for (const unsigned char fill : StorageFills)
    {
        PoisonedStorage<SignedQ8_16Type> storage(fill);
        const SignedQ8_16Type minusOne = *storage.fromParts(-1, 0);
        const SignedQ8_16Type zero(0, 0);
        const SignedQ8_16Type half(0, 0x8000);

        BOOST_TEST(minusOne.getValue() == -65536);
        BOOST_TEST(minusOne.getFixedPart() == -1);
        BOOST_TEST(minusOne.getFractionalPart() == 0u);
        BOOST_TEST((minusOne < zero));
        BOOST_TEST((zero > minusOne));
        BOOST_TEST((minusOne != zero));
        BOOST_TEST((minusOne + half).getValue() == -32768);
        BOOST_TEST((minusOne - half).getValue() == -98304);

        // -1.5 is fixed part -2 (the floor) plus 0.5.
        PoisonedStorage<SignedQ8_16Type> storage2(fill);
        const SignedQ8_16Type minusOneAndHalf = *storage2.fromParts(-2, 0x8000);
        BOOST_TEST(minusOneAndHalf.getValue() == -98304);
        BOOST_TEST(minusOneAndHalf.getFixedPart() == -2);
        BOOST_TEST(minusOneAndHalf.getFractionalPart() == 0x8000u);

        PoisonedStorage<SignedQ4_8Type> storage3(fill);
        const SignedQ4_8Type minQ4_8 = *storage3.fromParts(-8, 0);
        BOOST_TEST(minQ4_8.getValue() == -2048);
        BOOST_TEST(minQ4_8.getFixedPart() == -8);

        PoisonedStorage<UnsignedQ4_8Type> storage4(fill);
        const UnsignedQ4_8Type maxUQ4_8 = *storage4.fromParts(15, 0xFF);
        BOOST_TEST(maxUQ4_8.getValue() == 4095u);
        BOOST_TEST(maxUQ4_8.getFixedPart() == 15u);
        BOOST_TEST(maxUQ4_8.getFractionalPart() == 0xFFu);
    }
}

BOOST_AUTO_TEST_CASE(test_case_narrow_format_default_constructor_and_set_value)
{
    for (const unsigned char fill : StorageFills)
    {
        PoisonedStorage<SignedQ8_16Type> storage(fill);
        SignedQ8_16Type* value = storage.defaultConstructed();
        BOOST_TEST(value->getValue() == 0);

        value->setValue(-1, 0);
        BOOST_TEST(value->getValue() == -65536);
        BOOST_TEST(value->getFixedPart() == -1);

        // A full-width write with the format's sign bit set but the upper
        // byte clear is the same 24-bit pattern as -1.0, and must read back
        // as -1.0 rather than +255.0.
        value->setValue(static_cast<SignedQ8_16Type::FullWidthValueType>(0x00FF0000));
        BOOST_TEST(value->getValue() == -65536);

        *value = static_cast<SignedQ8_16Type::FullWidthValueType>(0x00FF8000);
        BOOST_TEST(value->getValue() == -32768);

        const SignedQ8_16Type fromRaw(static_cast<SignedQ8_16Type::FullWidthFieldType>(0x00FF0000));
        BOOST_TEST(fromRaw.getValue() == -65536);

        PoisonedStorage<UnsignedQ4_8Type> storage2(fill);
        UnsignedQ4_8Type* unsignedValue = storage2.defaultConstructed();
        BOOST_TEST(unsignedValue->getValue() == 0u);
        unsignedValue->setValue(15, 0xFF);
        BOOST_TEST(unsignedValue->getValue() == 4095u);
    }
}

BOOST_AUTO_TEST_CASE(test_case_narrow_format_convert_from_other_qvalue_type)
{
    const SignedQ8_8Type minusOne(-1, 0);           // -256 in Q8.8
    const SignedQ8_8Type minusOneAndHalf(-2, 0x80); // -384 in Q8.8

    for (const unsigned char fill : StorageFills)
    {
        PoisonedStorage<SignedQ4_8Type> storage(fill);
        SignedQ4_8Type* value = storage.fromParts(0, 0);

        value->convertFromOtherQValueType(minusOne);
        BOOST_TEST(value->getValue() == -256);

        value->convertFromOtherQValueType(minusOneAndHalf);
        BOOST_TEST(value->getValue() == -384);
        BOOST_TEST(value->getFixedPart() == -2);
        BOOST_TEST(value->getFractionalPart() == 0x80u);

        // Widening the fractional part: Q8.8 -1.5 into Q8.16.
        PoisonedStorage<SignedQ8_16Type> storage2(fill);
        SignedQ8_16Type* wide = storage2.fromParts(0, 0);
        wide->convertFromOtherQValueType(minusOneAndHalf);
        BOOST_TEST(wide->getValue() == -98304);
    }
}

BOOST_AUTO_TEST_CASE(test_case_narrow_format_multiply_divide)
{
    for (const unsigned char fill : StorageFills)
    {
        PoisonedStorage<SignedQ8_16Type> storage(fill);
        const SignedQ8_16Type minusOne = *storage.fromParts(-1, 0);
        const SignedQ8_16Type oneAndHalf(1, 0x8000);

        BOOST_TEST((minusOne * oneAndHalf).getValue() == -98304);
        BOOST_TEST((minusOne / oneAndHalf).getValue() == -43690); // -0.6666 truncated
        BOOST_TEST((oneAndHalf * minusOne).getValue() == -98304);
    }
}

BOOST_AUTO_TEST_CASE(test_case_narrow_format_wraps_at_format_width)
{
    // WrapPolicy must wrap at the format's width, not the storage word's.
    // Before the fix, Q8.16 max + 1 LSB read back as +128.0 (0x800000) while
    // its fixed part read -128: the two views of one value disagreed.
    const SignedQ8_16Type maxQ8_16(static_cast<SignedQ8_16Type::FullWidthValueType>(0x7FFFFF));
    const SignedQ8_16Type lsb(static_cast<SignedQ8_16Type::FullWidthValueType>(1));
    const SignedQ8_16Type wrapped = maxQ8_16 + lsb;
    BOOST_TEST(wrapped.getValue() == -8388608);
    BOOST_TEST(wrapped.getFixedPart() == -128);
    BOOST_TEST((wrapped < maxQ8_16));
    BOOST_TEST((wrapped - lsb).getValue() == 0x7FFFFF);

    SignedQ8_16Type incremented(maxQ8_16);
    ++incremented;
    BOOST_TEST(incremented.getValue() == -8323073); // 127.99998 + 1.0 wraps to -127.00002

    // Q4.8: 7.0 * 2.0 = 14.0 does not fit in [-8, 8) and wraps to -2.0.
    const SignedQ4_8Type seven(7, 0);
    const SignedQ4_8Type two(2, 0);
    BOOST_TEST((seven * two).getValue() == -512);

    const UnsignedQ4_8Type maxUQ4_8(15, 0xFF);
    const UnsignedQ4_8Type uLsb(static_cast<UnsignedQ4_8Type::FullWidthValueType>(1));
    BOOST_TEST((maxUQ4_8 + uLsb).getValue() == 0u);
    BOOST_TEST((UnsignedQ4_8Type(0, 0) - uLsb).getValue() == 4095u);

    // Saturation keeps a narrow format inside its bounds as before.
    const SignedQ4_8SatPolicyType satSeven(7, 0);
    const SignedQ4_8SatPolicyType satTwo(2, 0);
    BOOST_TEST((satSeven + satTwo).getValue() == SignedQ4_8SatPolicyType::QFormatMaxValue());
}

BOOST_AUTO_TEST_CASE(test_case_parts_round_trip_exhaustive)
{
    // Every representable value must split into (fixed, fractional) parts that
    // rebuild the same value, for a narrow format (every 12-bit pattern) and a
    // word-filling one (every Q8.8 pattern, guarding the existing layout).
    for (int raw = -2048; raw < 2048; ++raw)
    {
        const SignedQ4_8Type value(static_cast<SignedQ4_8Type::FullWidthValueType>(raw));
        BOOST_TEST_REQUIRE(value.getValue() == raw);
        BOOST_TEST_REQUIRE(value.getFixedPart() == (raw >> 8));
        BOOST_TEST_REQUIRE(static_cast<int>(value.getFractionalPart()) == (raw & 0xFF));
        const SignedQ4_8Type rebuilt(value.getFixedPart(), value.getFractionalPart());
        BOOST_TEST_REQUIRE(rebuilt.getValue() == raw);
    }

    for (int raw = -32768; raw < 32768; ++raw)
    {
        const SignedQ8_8Type value(static_cast<SignedQ8_8Type::FullWidthValueType>(raw));
        BOOST_TEST_REQUIRE(value.getFixedPart() == (raw >> 8));
        BOOST_TEST_REQUIRE(static_cast<int>(value.getFractionalPart()) == (raw & 0xFF));
        const SignedQ8_8Type rebuilt(value.getFixedPart(), value.getFractionalPart());
        BOOST_TEST_REQUIRE(rebuilt.getValue() == raw);
    }

    for (unsigned raw = 0; raw < 4096; ++raw)
    {
        const UnsignedQ4_8Type value(static_cast<UnsignedQ4_8Type::FullWidthValueType>(raw));
        const UnsignedQ4_8Type rebuilt(value.getFixedPart(), value.getFractionalPart());
        BOOST_TEST_REQUIRE(rebuilt.getValue() == raw);
    }
}

BOOST_AUTO_TEST_CASE(test_case_sign_extender_mask_covers_wide_intermediate)
{
    // The mask must set every bit above the sign bit of the 64-bit
    // intermediate. It used to be computed through size_t and lost its upper
    // 32 bits on 32-bit size_t targets; embedded_smoke_test pins the same
    // check on the Cortex-M cross-compile.
    typedef SignedQ8_16Type::MultiplicationResultFullWidthValueType WideType;
    typedef tinymind::SignExtender<WideType, 8, 16, true> ExtenderType;
    static_assert(ExtenderType::SignExtensionBits == static_cast<WideType>(~static_cast<WideType>(0xFFFFFF)), "SignExtender mask truncated");

    WideType value = static_cast<WideType>(0x00FF0000); // -1.0 as a raw 24-bit pattern
    ExtenderType::signExtend(value);
    BOOST_TEST(value == static_cast<WideType>(-65536));
}

BOOST_AUTO_TEST_SUITE_END()
