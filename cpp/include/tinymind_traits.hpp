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

#include "tinymind_platform.hpp"

/*
 * Minimal type traits for tinymind's SFINAE dispatches.
 *
 * The standalone composable layers (BatchNorm1D, Dropout, BinaryLayer,
 * TernaryLayer, MaxPool*, BSpline) use enable_if + is_floating_point to
 * pick between a float/double constructor path and a QValue (raw, fractional)
 * constructor path inside their fromInteger() helpers.
 *
 * When TINYMIND_ENABLE_STD=1, these resolve to std::enable_if /
 * std::is_floating_point. When TINYMIND_ENABLE_STD=0, hand-rolled
 * equivalents are provided so headers used by embedded targets do not
 * include <type_traits> or reference namespace std::.
 *
 * Only the subset actually used by tinymind is provided.
 */

#if TINYMIND_ENABLE_STD
#include <type_traits>
#endif

namespace tinymind {

#if TINYMIND_ENABLE_STD

    template<bool B, typename T = void>
    using enable_if = std::enable_if<B, T>;

    template<typename T>
    using is_floating_point = std::is_floating_point<T>;

#else

    template<bool B, typename T = void>
    struct enable_if {};

    template<typename T>
    struct enable_if<true, T> { typedef T type; };

    template<typename T>
    struct is_floating_point { static const bool value = false; };

    template<> struct is_floating_point<float>       { static const bool value = true; };
    template<> struct is_floating_point<double>      { static const bool value = true; };
    template<> struct is_floating_point<long double> { static const bool value = true; };

    template<typename T> struct is_floating_point<const T>          : is_floating_point<T> {};
    template<typename T> struct is_floating_point<volatile T>       : is_floating_point<T> {};
    template<typename T> struct is_floating_point<const volatile T> : is_floating_point<T> {};

#endif

    /*
     * Fixed-point hyperparameter defaults.
     *
     * The optimizer, weight-decay and schedule policies take each constant as
     * an (IntegerPart, FractionalPart) pair for the QValue(int, fraction)
     * constructor, where FractionalPart is a raw count of LSBs in ValueType's
     * own format (230 means 230/256 in Q8.8 but 230/65536 in Q16.16). Their
     * defaults used to be written as Q8.8 raw counts, which made every other
     * format's defaults wrong (Adam beta1 = 0.0035 in Q16.16).
     *
     * QFractionLsbs<ValueType, Num, Den>::value is the raw fractional field
     * for Num/Den in ValueType's format: round(Num/Den * 2^F), clamped to
     * [1, 2^F - 1] so a rate never rounds to exactly 0 or 1 (0.999 is 1.0
     * in Q8.8, and 1e-8 is 0 in every format). Num must be less than Den.
     * For a type with no NumberOfFractionalBits (float, double) the value is
     * 1 and unused: the floating-point policies take real constants.
     */
    template<typename T>
    struct fractional_bits_void
    {
        typedef void type;
    };

    template<typename T, typename = void>
    struct fractional_bits_of
    {
        static const unsigned value = 0;
    };

    template<typename T>
    struct fractional_bits_of<T, typename fractional_bits_void<decltype(T::NumberOfFractionalBits)>::type>
    {
        static const unsigned value = static_cast<unsigned>(T::NumberOfFractionalBits);
    };

    namespace detail
    {
        // Long division, so 2^F never has to be formed (F may be 64).
        constexpr unsigned long long fractionLsbs(const unsigned long long num, const unsigned long long den, const unsigned bits)
        {
            const unsigned long long maxLsbs = (bits >= 64u) ? ~0ULL : ((1ULL << bits) - 1ULL);
            unsigned long long quotient = 0;
            unsigned long long remainder = num % den;
            for (unsigned bit = 0; bit < bits; ++bit)
            {
                remainder <<= 1;
                quotient <<= 1;
                if (remainder >= den)
                {
                    remainder -= den;
                    quotient |= 1ULL;
                }
            }
            // Round half up, without wrapping past the largest fraction.
            if (((remainder << 1) >= den) && (quotient < maxLsbs))
            {
                ++quotient;
            }
            if (quotient > maxLsbs)
            {
                quotient = maxLsbs;
            }
            if (quotient == 0)
            {
                quotient = 1;
            }
            return quotient;
        }
    }

    template<typename ValueType, unsigned long long Num, unsigned long long Den>
    struct QFractionLsbs
    {
        static_assert(Den > 0, "Denominator must be positive.");
        static_assert(Num < Den, "QFractionLsbs encodes a value in (0, 1).");

        static const unsigned long long value = detail::fractionLsbs(Num, Den, fractional_bits_of<ValueType>::value);
    };

    // Definitions, so value can also be bound to a reference (a static const
    // member is not implicitly inline in any standard). A template's static
    // member may be defined in a header.
    template<typename ValueType, unsigned long long Num, unsigned long long Den>
    const unsigned long long QFractionLsbs<ValueType, Num, Den>::value;

    template<typename T, typename Enable>
    const unsigned fractional_bits_of<T, Enable>::value;

    template<typename T>
    const unsigned fractional_bits_of<T, typename fractional_bits_void<decltype(T::NumberOfFractionalBits)>::type>::value;

}
