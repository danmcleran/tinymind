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

#include "activation.hpp"
#include "log.hpp"
#include "sin.hpp"
#include "cos.hpp"

// Fixed-point log, sin and cos over the whole range of a signed Q format.
//
// The activation tables (tanh, sigmoid, exp) cover [-6, 6) and clamp outside
// it, which suits functions that saturate. log and sin/cos do not saturate, so
// they reduce their input first and look up only the reduced value:
//
// - LogLookup<Q>::log(x) writes x = m * 2^e with m in [1, 2), reads log(m)
//   from a table of NUMBER_OF_LOG_TABLE_VALUES knots over [1, 2], and adds
//   e * ln 2. Inputs <= 0 return minus the largest value of the format, the
//   most negative value whose negation is representable, so a cross-entropy
//   that negates the sum does not wrap.
// - TrigLookup<Q>::sin(x) and ::cos(x) reduce x by pi/2, mirror the remainder
//   into [0, pi/2], read a quarter-wave table of NUMBER_OF_TRIG_TABLE_VALUES
//   knots, and set the sign from the quadrant.
//
// Each needs the matching TINYMIND_USE_LOG_<I>_<F>, TINYMIND_USE_SIN_<I>_<F> or
// TINYMIND_USE_COS_<I>_<F> macro and lookupTables.cpp linked, like every
// other lookup-table function. Only the table a call reads is required: sin
// alone needs only the sin table.
namespace tinymind {
    namespace detail
    {
        // ln 2 * 2^56 and pi/2 * 2^62, rounded to nearest. Both leave room in
        // a uint64_t for the products formed below.
        static const uint64_t LookupLn2Q56 = 0xB17217F7D1CF7AULL;
        static const uint64_t LookupHalfPiQ62 = 0x6487ED5110B4611AULL;

        // value * 2^-shift for shift >= 0 (rounded to nearest) or
        // value * 2^-shift for shift < 0, as long as the result stays below
        // 2^64; `overflow` reports when it would not. The shift is a run-time
        // argument so that a branch a format never takes does not compile a
        // shift by a negative or oversized constant.
        inline uint64_t lookupScaleUnsigned(const uint64_t value, const int shift, bool& overflow)
        {
            overflow = false;

            if (shift > 0)
            {
                const unsigned amount = static_cast<unsigned>(shift);
                if (amount >= 64u)
                {
                    return 0u;
                }

                return (value >> amount) + ((value >> (amount - 1u)) & 1u);
            }
            else if (shift < 0)
            {
                const unsigned amount = static_cast<unsigned>(-shift);
                if ((amount >= 64u) || (value > (UINT64_MAX >> amount)))
                {
                    overflow = true;
                    return UINT64_MAX;
                }

                return value << amount;
            }

            return value;
        }

        // numerator / denominator rounded to nearest, ties away from zero;
        // denominator > 0.
        template<typename Wide>
        Wide lookupRoundDivide(const Wide& numerator, const Wide& denominator)
        {
            const Wide half = static_cast<Wide>(denominator / 2);

            if (numerator < 0)
            {
                return static_cast<Wide>(-((-numerator + half) / denominator));
            }

            return static_cast<Wide>((numerator + half) / denominator);
        }

        // floor(numerator / denominator); denominator > 0.
        template<typename Wide>
        Wide lookupFloorDivide(const Wide& numerator, const Wide& denominator)
        {
            Wide quotient = static_cast<Wide>(numerator / denominator);

            if ((numerator < 0) && ((quotient * denominator) != numerator))
            {
                quotient = static_cast<Wide>(quotient - 1);
            }

            return quotient;
        }

        template<typename ValueType>
        struct LookupFormat
        {
            typedef typename ValueType::FullWidthValueType Raw;
            typedef typename ValueType::FullWidthFieldType Field;
            typedef typename ValueType::MultiplicationResultFullWidthValueType Wide;

            static const unsigned Bits = ValueType::NumberOfFixedBits + ValueType::NumberOfFractionalBits;
            static const unsigned FractionalBits = ValueType::NumberOfFractionalBits;

            static_assert(ValueType::IsSigned, "The log, sin and cos lookups are generated for signed formats only.");
            static_assert((8u * sizeof(Wide)) >= static_cast<size_t>(2u * Bits), "The lookups need an intermediate twice the width of the format.");

            static Wide maxRaw()
            {
                return static_cast<Wide>((static_cast<Wide>(1) << (Bits - 1u)) - 1);
            }

            static Wide minRaw()
            {
                return static_cast<Wide>(-maxRaw() - 1);
            }

            static Wide entry(Field const* const table, const size_t index)
            {
                return static_cast<Wide>(static_cast<Raw>(table[index]));
            }

            static ValueType fromWide(const Wide& raw)
            {
                ValueType result;
                result.setValue(static_cast<Raw>(raw));
                return result;
            }
        };
    }

    template<typename ValueType>
    struct LogLookup
    {
        typedef detail::LookupFormat<ValueType> Format;
        typedef typename Format::Raw Raw;
        typedef typename Format::Field Field;
        typedef typename Format::Wide Wide;

        static ValueType log(const ValueType& x)
        {
            typedef typename LogValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::LogTableType TableType;
            static const TableType table;

            const Raw raw = x.getValue();

            if (raw <= 0)
            {
                return Format::fromWide(static_cast<Wide>(-Format::maxRaw()));
            }

            // x = 2^(msb - FractionalBits) * (raw / 2^msb), with raw / 2^msb
            // in [1, 2). The bits below the leading one are the mantissa's
            // fraction: its top LOG_TABLE_SHIFT bits pick the knot and the
            // rest interpolate between that knot and the next.
            const Field bits = static_cast<Field>(raw);
            unsigned msb = 0;
            for (Field v = bits; v > 1u; v = static_cast<Field>(v >> 1))
            {
                ++msb;
            }

            const Field fraction = static_cast<Field>(bits - static_cast<Field>(static_cast<Field>(1) << msb));
            size_t index;
            Wide remainder = 0;
            Wide span = 1;

            if (msb >= LOG_TABLE_SHIFT)
            {
                const unsigned low = msb - LOG_TABLE_SHIFT;
                index = static_cast<size_t>(fraction >> low);
                span = static_cast<Wide>(static_cast<Wide>(1) << low);
                remainder = static_cast<Wide>(static_cast<Wide>(fraction) & static_cast<Wide>(span - 1));
            }
            else
            {
                index = static_cast<size_t>(fraction) << (LOG_TABLE_SHIFT - msb);
            }

            const Wide y0 = Format::entry(&table.values[0], index);
            const Wide y1 = Format::entry(&table.values[0], index + 1u);
            const Wide logMantissa = static_cast<Wide>(y0 + detail::lookupRoundDivide<Wide>(static_cast<Wide>((y1 - y0) * remainder), span));

            // exponent * ln 2, from ln 2 in Q56 rescaled to the format.
            const int exponent = static_cast<int>(msb) - static_cast<int>(Format::FractionalBits);
            const uint64_t magnitude = static_cast<uint64_t>(exponent < 0 ? -exponent : exponent);
            bool overflow;
            const uint64_t scaled = detail::lookupScaleUnsigned(magnitude * detail::LookupLn2Q56, 56 - static_cast<int>(Format::FractionalBits), overflow);
            // Past the format's range the exponent term is pinned at twice the
            // range, so adding log(m) in [0, ln 2) cannot pull it back inside
            // and the clamp below saturates it.
            const Wide limit = static_cast<Wide>(2 * (Format::maxRaw() + 1));
            Wide exponentPart = (overflow || (static_cast<Wide>(scaled >> (Format::Bits - 1u)) != 0)) ? limit : static_cast<Wide>(scaled);
            if (exponent < 0)
            {
                exponentPart = static_cast<Wide>(-exponentPart);
            }

            Wide result = static_cast<Wide>(exponentPart + logMantissa);
            if (result > Format::maxRaw())
            {
                result = Format::maxRaw();
            }
            else if (result < Format::minRaw())
            {
                result = Format::minRaw();
            }

            return Format::fromWide(result);
        }
    };

    template<typename ValueType>
    struct TrigLookup
    {
        typedef detail::LookupFormat<ValueType> Format;
        typedef typename Format::Raw Raw;
        typedef typename Format::Field Field;
        typedef typename Format::Wide Wide;

        static ValueType sin(const ValueType& x)
        {
            typedef typename SinValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::SinTableType TableType;
            static const TableType table;

            return evaluate(x, &table.values[0], false);
        }

        static ValueType cos(const ValueType& x)
        {
            typedef typename CosValuesTableSelector<ValueType::NumberOfFixedBits, ValueType::NumberOfFractionalBits, ValueType::IsSigned>::CosTableType TableType;
            static const TableType table;

            return evaluate(x, &table.values[0], true);
        }

    private:
        // The reduction works in Q(ReductionBits) inside the double-width
        // type: the input is scaled up to that many fractional bits and pi/2
        // is carried to the same precision, so the remainder of a large input
        // is off by well under one LSB of the format.
        static const unsigned ReductionBits = (Format::FractionalBits > (Format::Bits - 2u)) ? Format::FractionalBits : (Format::Bits - 2u);

        static ValueType evaluate(const ValueType& x, Field const* const table, const bool isCosine)
        {
            bool overflow;
            const Wide halfPi = static_cast<Wide>(detail::lookupScaleUnsigned(detail::LookupHalfPiQ62, 62 - static_cast<int>(ReductionBits), overflow));
            const Wide scale = static_cast<Wide>(static_cast<Wide>(1) << (ReductionBits - Format::FractionalBits));
            const Wide input = static_cast<Wide>(static_cast<Wide>(x.getValue()) * scale);

            const Wide quarterTurns = detail::lookupFloorDivide<Wide>(input, halfPi);
            Wide remainder = static_cast<Wide>(input - (quarterTurns * halfPi));
            const unsigned quadrant = static_cast<unsigned>(((quarterTurns % 4) + 4) % 4);

            // sin and cos over [pi/2, pi) and [3pi/2, 2pi) mirror the first
            // quarter: f(pi/2 + r) uses the table at pi/2 - r.
            if ((quadrant & 1u) != 0u)
            {
                remainder = static_cast<Wide>(halfPi - remainder);
            }

            const Wide position = static_cast<Wide>(remainder * static_cast<Wide>(1 << TRIG_TABLE_SHIFT));
            const Wide knot = static_cast<Wide>(position / halfPi);
            Wide value;

            if (knot >= static_cast<Wide>(NUMBER_OF_TRIG_TABLE_VALUES - 1))
            {
                value = Format::entry(table, NUMBER_OF_TRIG_TABLE_VALUES - 1);
            }
            else
            {
                const size_t index = static_cast<size_t>(knot);
                const Wide offset = static_cast<Wide>(position - (knot * halfPi));
                const Wide y0 = Format::entry(table, index);
                const Wide y1 = Format::entry(table, index + 1u);
                value = static_cast<Wide>(y0 + detail::lookupRoundDivide<Wide>(static_cast<Wide>((y1 - y0) * offset), halfPi));
            }

            // sin is negative in quadrants 2 and 3, cos in quadrants 1 and 2.
            const bool negate = isCosine ? ((quadrant == 1u) || (quadrant == 2u)) : (quadrant >= 2u);

            return Format::fromWide(negate ? static_cast<Wide>(-value) : value);
        }
    };
}
