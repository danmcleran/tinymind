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

// qformat_oracle_unit_test.cpp : Differential oracle for tinymind::QValue.
//
// Every public QValue operation is run against an exact reference computed in
// 128-bit integers, across a matrix of formats (narrow and word-filling,
// signed and unsigned, 1-bit fixed or fractional parts, 8- to 64-bit storage)
// and all four rounding x saturation policy combinations. Formats of 8 bits or
// fewer are checked exhaustively over every operand pair; wider formats over
// edge values plus a fixed-seed random sample.
//
// The reference states what each operation should compute. Where QValue is
// known to compute something else, the difference is written down as a
// KnownDefect with a model that reproduces the current behaviour exactly. Each
// mismatch is classified against those models:
//
//   - matches the intended result            -> pass
//   - matches a combination of known defects -> charged to every defect that
//                                               all such combinations share
//   - matches nothing                        -> FAIL (a new defect)
//
// and a known defect that no result needs any more also FAILS, so fixing one
// forces its model out of this file. The list below is therefore a ratchet:
// it can only shrink, and it is always an exact description of how QValue
// departs from its intended arithmetic.
//
// Inputs that are undefined behaviour in QValue would be skipped rather than
// modelled and listed in skippedInputs(); there are none at present.
//
// Besides single operations, the oracle runs chains: one value carried
// through a random sequence of compound assignments (including aliased ones
// such as q *= q), integer operands, steps and a parts round trip, each step
// checked against the reference. Chains reach values that arithmetic
// produces (rounded products, saturated bounds, repeated quotients) rather
// than only values drawn directly.
//
// Soak mode. The default run uses fixed seeds, so every PR checks the same
// inputs. Setting TINYMIND_ORACLE_SEED to a nonzero value draws a different
// random sample, and TINYMIND_ORACLE_SCALE multiplies every random count; the
// edge values and the reference are unchanged. The nightly soak job runs with
// a fresh seed and a large scale. Every run prints its seed and scale, and
// rerunning with the same two values reproduces it exactly.

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "qformat.hpp"

#define BOOST_TEST_MODULE qformat oracle
#include <boost/test/included/unit_test.hpp>

#ifndef __SIZEOF_INT128__
#error "The QValue oracle needs 128-bit integers for its exact reference."
#endif

namespace
{
    typedef __int128_t  Wide;
    typedef __uint128_t UWide;

    // === Known defects =======================================================

    // Every defect this oracle has found so far is fixed, so the list is
    // empty. When a new one is found and its fix deferred, add an enumerator
    // here (before NumberOfKnownDefects), its name to KnownDefectNames, and a
    // `has(...)` branch in Model that reproduces the current behaviour
    // exactly; the ratchet then tolerates exactly that behaviour until the
    // fix lands.
    enum KnownDefect
    {
        NumberOfKnownDefects
    };

    const std::vector<const char*> KnownDefectNames;

    typedef unsigned DefectMask;

    const char* skippedInputs()
    {
        return "none";
    }

    // === Exact reference arithmetic ==========================================

    struct Format
    {
        unsigned fixedBits;
        unsigned fractionalBits;
        unsigned storageBits;
        bool     isSigned;
        bool     roundUp;
        bool     saturate;

        unsigned width() const
        {
            return fixedBits + fractionalBits;
        }
    };

    std::string toString(Wide value)
    {
        if (value == 0)
        {
            return "0";
        }

        const bool negative = (value < 0);
        UWide magnitude = negative ? (static_cast<UWide>(0) - static_cast<UWide>(value)) : static_cast<UWide>(value);
        std::string digits;

        while (magnitude != 0)
        {
            digits.insert(digits.begin(), static_cast<char>('0' + static_cast<int>(magnitude % 10)));
            magnitude /= 10;
        }

        return negative ? ("-" + digits) : digits;
    }

    std::string formatName(const Format& f)
    {
        char buffer[96];
        std::snprintf(buffer, sizeof(buffer), "%sQ%u.%u/%s/%s",
                      f.isSigned ? "" : "U", f.fixedBits, f.fractionalBits,
                      f.roundUp ? "RoundUp" : "Truncate",
                      f.saturate ? "MinMaxSaturate" : "Wrap");
        return buffer;
    }

    Wide powerOfTwo(const unsigned exponent)
    {
        return static_cast<Wide>(1) << exponent;
    }

    // x reduced modulo 2^bits into the signed or unsigned range of that width.
    Wide wrapTo(const Wide x, const unsigned bits, const bool isSigned)
    {
        const UWide mask = (static_cast<UWide>(1) << bits) - 1;
        const UWide low = static_cast<UWide>(x) & mask;

        if (isSigned && (((low >> (bits - 1)) & 1) != 0))
        {
            return static_cast<Wide>(low) - powerOfTwo(bits);
        }

        return static_cast<Wide>(low);
    }

    Wide floorDiv(const Wide x, const Wide divisor)
    {
        Wide quotient = x / divisor;

        if (((x % divisor) != 0) && (x < 0))
        {
            --quotient;
        }

        return quotient;
    }

    Wide floorMod(const Wide x, const Wide divisor)
    {
        return x - (floorDiv(x, divisor) * divisor);
    }

    Wide trueMin(const Format& f)
    {
        return f.isSigned ? -powerOfTwo(f.width() - 1) : 0;
    }

    Wide trueMax(const Format& f)
    {
        return f.isSigned ? (powerOfTwo(f.width() - 1) - 1) : (powerOfTwo(f.width()) - 1);
    }

    // What QValue should compute, optionally with a set of known defects
    // switched on so their current behaviour can be reproduced exactly.
    struct Model
    {
        const Format& f;
        DefectMask    defects;

        Model(const Format& format, const DefectMask mask) : f(format), defects(mask)
        {
        }

        bool has(const KnownDefect defect) const
        {
            return ((defects >> defect) & 1u) != 0;
        }

        Wide minValue() const
        {
            return trueMin(f);
        }

        Wide maxValue() const
        {
            return trueMax(f);
        }

        Wide wrap(const Wide x) const
        {
            return wrapTo(x, f.width(), f.isSigned);
        }

        Wide clamp(const Wide x) const
        {
            const Wide lo = minValue();
            const Wide hi = maxValue();

            return (x < lo) ? lo : ((x > hi) ? hi : x);
        }

        // Result of an operation whose exact value is known: wrap or clamp.
        Wide finish(const Wide exact) const
        {
            return f.saturate ? clamp(exact) : wrap(exact);
        }

        // Addition and subtraction take the accumulator `o` (always in range)
        // and an addend `t`. MinMaxSaturatePolicy clamps only towards the bound
        // the addend moves to; with the true bounds that equals clamp(o + t),
        // and the transcription keeps the known-defect bounds exact too.
        Wide add(const Wide o, const Wide t) const
        {
            if (!f.saturate)
            {
                return wrap(o + t);
            }

            const Wide lo = minValue();
            const Wide hi = maxValue();

            if (t > 0)
            {
                return (o > (hi - t)) ? hi : (o + t);
            }
            else if (t < 0)
            {
                return (o < (lo - t)) ? lo : (o + t);
            }

            return o;
        }

        Wide sub(const Wide o, const Wide t) const
        {
            if (!f.saturate)
            {
                return wrap(o - t);
            }

            const Wide lo = minValue();
            const Wide hi = maxValue();

            if (t > 0)
            {
                return (o < (lo + t)) ? lo : (o - t);
            }
            else if (t < 0)
            {
                return (o > (hi + t)) ? hi : (o - t);
            }

            return o;
        }

        // The full product rounded to the format: floor for TruncatePolicy,
        // round half up for RoundUpPolicy.
        Wide roundedProduct(const Wide a, const Wide b) const
        {
            const unsigned frac = f.fractionalBits;

            if (!f.isSigned)
            {
                // Two 64-bit unsigned operands overflow a signed 128-bit
                // product, so the unsigned case multiplies unsigned.
                const UWide product = static_cast<UWide>(a) * static_cast<UWide>(b);
                UWide result = product >> frac;

                if (f.roundUp && (((product >> (frac - 1)) & 1) != 0))
                {
                    ++result;
                }

                return static_cast<Wide>(result);
            }

            const Wide product = a * b;
            const Wide scale = powerOfTwo(frac);
            Wide result = floorDiv(product, scale);

            if (f.roundUp && (floorMod(product, scale) >= (scale / 2)))
            {
                ++result;
            }

            return result;
        }

        Wide mul(const Wide a, const Wide b) const
        {
            return finish(roundedProduct(a, b));
        }

        // A zero divisor under both policies: the bound for the numerator's
        // sign, and 0 for 0 / 0.
        Wide divByZero(const Wide a) const
        {
            return (a > 0) ? maxValue() : ((a < 0) ? minValue() : 0);
        }

        // Quotient truncated towards zero, as integer division does.
        Wide div(const Wide a, const Wide b) const
        {
            if (b == 0)
            {
                return divByZero(a);
            }

            return finish((a * powerOfTwo(f.fractionalBits)) / b);
        }

        Wide one() const
        {
            return powerOfTwo(f.fractionalBits);
        }


        // An int operand of the whole-number operators: exact.
        Wide intOperand(const Wide n) const
        {
            return n;
        }

        Wide addInt(const Wide a, const Wide n) const
        {
            return add(a, intOperand(n) * one());
        }

        Wide subInt(const Wide a, const Wide n) const
        {
            return sub(a, intOperand(n) * one());
        }

        Wide mulInt(const Wide a, const Wide n) const
        {
            return finish(a * intOperand(n));
        }

        // Integer divisor: the scaled quotient a*2^F / (n*2^F) is a / n.
        Wide divInt(const Wide a, const Wide n) const
        {
            const Wide divisor = intOperand(n);

            if (divisor == 0)
            {
                return divByZero(a);
            }

            return finish(a / divisor);
        }

        Wide compareMask(const Wide a, const Wide b) const
        {
            return (a == b ? 1 : 0) | (a != b ? 2 : 0) | (a < b ? 4 : 0) |
                   (a > b ? 8 : 0) | (a <= b ? 16 : 0) | (a >= b ? 32 : 0);
        }

        // <, >, <=, >= against an int (bits 2..5 of compareMask).
        Wide compareInt(const Wide a, const Wide n) const
        {
            return compareMask(a, intOperand(n) * one()) & 60;
        }

        Wide fromParts(const Wide fixedPart, const Wide fractionalPart) const
        {
            const Wide fixedValue = wrapTo(fixedPart, f.fixedBits, f.isSigned);
            const Wide fractionalValue = floorMod(fractionalPart, one());

            return wrap((fixedValue * one()) + fractionalValue);
        }
    };

    // === Classification ======================================================

    struct Tally
    {
        unsigned long long checks;
        unsigned long long explained;
        unsigned long long unknown;
        std::vector<unsigned long long> hits;
        std::vector<std::string> firstHit;
        std::vector<std::string> unknownExamples;

        Tally() : checks(0), explained(0), unknown(0), hits(NumberOfKnownDefects, 0), firstHit(NumberOfKnownDefects)
        {
        }

        void merge(const Tally& other)
        {
            checks += other.checks;
            explained += other.explained;
            unknown += other.unknown;

            for (unsigned d = 0; d < NumberOfKnownDefects; ++d)
            {
                hits[d] += other.hits[d];

                if (firstHit[d].empty())
                {
                    firstHit[d] = other.firstHit[d];
                }
            }

            for (const std::string& example : other.unknownExamples)
            {
                if (unknownExamples.size() < 40)
                {
                    unknownExamples.push_back(example);
                }
            }
        }
    };

    std::string describe(const Format& f, const char* property, const std::vector<Wide>& inputs, const Wide actual, const Wide expected)
    {
        std::string text = formatName(f) + " " + property + "(";

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            text += (i == 0 ? "" : ", ") + toString(inputs[i]);
        }

        return text + ") = " + toString(actual) + ", expected " + toString(expected);
    }

    template<typename Expectation>
    void check(Tally& tally, const Format& f, const char* property, const std::vector<Wide>& inputs, const Wide actual, Expectation expectation)
    {
        ++tally.checks;

        const Wide expected = expectation(Model(f, 0));

        if (actual == expected)
        {
            return;
        }

        // Try every combination of known defects. A mismatch is explained if
        // any combination reproduces it, and a defect is charged only when it
        // is in every combination that does: models can coincide (Q1.7's
        // clipped maximum is 0, so a clamp to it looks like a wrapped int
        // operand), and charging the first match would keep a fixed defect
        // looking live. Charging only required defects is what makes the
        // staleness check sound.
        const DefectMask allDefects = (1u << NumberOfKnownDefects) - 1;
        DefectMask required = allDefects;
        bool explained = false;

        for (DefectMask mask = 1; mask <= allDefects; ++mask)
        {
            if (actual == expectation(Model(f, mask)))
            {
                explained = true;
                required &= mask;
            }
        }

        if (explained)
        {
            ++tally.explained;

            for (unsigned d = 0; d < NumberOfKnownDefects; ++d)
            {
                if (((required >> d) & 1u) != 0)
                {
                    ++tally.hits[d];

                    if (tally.firstHit[d].empty())
                    {
                        tally.firstHit[d] = describe(f, property, inputs, actual, expected);
                    }
                }
            }

            return;
        }

        ++tally.unknown;

        // One example per format and property, so distinct defects stay visible.
        const std::string key = formatName(f) + " " + property + "(";

        for (const std::string& example : tally.unknownExamples)
        {
            if (example.compare(0, key.size(), key) == 0)
            {
                return;
            }
        }

        if (tally.unknownExamples.size() < 40)
        {
            tally.unknownExamples.push_back(describe(f, property, inputs, actual, expected));
        }
    }

    // === Adapters: one QValue instantiation behind plain functions ===========
    // Keeping the per-format template code to these thin wrappers lets the
    // engine below stay a single non-template function, which keeps a matrix
    // of ~130 instantiations cheap to compile.

    struct Ops
    {
        Wide (*minValue)();
        Wide (*maxValue)();
        Wide (*defaultValue)();
        Wide (*fromRawField)(Wide);
        Wide (*setRaw)(Wide);
        Wide (*assignRaw)(Wide);
        Wide (*fromParts)(Wide, Wide);
        Wide (*setParts)(Wide, Wide);
        Wide (*fixedPart)(Wide);
        Wide (*fractionalPart)(Wide);
        Wide (*add)(Wide, Wide);
        Wide (*sub)(Wide, Wide);
        Wide (*mul)(Wide, Wide);
        Wide (*div)(Wide, Wide);
        Wide (*compare)(Wide, Wide);
        Wide (*preIncrement)(Wide);
        Wide (*preDecrement)(Wide);
        Wide (*postIncrementResult)(Wide);
        Wide (*postIncrementAfter)(Wide);
        Wide (*postDecrementResult)(Wide);
        Wide (*postDecrementAfter)(Wide);
        Wide (*addInt)(Wide, int);
        Wide (*subInt)(Wide, int);
        Wide (*mulInt)(Wide, int);
        Wide (*divInt)(Wide, int);
        Wide (*compareInt)(Wide, int);
        Wide (*mulAssignInt)(Wide, int);
        Wide (*divAssignInt)(Wide, int);
        Wide (*addAssign)(Wide, Wide);
        Wide (*subAssign)(Wide, Wide);
        Wide (*mulAssign)(Wide, Wide);
        Wide (*divAssign)(Wide, Wide);
        Wide (*selfAdd)(Wide);
        Wide (*selfSub)(Wide);
        Wide (*selfMul)(Wide);
        Wide (*selfDiv)(Wide);
        Wide (*partsRoundTrip)(Wide);
    };

    template<typename Q>
    struct Adapter
    {
        typedef typename Q::FullWidthValueType      ValueType;
        typedef typename Q::FullWidthFieldType      FieldType;
        typedef typename Q::FixedPartFieldType      FixedType;
        typedef typename Q::FractionalPartFieldType FractionalType;

        static Format format()
        {
            Format f;
            f.fixedBits = Q::NumberOfFixedBits;
            f.fractionalBits = Q::NumberOfFractionalBits;
            f.storageBits = static_cast<unsigned>(8 * sizeof(ValueType));
            f.isSigned = Q::IsSigned;
            f.roundUp = std::is_same<typename Q::RoundingPolicy,
                                     tinymind::RoundUpPolicy<typename Q::MultiplicationResultFullWidthValueType, Q::NumberOfFractionalBits> >::value;
            f.saturate = std::is_same<typename Q::SaturatePolicy, tinymind::MinMaxSaturatePolicy<ValueType> >::value;
            return f;
        }

        // An in-range raw value as a QValue.
        static Q make(const Wide v)
        {
            Q q;
            q.setValue(static_cast<ValueType>(v));
            return q;
        }

        static Wide value(const Q& q)
        {
            return static_cast<Wide>(q.getValue());
        }

        static Wide minValue()                         { return static_cast<Wide>(Q::QFormatMinValue()); }
        static Wide maxValue()                         { return static_cast<Wide>(Q::QFormatMaxValue()); }
        static Wide defaultValue()                     { return value(Q()); }
        static Wide fromRawField(const Wide word)      { return value(Q(static_cast<FieldType>(word))); }

        static Wide setRaw(const Wide word)
        {
            Q q;
            q.setValue(static_cast<ValueType>(word));
            return value(q);
        }

        static Wide assignRaw(const Wide word)
        {
            Q q;
            q = static_cast<ValueType>(word);
            return value(q);
        }

        static Wide fromParts(const Wide fixedPart, const Wide fractionalPart)
        {
            return value(Q(static_cast<FixedType>(fixedPart), static_cast<FractionalType>(fractionalPart)));
        }

        static Wide setParts(const Wide fixedPart, const Wide fractionalPart)
        {
            Q q;
            q.setValue(static_cast<FixedType>(fixedPart), static_cast<FractionalType>(fractionalPart));
            return value(q);
        }

        static Wide fixedPart(const Wide v)            { return static_cast<Wide>(make(v).getFixedPart()); }
        static Wide fractionalPart(const Wide v)       { return static_cast<Wide>(make(v).getFractionalPart()); }
        static Wide add(const Wide a, const Wide b)    { return value(make(a) + make(b)); }
        static Wide sub(const Wide a, const Wide b)    { return value(make(a) - make(b)); }
        static Wide mul(const Wide a, const Wide b)    { return value(make(a) * make(b)); }
        static Wide div(const Wide a, const Wide b)    { return value(make(a) / make(b)); }

        static Wide compare(const Wide a, const Wide b)
        {
            const Q x = make(a);
            const Q y = make(b);

            return (x == y ? 1 : 0) | (x != y ? 2 : 0) | (x < y ? 4 : 0) |
                   (x > y ? 8 : 0) | (x <= y ? 16 : 0) | (x >= y ? 32 : 0);
        }

        static Wide preIncrement(const Wide a)         { Q q = make(a); ++q; return value(q); }
        static Wide preDecrement(const Wide a)         { Q q = make(a); --q; return value(q); }
        static Wide postIncrementResult(const Wide a)  { Q q = make(a); return value(q++); }
        static Wide postIncrementAfter(const Wide a)   { Q q = make(a); q++; return value(q); }
        static Wide postDecrementResult(const Wide a)  { Q q = make(a); return value(q--); }
        static Wide postDecrementAfter(const Wide a)   { Q q = make(a); q--; return value(q); }
        static Wide addInt(const Wide a, const int n)  { return value(make(a) + n); }
        static Wide subInt(const Wide a, const int n)  { return value(make(a) - n); }
        static Wide mulInt(const Wide a, const int n)  { return value(make(a) * n); }
        static Wide divInt(const Wide a, const int n)  { return value(make(a) / n); }

        static Wide compareInt(const Wide a, const int n)
        {
            const Q x = make(a);

            return (x < n ? 4 : 0) | (x > n ? 8 : 0) | (x <= n ? 16 : 0) | (x >= n ? 32 : 0);
        }

        static Wide mulAssignInt(const Wide a, const int n) { Q q = make(a); q *= n; return value(q); }
        static Wide divAssignInt(const Wide a, const int n) { Q q = make(a); q /= n; return value(q); }
        static Wide addAssign(const Wide a, const Wide b)   { Q q = make(a); q += make(b); return value(q); }
        static Wide subAssign(const Wide a, const Wide b)   { Q q = make(a); q -= make(b); return value(q); }
        static Wide mulAssign(const Wide a, const Wide b)   { Q q = make(a); q *= make(b); return value(q); }
        static Wide divAssign(const Wide a, const Wide b)   { Q q = make(a); q /= make(b); return value(q); }

        // The operand is the object itself, so a compound assignment that
        // wrote its result before reading the operand would show up here.
        static Wide selfAdd(const Wide a)                   { Q q = make(a); q += q; return value(q); }
        static Wide selfSub(const Wide a)                   { Q q = make(a); q -= q; return value(q); }
        static Wide selfMul(const Wide a)                   { Q q = make(a); q *= q; return value(q); }
        static Wide selfDiv(const Wide a)                   { Q q = make(a); q /= q; return value(q); }

        static Wide partsRoundTrip(const Wide a)
        {
            Q q = make(a);
            q.setValue(q.getFixedPart(), q.getFractionalPart());
            return value(q);
        }

        static Ops ops()
        {
            const Ops o =
            {
                &minValue, &maxValue, &defaultValue, &fromRawField, &setRaw, &assignRaw,
                &fromParts, &setParts, &fixedPart, &fractionalPart,
                &add, &sub, &mul, &div, &compare,
                &preIncrement, &preDecrement,
                &postIncrementResult, &postIncrementAfter, &postDecrementResult, &postDecrementAfter,
                &addInt, &subInt, &mulInt, &divInt, &compareInt,
                &mulAssignInt, &divAssignInt,
                &addAssign, &subAssign, &mulAssign, &divAssign,
                &selfAdd, &selfSub, &selfMul, &selfDiv, &partsRoundTrip,
            };
            return o;
        }
    };

    // === Run configuration ===================================================

    struct RunConfig
    {
        uint64_t seed;  // 0: the fixed per-PR sample
        unsigned scale; // multiplies every random count
    };

    // A whole-number environment variable, or `fallback` when it is unset or
    // empty. Anything else that does not parse aborts the run: a soak that
    // silently fell back to the default sample would pass while testing
    // nothing new.
    uint64_t environmentNumber(const char* name, const uint64_t fallback, const uint64_t maximum)
    {
        const char* text = std::getenv(name);

        if ((text == nullptr) || (*text == '\0'))
        {
            return fallback;
        }

        char* end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(text, &end, 0);

        if ((errno != 0) || (*end != '\0') || (*text == '-') || (value > maximum))
        {
            std::fprintf(stderr, "%s=\"%s\" is not a whole number in [0, %llu]\n", name, text, static_cast<unsigned long long>(maximum));
            std::exit(2);
        }

        return value;
    }

    const RunConfig& runConfig()
    {
        static const RunConfig config =
        {
            environmentNumber("TINYMIND_ORACLE_SEED", 0, UINT64_MAX),
            static_cast<unsigned>(environmentNumber("TINYMIND_ORACLE_SCALE", 1, 100000)),
        };

        if (config.scale == 0)
        {
            std::fprintf(stderr, "TINYMIND_ORACLE_SCALE must be at least 1\n");
            std::exit(2);
        }

        return config;
    }

    unsigned scaled(const unsigned count)
    {
        return count * runConfig().scale;
    }

    // === Input generation ====================================================

    // splitmix64: small, fixed-seed, identical on every platform.
    struct Rng
    {
        uint64_t state;

        explicit Rng(const uint64_t seed) : state(seed)
        {
        }

        uint64_t next()
        {
            uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }
    };

    // Seed 0 leaves the per-format seeds as they have always been, so the
    // default run keeps drawing the same sample.
    uint64_t seedFor(const Format& f)
    {
        const uint64_t base = 0x5EED0000ULL ^ (static_cast<uint64_t>(f.fixedBits) << 24) ^ (static_cast<uint64_t>(f.fractionalBits) << 16) ^
                              (f.isSigned ? 0x100ULL : 0) ^ (f.roundUp ? 0x10ULL : 0) ^ (f.saturate ? 0x1ULL : 0);

        return base ^ (runConfig().seed * 0x9E3779B97F4A7C15ULL);
    }

    bool inRange(const Format& f, const Wide v)
    {
        return (v >= trueMin(f)) && (v <= trueMax(f));
    }

    void addUnique(std::vector<Wide>& values, const Wide v)
    {
        for (const Wide existing : values)
        {
            if (existing == v)
            {
                return;
            }
        }

        values.push_back(v);
    }

    // Values where fixed-point arithmetic tends to break: zero, one LSB, one,
    // one half, the bounds and their neighbours, and the largest whole number
    // (the saturation bound before it gained its fractional bits, and still a
    // fixed-part boundary).
    std::vector<Wide> edgeValues(const Format& f)
    {
        const Wide one = powerOfTwo(f.fractionalBits);
        const Wide half = one / 2;
        const Wide lo = trueMin(f);
        const Wide hi = trueMax(f);
        const Wide clipped = (f.isSigned ? (powerOfTwo(f.fixedBits - 1) - 1) : (powerOfTwo(f.fixedBits) - 1)) * one;
        const Wide candidates[] =
        {
            0, 1, 2, 3, -1, -2, -3,
            one, -one, half, -half, one + 1, one - 1, -one - 1, 2 * one, -2 * one, one + half, -one - half,
            lo, lo + 1, lo + 2, hi, hi - 1, hi - 2, lo / 2, hi / 2,
            clipped, clipped + 1, clipped - 1, -clipped, -clipped - 1,
        };
        std::vector<Wide> values;

        for (const Wide v : candidates)
        {
            if (inRange(f, v))
            {
                addUnique(values, v);
            }
        }

        return values;
    }

    Wide randomValue(const Format& f, Rng& rng)
    {
        const Wide lo = trueMin(f);
        const Wide hi = trueMax(f);

        switch (rng.next() % 4)
        {
            case 0:
            {
                // Small magnitude: a few units either side of zero.
                const unsigned bits = (f.fractionalBits + 3 < f.width()) ? (f.fractionalBits + 3) : (f.width() - 1);
                const Wide span = powerOfTwo(bits);
                const Wide v = static_cast<Wide>(rng.next() % static_cast<uint64_t>(span)) - (f.isSigned ? (span / 2) : 0);
                return inRange(f, v) ? v : 0;
            }
            case 1:
            {
                // Near a bound.
                const Wide offset = static_cast<Wide>(rng.next() % 8);
                return ((rng.next() & 1) != 0) ? (hi - offset) : (lo + offset);
            }
            default:
                // Uniform over the format's bit patterns.
                return wrapTo(static_cast<Wide>(rng.next()), f.width(), f.isSigned);
        }
    }

    std::vector<Wide> intOperands(const Format& f)
    {
        const Wide candidates[] =
        {
            0, 1, -1, 2, -2, 3, -3, 7, -7, 100, -100, 200, -200, 1000, -1000,
            32767, -32768, 65536, -65536, INT_MAX, INT_MIN,
            powerOfTwo(f.fixedBits - 1) - 1, powerOfTwo(f.fixedBits - 1), -powerOfTwo(f.fixedBits - 1),
            -powerOfTwo(f.fixedBits - 1) - 1, powerOfTwo(f.fixedBits) - 1, powerOfTwo(f.fixedBits),
        };
        std::vector<Wide> values;

        for (const Wide v : candidates)
        {
            if ((v >= INT_MIN) && (v <= INT_MAX))
            {
                addUnique(values, v);
            }
        }

        return values;
    }

    // A random int operand: small, near a power of two, or anywhere in int.
    Wide randomInt(Rng& rng)
    {
        switch (rng.next() % 3)
        {
            case 0:
                return static_cast<Wide>(rng.next() % 2049) - 1024;
            case 1:
            {
                const Wide magnitude = powerOfTwo(static_cast<unsigned>(rng.next() % 31)) + (static_cast<Wide>(rng.next() % 3) - 1);
                return ((rng.next() & 1) != 0) ? magnitude : -magnitude;
            }
            default:
                return wrapTo(static_cast<Wide>(rng.next()), 32, true);
        }
    }

    // === Engine ==============================================================

    const unsigned ExhaustiveWidth = 8;
    const unsigned RandomPairs = 3000;
    const unsigned RandomValues = 48;
    const unsigned RandomWords = 400;
    const unsigned RandomIntPairs = 1000;
    const unsigned RandomPartPairs = 4096; // soak only: the default run crosses fixed lists
    const unsigned Chains = 100;
    const unsigned ChainLength = 24;

    void checkPair(Tally& t, const Format& f, const Ops& ops, const Wide a, const Wide b)
    {
        const std::vector<Wide> in = {a, b};

        check(t, f, "add", in, ops.add(a, b), [&](const Model& m) { return m.add(a, b); });
        check(t, f, "sub", in, ops.sub(a, b), [&](const Model& m) { return m.sub(a, b); });
        check(t, f, "mul", in, ops.mul(a, b), [&](const Model& m) { return m.mul(a, b); });
        check(t, f, "compare", in, ops.compare(a, b), [&](const Model& m) { return m.compareMask(a, b); });
        check(t, f, "div", in, ops.div(a, b), [&](const Model& m) { return m.div(a, b); });
    }

    void checkValue(Tally& t, const Format& f, const Ops& ops, const Wide a)
    {
        const std::vector<Wide> in = {a};

        check(t, f, "fixedPart", in, ops.fixedPart(a), [&](const Model& m) { return floorDiv(a, m.one()); });
        check(t, f, "fractionalPart", in, ops.fractionalPart(a), [&](const Model& m) { return floorMod(a, m.one()); });
        check(t, f, "preIncrement", in, ops.preIncrement(a), [&](const Model& m) { return m.add(a, m.one()); });
        check(t, f, "preDecrement", in, ops.preDecrement(a), [&](const Model& m) { return m.sub(a, m.one()); });
        check(t, f, "postIncrementResult", in, ops.postIncrementResult(a), [&](const Model&) { return a; });
        check(t, f, "postIncrementAfter", in, ops.postIncrementAfter(a), [&](const Model& m) { return m.add(a, m.one()); });
        check(t, f, "postDecrementResult", in, ops.postDecrementResult(a), [&](const Model&) { return a; });
        check(t, f, "postDecrementAfter", in, ops.postDecrementAfter(a), [&](const Model& m) { return m.sub(a, m.one()); });
    }

    void checkInt(Tally& t, const Format& f, const Ops& ops, const Wide a, const Wide n)
    {
        const std::vector<Wide> in = {a, n};
        const int i = static_cast<int>(n);

        check(t, f, "addInt", in, ops.addInt(a, i), [&](const Model& m) { return m.addInt(a, n); });
        check(t, f, "subInt", in, ops.subInt(a, i), [&](const Model& m) { return m.subInt(a, n); });
        check(t, f, "mulInt", in, ops.mulInt(a, i), [&](const Model& m) { return m.mulInt(a, n); });
        check(t, f, "mulAssignInt", in, ops.mulAssignInt(a, i), [&](const Model& m) { return m.mulInt(a, n); });
        check(t, f, "compareInt", in, ops.compareInt(a, i), [&](const Model& m) { return m.compareInt(a, n); });
        check(t, f, "divInt", in, ops.divInt(a, i), [&](const Model& m) { return m.divInt(a, n); });
        check(t, f, "divAssignInt", in, ops.divAssignInt(a, i), [&](const Model& m) { return m.divInt(a, n); });
    }

    void checkRepresentation(Tally& t, const Format& f, const Ops& ops, Rng& rng)
    {
        const std::vector<Wide> none;

        check(t, f, "QFormatMinValue", none, ops.minValue(), [&](const Model& m) { return m.minValue(); });
        check(t, f, "QFormatMaxValue", none, ops.maxValue(), [&](const Model& m) { return m.maxValue(); });
        check(t, f, "defaultValue", none, ops.defaultValue(), [&](const Model&) { return static_cast<Wide>(0); });

        // Raw words: any storage-word bit pattern reduces to the format width.
        std::vector<Wide> words = edgeValues(f);
        addUnique(words, wrapTo(powerOfTwo(f.width()), f.storageBits, false));
        addUnique(words, wrapTo(-powerOfTwo(f.width()), f.storageBits, false));
        addUnique(words, wrapTo(-1, f.storageBits, false));

        for (unsigned i = 0; i < scaled(RandomWords); ++i)
        {
            words.push_back(wrapTo(static_cast<Wide>(rng.next()), f.storageBits, false));
        }

        for (const Wide word : words)
        {
            const Wide unsignedWord = wrapTo(word, f.storageBits, false);
            const Wide valueWord = wrapTo(word, f.storageBits, f.isSigned);

            check(t, f, "fromRawField", {unsignedWord}, ops.fromRawField(unsignedWord), [&](const Model& m) { return m.wrap(unsignedWord); });
            check(t, f, "setRaw", {valueWord}, ops.setRaw(valueWord), [&](const Model& m) { return m.wrap(valueWord); });
            check(t, f, "assignRaw", {valueWord}, ops.assignRaw(valueWord), [&](const Model& m) { return m.wrap(valueWord); });
        }

        // Parts: any fixed-part-type and fractional-part-type values, including
        // ones outside the format's fixed and fractional widths.
        std::vector<Wide> fixedParts;
        const Wide fixedCandidates[] =
        {
            0, 1, -1, 2, -2, powerOfTwo(f.fixedBits - 1) - 1, -powerOfTwo(f.fixedBits - 1),
            powerOfTwo(f.fixedBits - 1), powerOfTwo(f.fixedBits) - 1, powerOfTwo(f.fixedBits),
        };

        for (const Wide v : fixedCandidates)
        {
            addUnique(fixedParts, wrapTo(v, f.storageBits, f.isSigned));
        }

        std::vector<Wide> fractionalParts;
        const Wide fractionalCandidates[] =
        {
            0, 1, powerOfTwo(f.fractionalBits - 1), powerOfTwo(f.fractionalBits) - 1, powerOfTwo(f.fractionalBits), -1,
        };

        for (const Wide v : fractionalCandidates)
        {
            addUnique(fractionalParts, wrapTo(v, f.storageBits, false));
        }

        for (unsigned i = 0; i < 64; ++i)
        {
            fixedParts.push_back(wrapTo(static_cast<Wide>(rng.next()), f.storageBits, f.isSigned));
            fractionalParts.push_back(wrapTo(static_cast<Wide>(rng.next()), f.storageBits, false));
        }

        for (const Wide fixedPart : fixedParts)
        {
            for (const Wide fractionalPart : fractionalParts)
            {
                const std::vector<Wide> in = {fixedPart, fractionalPart};

                check(t, f, "fromParts", in, ops.fromParts(fixedPart, fractionalPart), [&](const Model& m) { return m.fromParts(fixedPart, fractionalPart); });
                check(t, f, "setParts", in, ops.setParts(fixedPart, fractionalPart), [&](const Model& m) { return m.fromParts(fixedPart, fractionalPart); });
            }
        }

        // A soak adds independent random part pairs; crossing two scaled lists
        // would grow with the square of the scale.
        for (unsigned i = 0; i < (scaled(RandomPartPairs) - RandomPartPairs); ++i)
        {
            const Wide fixedPart = wrapTo(static_cast<Wide>(rng.next()), f.storageBits, f.isSigned);
            const Wide fractionalPart = wrapTo(static_cast<Wide>(rng.next()), f.storageBits, false);
            const std::vector<Wide> in = {fixedPart, fractionalPart};

            check(t, f, "fromParts", in, ops.fromParts(fixedPart, fractionalPart), [&](const Model& m) { return m.fromParts(fixedPart, fractionalPart); });
            check(t, f, "setParts", in, ops.setParts(fixedPart, fractionalPart), [&](const Model& m) { return m.fromParts(fixedPart, fractionalPart); });
        }
    }

    // One value carried through a random sequence of operations. Each step
    // starts from the value QValue actually produced, so a defect is reported
    // at the step that causes it instead of cascading through the rest.
    void checkChain(Tally& t, const Format& f, const Ops& ops, Rng& rng)
    {
        Wide a = randomValue(f, rng);

        for (unsigned step = 0; step < ChainLength; ++step)
        {
            const unsigned op = static_cast<unsigned>(rng.next() % 15);
            const Wide b = randomValue(f, rng);
            const Wide n = randomInt(rng);

            const std::vector<Wide> binary = {a, b};
            const std::vector<Wide> withInt = {a, n};
            const std::vector<Wide> unary = {a};
            const int i = static_cast<int>(n);
            Wide actual = a;

            switch (op)
            {
                case 0:
                    actual = ops.addAssign(a, b);
                    check(t, f, "chain addAssign", binary, actual, [&](const Model& m) { return m.add(a, b); });
                    break;
                case 1:
                    actual = ops.subAssign(a, b);
                    check(t, f, "chain subAssign", binary, actual, [&](const Model& m) { return m.sub(a, b); });
                    break;
                case 2:
                    actual = ops.mulAssign(a, b);
                    check(t, f, "chain mulAssign", binary, actual, [&](const Model& m) { return m.mul(a, b); });
                    break;
                case 3:
                    actual = ops.divAssign(a, b);
                    check(t, f, "chain divAssign", binary, actual, [&](const Model& m) { return m.div(a, b); });
                    break;
                case 4:
                    actual = ops.addInt(a, i);
                    check(t, f, "chain addInteger", withInt, actual, [&](const Model& m) { return m.addInt(a, n); });
                    break;
                case 5:
                    actual = ops.subInt(a, i);
                    check(t, f, "chain subtractInteger", withInt, actual, [&](const Model& m) { return m.subInt(a, n); });
                    break;
                case 6:
                    actual = ops.mulAssignInt(a, i);
                    check(t, f, "chain mulAssignInt", withInt, actual, [&](const Model& m) { return m.mulInt(a, n); });
                    break;
                case 7:
                    actual = ops.divAssignInt(a, i);
                    check(t, f, "chain divAssignInt", withInt, actual, [&](const Model& m) { return m.divInt(a, n); });
                    break;
                case 8:
                    actual = ops.preIncrement(a);
                    check(t, f, "chain preIncrement", unary, actual, [&](const Model& m) { return m.add(a, m.one()); });
                    break;
                case 9:
                    actual = ops.preDecrement(a);
                    check(t, f, "chain preDecrement", unary, actual, [&](const Model& m) { return m.sub(a, m.one()); });
                    break;
                case 10:
                    actual = ops.selfAdd(a);
                    check(t, f, "chain selfAdd", unary, actual, [&](const Model& m) { return m.add(a, a); });
                    break;
                case 11:
                    actual = ops.selfSub(a);
                    check(t, f, "chain selfSub", unary, actual, [&](const Model& m) { return m.sub(a, a); });
                    break;
                case 12:
                    actual = ops.selfMul(a);
                    check(t, f, "chain selfMul", unary, actual, [&](const Model& m) { return m.mul(a, a); });
                    break;
                case 13:
                    actual = ops.selfDiv(a);
                    check(t, f, "chain selfDiv", unary, actual, [&](const Model& m) { return m.div(a, a); });
                    break;
                default:
                    actual = ops.partsRoundTrip(a);
                    check(t, f, "chain partsRoundTrip", unary, actual, [&](const Model&) { return a; });
                    break;
            }

            // An out-of-range result has already failed its check; carry on
            // from a valid value so the rest of the chain still means something.
            a = inRange(f, actual) ? actual : 0;
        }
    }

    void runEngine(const Format& f, const Ops& ops, Tally& tally)
    {
        Rng rng(seedFor(f));

        checkRepresentation(tally, f, ops, rng);

        std::vector<Wide> values;

        if (f.width() <= ExhaustiveWidth)
        {
            for (Wide v = trueMin(f); v <= trueMax(f); ++v)
            {
                values.push_back(v);
            }

            for (const Wide a : values)
            {
                for (const Wide b : values)
                {
                    checkPair(tally, f, ops, a, b);
                }
            }
        }
        else
        {
            values = edgeValues(f);
            const std::vector<Wide> edges = values;

            for (unsigned i = 0; i < scaled(RandomValues); ++i)
            {
                values.push_back(randomValue(f, rng));
            }

            for (const Wide a : edges)
            {
                for (const Wide b : edges)
                {
                    checkPair(tally, f, ops, a, b);
                }
            }

            for (unsigned i = 0; i < scaled(RandomPairs); ++i)
            {
                // Sequenced: argument evaluation order is unspecified, and
                // both compilers must draw the same pairs.
                const Wide a = randomValue(f, rng);
                const Wide b = randomValue(f, rng);
                checkPair(tally, f, ops, a, b);
            }
        }

        const std::vector<Wide> ints = intOperands(f);

        for (const Wide a : values)
        {
            checkValue(tally, f, ops, a);

            for (const Wide n : ints)
            {
                checkInt(tally, f, ops, a, n);
            }
        }

        // The int list above is fixed; these pairs reach the rest of int.
        for (unsigned i = 0; i < scaled(RandomIntPairs); ++i)
        {
            const Wide a = randomValue(f, rng);
            const Wide n = randomInt(rng);
            checkInt(tally, f, ops, a, n);
        }

        for (unsigned i = 0; i < scaled(Chains); ++i)
        {
            checkChain(tally, f, ops, rng);
        }
    }

    template<typename Q>
    void runFormat(Tally& tally)
    {
        runEngine(Adapter<Q>::format(), Adapter<Q>::ops(), tally);
    }

    template<unsigned I, unsigned F, bool S>
    void runAllPolicies(Tally& tally)
    {
        runFormat<tinymind::QValue<I, F, S, tinymind::TruncatePolicy, tinymind::WrapPolicy> >(tally);
        runFormat<tinymind::QValue<I, F, S, tinymind::TruncatePolicy, tinymind::MinMaxSaturatePolicy> >(tally);
        runFormat<tinymind::QValue<I, F, S, tinymind::RoundUpPolicy, tinymind::WrapPolicy> >(tally);
        runFormat<tinymind::QValue<I, F, S, tinymind::RoundUpPolicy, tinymind::MinMaxSaturatePolicy> >(tally);
    }

    // convertFromOtherQValueType: rescale the fraction (floor when narrowing),
    // then wrap to the destination's width.
    template<typename Dst, typename Src>
    void runConversion(Tally& tally)
    {
        const Format src = Adapter<Src>::format();
        const Format dst = Adapter<Dst>::format();
        Rng rng(seedFor(src) ^ (seedFor(dst) << 1));
        std::vector<Wide> values = edgeValues(src);

        for (unsigned i = 0; i < scaled(RandomPairs); ++i)
        {
            values.push_back(randomValue(src, rng));
        }

        for (const Wide v : values)
        {
            Src s;
            s.setValue(static_cast<typename Src::FullWidthValueType>(v));
            Dst d;
            d.convertFromOtherQValueType(s);

            const Wide actual = static_cast<Wide>(d.getValue());

            check(tally, dst, "convertFrom", {v}, actual, [&](const Model& m)
            {
                const Wide scaled = (dst.fractionalBits >= src.fractionalBits)
                                    ? (v * powerOfTwo(dst.fractionalBits - src.fractionalBits))
                                    : floorDiv(v, powerOfTwo(src.fractionalBits - dst.fractionalBits));
                return m.wrap(scaled);
            });
        }
    }

    Tally gTotal;

    void report(const char* family, const Tally& tally)
    {
        std::printf("%s: %llu checks, %llu explained by known defects, %llu unexplained", family, tally.checks, tally.explained, tally.unknown);

        for (unsigned d = 0; d < NumberOfKnownDefects; ++d)
        {
            std::printf(", %s=%llu", KnownDefectNames[d], tally.hits[d]);
        }

        std::printf("\n");

        std::string examples;

        for (const std::string& example : tally.unknownExamples)
        {
            examples += "\n  " + example;
        }

        BOOST_CHECK_MESSAGE(tally.unknown == 0, family << ": " << tally.unknown << " results match neither the intended arithmetic nor any known defect"
                            << " (reproduce with TINYMIND_ORACLE_SEED=" << runConfig().seed << " TINYMIND_ORACLE_SCALE=" << runConfig().scale << "):" << examples);

        gTotal.merge(tally);
    }
}

BOOST_AUTO_TEST_SUITE(test_suite_qformat_oracle)

BOOST_AUTO_TEST_CASE(test_case_oracle_8bit_exhaustive)
{
    Tally tally;
    runAllPolicies<1, 7, true>(tally);
    runAllPolicies<2, 6, true>(tally);
    runAllPolicies<4, 4, true>(tally);
    runAllPolicies<7, 1, true>(tally);
    runAllPolicies<1, 7, false>(tally);
    runAllPolicies<4, 4, false>(tally);
    runAllPolicies<7, 1, false>(tally);
    report("8-bit formats (exhaustive)", tally);
}

BOOST_AUTO_TEST_CASE(test_case_oracle_16bit_storage)
{
    Tally tally;
    runAllPolicies<8, 8, true>(tally);
    runAllPolicies<1, 15, true>(tally);
    runAllPolicies<7, 9, true>(tally);
    runAllPolicies<15, 1, true>(tally);
    runAllPolicies<4, 8, true>(tally);
    runAllPolicies<8, 8, false>(tally);
    runAllPolicies<4, 8, false>(tally);
    report("16-bit storage", tally);
}

BOOST_AUTO_TEST_CASE(test_case_oracle_32bit_storage)
{
    Tally tally;
    runAllPolicies<16, 16, true>(tally);
    runAllPolicies<24, 8, true>(tally);
    runAllPolicies<8, 24, true>(tally);
    runAllPolicies<1, 31, true>(tally);
    runAllPolicies<31, 1, true>(tally);
    runAllPolicies<8, 16, true>(tally);
    runAllPolicies<16, 16, false>(tally);
    runAllPolicies<24, 8, false>(tally);
    runAllPolicies<8, 24, false>(tally);
    runAllPolicies<8, 16, false>(tally);
    report("32-bit storage", tally);
}

BOOST_AUTO_TEST_CASE(test_case_oracle_64bit_storage)
{
    Tally tally;
    runAllPolicies<32, 32, true>(tally);
    runAllPolicies<24, 40, true>(tally);
    runAllPolicies<16, 32, true>(tally);
    runAllPolicies<1, 63, true>(tally);
    runAllPolicies<63, 1, true>(tally);
    runAllPolicies<32, 32, false>(tally);
    runAllPolicies<16, 32, false>(tally);
    runAllPolicies<1, 63, false>(tally);
    runAllPolicies<63, 1, false>(tally);
    report("64-bit storage", tally);
}

BOOST_AUTO_TEST_CASE(test_case_oracle_conversions)
{
    typedef tinymind::QValue<8, 8, true>   S8_8;
    typedef tinymind::QValue<4, 8, true>   S4_8;
    typedef tinymind::QValue<8, 16, true>  S8_16;
    typedef tinymind::QValue<16, 16, true> S16_16;
    typedef tinymind::QValue<8, 24, true>  S8_24;
    typedef tinymind::QValue<1, 15, true>  S1_15;
    typedef tinymind::QValue<1, 7, true>   S1_7;
    typedef tinymind::QValue<32, 32, true> S32_32;
    typedef tinymind::QValue<24, 40, true> S24_40;
    typedef tinymind::QValue<8, 8, false>  U8_8;
    typedef tinymind::QValue<16, 16, false> U16_16;

    Tally tally;
    runConversion<S4_8, S8_8>(tally);
    runConversion<S8_8, S4_8>(tally);
    runConversion<S8_16, S8_8>(tally);
    runConversion<S8_8, S8_16>(tally);
    runConversion<S8_8, S16_16>(tally);
    runConversion<S16_16, S8_8>(tally);
    runConversion<S16_16, S8_24>(tally);
    runConversion<S8_24, S16_16>(tally);
    runConversion<S8_8, S1_15>(tally);
    runConversion<S1_7, S8_8>(tally);
    runConversion<S16_16, S32_32>(tally);
    runConversion<S32_32, S16_16>(tally);
    runConversion<S8_24, S24_40>(tally);
    runConversion<S8_8, U8_8>(tally);
    runConversion<U8_8, S8_8>(tally);
    runConversion<U16_16, S16_16>(tally);
    report("conversions", tally);
}

// Runs last (Boost runs cases in declaration order): every known defect must
// still be required to explain some result, or its model is stale and must be
// removed. Run the whole module for this check to mean anything; a filtered
// run skips the families that produce the hits.
BOOST_AUTO_TEST_CASE(test_case_known_defects_still_reproduce)
{
    std::printf("oracle total: %llu checks (TINYMIND_ORACLE_SEED=%llu TINYMIND_ORACLE_SCALE=%u). Skipped: %s.\n",
                gTotal.checks, static_cast<unsigned long long>(runConfig().seed), runConfig().scale, skippedInputs());

    for (unsigned d = 0; d < NumberOfKnownDefects; ++d)
    {
        std::printf("  known defect %s: required by %llu results, e.g. %s\n", KnownDefectNames[d], gTotal.hits[d], gTotal.firstHit[d].c_str());
        BOOST_CHECK_MESSAGE(gTotal.hits[d] > 0, "known defect " << KnownDefectNames[d] << " no longer reproduces: remove it and its model from this oracle");
    }
}

BOOST_AUTO_TEST_SUITE_END()
