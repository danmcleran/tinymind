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
#include "include/tinymind_traits.hpp"

#include "lookupTable.hpp"
#include "exp.hpp"

#include <cstddef>
#include <cstdint>

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>
#endif

namespace tinymind {

    namespace anfis_detail {
        // Construct a ValueType representing the integer v. FPU-free for QValue:
        // QValue(int) treats its argument as the raw fixed-point bit pattern,
        // so the (fixed, fractional) constructor is required instead.
        template<typename T>
        inline typename tinymind::enable_if<tinymind::is_floating_point<T>::value, T>::type
        fromInteger(const int v)
        {
            return static_cast<T>(v);
        }

        template<typename T>
        inline typename tinymind::enable_if<!tinymind::is_floating_point<T>::value, T>::type
        fromInteger(const int v)
        {
            return T(static_cast<typename T::FixedPartFieldType>(v), 0u);
        }

        template<typename T>
        inline T zero()
        {
            static const T z = fromInteger<T>(0);
            return z;
        }

        template<typename T>
        inline T one()
        {
            static const T o = fromInteger<T>(1);
            return o;
        }

        template<typename T>
        inline T absoluteValue(const T& v)
        {
            return (v < zero<T>()) ? (zero<T>() - v) : v;
        }

        // Compile-time Base^Exponent, used to size a full-grid rule table.
        template<size_t Base, size_t Exponent>
        struct Power
        {
            static const size_t value = Base * Power<Base, Exponent - 1>::value;
        };

        template<size_t Base>
        struct Power<Base, 0>
        {
            static const size_t value = 1;
        };
    }

    /**
     * Triangular membership function.
     *
     * Parameters {a, b, c} are the left foot, the peak, and the right foot:
     *
     *   mu(x) = 0                  x <= a or x >= c
     *           (x - a) / (b - a)  a <  x <  b
     *           1                  x == b
     *           (c - x) / (c - b)  b <  x <  c
     *
     * Degenerate shoulders are legal: a == b gives a left shoulder that steps
     * to 1 immediately, b == c gives a right shoulder. Both branches are only
     * reachable with a non-zero denominator, so no divide-by-zero guard is
     * needed.
     *
     * Uses only compare, subtract, and divide, so it is the membership
     * function to reach for at TINYMIND_ENABLE_FLOAT=0 / TINYMIND_ENABLE_STD=0.
     */
    template<typename ValueType>
    struct TriangularMembershipFunction
    {
        typedef ValueType MembershipFunctionValueType;

        static const size_t NumberOfParameters = 3;

        static ValueType evaluate(ValueType const* const parameters, const ValueType& x)
        {
            const ValueType a = parameters[0];
            const ValueType b = parameters[1];
            const ValueType c = parameters[2];

            if ((x <= a) || (x >= c))
            {
                return anfis_detail::zero<ValueType>();
            }

            if (x < b)
            {
                return (x - a) / (b - a);
            }

            if (x > b)
            {
                return (c - x) / (c - b);
            }

            return anfis_detail::one<ValueType>();
        }
    };

    /**
     * Trapezoidal membership function.
     *
     * Parameters {a, b, c, d} are the left foot, the start of the plateau, the
     * end of the plateau, and the right foot:
     *
     *   mu(x) = 0                  x <= a or x >= d
     *           (x - a) / (b - a)  a <  x <  b
     *           1                  b <= x <= c
     *           (d - x) / (d - c)  c <  x <  d
     *
     * As with the triangle, a == b and c == d produce shoulders and cannot
     * reach a zero denominator.
     *
     * Arithmetic-only: safe at TINYMIND_ENABLE_FLOAT=0 / TINYMIND_ENABLE_STD=0.
     */
    template<typename ValueType>
    struct TrapezoidalMembershipFunction
    {
        typedef ValueType MembershipFunctionValueType;

        static const size_t NumberOfParameters = 4;

        static ValueType evaluate(ValueType const* const parameters, const ValueType& x)
        {
            const ValueType a = parameters[0];
            const ValueType b = parameters[1];
            const ValueType c = parameters[2];
            const ValueType d = parameters[3];

            if ((x <= a) || (x >= d))
            {
                return anfis_detail::zero<ValueType>();
            }

            if (x < b)
            {
                return (x - a) / (b - a);
            }

            if (x <= c)
            {
                return anfis_detail::one<ValueType>();
            }

            return (d - x) / (d - c);
        }
    };

    /**
     * Generalized bell membership function with a compile-time integer
     * exponent.
     *
     * Parameters {a, c} are the width and the center:
     *
     *   t      = ((x - c) / a)^2
     *   mu(x)  = 1 / (1 + t^BellExponent)
     *
     * The classic Jang form carries a runtime exponent b and evaluates
     * |(x-c)/a|^(2b) with pow(). Pinning b at compile time turns that into
     * BellExponent-1 multiplies, so the bell needs no transcendental function
     * and holds at TINYMIND_ENABLE_FLOAT=0 / TINYMIND_ENABLE_STD=0 — unlike
     * the Gaussian, which needs an exp lookup table.
     *
     * t is clamped at 64 (mu < 1/65) and the result reported as 0, which both
     * keeps the rule base sparse and stops t^BellExponent from overflowing a
     * narrow Q format. a == 0 degenerates to an impulse at c.
     *
     * @tparam BellExponent  b in the formula above; the effective slope
     *                       exponent is 2 * BellExponent. Must be >= 1.
     */
    template<typename ValueType, size_t BellExponent = 1>
    struct GeneralizedBellMembershipFunction
    {
        typedef ValueType MembershipFunctionValueType;

        static const size_t NumberOfParameters = 2;

        static ValueType evaluate(ValueType const* const parameters, const ValueType& x)
        {
            const ValueType a = parameters[0];
            const ValueType c = parameters[1];
            const ValueType zeroValue = anfis_detail::zero<ValueType>();
            const ValueType oneValue = anfis_detail::one<ValueType>();
            static const ValueType limit = anfis_detail::fromInteger<ValueType>(64);

            if (a == zeroValue)
            {
                return (x == c) ? oneValue : zeroValue;
            }

            const ValueType ratio = (x - c) / a;
            const ValueType t = ratio * ratio;

            if (t >= limit)
            {
                return zeroValue;
            }

            ValueType accumulator = t;

            for (size_t power = 1; power < BellExponent; ++power)
            {
                accumulator = accumulator * t;

                if (accumulator >= limit)
                {
                    return zeroValue;
                }
            }

            return oneValue / (oneValue + accumulator);
        }

        static_assert(BellExponent >= 1, "BellExponent must be >= 1.");
    };

    /**
     * Gaussian membership function, evaluated through the fixed-point exp
     * lookup table.
     *
     * Parameters {c, sigma, inverseSigma} are the center, the width, and the
     * precomputed reciprocal of the width:
     *
     *   z      = (x - c) * inverseSigma
     *   mu(x)  = exp(-z^2 / 2)     |z| <  4
     *            0                 |z| >= 4
     *
     * sigma is carried alongside its own reciprocal so the support test can be
     * written as |x - c| >= 4 * sigma. That comparison happens before the
     * multiply, which is what keeps z bounded and the subsequent square inside
     * a narrow Q format. The caller (or the offline exporter) must supply both
     * and must keep 4 * sigma representable in ValueType.
     *
     * Truncating at 4 sigma (mu ~= 3.4e-4) keeps the rule base sparse; the
     * float specializations below apply the identical cut so a model behaves
     * the same in float and in fixed point.
     *
     * The QValue form needs the matching TINYMIND_USE_EXP_<F>_<Fr> build macro
     * for its Q format, exactly as SoftmaxActivationPolicy does. It needs no
     * FPU and no stdlib: the table is integer data.
     */
    template<typename ValueType>
    struct GaussianMembershipFunction
    {
        typedef ValueType MembershipFunctionValueType;
        typedef typename ValueType::FullWidthFieldType FullWidthFieldType;
        typedef LookupTable<ValueType> LookupTableType;
        typedef typename ExpValuesTableSelector<ValueType::NumberOfFixedBits,
                                                ValueType::NumberOfFractionalBits,
                                                ValueType::IsSigned>::ExpTableType ExpTableType;

        static const size_t NumberOfParameters = 3;

        static ValueType evaluate(ValueType const* const parameters, const ValueType& x)
        {
            static const ptrdiff_t MAX_ACTIVATION_INDEX =
                (((sizeof(FullWidthFieldType) * NUMBER_OF_ACTIVATION_TABLE_VALUES) /
                  sizeof(expActivationTable.values[0])) - 1);
            static const ValueType four = anfis_detail::fromInteger<ValueType>(4);
            static const ValueType two = anfis_detail::fromInteger<ValueType>(2);

            const ValueType c = parameters[0];
            const ValueType sigma = parameters[1];
            const ValueType inverseSigma = parameters[2];

            const ValueType deviation = anfis_detail::absoluteValue(x - c);

            if (deviation >= (four * sigma))
            {
                return anfis_detail::zero<ValueType>();
            }

            const ValueType z = deviation * inverseSigma;
            const ValueType argument = anfis_detail::zero<ValueType>() - ((z * z) / two);

            return LookupTableType::getValue(argument, &expActivationTable.values[0], MAX_ACTIVATION_INDEX);
        }

    private:
        static const ExpTableType expActivationTable;

        static_assert(ValueType::IsSigned, "Gaussian membership functions require a signed type.");
    };

    template<typename ValueType>
    const typename ExpValuesTableSelector<ValueType::NumberOfFixedBits,
                                          ValueType::NumberOfFractionalBits,
                                          ValueType::IsSigned>::ExpTableType
        GaussianMembershipFunction<ValueType>::expActivationTable;

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
    template<>
    struct GaussianMembershipFunction<float>
    {
        typedef float MembershipFunctionValueType;

        static const size_t NumberOfParameters = 3;

        static float evaluate(float const* const parameters, const float& x)
        {
            const float deviation = std::fabs(x - parameters[0]);

            if (deviation >= (4.0f * parameters[1]))
            {
                return 0.0f;
            }

            const float z = deviation * parameters[2];

            return std::exp(-(z * z) / 2.0f);
        }
    };

    template<>
    struct GaussianMembershipFunction<double>
    {
        typedef double MembershipFunctionValueType;

        static const size_t NumberOfParameters = 3;

        static double evaluate(double const* const parameters, const double& x)
        {
            const double deviation = std::fabs(x - parameters[0]);

            if (deviation >= (4.0 * parameters[1]))
            {
                return 0.0;
            }

            const double z = deviation * parameters[2];

            return std::exp(-(z * z) / 2.0);
        }
    };
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

    /**
     * Adaptive Neuro-Fuzzy Inference System (Jang, 1993) — inference only.
     *
     * A Takagi-Sugeno fuzzy system whose premise and consequent parameters are
     * learned offline. Five feed-forward stages, no state carried between
     * calls, no dynamic allocation, and a latency that does not depend on the
     * input data:
     *
     *   1. fuzzify    mu[i][m]  = MembershipFunctionPolicy(x[i])
     *   2. fire       w[r]      = product over i of mu[i][ruleTable[r][i]]
     *   3. normalize  wbar[r]   = w[r] / sum(w)
     *   4. consequent f[r][o]   = p[r][o] . x + q[r][o]   (first order)
     *                           = c[r][o]                 (zeroth order)
     *   5. output     y[o]      = sum over r of wbar[r] * f[r][o]
     *
     * Stages 3 and 5 are fused into a single divide per output —
     * y[o] = sum(w[r] * f[r][o]) / sum(w) — because a per-rule reciprocal is
     * both slower and, in a narrow Q format, far less accurate. Q16.16 is the
     * recommended ValueType; Q8.8 will collapse for anything but a handful of
     * rules with small consequents.
     *
     * The rule base is an explicit index table rather than an implicit full
     * grid. A full grid costs
     * NumberOfMembershipFunctionsPerInput^NumberOfInputs rules and is the
     * reason plain ANFIS does not scale past a handful of inputs; a table lets
     * an offline pruning pass ship only the rules that carry weight. Use
     * AnfisFullGridRuleTable to fill a grid when that is what you want.
     * An antecedent entry of DontCareIndex (or any index at or above
     * NumberOfMembershipFunctionsPerInput) drops that input from the rule,
     * contributing a membership grade of 1.
     *
     * The firing strengths survive the call, so the caller can ask which rules
     * explain a given output. That readability is the reason to pick ANFIS
     * over a dense net of the same size in the first place.
     *
     * Training is deliberately absent. The classic hybrid scheme solves the
     * consequents by least squares — a pseudo-inverse — and only then descends
     * on the premises, which belongs on a host, not on a microcontroller.
     * Parameters arrive frozen, matching the rest of the deployment path.
     *
     * @tparam ValueType                          Numeric type (QValue or float/double)
     * @tparam NumberOfInputs                     Number of crisp inputs
     * @tparam NumberOfMembershipFunctionsPerInput Membership functions per input
     * @tparam NumberOfRules                      Number of rules in the rule table
     * @tparam MembershipFunctionPolicy           Membership function policy (see above)
     * @tparam FirstOrderConsequent               true for a linear (order 1) TSK
     *                                            consequent, false for a constant
     *                                            (order 0) consequent
     * @tparam NumberOfOutputs                    Number of outputs sharing the premise
     */
    template<
        typename ValueType,
        size_t NumberOfInputs,
        size_t NumberOfMembershipFunctionsPerInput,
        size_t NumberOfRules,
        typename MembershipFunctionPolicy,
        bool FirstOrderConsequent = true,
        size_t NumberOfOutputs = 1>
    class Anfis
    {
    public:
        typedef ValueType AnfisValueType;
        typedef MembershipFunctionPolicy MembershipFunctionPolicyType;

        static const size_t InputSize = NumberOfInputs;
        static const size_t OutputSize = NumberOfOutputs;
        static const size_t NumberOfParametersPerMembershipFunction = MembershipFunctionPolicy::NumberOfParameters;
        static const size_t NumberOfMembershipFunctions = NumberOfInputs * NumberOfMembershipFunctionsPerInput;
        static const size_t NumberOfPremiseParameters = NumberOfMembershipFunctions * NumberOfParametersPerMembershipFunction;
        static const size_t NumberOfConsequentParametersPerRule = FirstOrderConsequent ? (NumberOfInputs + 1) : 1;
        static const size_t NumberOfConsequentParameters = NumberOfRules * NumberOfOutputs * NumberOfConsequentParametersPerRule;
        static const size_t RuleTableSize = NumberOfRules * NumberOfInputs;
        static const uint8_t DontCareIndex = static_cast<uint8_t>(NumberOfMembershipFunctionsPerInput);

        /**
         * @param premiseParameters    NumberOfPremiseParameters values, laid out as
         *                             [input][membershipFunction][parameter]
         * @param ruleTable            RuleTableSize antecedent indices, laid out as
         *                             [rule][input]
         * @param consequentParameters NumberOfConsequentParameters values, laid out as
         *                             [rule][output][parameter]; for a first-order
         *                             consequent the first NumberOfInputs parameters
         *                             are the input coefficients and the last is the
         *                             constant term
         */
        Anfis(ValueType const* const premiseParameters,
              uint8_t const* const ruleTable,
              ValueType const* const consequentParameters)
            : mPremiseParameters(premiseParameters),
              mRuleTable(ruleTable),
              mConsequentParameters(consequentParameters),
              mTotalFiringStrength(anfis_detail::zero<ValueType>())
        {
            for (size_t i = 0; i < NumberOfMembershipFunctions; ++i)
            {
                mMembershipGrades[i] = anfis_detail::zero<ValueType>();
            }

            for (size_t rule = 0; rule < NumberOfRules; ++rule)
            {
                mFiringStrengths[rule] = anfis_detail::zero<ValueType>();
            }
        }

        /**
         * Forward pass.
         *
         * @param input  Array of NumberOfInputs crisp values
         * @param output Array of NumberOfOutputs defuzzified values
         *
         * If no rule fires — every input fell outside the support of every
         * membership function it is tested against — the total firing strength
         * is zero, every output is set to zero, and no division is attempted.
         * Check getTotalFiringStrength() to distinguish that case from a
         * genuine zero output.
         */
        void forward(ValueType const* const input, ValueType* output)
        {
            const ValueType zeroValue = anfis_detail::zero<ValueType>();

            computeMembershipGrades(input);

            mTotalFiringStrength = zeroValue;

            for (size_t rule = 0; rule < NumberOfRules; ++rule)
            {
                const ValueType firingStrength = computeFiringStrength(rule);

                mFiringStrengths[rule] = firingStrength;
                mTotalFiringStrength += firingStrength;
            }

            if (mTotalFiringStrength == zeroValue)
            {
                for (size_t out = 0; out < NumberOfOutputs; ++out)
                {
                    output[out] = zeroValue;
                }

                return;
            }

            for (size_t out = 0; out < NumberOfOutputs; ++out)
            {
                ValueType weightedSum = zeroValue;

                for (size_t rule = 0; rule < NumberOfRules; ++rule)
                {
                    if (mFiringStrengths[rule] != zeroValue)
                    {
                        weightedSum += (mFiringStrengths[rule] * evaluateConsequent(rule, out, input));
                    }
                }

                output[out] = weightedSum / mTotalFiringStrength;
            }
        }

        /**
         * Raw (unnormalized) firing strength of a rule from the last forward().
         */
        ValueType getFiringStrength(const size_t rule) const
        {
            return mFiringStrengths[rule];
        }

        /**
         * Normalized firing strength of a rule from the last forward() — the
         * fraction of the output that rule is responsible for. Computed on
         * demand so the forward path pays for one divide per output rather
         * than one per rule.
         */
        ValueType getNormalizedFiringStrength(const size_t rule) const
        {
            const ValueType zeroValue = anfis_detail::zero<ValueType>();

            if (mTotalFiringStrength == zeroValue)
            {
                return zeroValue;
            }

            return mFiringStrengths[rule] / mTotalFiringStrength;
        }

        ValueType getTotalFiringStrength() const
        {
            return mTotalFiringStrength;
        }

        /**
         * Membership grade of an input in one of its membership functions from
         * the last forward().
         */
        ValueType getMembershipGrade(const size_t input, const size_t membershipFunction) const
        {
            return mMembershipGrades[(input * NumberOfMembershipFunctionsPerInput) + membershipFunction];
        }

        /**
         * Index of the strongest rule from the last forward(). Rule 0 is
         * reported when nothing fired.
         */
        size_t getDominantRule() const
        {
            size_t dominant = 0;

            for (size_t rule = 1; rule < NumberOfRules; ++rule)
            {
                if (mFiringStrengths[rule] > mFiringStrengths[dominant])
                {
                    dominant = rule;
                }
            }

            return dominant;
        }

    private:
        ValueType const* const mPremiseParameters;
        uint8_t const* const mRuleTable;
        ValueType const* const mConsequentParameters;
        ValueType mMembershipGrades[NumberOfMembershipFunctions];
        ValueType mFiringStrengths[NumberOfRules];
        ValueType mTotalFiringStrength;

        void computeMembershipGrades(ValueType const* const input)
        {
            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                const ValueType x = input[i];

                for (size_t mf = 0; mf < NumberOfMembershipFunctionsPerInput; ++mf)
                {
                    const size_t index = (i * NumberOfMembershipFunctionsPerInput) + mf;

                    mMembershipGrades[index] = MembershipFunctionPolicy::evaluate(
                        &mPremiseParameters[index * NumberOfParametersPerMembershipFunction], x);
                }
            }
        }

        // Product t-norm over the rule's antecedents. Bails out on the first
        // zero grade, which is the common case for compact-support membership
        // functions and is what makes a large rule table affordable.
        ValueType computeFiringStrength(const size_t rule) const
        {
            const ValueType zeroValue = anfis_detail::zero<ValueType>();
            ValueType firingStrength = anfis_detail::one<ValueType>();

            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                const uint8_t membershipFunction = mRuleTable[(rule * NumberOfInputs) + i];

                if (membershipFunction >= static_cast<uint8_t>(NumberOfMembershipFunctionsPerInput))
                {
                    continue; // don't care: contributes a grade of 1
                }

                const ValueType grade =
                    mMembershipGrades[(i * NumberOfMembershipFunctionsPerInput) + membershipFunction];

                if (grade == zeroValue)
                {
                    return zeroValue;
                }

                firingStrength = firingStrength * grade;
            }

            return firingStrength;
        }

        ValueType evaluateConsequent(const size_t rule, const size_t out, ValueType const* const input) const
        {
            const size_t base = ((rule * NumberOfOutputs) + out) * NumberOfConsequentParametersPerRule;

            if (!FirstOrderConsequent)
            {
                return mConsequentParameters[base];
            }

            ValueType result = mConsequentParameters[base + NumberOfInputs]; // constant term

            for (size_t i = 0; i < NumberOfInputs; ++i)
            {
                result += (mConsequentParameters[base + i] * input[i]);
            }

            return result;
        }

        static_assert(NumberOfInputs > 0, "ANFIS requires at least one input.");
        static_assert(NumberOfOutputs > 0, "ANFIS requires at least one output.");
        static_assert(NumberOfRules > 0, "ANFIS requires at least one rule.");
        static_assert(NumberOfMembershipFunctionsPerInput > 0,
                      "ANFIS requires at least one membership function per input.");
        static_assert(NumberOfMembershipFunctionsPerInput < 256,
                      "Rule table antecedent indices are uint8_t; membership functions per input must be < 256.");
        static_assert(MembershipFunctionPolicy::NumberOfParameters > 0,
                      "A membership function policy must declare at least one parameter.");
    };

    /**
     * Full-grid rule table: every combination of one membership function per
     * input, NumberOfMembershipFunctionsPerInput^NumberOfInputs rules.
     *
     * This is the classic ANFIS rule base and the source of its curse of
     * dimensionality — 4 inputs with 3 membership functions is 81 rules, 8
     * inputs is 6561. Published fuzzy hardware overwhelmingly stays under ~25
     * rules, so treat the full grid as a starting point for an offline pruning
     * pass rather than as a deployment target.
     *
     * The generated antecedent for the last input varies fastest.
     */
    template<size_t NumberOfInputs, size_t NumberOfMembershipFunctionsPerInput>
    struct AnfisFullGridRuleTable
    {
        static const size_t NumberOfRules =
            anfis_detail::Power<NumberOfMembershipFunctionsPerInput, NumberOfInputs>::value;
        static const size_t RuleTableSize = NumberOfRules * NumberOfInputs;

        /**
         * @param ruleTable Caller-owned buffer of RuleTableSize entries
         */
        static void generate(uint8_t* ruleTable)
        {
            for (size_t rule = 0; rule < NumberOfRules; ++rule)
            {
                size_t remainder = rule;

                for (size_t i = NumberOfInputs; i > 0; --i)
                {
                    ruleTable[(rule * NumberOfInputs) + (i - 1)] =
                        static_cast<uint8_t>(remainder % NumberOfMembershipFunctionsPerInput);
                    remainder /= NumberOfMembershipFunctionsPerInput;
                }
            }
        }

        static_assert(NumberOfInputs > 0, "A rule table requires at least one input.");
        static_assert(NumberOfMembershipFunctionsPerInput > 0,
                      "A rule table requires at least one membership function per input.");
        static_assert(NumberOfMembershipFunctionsPerInput < 256,
                      "Rule table antecedent indices are uint8_t; membership functions per input must be < 256.");
    };
}
