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

#include "qaffine.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized Takagi-Sugeno ANFIS (int8 affine tier).
 *
 * Integer counterpart of cpp/anfis.hpp. Same five stages, same explicit rule
 * table, same readable firing strengths -- but the runtime path is pure
 * integer, so it composes with the rest of the q*.hpp family without a bridge.
 *
 * Three design points are worth stating up front, because each was measured
 * rather than assumed.
 *
 * 1. GRADE PRECISION. Stage 2 multiplies N membership grades, each in [0, 1],
 *    so the product shrinks geometrically -- the opposite of a QDense
 *    accumulator, which only grows. Grades are therefore carried as uint16 on
 *    a 2^-16 grid, and the running product is renormalized back to 16 bits
 *    after each multiply (`(w * g) >> 16`), which keeps every intermediate
 *    inside uint32: 65535 * 65535 = 4294836225 < 2^32.
 *
 *    8-bit grades were tried and rejected. On a 4-input bell model the error
 *    grew 21x; on a 3-membership-function triangular model, 82x. The damage is
 *    not mainly rules underflowing to zero -- it is resolution collapse in the
 *    products that survive.
 *
 * 2. THE GRADE SCALE CANCELS. The fused defuzzification is
 *
 *        y = sum_r(w_r * f_r) / sum_r(w_r)
 *
 *    and a factor applied to every w_r divides out exactly. So the 2^-16 grid
 *    never has to be tracked through calibration: the quotient lands at
 *    whatever scale f_r carries, and the ordinary output Requantizer finishes
 *    the job. No new calibration machinery is needed for the divide itself.
 *
 * 3. CONSEQUENTS ARE PER-RULE QUANTIZED. A least-squares consequent solve
 *    produces coefficients spanning orders of magnitude -- on the bundled
 *    benchmark, a 49000x ratio between the largest and smallest. A single
 *    per-tensor weight scale annihilates the small-coefficient rules (the
 *    largest coefficient of one rule quantized to 1). Each rule therefore
 *    carries its own weight scale and its own rescaler onto a shared
 *    consequent grid, where the weighted sum happens. This is the same
 *    argument per-channel quantization makes for convolution, applied per
 *    rule.
 *
 *    Note for whoever trains these: even with per-rule scales, an
 *    unregularized solve is hostile to int8. Large cancelling coefficients
 *    amplify input quantization error. A small ridge penalty on the consequent
 *    solve costs ~20% float accuracy and buys ~4.7x int8 accuracy.
 *    apps/anfis_train exposes this as `ridge`.
 *
 * Pure integer at runtime; no float, no <cmath>, no stdlib. Safe at
 * TINYMIND_ENABLE_FLOAT=0 / TINYMIND_ENABLE_STD=0. Host-side parameter
 * construction lives in cpp/include/qcalibration.hpp.
 */

namespace tinymind {

    /**
     * Number of fractional bits in a membership grade. A grade is stored as an
     * unsigned integer on a 2^-QAnfisGradeBits grid; QAnfisGradeOne is the
     * largest representable value, standing in for mu = 1.
     *
     * 16 bits is not a free parameter -- see the header comment. It is the
     * smallest width that held on every model measured, and the largest that
     * keeps the renormalizing multiply inside uint32.
     */
    static const unsigned QAnfisGradeBits = 16u;
    static const uint32_t QAnfisGradeOne = 0xFFFFu;

    /**
     * Index into a membership grade lookup table for a given quantized input.
     *
     * The table is indexed by the raw int8 pattern shifted into [0, 255], so
     * the input zero_point cancels -- the table was built over the same grid
     * the input arrives on. Centralized so the layer and the host-side builder
     * cannot disagree.
     */
    inline std::size_t qAnfisGradeIndex(int32_t q_in)
    {
        int32_t i = q_in + 128;
        if (i < 0)   { i = 0; }
        if (i > 255) { i = 255; }
        return static_cast<std::size_t>(i);
    }

    /**
     * Antecedent index meaning "this rule ignores this input", contributing a
     * grade of 1. Matches the DontCareIndex convention in cpp/anfis.hpp: any
     * index at or above the membership function count is a don't-care.
     */
    static const uint8_t QAnfisDontCare = 0xFFu;

    /**
     * Per-rule rescaler onto the shared consequent grid.
     *
     * Deliberately not a Requantizer: that saturates into a narrow destination
     * type, and squeezing each rule's consequent to int8 before the weighted
     * sum would throw away exactly the precision the sum needs. This keeps the
     * result int32.
     */
    struct QAnfisRuleScale
    {
        int32_t multiplier;
        int32_t shift;

        int32_t apply(int32_t acc) const
        {
            return multiplyByQuantizedMultiplier(acc, multiplier, shift);
        }
    };

    /**
     * Quantized ANFIS with first-order (linear) Takagi-Sugeno consequents.
     *
     * @tparam InputStorage_   int8_t activations
     * @tparam WeightStorage_  int8_t consequent coefficients (symmetric, per rule)
     * @tparam GradeStorage_   uint16_t membership grades
     * @tparam AccumType_      int32_t accumulator / bias
     * @tparam OutputStorage_  int8_t output
     * @tparam NumInputs_      crisp inputs
     * @tparam NumMfsPerInput_ membership functions per input
     * @tparam NumRules_       rules in the table
     *
     * Caller-owned buffers, matching the rest of the q*.hpp family:
     *
     *   grade_lut     [NumInputs][NumMfsPerInput][256]  membership grades
     *   rule_table    [NumRules][NumInputs]             antecedent indices
     *   weights       [NumRules][NumInputs]             consequent coefficients
     *   biases        [NumRules]                        at rule r's own scale
     *   rule_scales   [NumRules]                        onto the shared grid
     */
    template<typename InputStorage_, typename WeightStorage_,
             typename GradeStorage_, typename AccumType_, typename OutputStorage_,
             std::size_t NumInputs_, std::size_t NumMfsPerInput_,
             std::size_t NumRules_>
    struct QAnfis
    {
        typedef InputStorage_  InputType;
        typedef WeightStorage_ WeightType;
        typedef GradeStorage_  GradeType;
        typedef AccumType_     AccumulatorType;
        typedef OutputStorage_ OutputType;

        static constexpr std::size_t NumInputs      = NumInputs_;
        static constexpr std::size_t NumMfsPerInput = NumMfsPerInput_;
        static constexpr std::size_t NumRules       = NumRules_;
        static constexpr std::size_t GradeLutSize   = NumInputs_ * NumMfsPerInput_ * 256u;
        static constexpr std::size_t RuleTableSize  = NumRules_ * NumInputs_;
        static constexpr std::size_t WeightsSize    = NumRules_ * NumInputs_;

        const GradeType*       grade_lut;
        const uint8_t*         rule_table;
        const WeightType*      weights;
        const AccumulatorType* biases;
        const QAnfisRuleScale* rule_scales;
        Requantizer<AccumulatorType, OutputType> output_requantizer;
        InputType input_zero_point;

        /**
         * Raw firing strength of a rule, on the 2^-QAnfisGradeBits grid.
         *
         * Bails out on the first zero grade: with compact-support membership
         * functions most rules contribute nothing for most inputs, and this is
         * what makes a large rule table affordable.
         */
        uint32_t firingStrength(const InputType* input, std::size_t rule) const
        {
            uint32_t w = 0u;
            bool seeded = false;

            for (std::size_t i = 0; i < NumInputs_; ++i)
            {
                const uint8_t mf = rule_table[(rule * NumInputs_) + i];

                if (mf >= static_cast<uint8_t>(NumMfsPerInput_))
                {
                    continue;   // don't care: grade of 1
                }

                const std::size_t idx =
                    ((i * NumMfsPerInput_) + static_cast<std::size_t>(mf)) * 256u
                    + qAnfisGradeIndex(static_cast<int32_t>(input[i]));

                const uint32_t g = static_cast<uint32_t>(grade_lut[idx]);

                if (g == 0u)
                {
                    return 0u;
                }

                if (!seeded)
                {
                    // Take the first live grade directly rather than
                    // multiplying it into a seed. Seeding with QAnfisGradeOne
                    // and multiplying would compute g * 65535 >> 16, which is
                    // g - 1 for most g: one ULP lost per rule, before the
                    // product has done any real work.
                    w = g;
                    seeded = true;
                    continue;
                }

                // Renormalize back onto the grade grid. Both operands are
                // <= 65535, so the product cannot exceed 2^32 - 1.
                w = static_cast<uint32_t>((w * g) >> QAnfisGradeBits);

                if (w == 0u)
                {
                    return 0u;
                }
            }

            // A rule whose every antecedent is don't-care matches everything.
            return seeded ? w : QAnfisGradeOne;
        }

        /**
         * Rule r's linear consequent, rescaled onto the shared consequent grid.
         */
        int32_t consequent(const InputType* input, std::size_t rule) const
        {
            const int32_t zp = static_cast<int32_t>(input_zero_point);

            AccumulatorType acc = (biases != nullptr)
                ? biases[rule]
                : static_cast<AccumulatorType>(0);

            const WeightType* row = weights + (rule * NumInputs_);

            for (std::size_t i = 0; i < NumInputs_; ++i)
            {
                acc += static_cast<AccumulatorType>(
                    static_cast<int32_t>(row[i]) *
                    (static_cast<int32_t>(input[i]) - zp));
            }

            return rule_scales[rule].apply(static_cast<int32_t>(acc));
        }

        /**
         * Forward pass. Returns the quantized output.
         *
         * Stages 3 and 5 are fused into one divide, exactly as the float layer
         * does -- and here the grade scale cancels in that divide, so the
         * quotient arrives on the shared consequent grid and the output
         * requantizer is an ordinary one.
         *
         * If nothing fires, the output is the zero point (the quantized
         * representation of 0.0), never a divide by zero.
         */
        OutputType forward(const InputType* input) const
        {
            int64_t  numerator   = 0;
            uint64_t denominator = 0;

            for (std::size_t r = 0; r < NumRules_; ++r)
            {
                const uint32_t w = firingStrength(input, r);

                if (w == 0u)
                {
                    continue;
                }

                numerator += static_cast<int64_t>(w) *
                             static_cast<int64_t>(consequent(input, r));
                denominator += static_cast<uint64_t>(w);
            }

            if (denominator == 0u)
            {
                return output_requantizer.zero_point;
            }

            const int64_t den  = static_cast<int64_t>(denominator);
            const int64_t half = den / 2;

            // Round to nearest, symmetric about zero: a consequent can be
            // negative, and truncating toward zero would bias the output.
            const int64_t quotient = (numerator >= 0)
                ? ((numerator + half) / den)
                : -(((-numerator) + half) / den);

            return output_requantizer.apply(
                static_cast<AccumulatorType>(quotient));
        }

        /**
         * Index of the strongest rule for this input -- the int8 counterpart of
         * the float layer's interpretability accessors. Rule 0 is reported when
         * nothing fires.
         */
        std::size_t dominantRule(const InputType* input) const
        {
            std::size_t best = 0;
            uint32_t best_w = 0;

            for (std::size_t r = 0; r < NumRules_; ++r)
            {
                const uint32_t w = firingStrength(input, r);
                if (w > best_w)
                {
                    best_w = w;
                    best = r;
                }
            }

            return best;
        }

        static_assert(NumInputs_ > 0, "QAnfis requires at least one input.");
        static_assert(NumRules_ > 0, "QAnfis requires at least one rule.");
        static_assert(NumMfsPerInput_ > 0,
                      "QAnfis requires at least one membership function per input.");
        static_assert(NumMfsPerInput_ < 255,
                      "Antecedent indices are uint8_t; membership functions per input must be < 255.");
    };

} // namespace tinymind
