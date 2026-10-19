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
 * Fixed-point int8 softmax, bit-exact with the TFLite int8 softmax.
 *
 * QSoftmax1D (qsoftmax.hpp) normalizes through a host-built exp table and
 * an int64 division, so it agrees with TFLite only to within rounding. A
 * model imported from TFLite whose graph ends in SOFTMAX needs the exact
 * output, or its int8 results cannot be checked against TFLite Micro bit
 * for bit. This layer reproduces the TFLite reference kernel (gemmlowp
 * fixed point), which CMSIS-NN's arm_softmax_s8 also reproduces:
 *
 *   1. max over the row;
 *   2. for each x with d = x - max >= diff_min:
 *        e = exp(d * input_scale * beta), computed in Q5.26 -> Q0.31 by
 *        gemmlowp's exp_on_negative_values;
 *        sum += e rounded to Q12.19;
 *   3. 1 / sum by three Newton-Raphson steps (one_over_one_plus_x);
 *   4. y = e / sum, rounded onto the TFLite int8 output convention
 *      (scale 1/256, zero point -128); y = -128 where d < diff_min.
 *
 * The three parameters come from the input scale and beta exactly as
 * TFLite Micro computes them (PreprocessSoftmaxScaling, then
 * CalculateInputRadius); buildQSoftmaxFixedPointParams in qcalibration.hpp
 * derives them on the host.
 *
 * Pure integer at runtime; freestanding-safe.
 */

namespace tinymind {

    namespace detail {

        // gemmlowp SaturatingRoundingMultiplyByPOT<Exponent> for Exponent > 0.
        inline int32_t fixedPointMulByPOTSaturating(int32_t x, int32_t exponent)
        {
            const int32_t threshold = static_cast<int32_t>(
                (static_cast<uint32_t>(1) << (31 - exponent)) - 1u);
            if (x > threshold)
            {
                return static_cast<int32_t>(0x7FFFFFFF);
            }
            if (x < -threshold)
            {
                return static_cast<int32_t>(0x80000000);
            }
            return x * (static_cast<int32_t>(1) << exponent);
        }

        // gemmlowp exp_on_negative_values: a <= 0 in Q5.26, exp(a) in Q0.31.
        inline int32_t fixedPointExpOnNegativeValues(int32_t a)
        {
            const int32_t oneQuarter = static_cast<int32_t>(1) << 24;
            const int32_t aModQuarterMinusOneQuarter = (a & (oneQuarter - 1)) - oneQuarter;

            // exp on [-1/4, 0): Taylor series around -1/8, in Q0.31.
            const int32_t x = aModQuarterMinusOneQuarter * 32 + (static_cast<int32_t>(1) << 28);
            const int32_t x2 = saturatingRoundingDoublingHighMul(x, x);
            const int32_t x3 = saturatingRoundingDoublingHighMul(x2, x);
            const int32_t x4 = saturatingRoundingDoublingHighMul(x2, x2);
            const int32_t x4Over4 = roundingDivideByPOT(x4, 2);
            const int32_t series = roundingDivideByPOT(
                saturatingRoundingDoublingHighMul(x4Over4 + x3, 715827883) + x2, 1);
            const int32_t expOfMinusOneEighth = 1895147668;
            int32_t result = expOfMinusOneEighth +
                saturatingRoundingDoublingHighMul(expOfMinusOneEighth, x + series);

            // The whole quarters: multiply in exp(-2^k) for every set bit.
            static const int32_t expOfMinusPowerOfTwo[7] =
            {
                1672461947, // exp(-1/4)
                1302514674, // exp(-1/2)
                790015084,  // exp(-1)
                290630308,  // exp(-2)
                39332535,   // exp(-4)
                720401,     // exp(-8)
                242,        // exp(-16)
            };
            const int32_t remainder = aModQuarterMinusOneQuarter - a;
            for (int32_t k = 0; k < 7; ++k)
            {
                if ((remainder & (static_cast<int32_t>(1) << (24 + k))) != 0)
                {
                    result = saturatingRoundingDoublingHighMul(result, expOfMinusPowerOfTwo[k]);
                }
            }

            return (a == 0) ? static_cast<int32_t>(0x7FFFFFFF) : result;
        }

        // gemmlowp one_over_one_plus_x_for_x_in_0_1: a in Q0.31, 1 / (1 + a) in Q0.31.
        inline int32_t fixedPointOneOverOnePlusX(int32_t a)
        {
            const int64_t sum = static_cast<int64_t>(a) + static_cast<int64_t>(0x7FFFFFFF);
            const int32_t halfDenominator = static_cast<int32_t>((sum + ((sum >= 0) ? 1 : -1)) / 2);

            // Newton-Raphson in Q2.29, from the initial guess 48/17 - 32/17 * d.
            int32_t x = 1515870810 + saturatingRoundingDoublingHighMul(halfDenominator, -1010580540);
            const int32_t one = static_cast<int32_t>(1) << 29;
            for (int i = 0; i < 3; ++i)
            {
                const int32_t oneMinusHalfDenominatorTimesX =
                    one - saturatingRoundingDoublingHighMul(halfDenominator, x);
                x += fixedPointMulByPOTSaturating(
                    saturatingRoundingDoublingHighMul(x, oneMinusHalfDenominatorTimesX), 2);
            }
            return fixedPointMulByPOTSaturating(x, 1);
        }

        inline int32_t countLeadingZeros32(uint32_t x)
        {
#if defined(__GNUC__) || defined(__clang__)
            return (x == 0u) ? 32 : static_cast<int32_t>(__builtin_clz(x));
#else
            int32_t n = 0;
            for (uint32_t bit = static_cast<uint32_t>(1) << 31; bit != 0u && (x & bit) == 0u; bit >>= 1)
            {
                ++n;
            }
            return n;
#endif
        }

    } // namespace detail

    template<std::size_t Rows_, std::size_t Features_>
    struct QSoftmaxFixedPoint1D
    {
        typedef int8_t InputType;
        typedef int8_t OutputType;

        static constexpr std::size_t Rows     = Rows_;
        static constexpr std::size_t Features = Features_;
        static constexpr std::size_t Size     = Rows_ * Features_;

        static_assert(Rows_ > 0, "Rows must be > 0.");
        static_assert(Features_ > 0, "Features must be > 0.");
        // Each term of the sum of exps is at most 1 (2^19 in Q12.19). TFLite
        // requires the final rounding shift to stay <= 31, which holds while
        // the sum is below 512.
        static_assert(Features_ < 512, "Features must be < 512.");

        int32_t input_multiplier;  // Q0.31
        int32_t input_left_shift;  // >= 0
        int32_t diff_min;          // <= 0; x - max below this gives probability 0

        void forward(const InputType* input, OutputType* output) const
        {
            const int32_t multiplier = input_multiplier;
            const int32_t leftShift = input_left_shift;
            const int32_t diffMin = diff_min;
            const int32_t accumulationIntegerBits = 12;

            for (std::size_t r = 0; r < Rows_; ++r)
            {
                const InputType* const rowIn = input + r * Features_;
                OutputType* const rowOut = output + r * Features_;

                int32_t maxIn = static_cast<int32_t>(rowIn[0]);
                for (std::size_t i = 1; i < Features_; ++i)
                {
                    if (static_cast<int32_t>(rowIn[i]) > maxIn)
                    {
                        maxIn = static_cast<int32_t>(rowIn[i]);
                    }
                }

                int32_t sumOfExps = 0;
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const int32_t diff = static_cast<int32_t>(rowIn[i]) - maxIn;
                    if (diff >= diffMin)
                    {
                        const int32_t scaledDiff = saturatingRoundingDoublingHighMul(
                            diff * (static_cast<int32_t>(1) << leftShift), multiplier);
                        sumOfExps += roundingDivideByPOT(
                            detail::fixedPointExpOnNegativeValues(scaledDiff), accumulationIntegerBits);
                    }
                }

                // The max term contributes exactly 1, so sumOfExps >= 2^19 > 0.
                const int32_t headroomPlusOne =
                    detail::countLeadingZeros32(static_cast<uint32_t>(sumOfExps));
                const int32_t shiftedSumMinusOne = static_cast<int32_t>(
                    (static_cast<uint32_t>(sumOfExps) << headroomPlusOne) -
                    (static_cast<uint32_t>(1) << 31));
                const int32_t shiftedScale = detail::fixedPointOneOverOnePlusX(shiftedSumMinusOne);
                const int32_t exponent = (accumulationIntegerBits - headroomPlusOne) + 31 - 8;

                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const int32_t diff = static_cast<int32_t>(rowIn[i]) - maxIn;
                    int32_t q = -128;
                    if (diff >= diffMin)
                    {
                        const int32_t scaledDiff = saturatingRoundingDoublingHighMul(
                            diff * (static_cast<int32_t>(1) << leftShift), multiplier);
                        const int32_t expValue = detail::fixedPointExpOnNegativeValues(scaledDiff);
                        q = roundingDivideByPOT(
                                saturatingRoundingDoublingHighMul(shiftedScale, expValue), exponent) - 128;
                        if (q > 127)
                        {
                            q = 127;
                        }
                        if (q < -128)
                        {
                            q = -128;
                        }
                    }
                    rowOut[i] = static_cast<OutputType>(q);
                }
            }
        }
    };

} // namespace tinymind
