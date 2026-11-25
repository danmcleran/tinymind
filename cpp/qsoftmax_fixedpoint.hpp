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
#include "qactivations.hpp"
#include "qaffine.hpp"
#include "qsoftmax16_tables.hpp"

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
 * QSoftmaxInt16 is the int16 counterpart, bit-exact with TFLite's int16
 * softmax (reference_ops::SoftmaxInt16, the 16x8 scheme): int16 input with
 * zero point 0, int16 output on [0, 1) with scale 2^-15 and zero point 0.
 * It takes the exponentials from a 513-entry exp table over [-10, 0] and
 * the reciprocal of their sum from a 1 / (1 + x) table, both interpolated
 * by qApplyLUT16 (TFLite's LUTLookup), with TFLite's own tables
 * (qsoftmax16_tables.hpp). buildQSoftmaxInt16Params derives its two
 * parameters on the host.
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

    /**
     * int16 softmax over Rows rows of Features values, bit-exact with TFLite's
     * int16 SOFTMAX (reference_ops::SoftmaxInt16). For each row:
     *
     *   1. max over the row;
     *   2. for each x: d = x - max, rescaled by (input_multiplier,
     *      input_shift) so that -65535 maps to -10 and recentred by 32767
     *      onto the exp table's input, saturated to int16; e = the exp
     *      table (Q0.15) at it; sum += e;
     *   3. the sum normalized into [1, 2) and looked up in the 1 / (1 + x)
     *      table, giving its reciprocal in Q0.15 and a shift;
     *   4. y = e * reciprocal, rounded and shifted, clamped to [0, 32767].
     *
     * The output's scale is 2^-15 and its zero point 0, as TFLite requires
     * of an int16 softmax; the input's zero point must be 0. Every input
     * more than about 10 below its row's maximum (in units of
     * 1 / (input_scale * beta)) contributes the table's floor rather than 0,
     * as in TFLite. output may be input: each element is read before it is
     * written.
     *
     * input_shift has this library's sign (negative shifts left; TFLite's
     * input_left_shift is -input_shift). Out of TFLite's contract, a
     * difference that overflows int32 when shifted left saturates rather
     * than wrapping.
     */
    template<std::size_t Rows_, std::size_t Features_>
    struct QSoftmaxInt16
    {
        typedef int16_t InputType;
        typedef int16_t OutputType;

        static constexpr std::size_t Rows     = Rows_;
        static constexpr std::size_t Features = Features_;
        static constexpr std::size_t Size     = Rows_ * Features_;

        static_assert(Rows_ > 0, "Rows must be > 0.");
        static_assert(Features_ > 0, "Features must be > 0.");
        // The sum of a row's exponentials (each at most 32767) is an int32.
        static_assert(Features_ < 65536, "Features must be < 65536.");

        int32_t input_multiplier; // Q0.31
        int32_t input_shift;      // this library's sign: negative shifts left

        void forward(const InputType* input, OutputType* output) const
        {
            const int32_t multiplier = input_multiplier;
            const int32_t shift = input_shift;
            const int16_t* const expTable = detail::kQSoftmaxInt16ExpLUT;
            const int16_t* const reciprocalTable = detail::kQSoftmaxInt16ReciprocalLUT;

            for (std::size_t r = 0; r < Rows_; ++r)
            {
                const InputType* const rowIn = input + r * Features_;
                OutputType* const rowOut = output + r * Features_;

                int32_t maxIn = static_cast<int32_t>(rowIn[0]);
                for (std::size_t i = 1; i < Features_; ++i)
                {
                    const int32_t v = static_cast<int32_t>(rowIn[i]);
                    maxIn = (v > maxIn) ? v : maxIn;
                }

                // The exponentials go in the output row (as in TFLite), so
                // the row is read once and the second pass needs no buffer.
                int32_t sumOfExps = 0; // Q16.15
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    const int32_t diff = static_cast<int32_t>(rowIn[i]) - maxIn;
                    const int32_t scaled = multiplyByQuantizedMultiplier(diff, multiplier, shift);
                    // diff <= 0, so scaled <= 0 and scaled + 32767 cannot overflow.
                    int32_t centred = scaled + 32767;
                    centred = (centred < -32768) ? -32768 : centred;
                    const int16_t e = qApplyLUT16(static_cast<int16_t>(centred), expTable);
                    rowOut[i] = e;
                    sumOfExps += static_cast<int32_t>(e);
                }

                // The row's maximum contributes the table's top entry, 32767,
                // so the sum is at least that and headroomPlusOne <= 17.
                const int32_t headroomPlusOne = detail::countLeadingZeros32(static_cast<uint32_t>(sumOfExps));
                // The sum normalized to [2^30, 2^31) and taken to Q1.15 (in
                // uint32: the rounding term can carry past 2^31).
                const int32_t shiftedSum = static_cast<int32_t>(
                    ((static_cast<uint32_t>(sumOfExps) << (headroomPlusOne - 1)) + (1u << 13)) >> 14);
                // 1 / (1 + x) with x = shiftedSum / 2^16 - 1, recentred onto
                // the table's input [-32768, 32767].
                int32_t centredSum = shiftedSum - ((1 << 15) + (1 << 16));
                centredSum = (centredSum < -32768) ? -32768 : ((centredSum > 32767) ? 32767 : centredSum);
                const int32_t reciprocal = static_cast<int32_t>(
                    qApplyLUT16(static_cast<int16_t>(centredSum), reciprocalTable)); // Q0.15

                // e and reciprocal are at most 32767, so their product and
                // the rounding term (at most 2^29) stay inside int32.
                const int32_t rightShift = 31 - headroomPlusOne;
                const int32_t round = static_cast<int32_t>(1) << (rightShift - 1);
                for (std::size_t i = 0; i < Features_; ++i)
                {
                    int32_t y = (static_cast<int32_t>(rowOut[i]) * reciprocal + round) >> rightShift;
                    y = (y < 0) ? 0 : ((y > 32767) ? 32767 : y);
                    rowOut[i] = static_cast<OutputType>(y);
                }
            }
        }
    };

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    namespace detail {

        /**
         * TFLite's LUTPopulateInt16<double> (tensorflow/lite/kernels/internal/
         * common.h), operation for operation, so that a table it builds equals
         * TFLite's entry for entry: 512 samples spaced (input_max - input_min)
         * / 512 apart, each less half the linear interpolation's error at its
         * interval's midpoint, and a 513th at input_max. (buildQLUT16 in
         * qactivations.hpp puts its samples exactly on qApplyLUT16's knots
         * instead; this one keeps TFLite's spacing, which the softmax tables
         * must match to give TFLite's results.)
         */
        inline void populateTFLiteLUT16(double input_scale, int32_t input_zero_point, double output_scale,
                                        int32_t output_zero_point, double (*transform)(double), int16_t* lut)
        {
            const double input_min = input_scale * static_cast<double>(-32768 - input_zero_point);
            const double input_max = input_scale * static_cast<double>(32767 - input_zero_point);
            const double output_min = output_scale * static_cast<double>(-32768 - output_zero_point);
            const double output_max = output_scale * static_cast<double>(32767 - output_zero_point);

            const int nb_steps = 512;
            const double step = (input_max - input_min) / nb_steps;
            const double half_step = step / 2;
            const double output_scaling_inv = static_cast<double>(32767 - (-32768) + 1) / (output_max - output_min);
            const double table_min = -32768.0;
            const double table_max = 32767.0;

            for (int i = 0; i < nb_steps; ++i)
            {
                const double val = transform(input_min + i * step);
                const double val_midpoint = transform(input_min + i * step + half_step);
                const double val_next = transform(input_min + (i + 1) * step);

                const double sample_val = std::round(val * output_scaling_inv);
                const double midpoint_interp_val =
                    std::round((val_next * output_scaling_inv + std::round(val * output_scaling_inv)) / 2);
                const double midpoint_val = std::round(val_midpoint * output_scaling_inv);
                const double midpoint_err = midpoint_interp_val - midpoint_val;
                const double bias = std::round(midpoint_err / 2);

                const double v = sample_val - bias;
                lut[i] = static_cast<int16_t>((v < table_min) ? table_min : ((v > table_max) ? table_max : v));
            }
            const double last = std::round(transform(input_max) * output_scaling_inv);
            lut[nb_steps] = static_cast<int16_t>((last < table_min) ? table_min : ((last > table_max) ? table_max : last));
        }

        inline double softmaxExpTransform(double x)
        {
            return std::exp(x);
        }

        inline double softmaxReciprocalTransform(double x)
        {
            return 1.0 / (1.0 + x);
        }

    } // namespace detail

    /**
     * Build QSoftmaxInt16's two tables as TFLite's SoftmaxPrepare does for an
     * int16 softmax: exp(x) over [-10, 0] and 1 / (1 + x) over [0, 1], both
     * onto the symmetric [-1, 1] int16 output range. Host-side; the layer
     * uses the copies in qsoftmax16_tables.hpp, which a unit test checks
     * against this builder.
     */
    inline void buildQSoftmaxInt16LUTs(int16_t* exp_lut, int16_t* reciprocal_lut)
    {
        const double range = static_cast<double>(32767 - (-32768));
        detail::populateTFLiteLUT16(10.0 / range, 32767, 2.0 / range, 0, detail::softmaxExpTransform, exp_lut);
        detail::populateTFLiteLUT16(1.0 / range, -32768, 2.0 / range, 0, detail::softmaxReciprocalTransform,
                                    reciprocal_lut);
    }

    /**
     * QSoftmaxInt16's input multiplier and shift for an int16 input of scale
     * input_scale (zero point 0) and softmax beta, as TFLite computes them:
     * input_scale * beta / (10 / 65535), so that a difference from the row's
     * maximum of -65535 steps maps to -10. Pass the float32 scale and beta
     * widened to double, as TFLite does, to get its multiplier exactly.
     * input_shift has this library's sign (negative shifts left).
     */
    inline void buildQSoftmaxInt16Params(double input_scale, double beta, int32_t& input_multiplier,
                                         int32_t& input_shift)
    {
        quantizeMultiplier(input_scale * beta / (10.0 / 65535.0), input_multiplier, input_shift);
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
