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


// Tests for QSoftmaxInt16 (cpp/qsoftmax_fixedpoint.hpp). Included by
// quantization_unit_test.cpp inside its test suite, after <algorithm>,
// <cmath>, <random> and <vector>.

#include "qsoftmax16_tflite_golden.hpp"

#include <cstring>

namespace qsoftmax16_test
{
    template<std::size_t F, std::size_t N>
    void checkGolden(uint32_t scaleBits, const int16_t (&input)[N], const int16_t (&output)[N])
    {
        static_assert(N % F == 0, "whole rows");
        float scale;
        std::memcpy(&scale, &scaleBits, sizeof(scale));
        tinymind::QSoftmaxInt16<1, F> layer;
        // TFLite widens its float32 scale and beta to double.
        tinymind::buildQSoftmaxInt16Params(static_cast<double>(scale), 1.0, layer.input_multiplier, layer.input_shift);
        std::size_t mismatches = 0;
        for (std::size_t r = 0; r < N / F; ++r)
        {
            int16_t got[F];
            layer.forward(input + r * F, got);
            for (std::size_t i = 0; i < F; ++i)
            {
                mismatches += (got[i] != output[r * F + i]) ? 1u : 0u;
            }
        }
        BOOST_TEST(mismatches == 0u, "features " << F << " scale " << scale);
    }
}

BOOST_AUTO_TEST_CASE(qsoftmax16_tables_match_tflite_builder)
{
    // The shipped tables are what buildQSoftmaxInt16LUTs (TFLite's
    // LUTPopulateInt16<double>, operation for operation) builds.
    std::vector<int16_t> expLut(tinymind::detail::kQSoftmaxInt16LUTSize);
    std::vector<int16_t> reciprocalLut(tinymind::detail::kQSoftmaxInt16LUTSize);
    tinymind::buildQSoftmaxInt16LUTs(expLut.data(), reciprocalLut.data());
    std::size_t differences = 0;
    for (std::size_t i = 0; i < expLut.size(); ++i)
    {
        differences += (expLut[i] != tinymind::detail::kQSoftmaxInt16ExpLUT[i]) ? 1u : 0u;
        differences += (reciprocalLut[i] != tinymind::detail::kQSoftmaxInt16ReciprocalLUT[i]) ? 1u : 0u;
    }
    BOOST_TEST(differences == 0u);
    // Their ends: exp(0) and 1 / (1 + 0) are 1, clamped to 32767; 1 / 2 is 16384.
    BOOST_TEST(tinymind::detail::kQSoftmaxInt16ExpLUT[512] == 32767);
    BOOST_TEST(tinymind::detail::kQSoftmaxInt16ReciprocalLUT[0] == 32767);
    BOOST_TEST(tinymind::detail::kQSoftmaxInt16ReciprocalLUT[512] == 16384);
}

BOOST_AUTO_TEST_CASE(qsoftmax16_matches_tflite)
{
    // Rows TFLite's own int16 SOFTMAX produced (gen_qsoftmax16_golden.py):
    // uniform rows at the int16 extremes, one class at the top, random rows.
    using namespace qsoftmax16_golden;
    qsoftmax16_test::checkGolden<3>(kCase0ScaleBits, kCase0Input, kCase0Output);
    qsoftmax16_test::checkGolden<10>(kCase1ScaleBits, kCase1Input, kCase1Output);
    qsoftmax16_test::checkGolden<10>(kCase2ScaleBits, kCase2Input, kCase2Output);
    qsoftmax16_test::checkGolden<64>(kCase3ScaleBits, kCase3Input, kCase3Output);
    qsoftmax16_test::checkGolden<257>(kCase4ScaleBits, kCase4Input, kCase4Output);
}

BOOST_AUTO_TEST_CASE(qsoftmax16_rows_in_place_and_float)
{
    std::mt19937 rng(31);
    std::normal_distribution<double> logit(0.0, 3.0);
    constexpr std::size_t R = 5;
    constexpr std::size_t F = 12;
    const double scale = 12.0 / 32768.0; // logits up to +-12
    tinymind::QSoftmaxInt16<R, F> layer;
    tinymind::buildQSoftmaxInt16Params(scale, 1.0, layer.input_multiplier, layer.input_shift);
    tinymind::QSoftmaxInt16<1, F> one = {layer.input_multiplier, layer.input_shift};

    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial)
    {
        int16_t x[R * F];
        for (int16_t& v : x)
        {
            const double q = std::round(logit(rng) / scale);
            v = static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, q)));
        }
        int16_t y[R * F];
        layer.forward(x, y);

        // Row by row, and in place, give the same outputs.
        int16_t inPlace[R * F];
        std::copy(x, x + R * F, inPlace);
        layer.forward(inPlace, inPlace);
        for (std::size_t r = 0; r < R; ++r)
        {
            int16_t row[F];
            one.forward(x + r * F, row);
            BOOST_TEST(std::equal(row, row + F, y + r * F));
        }
        BOOST_TEST(std::equal(inPlace, inPlace + R * F, y));

        // Close to the float softmax of the dequantized logits.
        for (std::size_t r = 0; r < R; ++r)
        {
            double m = -1e9;
            for (std::size_t i = 0; i < F; ++i)
            {
                m = std::max(m, x[r * F + i] * scale);
            }
            double sum = 0.0;
            for (std::size_t i = 0; i < F; ++i)
            {
                sum += std::exp(x[r * F + i] * scale - m);
            }
            for (std::size_t i = 0; i < F; ++i)
            {
                const double want = std::exp(x[r * F + i] * scale - m) / sum;
                worst = std::max(worst, std::fabs(y[r * F + i] / 32768.0 - want));
            }
        }
    }
    BOOST_TEST_MESSAGE("QSoftmaxInt16 largest error against float: " << worst);
    BOOST_TEST(worst < 1e-3); // measured: 4.0e-4
}
