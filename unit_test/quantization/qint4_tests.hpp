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

// Tests for cpp/qint4.hpp. Included by quantization_unit_test.cpp inside its
// test suite, after <algorithm>, <cmath>, <random> and <vector>.

BOOST_AUTO_TEST_CASE(qint4_nibbles_and_packing)
{
    // Every byte: the two nibbles sign-extend as two's complement 4-bit values.
    for (int b = 0; b < 256; ++b)
    {
        const uint8_t byte = static_cast<uint8_t>(b);
        const int low = b & 15;
        const int high = b >> 4;
        BOOST_TEST(tinymind::unpackInt4(&byte, 0) == ((low >= 8) ? low - 16 : low));
        BOOST_TEST(tinymind::unpackInt4(&byte, 1) == ((high >= 8) ? high - 16 : high));
    }

    // Round trip of every value, at every position, for even and odd counts.
    for (std::size_t n = 1; n <= 33; ++n)
    {
        std::vector<int8_t> values(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            values[i] = static_cast<int8_t>(static_cast<int>((i * 7 + n) % 16) - 8);
        }
        std::vector<uint8_t> packed(tinymind::int4PackedBytes(n), 0xAA);
        tinymind::packInt4(values.data(), n, packed.data());
        BOOST_TEST(packed.size() == (n + 1) / 2);
        for (std::size_t i = 0; i < n; ++i)
        {
            BOOST_TEST(tinymind::unpackInt4(packed.data(), i) == values[i]);
        }
        if (n % 2 == 1)
        {
            BOOST_TEST((packed.back() >> 4) == 0); // trailing nibble cleared
        }
    }
    // TFLite / ONNX order: element 0 in the low nibble.
    const int8_t pair[2] = {-3, 5};
    uint8_t byte = 0;
    tinymind::packInt4(pair, 2, &byte);
    BOOST_TEST(byte == 0x5D);
}

namespace qint4_test
{
    // QDenseInt4 against an int8 reference with the same weight values, for
    // one shape. The int4 layer must be bit-exact with the int8 kernels.
    template<std::size_t In, std::size_t Out>
    void checkAgainstInt8(std::mt19937& rng)
    {
        std::uniform_int_distribution<int> code(-8, 7);
        std::uniform_int_distribution<int> value(-128, 127);
        std::uniform_int_distribution<int32_t> bias(-20000, 20000);
        std::uniform_int_distribution<int32_t> shift(4, 12);
        std::uniform_int_distribution<int32_t> mult(1 << 29, (1 << 30) + (1 << 29));
        std::vector<int8_t> w(In * Out);
        for (std::size_t i = 0; i < w.size(); ++i)
        {
            w[i] = static_cast<int8_t>(code(rng));
        }
        std::vector<uint8_t> packed(tinymind::int4PackedBytes(In * Out));
        tinymind::packInt4(w.data(), w.size(), packed.data());
        std::vector<int32_t> biases(Out);
        std::vector<tinymind::Requantizer<int32_t, int8_t> > rqs(Out);
        for (std::size_t o = 0; o < Out; ++o)
        {
            biases[o] = bias(rng);
            rqs[o].multiplier = mult(rng);
            rqs[o].shift = shift(rng);
            rqs[o].zero_point = static_cast<int8_t>(value(rng) / 4);
            rqs[o].qmin = -128;
            rqs[o].qmax = 127;
        }
        for (int withBias = 0; withBias < 2; ++withBias)
        {
            const int8_t zp = static_cast<int8_t>(value(rng));
            const tinymind::QDenseInt4<int8_t, int32_t, int8_t, In, Out> int4Layer = {
                packed.data(), withBias ? biases.data() : nullptr, zp, rqs.data()};
            // The int8 path: a 1x1 per-channel convolution is a dense layer
            // with one requantizer per output.
            tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, In, 1, 1, 1, 1, Out> int8Layer;
            int8Layer.weights = w.data();
            int8Layer.biases = withBias ? biases.data() : nullptr;
            int8Layer.input_zero_point = zp;
            int8Layer.requantizers = rqs.data();
            for (int trial = 0; trial < 20; ++trial)
            {
                int8_t x[In];
                for (std::size_t i = 0; i < In; ++i)
                {
                    x[i] = static_cast<int8_t>(value(rng));
                }
                int8_t got[Out];
                int8_t want[Out];
                int4Layer.forward(x, got);
                int8Layer.forward(x, want);
                for (std::size_t o = 0; o < Out; ++o)
                {
                    BOOST_TEST(got[o] == want[o], "In " << In << " Out " << Out << " o " << o);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(qint4_dense_matches_int8_kernel)
{
    std::mt19937 rng(21);
    // Even and odd input counts: with an odd count every second row starts
    // in the high nibble of a byte.
    qint4_test::checkAgainstInt8<1, 3>(rng);
    qint4_test::checkAgainstInt8<2, 5>(rng);
    qint4_test::checkAgainstInt8<7, 4>(rng);
    qint4_test::checkAgainstInt8<16, 10>(rng);
    qint4_test::checkAgainstInt8<33, 9>(rng);
    qint4_test::checkAgainstInt8<256, 10>(rng);

    // Extremes: every input -128 against zero point 127 and every weight -8.
    const std::size_t In = 64;
    std::vector<int8_t> w(In, -8);
    std::vector<uint8_t> packed(tinymind::int4PackedBytes(In));
    tinymind::packInt4(w.data(), In, packed.data());
    tinymind::Requantizer<int32_t, int8_t> rq;
    rq.multiplier = 1 << 30;
    rq.shift = 20;
    rq.zero_point = 0;
    rq.qmin = -128;
    rq.qmax = 127;
    const tinymind::QDenseInt4<int8_t, int32_t, int8_t, In, 1> layer = {packed.data(), nullptr, 127, &rq};
    int8_t x[In];
    std::fill(x, x + In, static_cast<int8_t>(-128));
    int8_t out[1];
    layer.forward(x, out);
    // acc = 64 * (-255) * (-8) = 130560; * 2^30 / 2^31 / 2^20, rounded.
    BOOST_TEST(out[0] == 0);
    rq.shift = 8;
    const tinymind::QDenseInt4<int8_t, int32_t, int8_t, In, 1> wide = {packed.data(), nullptr, 127, &rq};
    wide.forward(x, out);
    BOOST_TEST(out[0] == 127); // 255 saturates
}

BOOST_AUTO_TEST_CASE(qint4_per_channel_quantization)
{
    std::mt19937 rng(22);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const std::size_t channels = 6;
    const std::size_t per = 41;
    std::vector<float> w(channels * per);
    for (std::size_t c = 0; c < channels; ++c)
    {
        for (std::size_t i = 0; i < per; ++i)
        {
            w[c * per + i] = (c == 5) ? 0.0f : gauss(rng) * static_cast<float>(c + 1);
        }
    }
    std::vector<int8_t> codes(channels * per);
    std::vector<float> scales(channels);
    tinymind::quantizeInt4PerChannel(w.data(), channels, per, codes.data(), scales.data());
    for (std::size_t c = 0; c < channels; ++c)
    {
        bool reachesSeven = false;
        for (std::size_t i = 0; i < per; ++i)
        {
            const int q = codes[c * per + i];
            BOOST_TEST(q >= -7);
            BOOST_TEST(q <= 7);
            reachesSeven = reachesSeven || (q == 7 || q == -7);
            // Rounded to nearest on the channel's grid.
            BOOST_TEST(std::fabs(q * scales[c] - w[c * per + i]) <= 0.5f * scales[c] * 1.0001f);
        }
        if (c < 5)
        {
            BOOST_TEST(reachesSeven); // the largest weight uses the full range
        }
    }
    BOOST_TEST(scales[5] == 1.0f); // all-zero channel
}
