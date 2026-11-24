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

namespace qint4_test
{
    template<typename Layer>
    void fillRequantizers(std::mt19937& rng, std::vector<int32_t>& biases,
                          std::vector<tinymind::Requantizer<int32_t, int8_t> >& rqs, std::size_t n)
    {
        std::uniform_int_distribution<int32_t> bias(-20000, 20000);
        std::uniform_int_distribution<int32_t> shift(4, 12);
        std::uniform_int_distribution<int32_t> mult(1 << 29, (1 << 30) + (1 << 29));
        std::uniform_int_distribution<int> zp(-32, 32);
        biases.resize(n);
        rqs.resize(n);
        for (std::size_t o = 0; o < n; ++o)
        {
            biases[o] = bias(rng);
            rqs[o].multiplier = mult(rng);
            rqs[o].shift = shift(rng);
            rqs[o].zero_point = static_cast<int8_t>(zp(rng));
            rqs[o].qmin = (o % 3 == 0) ? 0 : -128; // some outputs with a fused ReLU
            rqs[o].qmax = 127;
        }
    }

    // QConv2DInt4 against QConv2DPerChannel with the same weights as int8.
    template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW,
             std::size_t SH, std::size_t SW, std::size_t F,
             std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
    void checkConv(std::mt19937& rng)
    {
        typedef tinymind::QConv2DInt4<int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F, PT, PB, PL, PR> Int4;
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F,
                                            PT, PB, PL, PR> Int8;
        static_assert(Int4::OutputSize == Int8::OutputSize, "same output shape");
        const std::size_t total = F * KH * KW * C;
        std::uniform_int_distribution<int> code(-8, 7);
        std::uniform_int_distribution<int> value(-128, 127);
        std::vector<int8_t> w(total);
        for (int8_t& v : w)
        {
            v = static_cast<int8_t>(code(rng));
        }
        std::vector<uint8_t> packed(Int4::PackedWeightBytes);
        BOOST_TEST(packed.size() == tinymind::int4PackedBytes(total));
        tinymind::packInt4(w.data(), total, packed.data());
        std::vector<int32_t> biases;
        std::vector<tinymind::Requantizer<int32_t, int8_t> > rqs;
        fillRequantizers<Int4>(rng, biases, rqs, F);
        std::vector<int8_t> x(H * W * C);
        std::vector<int8_t> got(Int4::OutputSize);
        std::vector<int8_t> want(Int4::OutputSize);
        for (int trial = 0; trial < 6; ++trial)
        {
            const bool withBias = (trial % 2) == 0;
            const int8_t zp = static_cast<int8_t>((trial == 1) ? -128 : ((trial == 3) ? 127 : value(rng)));
            for (int8_t& v : x)
            {
                v = static_cast<int8_t>(value(rng));
            }
            const Int4 int4Layer = {packed.data(), withBias ? biases.data() : nullptr, zp, rqs.data()};
            Int8 int8Layer;
            int8Layer.weights = w.data();
            int8Layer.biases = withBias ? biases.data() : nullptr;
            int8Layer.input_zero_point = zp;
            int8Layer.requantizers = rqs.data();
            int4Layer.forward(x.data(), got.data());
            int8Layer.forward(x.data(), want.data());
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < got.size(); ++i)
            {
                mismatches += (got[i] != want[i]) ? 1u : 0u;
            }
            BOOST_TEST(mismatches == 0u, "conv " << H << "x" << W << "x" << C << " k" << KH << "x" << KW
                                                 << " f" << F << " trial " << trial);
        }
    }

    // QDepthwiseConv2DInt4 against QDepthwiseConv2D with int8 weights.
    template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW,
             std::size_t SH, std::size_t SW, std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
    void checkDepthwise(std::mt19937& rng)
    {
        typedef tinymind::QDepthwiseConv2DInt4<int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, PT, PB, PL, PR> Int4;
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW,
                                           PT, PB, PL, PR> Int8;
        const std::size_t total = C * KH * KW;
        std::uniform_int_distribution<int> code(-8, 7);
        std::uniform_int_distribution<int> value(-128, 127);
        std::vector<int8_t> w(total);
        for (int8_t& v : w)
        {
            v = static_cast<int8_t>(code(rng));
        }
        std::vector<uint8_t> packed(tinymind::int4PackedBytes(total));
        tinymind::packInt4(w.data(), total, packed.data());
        std::vector<int32_t> biases;
        std::vector<tinymind::Requantizer<int32_t, int8_t> > rqs;
        fillRequantizers<Int4>(rng, biases, rqs, C);
        std::vector<int8_t> x(H * W * C);
        std::vector<int8_t> got(Int4::OutputSize);
        std::vector<int8_t> want(Int4::OutputSize);
        for (int trial = 0; trial < 6; ++trial)
        {
            const bool withBias = (trial % 2) == 0;
            const int8_t zp = static_cast<int8_t>((trial == 1) ? -128 : ((trial == 3) ? 127 : value(rng)));
            for (int8_t& v : x)
            {
                v = static_cast<int8_t>(value(rng));
            }
            const Int4 int4Layer = {packed.data(), withBias ? biases.data() : nullptr, zp, rqs.data()};
            const Int8 int8Layer = {w.data(), withBias ? biases.data() : nullptr, zp, rqs.data()};
            int4Layer.forward(x.data(), got.data());
            int8Layer.forward(x.data(), want.data());
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < got.size(); ++i)
            {
                mismatches += (got[i] != want[i]) ? 1u : 0u;
            }
            BOOST_TEST(mismatches == 0u, "depthwise " << H << "x" << W << "x" << C << " k" << KH << "x" << KW
                                                      << " trial " << trial);
        }
    }
}

BOOST_AUTO_TEST_CASE(qint4_conv_matches_int8_kernel)
{
    std::mt19937 rng(23);
    // Patch path, an odd 9-weight window (every second filter mid-byte), SAME.
    qint4_test::checkConv<8, 8, 1, 3, 3, 1, 1, 8, 1, 1, 1, 1>(rng);
    // Patch path, a 72-weight window, two filters per pass on the Arm DSP backend.
    qint4_test::checkConv<8, 8, 8, 3, 3, 1, 1, 8, 1, 1, 1, 1>(rng);
    // Odd filter count and an odd row length in an even window.
    qint4_test::checkConv<7, 5, 3, 2, 3, 1, 1, 3, 0, 0, 0, 0>(rng);
    // Row path (a 144-weight window), stride 2, asymmetric padding.
    qint4_test::checkConv<6, 6, 16, 3, 3, 2, 2, 5, 0, 1, 0, 1>(rng);
    // Pointwise: a 1x1 kernel reads each pixel in place.
    qint4_test::checkConv<4, 4, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0>(rng);
    qint4_test::checkConv<3, 5, 7, 1, 1, 1, 1, 3, 0, 0, 0, 0>(rng);
    // A 1-row kernel with horizontal padding and stride.
    qint4_test::checkConv<5, 7, 3, 1, 3, 1, 2, 4, 0, 0, 1, 1>(rng);
}

BOOST_AUTO_TEST_CASE(qint4_conv_direct_path_matches_int8_kernel)
{
    // Layers above TINYMIND_INT4_CONV_UNPACK_MAX_BYTES compute from the
    // packed weights. The three forms of that path:
    std::mt19937 rng(25);
    const std::size_t unpackLimit = TINYMIND_INT4_CONV_UNPACK_MAX_BYTES;
    BOOST_TEST(unpackLimit == 1024u);
    // Copied to locals: BOOST_TEST binds a static constexpr member by reference.
    const bool sixteenFiltersUnpack = tinymind::QConv2DInt4<int8_t, int32_t, int8_t, 8, 8, 8, 3, 3, 1, 1, 16>::UnpacksWeights;
    const bool eightFiltersUnpack = tinymind::QConv2DInt4<int8_t, int32_t, int8_t, 8, 8, 8, 3, 3, 1, 1, 8>::UnpacksWeights;
    BOOST_TEST(!sixteenFiltersUnpack);
    BOOST_TEST(eightFiltersUnpack);
    // A gathered 72-weight patch, two filters per pass on the Arm DSP backend, SAME.
    qint4_test::checkConv<8, 8, 8, 3, 3, 1, 1, 16, 1, 1, 1, 1>(rng);
    // An odd 9-weight window over many filters (every second one mid-byte).
    qint4_test::checkConv<6, 6, 1, 3, 3, 1, 1, 121, 0, 0, 0, 0>(rng);
    // One dot product per kernel row (a 144-weight window), stride 2, asymmetric padding.
    qint4_test::checkConv<6, 6, 16, 3, 3, 2, 2, 9, 0, 1, 0, 1>(rng);
    // Pointwise, read in place.
    qint4_test::checkConv<4, 4, 32, 1, 1, 1, 1, 33, 0, 0, 0, 0>(rng);
}

BOOST_AUTO_TEST_CASE(qint4_depthwise_matches_int8_kernel)
{
    std::mt19937 rng(24);
    qint4_test::checkDepthwise<8, 8, 8, 3, 3, 1, 1, 0, 0, 0, 0>(rng);
    qint4_test::checkDepthwise<8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1>(rng);
    qint4_test::checkDepthwise<9, 7, 5, 3, 3, 2, 2, 1, 1, 1, 1>(rng);
    qint4_test::checkDepthwise<6, 6, 3, 2, 2, 1, 1, 0, 0, 0, 0>(rng);
}
