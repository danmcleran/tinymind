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

// Tests for the int16-activation, int8-weight ("16x8") path: the 64-bit
// Requantizer, the blocked int16 x int8 dot product, the layers over them,
// and the 513-entry int16 activation tables. Included by
// quantization_unit_test.cpp inside its test suite, after <algorithm>,
// <cmath>, <random> and <vector>.

namespace qint16x8_test
{
    typedef tinymind::Requantizer<int64_t, int16_t> Rq64;

    // TFLite's int64 MultiplyByQuantizedMultiplier, as written there
    // (tensorflow/lite/kernels/internal/common.cc), with TFLite's shift sign:
    // negative shifts right.
    int32_t tfliteMultiply64(int64_t x, int32_t quantized_multiplier, int shift)
    {
        const int32_t reduced_multiplier = (quantized_multiplier < 0x7FFF0000)
            ? ((quantized_multiplier + (1 << 15)) >> 16)
            : 0x7FFF;
        const int64_t total_shift = 15 - shift;
        const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
        int64_t result = x * static_cast<int64_t>(reduced_multiplier) + round;
        result = result >> total_shift;
        return static_cast<int32_t>(result);
    }

    int16_t tfliteRequantize16(int64_t acc, const Rq64& rq)
    {
        int32_t v = tfliteMultiply64(acc, rq.multiplier, -rq.shift) + rq.zero_point;
        v = std::max<int32_t>(v, rq.qmin);
        v = std::min<int32_t>(v, rq.qmax);
        return static_cast<int16_t>(v);
    }

    // TFLite's LUTLookup(int16_t, const int16_t*), as written there
    // (tensorflow/lite/kernels/internal/common.h).
    int16_t tfliteLutLookup(int16_t value, const int16_t* lut)
    {
        const uint16_t index = static_cast<uint16_t>(256 + (value >> 7));
        const int16_t offset = value & 0x7f;
        const int16_t base = lut[index];
        const int16_t slope = static_cast<int16_t>(lut[index + 1] - lut[index]);
        const int delta = (slope * offset + 64) >> 7;
        return static_cast<int16_t>(base + delta);
    }

    Rq64 randomRequantizer(std::mt19937& rng)
    {
        std::uniform_int_distribution<int32_t> mult(1 << 30, 0x7FFFFFFF);
        std::uniform_int_distribution<int32_t> shift(-7, 31);
        Rq64 rq;
        rq.multiplier = mult(rng);
        rq.shift = shift(rng);
        rq.zero_point = 0;
        rq.qmin = -32768;
        rq.qmax = 32767;
        return rq;
    }

    // A layer's per-channel requantizers, picked so that the outputs spread
    // over the int16 range: the accumulator scale times 2^14 / bound.
    std::vector<Rq64> spreadRequantizers(std::size_t n, double bound, std::mt19937& rng)
    {
        std::uniform_real_distribution<double> jitter(0.5, 2.0);
        std::vector<Rq64> rqs(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            int32_t m = 0;
            int32_t s = 0;
            tinymind::quantizeMultiplier(16384.0 / bound * jitter(rng), m, s);
            rqs[i].multiplier = m;
            rqs[i].shift = s;
            rqs[i].zero_point = 0;
            rqs[i].qmin = -32768;
            rqs[i].qmax = 32767;
        }
        return rqs;
    }

    // NHWC convolution with zero-point padding, one requantizer per filter.
    // Depthwise when Depthwise: weights [C][KH * KW], one filter per channel.
    template<bool Depthwise>
    std::vector<int16_t> naiveConv(const std::vector<int16_t>& x, std::size_t H, std::size_t W, std::size_t C,
                                   const std::vector<int8_t>& w, const std::vector<int64_t>& bias,
                                   std::size_t KH, std::size_t KW, std::size_t S, std::size_t F,
                                   std::size_t padT, std::size_t padB, std::size_t padL, std::size_t padR,
                                   int16_t zp, const std::vector<Rq64>& rqs)
    {
        const std::size_t OH = (H + padT + padB - KH) / S + 1;
        const std::size_t OW = (W + padL + padR - KW) / S + 1;
        const std::size_t outC = Depthwise ? C : F;
        std::vector<int16_t> out(OH * OW * outC);
        for (std::size_t oh = 0; oh < OH; ++oh)
        {
            for (std::size_t ow = 0; ow < OW; ++ow)
            {
                for (std::size_t f = 0; f < outC; ++f)
                {
                    int64_t acc = bias.empty() ? 0 : bias[f];
                    for (std::size_t kh = 0; kh < KH; ++kh)
                    {
                        for (std::size_t kw = 0; kw < KW; ++kw)
                        {
                            const long ih = static_cast<long>(oh * S + kh) - static_cast<long>(padT);
                            const long iw = static_cast<long>(ow * S + kw) - static_cast<long>(padL);
                            const bool inside = ih >= 0 && iw >= 0 && ih < static_cast<long>(H) &&
                                                iw < static_cast<long>(W);
                            for (std::size_t c = 0; c < (Depthwise ? 1u : C); ++c)
                            {
                                const std::size_t ch = Depthwise ? f : c;
                                const int64_t xv = inside
                                    ? static_cast<int64_t>(x[(static_cast<std::size_t>(ih) * W +
                                                              static_cast<std::size_t>(iw)) * C + ch])
                                    : static_cast<int64_t>(zp);
                                const int64_t wv = Depthwise
                                    ? w[f * KH * KW + kh * KW + kw]
                                    : w[((f * KH + kh) * KW + kw) * C + c];
                                acc += (xv - zp) * wv;
                            }
                        }
                    }
                    out[(oh * OW + ow) * outC + f] = tfliteRequantize16(acc, rqs[f]);
                }
            }
        }
        return out;
    }

    template<typename T>
    std::vector<T> randomVector(std::size_t n, int lo, int hi, std::mt19937& rng)
    {
        std::uniform_int_distribution<int> d(lo, hi);
        std::vector<T> v(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            v[i] = static_cast<T>(d(rng));
        }
        return v;
    }

    template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW, std::size_t S,
             std::size_t F, std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
    void checkConv(std::mt19937& rng)
    {
        typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int64_t, int16_t, H, W, C, KH, KW, S, S, F,
                                            PT, PB, PL, PR> Layer;
        const std::vector<int8_t> w = randomVector<int8_t>(F * KH * KW * C, -127, 127, rng);
        const std::vector<int64_t> bias = randomVector<int64_t>(F, -2000000, 2000000, rng);
        const std::vector<Rq64> rqs = spreadRequantizers(F, 32768.0 * 127.0 * std::sqrt(KH * KW * C * 1.0), rng);
        for (int trial = 0; trial < 4; ++trial)
        {
            const std::vector<int16_t> x = randomVector<int16_t>(H * W * C, -32768, 32767, rng);
            const int16_t zp = (trial == 3) ? static_cast<int16_t>(-1234) : static_cast<int16_t>(0);
            Layer layer;
            layer.weights = w.data();
            layer.biases = (trial == 1) ? nullptr : bias.data();
            layer.input_zero_point = zp;
            layer.requantizers = rqs.data();
            std::vector<int16_t> got(Layer::OutputSize);
            layer.forward(x.data(), got.data());
            const std::vector<int16_t> want = naiveConv<false>(
                x, H, W, C, w, (trial == 1) ? std::vector<int64_t>() : bias, KH, KW, S, F, PT, PB, PL, PR, zp, rqs);
            BOOST_TEST(got == want, "conv " << H << "x" << W << "x" << C << " k" << KH << " trial " << trial);
        }
    }

    template<std::size_t H, std::size_t W, std::size_t C, std::size_t K, std::size_t S, std::size_t P>
    void checkDepthwise(std::mt19937& rng)
    {
        typedef tinymind::QDepthwiseConv2D<int16_t, int8_t, int64_t, int16_t, H, W, C, K, K, S, S, P, P, P, P> Layer;
        const std::vector<int8_t> w = randomVector<int8_t>(C * K * K, -127, 127, rng);
        const std::vector<int64_t> bias = randomVector<int64_t>(C, -200000, 200000, rng);
        const std::vector<Rq64> rqs = spreadRequantizers(C, 32768.0 * 127.0 * K, rng);
        for (int trial = 0; trial < 3; ++trial)
        {
            const std::vector<int16_t> x = randomVector<int16_t>(H * W * C, -32768, 32767, rng);
            const int16_t zp = (trial == 2) ? static_cast<int16_t>(777) : static_cast<int16_t>(0);
            Layer layer;
            layer.weights = w.data();
            layer.biases = bias.data();
            layer.input_zero_point = zp;
            layer.requantizers = rqs.data();
            std::vector<int16_t> got(Layer::OutputSize);
            layer.forward(x.data(), got.data());
            const std::vector<int16_t> want =
                naiveConv<true>(x, H, W, C, w, bias, K, K, S, C, P, P, P, P, zp, rqs);
            BOOST_TEST(got == want, "depthwise pad " << P << " trial " << trial);
        }
    }
}

BOOST_AUTO_TEST_CASE(qint16x8_requantizer_matches_tflite)
{
    std::mt19937 rng(31);
    std::uniform_int_distribution<int64_t> wide(-(static_cast<int64_t>(1) << 47),
                                                (static_cast<int64_t>(1) << 47) - 1);
    std::uniform_int_distribution<int64_t> narrow(-(static_cast<int64_t>(1) << 24), static_cast<int64_t>(1) << 24);
    for (int i = 0; i < 200000; ++i)
    {
        qint16x8_test::Rq64 rq = qint16x8_test::randomRequantizer(rng);
        if (i % 4 == 0)
        {
            rq.zero_point = static_cast<int16_t>(static_cast<int32_t>(rng() % 2001) - 1000);
            rq.qmin = -1000;
            rq.qmax = 20000;
        }
        // Accumulators whose result lands inside int32, as TFLite requires.
        const int64_t acc = (i % 2 == 0) ? narrow(rng) : (wide(rng) >> (rq.shift < 16 ? 16 - rq.shift : 0));
        BOOST_TEST(rq.apply(acc) == qint16x8_test::tfliteRequantize16(acc, rq), "acc " << acc << " m "
                   << rq.multiplier << " s " << rq.shift);
        BOOST_TEST(rq.applyOneRounding(acc) == rq.apply(acc));
    }

    // Hand-worked values. m = 2^30 (0.5) reduces to 2^14: (1001 * 2^14 + 2^14) >> 15 rounds 500.5 up.
    qint16x8_test::Rq64 half = {1 << 30, 0, 0, -32768, 32767};
    BOOST_TEST(half.apply(1001) == 501);
    BOOST_TEST(half.apply(-1001) == -500); // ties round up, toward +infinity
    BOOST_TEST(half.apply(-1003) == -501);
    BOOST_TEST(half.oneRounding());
    // The largest multipliers reduce to 0x7FFF rather than overflowing to 2^15.
    const qint16x8_test::Rq64 top = {0x7FFFFFFF, 15, 0, -32768, 32767};
    BOOST_TEST(top.apply(static_cast<int64_t>(1) << 30) == 32767);
    BOOST_TEST(top.apply(static_cast<int64_t>(1) << 29) == 16384); // 16383.5 rounds up
    BOOST_TEST(top.apply(-(static_cast<int64_t>(1) << 30)) == -32767); // 2^15 would give -32768
    // Saturation and a left shift.
    const qint16x8_test::Rq64 left = {1 << 30, -7, 5, -32768, 32767};
    BOOST_TEST(left.apply(100) == 6405); // 100 * 0.5 * 2^7 + 5
    BOOST_TEST(left.apply(1000) == 32767);
    BOOST_TEST(left.apply(-1000) == -32768);

    // A quotient far beyond int32 (out of contract, as TFLite checks) must
    // saturate towards its own sign rather than wrap.
    const int64_t huge[] = {static_cast<int64_t>(1) << 46, (static_cast<int64_t>(1) << 47) - 1, INT64_MAX};
    const int32_t steep[] = {-14, -10, -7, -1, 0, 5};
    for (int64_t a : huge)
    {
        for (int32_t s : steep)
        {
            const qint16x8_test::Rq64 rq = {0x7FFFFFFF, s, 0, -32768, 32767};
            BOOST_TEST(rq.apply(a) == 32767, "acc " << a << " s " << s);
            BOOST_TEST(rq.apply(-a) == -32768, "acc " << -a << " s " << s);
        }
    }

    // Out of contract: defined, and still clamped to [qmin, qmax].
    const int64_t extremes[] = {INT64_MIN, INT64_MAX, -(static_cast<int64_t>(1) << 50), static_cast<int64_t>(1) << 50, 0};
    const int32_t shifts[] = {-100, -31, -15, -14, 31, 47, 48, 100};
    const int32_t mults[] = {INT32_MIN, -1, 0, 1, 0x7FFFFFFF};
    for (int64_t acc : extremes)
    {
        for (int32_t s : shifts)
        {
            for (int32_t m : mults)
            {
                const qint16x8_test::Rq64 rq = {m, s, 3, -100, 100};
                const int16_t v = rq.apply(acc);
                BOOST_TEST((v >= -100 && v <= 100));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(qint16x8_dot_product_blocks)
{
    std::mt19937 rng(32);
    const std::size_t lengths[] = {0, 1, 7, 255, 256, 257, 511, 512, 513, 1000};
    for (std::size_t n : lengths)
    {
        for (int trial = 0; trial < 3; ++trial)
        {
            std::vector<int16_t> x = qint16x8_test::randomVector<int16_t>(n, -32768, 32767, rng);
            std::vector<int8_t> w = qint16x8_test::randomVector<int8_t>(n, -128, 127, rng);
            const int16_t zp = static_cast<int16_t>(trial == 0 ? 0 : (trial == 1 ? 32767 : -32768));
            if (trial > 0)
            {
                // The block bound's worst case: every term at +-65535 * 128 with one sign.
                std::fill(x.begin(), x.end(), static_cast<int16_t>(trial == 1 ? -32768 : 32767));
                std::fill(w.begin(), w.end(), static_cast<int8_t>(-128));
            }
            int64_t want = 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                want += (static_cast<int64_t>(x[i]) - zp) * static_cast<int64_t>(w[i]);
            }
            const int64_t got = tinymind::simd::dotProductWithZeroPoint<int16_t, int8_t, int64_t>(
                x.data(), w.data(), n, zp);
            BOOST_TEST(got == want, "n " << n << " trial " << trial);
        }
    }
}

BOOST_AUTO_TEST_CASE(qint16x8_dense_conv_depthwise_match_reference)
{
    std::mt19937 rng(33);

    // Dense, against the 1x1 convolution reference. 300 inputs crosses a
    // dot-product block boundary.
    {
        const std::size_t In = 300;
        const std::size_t Out = 7;
        const std::vector<int8_t> w = qint16x8_test::randomVector<int8_t>(In * Out, -127, 127, rng);
        const std::vector<int64_t> bias = qint16x8_test::randomVector<int64_t>(Out, -3000000, 3000000, rng);
        const std::vector<qint16x8_test::Rq64> rqs = qint16x8_test::spreadRequantizers(1, 32768.0 * 127.0 * 17.0, rng);
        const std::vector<qint16x8_test::Rq64> perOutput(Out, rqs[0]);
        tinymind::QDense<int16_t, int8_t, int64_t, int16_t, In, Out> layer;
        layer.weights = w.data();
        layer.biases = bias.data();
        layer.input_zero_point = 0;
        layer.requantizer = rqs[0];
        for (int trial = 0; trial < 10; ++trial)
        {
            const std::vector<int16_t> x = qint16x8_test::randomVector<int16_t>(In, -32768, 32767, rng);
            int16_t got[Out];
            layer.forward(x.data(), got);
            const std::vector<int16_t> want = qint16x8_test::naiveConv<false>(
                x, 1, 1, In, w, bias, 1, 1, 1, Out, 0, 0, 0, 0, 0, perOutput);
            for (std::size_t o = 0; o < Out; ++o)
            {
                BOOST_TEST(got[o] == want[o]);
            }
        }
    }

    // Convolutions: the patch path (window of 18 int16 values) and the row
    // path (72), padded and not, strided and not, odd filter counts.
    qint16x8_test::checkConv<6, 5, 2, 3, 3, 1, 5, 0, 0, 0, 0>(rng);
    qint16x8_test::checkConv<6, 5, 2, 3, 3, 1, 4, 1, 1, 1, 1>(rng);
    qint16x8_test::checkConv<8, 8, 8, 3, 3, 1, 3, 1, 1, 1, 1>(rng);
    qint16x8_test::checkConv<9, 7, 8, 3, 3, 2, 6, 0, 0, 0, 0>(rng);

    // Depthwise: the unpadded and padded loops.
    qint16x8_test::checkDepthwise<7, 6, 3, 3, 1, 0>(rng);
    qint16x8_test::checkDepthwise<7, 6, 3, 3, 2, 1>(rng);
    // 289 taps: past the 256 an int32 window holds, so summed in int64.
    qint16x8_test::checkDepthwise<17, 17, 2, 17, 1, 0>(rng);

    // Max pooling compares in the storage type.
    const std::vector<int16_t> x = qint16x8_test::randomVector<int16_t>(4 * 4 * 2, -32768, 32767, rng);
    const tinymind::QMaxPool2D<int16_t, 4, 4, 2, 2, 2> pool = {};
    int16_t pooled[8];
    pool.forward(x.data(), pooled);
    for (std::size_t oh = 0; oh < 2; ++oh)
    {
        for (std::size_t ow = 0; ow < 2; ++ow)
        {
            for (std::size_t c = 0; c < 2; ++c)
            {
                int16_t m = -32768;
                for (std::size_t k = 0; k < 4; ++k)
                {
                    m = std::max(m, x[((oh * 2 + k / 2) * 4 + ow * 2 + k % 2) * 2 + c]);
                }
                BOOST_TEST(pooled[(oh * 2 + ow) * 2 + c] == m);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(qint16x8_int32_accumulator_variant)
{
    // TFLite's other 16x8 variant keeps int32 biases and accumulators; the
    // int32 Requantizer then writes int16 and the kernels take their fast
    // requantization form, which must agree with the reference form.
    std::mt19937 rng(34);
    typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int32_t, int16_t, 5, 5, 2, 3, 3, 1, 1, 3> Layer;
    const std::vector<int8_t> w = qint16x8_test::randomVector<int8_t>(3 * 9 * 2, -127, 127, rng);
    std::vector<tinymind::Requantizer<int32_t, int16_t> > rqs(3);
    for (std::size_t f = 0; f < 3; ++f)
    {
        int32_t m = 0;
        int32_t s = 0;
        tinymind::quantizeMultiplier(16384.0 / (32768.0 * 127.0 * 4.0), m, s);
        rqs[f].multiplier = m;
        rqs[f].shift = s;
        rqs[f].zero_point = static_cast<int16_t>(f == 2 ? 100 : 0);
        rqs[f].qmin = -32768;
        rqs[f].qmax = (f == 1) ? static_cast<int16_t>(1000) : static_cast<int16_t>(32767); // f == 1 takes apply()
    }
    tinymind::detail::FastRequantizer<int16_t> fast;
    BOOST_TEST(fast.init(rqs[0]));
    BOOST_TEST(!fast.init(rqs[1]));
    Layer layer;
    layer.weights = w.data();
    layer.biases = nullptr;
    layer.input_zero_point = 0;
    layer.requantizers = rqs.data();
    const std::vector<int16_t> x = qint16x8_test::randomVector<int16_t>(5 * 5 * 2, -32768, 32767, rng);
    std::vector<int16_t> got(Layer::OutputSize);
    layer.forward(x.data(), got.data());
    for (std::size_t p = 0; p < 9; ++p)
    {
        const std::size_t oh = p / 3;
        const std::size_t ow = p % 3;
        for (std::size_t f = 0; f < 3; ++f)
        {
            int32_t acc = 0;
            for (std::size_t k = 0; k < 18; ++k)
            {
                const std::size_t kh = k / 6;
                const std::size_t kw = (k / 2) % 3;
                const std::size_t c = k % 2;
                acc += static_cast<int32_t>(x[((oh + kh) * 5 + ow + kw) * 2 + c]) * w[f * 18 + k];
            }
            BOOST_TEST(got[p * 3 + f] == rqs[f].apply(acc));
        }
    }
    // A 64-bit requantizer never takes the 32-bit fast form.
    const qint16x8_test::Rq64 wide = {1 << 30, 4, 0, -32768, 32767};
    BOOST_TEST(!fast.init(wide));
}

BOOST_AUTO_TEST_CASE(qint16x8_add_uses_tflite_int16_parameters)
{
    // TFLite's int16 ADD: zero points 0, left_shift 15, the int8 arithmetic
    // otherwise. QAdd's wide form (FastAddInput64) runs it; the int8 fast
    // form needs more headroom than 15 bits leave.
    const float scaleA = 0.0021f;
    const float scaleB = 0.0007f;
    const float scaleOut = 0.0025f;
    const int32_t leftShift = 15;
    const double twiceMax = 2.0 * std::max(scaleA, scaleB);
    int32_t ma = 0, sa = 0, mb = 0, sb = 0, mo = 0, so = 0;
    tinymind::quantizeMultiplier(scaleA / twiceMax, ma, sa);
    tinymind::quantizeMultiplier(scaleB / twiceMax, mb, sb);
    tinymind::quantizeMultiplier(twiceMax / ((1 << leftShift) * static_cast<double>(scaleOut)), mo, so);
    const tinymind::QAdd<int16_t, int16_t, int16_t, 64> add = {
        0, 0, leftShift, ma, sa, mb, sb, {mo, so, 0, -32768, 32767}};
    std::mt19937 rng(35);
    std::vector<int16_t> a = qint16x8_test::randomVector<int16_t>(64, -32768, 32767, rng);
    std::vector<int16_t> b = qint16x8_test::randomVector<int16_t>(64, -32768, 32767, rng);
    a[0] = -32768;
    b[0] = -32768;
    a[1] = 32767;
    b[1] = 32767;
    int16_t out[64];
    add.forward(a.data(), b.data(), out);
    for (std::size_t i = 0; i < 64; ++i)
    {
        const int32_t sumA = tinymind::multiplyByQuantizedMultiplier(a[i] * (1 << leftShift), ma, sa);
        const int32_t sumB = tinymind::multiplyByQuantizedMultiplier(b[i] * (1 << leftShift), mb, sb);
        int32_t v = tinymind::multiplyByQuantizedMultiplier(sumA + sumB, mo, so);
        v = std::max(-32768, std::min(32767, v));
        BOOST_TEST(out[i] == v);
        // And it is the float sum, to within the rounding of three requantizations.
        const double exact = (a[i] * static_cast<double>(scaleA) + b[i] * static_cast<double>(scaleB)) / scaleOut;
        if (std::fabs(exact) < 32767.0)
        {
            BOOST_TEST(std::fabs(out[i] - exact) <= 1.5);
        }
    }

    // With 15 bits of left shift, the int8 fast form refuses an int16 input;
    // at 13 it has the headroom and must agree with the reference.
    tinymind::detail::FastAddInput<int16_t> fastIn;
    BOOST_TEST(!fastIn.init(0, 15, ma, sa));
    BOOST_TEST(fastIn.init(0, 13, ma, sa));
    const int32_t shift13 = 13;
    tinymind::quantizeMultiplier(twiceMax / ((1 << shift13) * static_cast<double>(scaleOut)), mo, so);
    const tinymind::QAdd<int16_t, int16_t, int16_t, 64> add13 = {
        0, 0, shift13, ma, sa, mb, sb, {mo, so, 0, -32768, 32767}};
    add13.forward(a.data(), b.data(), out);
    for (std::size_t i = 0; i < 64; ++i)
    {
        const int32_t sumA = tinymind::multiplyByQuantizedMultiplier(a[i] * (1 << shift13), ma, sa);
        const int32_t sumB = tinymind::multiplyByQuantizedMultiplier(b[i] * (1 << shift13), mb, sb);
        int32_t v = tinymind::multiplyByQuantizedMultiplier(sumA + sumB, mo, so);
        v = std::max(-32768, std::min(32767, v));
        BOOST_TEST(out[i] == v);
    }

    // A zero point of 1 with 15 bits of left shift leaves FastAddInput64 too
    // little headroom (|q - zp| reaches 32769), so QAdd falls back to its
    // reference loop, which must still give the TFLite arithmetic.
    tinymind::detail::FastAddInput64<int16_t> wideIn;
    BOOST_TEST(!wideIn.init(1, leftShift, ma, sa));
    tinymind::quantizeMultiplier(twiceMax / ((1 << leftShift) * static_cast<double>(scaleOut)), mo, so);
    const tinymind::QAdd<int16_t, int16_t, int16_t, 64> addZp = {
        1, 0, leftShift, ma, sa, mb, sb, {mo, so, 0, -32768, 32767}};
    addZp.forward(a.data(), b.data(), out);
    for (std::size_t i = 0; i < 64; ++i)
    {
        const int32_t sumA = tinymind::multiplyByQuantizedMultiplier((a[i] - 1) * (1 << leftShift), ma, sa);
        const int32_t sumB = tinymind::multiplyByQuantizedMultiplier(b[i] * (1 << leftShift), mb, sb);
        int32_t v = tinymind::multiplyByQuantizedMultiplier(sumA + sumB, mo, so);
        v = std::max(-32768, std::min(32767, v));
        BOOST_TEST(out[i] == v);
    }
}

BOOST_AUTO_TEST_CASE(qint16x8_add_wide_rescale_matches_reference)
{
    // FastAddInput64, QAdd's one-rounding input rescale for int16, against
    // the reference multiplyByQuantizedMultiplier((q - zp) << left_shift, m,
    // e) for every int16 input. It must accept every left shift up to 15
    // with a zero point of 0 (TFLite's int16 ADD), and refuse what would
    // overflow.
    const int32_t multipliers[] = {0, 1, 1 << 30, 1518500250, 2147483647};
    const int32_t shifts[] = {0, 1, 2, 3, 7, 15, 16, 30, 31};
    const int32_t leftShifts[] = {0, 1, 8, 13, 14, 15};
    const int16_t zeroPoints[] = {0, -1, 1, 300};
    std::size_t configurations = 0;
    for (int32_t m : multipliers)
    {
        for (int32_t e : shifts)
        {
            for (int32_t ls : leftShifts)
            {
                for (int16_t zp : zeroPoints)
                {
                    tinymind::detail::FastAddInput64<int16_t> fast;
                    // The largest |q - zp| over int16.
                    const int32_t span = std::max(static_cast<int32_t>(zp) + 32768, 32767 - static_cast<int32_t>(zp));
                    const bool fits = ((static_cast<int64_t>(span) << ls) <= (static_cast<int64_t>(1) << 30)) && (e <= 30);
                    BOOST_TEST(fast.init(zp, ls, m, e) == fits, "m " << m << " e " << e << " ls " << ls << " zp " << zp);
                    if (!fits)
                    {
                        continue;
                    }
                    ++configurations;
                    bool same = true;
                    for (int32_t q = -32768; q <= 32767 && same; ++q)
                    {
                        const int32_t x = (q - zp) * (1 << ls);
                        same = fast.apply(q) == tinymind::multiplyByQuantizedMultiplier(x, m, e);
                    }
                    BOOST_TEST(same, "m " << m << " e " << e << " ls " << ls << " zp " << zp);
                }
            }
        }
    }
    BOOST_TEST(configurations >= 200u);
    // Out of its domain: a negative multiplier or shift.
    tinymind::detail::FastAddInput64<int16_t> fast;
    BOOST_TEST(!fast.init(0, 15, -1, 3));
    BOOST_TEST(!fast.init(0, 15, 1 << 30, -1));
}

BOOST_AUTO_TEST_CASE(qint16x8_lut16)
{
    // The lookup, against TFLite's, over every input of a sigmoid table and
    // of a hand-made table with steep (int16-wrapping) slopes.
    int16_t sig[tinymind::kQActivationLUT16Size];
    const double inScale = 8.0 / 32768.0; // inputs span [-8, 8)
    const double outScale = 1.0 / 32768.0;
    tinymind::buildQSigmoidLUT16(inScale, 0, outScale, sig);
    int16_t steep[tinymind::kQActivationLUT16Size];
    for (std::size_t i = 0; i < tinymind::kQActivationLUT16Size; ++i)
    {
        steep[i] = static_cast<int16_t>((i % 2 == 0) ? -30000 : 30000);
    }
    for (int32_t v = -32768; v <= 32767; ++v)
    {
        const int16_t x = static_cast<int16_t>(v);
        BOOST_TEST(tinymind::qApplyLUT16(x, sig) == qint16x8_test::tfliteLutLookup(x, sig));
        BOOST_TEST(tinymind::qApplyLUT16(x, steep) == qint16x8_test::tfliteLutLookup(x, steep));
    }

    // Accuracy: within 2 output steps of the function everywhere, for
    // sigmoid, tanh and an input grid with a zero point. The bound is the
    // knot and interpolation roundings plus the curvature of 512 intervals:
    // tanh on a 6 / 32768 grid reaches 1.86 steps, and 4.5 with TFLite's
    // knot spacing.
    int16_t tnh[tinymind::kQActivationLUT16Size];
    tinymind::buildQTanhLUT16(6.0 / 32768.0, 0, outScale, tnh);
    int16_t offset[tinymind::kQActivationLUT16Size];
    tinymind::buildQSigmoidLUT16(4.0 / 32768.0, -16384, outScale, offset);
    double worstSig = 0.0, worstTanh = 0.0, worstOffset = 0.0;
    for (int32_t v = -32768; v <= 32767; ++v)
    {
        const int16_t x = static_cast<int16_t>(v);
        const double s = 1.0 / (1.0 + std::exp(-v * inScale));
        worstSig = std::max(worstSig, std::fabs(tinymind::qApplyLUT16(x, sig) - std::min(s / outScale, 32767.0)));
        const double t = std::tanh(v * 6.0 / 32768.0);
        worstTanh = std::max(worstTanh, std::fabs(tinymind::qApplyLUT16(x, tnh) -
                                                  std::max(-32768.0, std::min(t / outScale, 32767.0))));
        const double so = 1.0 / (1.0 + std::exp(-(v + 16384) * 4.0 / 32768.0));
        worstOffset = std::max(worstOffset, std::fabs(tinymind::qApplyLUT16(x, offset) -
                                                      std::min(so / outScale, 32767.0)));
    }
    BOOST_TEST(worstSig <= 2.0, "sigmoid " << worstSig);
    BOOST_TEST(worstTanh <= 2.0, "tanh " << worstTanh);
    BOOST_TEST(worstOffset <= 2.0, "offset sigmoid " << worstOffset);
    // Entries sit on the knots: tanh(0) is 0 and sigmoid(0) one half.
    BOOST_TEST(tnh[256] == 0);
    BOOST_TEST(sig[256] == 16384);
    BOOST_TEST(tnh[0] == -32768);
}
