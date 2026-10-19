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

// Bit-exactness test for the TINYMIND_ENABLE_SIMD_ARM_DSP backend, run on
// an emulated Cortex-M4 (QEMU mps2-an386) so the SMLAD / SXTB16 / SSUB16
// instructions actually execute. No host can run them: the backend is
// AArch32-only and the hosted suites build for x86-64.
//
// Every case compares the dispatched dot product against the scalar
// reference. The sweep covers every length 0..kMaxN (so every n % 4 tail),
// every pair of byte misalignments for x and w (the kernel loads words from
// int8 pointers), zero points at both int8 extremes, and data sets chosen
// to hit the widest |w * (x - zp)| = 128 * 255. The 2x2 kernel (two inputs
// against two weight rows) gets the same sweep, and the out-of-line
// convolution kernel a few shapes. QPointwiseConv2D (narrow and wide enough
// for the 2x2 path), padded and unpadded QConv2DPerChannel on both its patch
// and row paths, and QDense are then checked end to end against direct
// evaluation.

#include "include/tinymind_platform.hpp"

#if !TINYMIND_ENABLE_SIMD_ARM_DSP
#error "arm_dsp_qemu_test must be built with TINYMIND_ENABLE_SIMD_ARM_DSP=1"
#endif

#include "include/simd/simd_dispatch.hpp"
#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdense.hpp"
#include "qpointwiseconv2d.hpp"

#include "semihost.hpp"

#include <cstddef>
#include <cstdint>

namespace {

    constexpr std::size_t kMaxN = 80;
    constexpr std::size_t kSlack = 4;

    uint32_t g_rng = 0x2545F491u;
    uint32_t g_failures = 0;
    uint32_t g_cases = 0;

    int8_t nextByte()
    {
        g_rng ^= g_rng << 13;
        g_rng ^= g_rng >> 17;
        g_rng ^= g_rng << 5;
        return static_cast<int8_t>(g_rng >> 24);
    }

    bool sameString(const char* a, const char* b)
    {
        while (*a && (*a == *b))
        {
            ++a;
            ++b;
        }
        return *a == *b;
    }

    void fail(const char* what, uint32_t a, uint32_t b, int32_t got, int32_t want)
    {
        if (g_failures < 8)
        {
            semihost::write("MISMATCH ");
            semihost::write(what);
            semihost::write(" [");
            semihost::writeUnsigned(a);
            semihost::write(",");
            semihost::writeUnsigned(b);
            semihost::write("] got ");
            semihost::writeSigned(got);
            semihost::write(" want ");
            semihost::writeSigned(want);
            semihost::write("\n");
        }
        ++g_failures;
    }

    enum class Fill { Random, MaxPositive, MaxNegative, Alternating };

    void fill(int8_t* x, int8_t* w, std::size_t len, Fill mode)
    {
        for (std::size_t i = 0; i < len; ++i)
        {
            switch (mode)
            {
            case Fill::Random:
                x[i] = nextByte();
                w[i] = nextByte();
                break;
            case Fill::MaxPositive:
                x[i] = -128;
                w[i] = -128;
                break;
            case Fill::MaxNegative:
                x[i] = 127;
                w[i] = -128;
                break;
            case Fill::Alternating:
                x[i] = (i & 1u) ? static_cast<int8_t>(127) : static_cast<int8_t>(-128);
                w[i] = (i & 2u) ? static_cast<int8_t>(-128) : static_cast<int8_t>(127);
                break;
            }
        }
    }

    void testDotSweep()
    {
        static const int8_t kZeroPoints[] = {-128, -127, -1, 0, 1, 77, 127};
        static const Fill kFills[] =
            {Fill::Random, Fill::MaxPositive, Fill::MaxNegative, Fill::Alternating};

        // Word-aligned backing storage, so offsets 0..3 give every alignment.
        alignas(4) static int8_t xbuf[kMaxN + kSlack];
        alignas(4) static int8_t wbuf[kMaxN + kSlack];

        for (const Fill mode : kFills)
        {
            fill(xbuf, wbuf, kMaxN + kSlack, mode);

            for (const int8_t zp : kZeroPoints)
            {
                for (std::size_t xo = 0; xo < kSlack; ++xo)
                {
                    for (std::size_t wo = 0; wo < kSlack; ++wo)
                    {
                        for (std::size_t n = 0; n <= kMaxN; ++n)
                        {
                            const int32_t want = tinymind::simd::int8DotWithZeroPointScalar(
                                xbuf + xo, wbuf + wo, n, zp);
                            const int32_t got = tinymind::simd::int8DotWithZeroPoint(
                                xbuf + xo, wbuf + wo, n, zp);
                            ++g_cases;
                            if (got != want)
                            {
                                fail("dot n,offset", static_cast<uint32_t>(n),
                                     static_cast<uint32_t>(xo * kSlack + wo), got, want);
                            }
                        }
                    }
                }
            }
        }
    }

    // The 2x2 kernel: two inputs PixelGap apart against two weight rows
    // WeightGap apart, accumulated onto nonzero starting values, for every
    // length 0..80, every x/w misalignment and the int8 extremes, against
    // four scalar dot products.
    void testDot2x2Sweep()
    {
        static const int8_t kZeroPoints[] = {-128, -1, 0, 77, 127};
        static const Fill kFills[] =
            {Fill::Random, Fill::MaxPositive, Fill::MaxNegative, Fill::Alternating};
        constexpr std::size_t kPixelGap = 37;
        constexpr std::size_t kWeightGap = 41;

        alignas(4) static int8_t xbuf[kMaxN + kPixelGap + kSlack];
        alignas(4) static int8_t wbuf[kMaxN + kWeightGap + kSlack];

        for (const Fill mode : kFills)
        {
            fill(xbuf, wbuf, kMaxN + kSlack, mode);
            fill(xbuf + kMaxN + kSlack, wbuf + kMaxN + kSlack, kPixelGap, mode);
            for (std::size_t i = kMaxN + kSlack + kPixelGap; i < sizeof(wbuf); ++i)
            {
                wbuf[i] = nextByte();
            }

            for (const int8_t zp : kZeroPoints)
            {
                for (std::size_t xo = 0; xo < kSlack; ++xo)
                {
                    for (std::size_t wo = 0; wo < kSlack; ++wo)
                    {
                        for (std::size_t n = 0; n <= kMaxN; ++n)
                        {
                            const int8_t* const x0 = xbuf + xo;
                            const int8_t* const w0 = wbuf + wo;
                            int32_t acc[4] = {11, -7, 123456, -98765};
                            const int32_t want[4] =
                            {
                                11 + tinymind::simd::int8DotWithZeroPointScalar(x0, w0, n, zp),
                                -7 + tinymind::simd::int8DotWithZeroPointScalar(x0, w0 + kWeightGap, n, zp),
                                123456 + tinymind::simd::int8DotWithZeroPointScalar(x0 + kPixelGap, w0, n, zp),
                                -98765 + tinymind::simd::int8DotWithZeroPointScalar(
                                    x0 + kPixelGap, w0 + kWeightGap, n, zp),
                            };
                            tinymind::simd::int8Dot2x2WithZeroPoint(
                                x0, x0 + kPixelGap, w0, w0 + kWeightGap, n, zp, acc);
                            for (std::size_t k = 0; k < 4; ++k)
                            {
                                ++g_cases;
                                if (acc[k] != want[k])
                                {
                                    fail("dot2x2 n,lane", static_cast<uint32_t>(n),
                                         static_cast<uint32_t>(k), acc[k], want[k]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // The out-of-line convolution kernel, one shape per call: KH rows read
    // from three separate, misaligned row pointers.
    template<std::size_t KH, std::size_t RowLength, std::size_t PixelStep, std::size_t WindowLength>
    void testConvRows2x2Shape()
    {
        alignas(4) static int8_t rowbuf[3][RowLength + PixelStep + kSlack];
        alignas(4) static int8_t wbuf[WindowLength * 2 + kSlack];
        static const int8_t kZeroPoints[] = {-128, 0, 101};

        for (std::size_t r = 0; r < 3; ++r)
        {
            for (int8_t& v : rowbuf[r]) { v = nextByte(); }
        }
        for (int8_t& v : wbuf) { v = nextByte(); }

        for (const int8_t zp : kZeroPoints)
        {
            for (std::size_t o = 0; o < kSlack; ++o)
            {
                const int8_t* const rows[3] = {rowbuf[0] + o, rowbuf[2] + (o ^ 1u), rowbuf[1] + (o ^ 3u)};
                const int8_t* const w0 = wbuf + (o ^ 2u);
                int32_t acc[4] = {5, 6, 7, 8};
                int32_t want[4] = {5, 6, 7, 8};
                for (std::size_t kh = 0; kh < KH; ++kh)
                {
                    const int8_t* const x = rows[kh];
                    const int8_t* const w = w0 + kh * RowLength;
                    want[0] += tinymind::simd::int8DotWithZeroPointScalar(x, w, RowLength, zp);
                    want[1] += tinymind::simd::int8DotWithZeroPointScalar(x, w + WindowLength, RowLength, zp);
                    want[2] += tinymind::simd::int8DotWithZeroPointScalar(x + PixelStep, w, RowLength, zp);
                    want[3] += tinymind::simd::int8DotWithZeroPointScalar(
                        x + PixelStep, w + WindowLength, RowLength, zp);
                }
                tinymind::simd::ConvRows2x2<int8_t, int8_t, int32_t, KH, RowLength, PixelStep, WindowLength>::apply(
                    rows, 0, w0, zp, acc);
                for (std::size_t k = 0; k < 4; ++k)
                {
                    ++g_cases;
                    if (acc[k] != want[k])
                    {
                        fail("convrows2x2 rowlen,lane", static_cast<uint32_t>(RowLength),
                             static_cast<uint32_t>(k), acc[k], want[k]);
                    }
                }
            }
        }
    }

    void testConvRows2x2()
    {
        testConvRows2x2Shape<3, 48, 16, 144>();
        testConvRows2x2Shape<3, 13, 6, 39>();   // tail of 1 per row
        testConvRows2x2Shape<1, 64, 64, 64>();  // a pointwise layer
        testConvRows2x2Shape<2, 30, 10, 60>();  // tail of 2
    }

    tinymind::Requantizer<int32_t, int8_t> makeRequantizer()
    {
        tinymind::Requantizer<int32_t, int8_t> r;
        r.multiplier = 1 << 30; // 0.5 in Q0.31
        r.shift = 9;            // overall acc / 1024: keeps outputs off the clamp
        r.zero_point = 3;
        r.qmin = -128;
        r.qmax = 127;
        return r;
    }

    void testPointwiseConv()
    {
        constexpr std::size_t H = 2, W = 3, C = 13, F = 5;
        typedef tinymind::QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, C, F> Layer;

        static int8_t input[Layer::InputSize];
        static int8_t weights[Layer::TotalWeights];
        static int32_t biases[F];
        static int8_t output[Layer::OutputSize];

        for (int8_t& v : input) { v = nextByte(); }
        for (int8_t& v : weights) { v = nextByte(); }
        for (std::size_t f = 0; f < F; ++f)
        {
            biases[f] = static_cast<int32_t>(f) * 1000 - 2000;
        }

        Layer layer;
        layer.weights = weights;
        layer.biases = biases;
        layer.input_zero_point = -11;
        layer.requantizer = makeRequantizer();
        layer.forward(input, output);

        for (std::size_t p = 0; p < H * W; ++p)
        {
            for (std::size_t f = 0; f < F; ++f)
            {
                int32_t acc = biases[f];
                for (std::size_t c = 0; c < C; ++c)
                {
                    acc += static_cast<int32_t>(weights[f * C + c]) *
                           (static_cast<int32_t>(input[p * C + c]) + 11);
                }
                const int32_t want = layer.requantizer.apply(acc);
                const int32_t got = output[p * F + f];
                ++g_cases;
                if (got != want)
                {
                    fail("pointwise p,f", static_cast<uint32_t>(p),
                         static_cast<uint32_t>(f), got, want);
                }
            }
        }
    }

    // Per-filter requantizers; every third has a narrowed qmin, which the
    // fast requantizer refuses, so filter pairs take both requantizer forms.
    template<std::size_t F>
    void makeRequantizers(tinymind::Requantizer<int32_t, int8_t>* rqs)
    {
        for (std::size_t f = 0; f < F; ++f)
        {
            rqs[f] = makeRequantizer();
            rqs[f].multiplier = (1 << 30) + static_cast<int32_t>(f) * 7919;
            rqs[f].zero_point = static_cast<int8_t>(static_cast<int>(f) - 2);
            if (f % 3 == 2)
            {
                rqs[f].qmin = -100;
            }
        }
    }

    // A padded per-channel convolution against a direct evaluation that
    // reads the input zero point outside the input.
    template<std::size_t H, std::size_t W, std::size_t C, std::size_t KH, std::size_t KW,
             std::size_t SH, std::size_t SW, std::size_t F,
             std::size_t PT, std::size_t PB, std::size_t PL, std::size_t PR>
    void testPaddedConvShape(const char* what)
    {
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, C, KH, KW, SH, SW, F,
                                            PT, PB, PL, PR> Layer;
        static int8_t input[Layer::InputSize];
        static int8_t weights[Layer::TotalWeights];
        static int32_t biases[F];
        static tinymind::Requantizer<int32_t, int8_t> rqs[F];
        static int8_t output[Layer::OutputSize];
        const int8_t zp = -9;

        for (int8_t& v : input) { v = nextByte(); }
        for (int8_t& v : weights) { v = nextByte(); }
        for (std::size_t f = 0; f < F; ++f)
        {
            biases[f] = static_cast<int32_t>(f) * 997 - 1500;
        }
        makeRequantizers<F>(rqs);

        const Layer layer = {weights, biases, zp, rqs};
        layer.forward(input, output);

        for (std::size_t oh = 0; oh < Layer::OutputHeight; ++oh)
        {
            for (std::size_t ow = 0; ow < Layer::OutputWidth; ++ow)
            {
                for (std::size_t f = 0; f < F; ++f)
                {
                    int32_t acc = biases[f];
                    for (std::size_t kh = 0; kh < KH; ++kh)
                    {
                        for (std::size_t kw = 0; kw < KW; ++kw)
                        {
                            const long ih = static_cast<long>(oh * SH + kh) - static_cast<long>(PT);
                            const long iw = static_cast<long>(ow * SW + kw) - static_cast<long>(PL);
                            const bool inside = (ih >= 0) && (ih < static_cast<long>(H)) &&
                                                (iw >= 0) && (iw < static_cast<long>(W));
                            for (std::size_t c = 0; c < C; ++c)
                            {
                                const int32_t x = inside
                                    ? input[(static_cast<std::size_t>(ih) * W + static_cast<std::size_t>(iw)) * C + c]
                                    : zp;
                                acc += static_cast<int32_t>(weights[((f * KH + kh) * KW + kw) * C + c]) * (x - zp);
                            }
                        }
                    }
                    const int32_t want = rqs[f].apply(acc);
                    const int32_t got = output[(oh * Layer::OutputWidth + ow) * F + f];
                    ++g_cases;
                    if (got != want)
                    {
                        fail(what, static_cast<uint32_t>(oh * Layer::OutputWidth + ow),
                             static_cast<uint32_t>(f), got, want);
                    }
                }
            }
        }
    }

    void testPaddedConv()
    {
        // Row path (window > 64 bytes), odd width and odd filter count.
        testPaddedConvShape<7, 7, 16, 3, 3, 1, 1, 5, 1, 1, 1, 1>("conv rows same p,f");
        // Row path, stride 2, padding at the bottom and right only.
        testPaddedConvShape<8, 9, 16, 3, 3, 2, 2, 4, 0, 1, 0, 1>("conv rows s2 p,f");
        // Unpadded row path.
        testPaddedConvShape<5, 6, 12, 3, 3, 1, 1, 3, 0, 0, 0, 0>("conv rows valid p,f");
        // Patch path (window <= 64 bytes) with uneven padding.
        testPaddedConvShape<9, 6, 1, 5, 2, 2, 1, 6, 2, 2, 1, 0>("conv patch pad p,f");
    }

    // A pointwise layer wide enough for the 2x2 path, with an odd pixel and
    // filter count.
    void testPointwiseWide()
    {
        constexpr std::size_t H = 3, W = 3, C = 40, F = 5;
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, C, F> Layer;
        static int8_t input[Layer::InputSize];
        static int8_t weights[Layer::TotalWeights];
        static int32_t biases[F];
        static tinymind::Requantizer<int32_t, int8_t> rqs[F];
        static int8_t output[Layer::OutputSize];

        for (int8_t& v : input) { v = nextByte(); }
        for (int8_t& v : weights) { v = nextByte(); }
        for (std::size_t f = 0; f < F; ++f)
        {
            biases[f] = 300 - static_cast<int32_t>(f) * 211;
        }
        makeRequantizers<F>(rqs);

        const Layer layer = {weights, biases, 21, rqs};
        layer.forward(input, output);

        for (std::size_t p = 0; p < H * W; ++p)
        {
            for (std::size_t f = 0; f < F; ++f)
            {
                int32_t acc = biases[f];
                for (std::size_t c = 0; c < C; ++c)
                {
                    acc += static_cast<int32_t>(weights[f * C + c]) *
                           (static_cast<int32_t>(input[p * C + c]) - 21);
                }
                const int32_t want = rqs[f].apply(acc);
                const int32_t got = output[p * F + f];
                ++g_cases;
                if (got != want)
                {
                    fail("pointwise wide p,f", static_cast<uint32_t>(p), static_cast<uint32_t>(f), got, want);
                }
            }
        }
    }

    void testDense()
    {
        constexpr std::size_t N = 37, O = 6;
        typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, N, O> Layer;

        static int8_t input[N];
        static int8_t weights[N * O];
        static int8_t output[O];

        for (int8_t& v : input) { v = nextByte(); }
        for (int8_t& v : weights) { v = nextByte(); }

        Layer layer;
        layer.weights = weights;
        layer.biases = nullptr;
        layer.input_zero_point = 5;
        layer.requantizer = makeRequantizer();
        layer.forward(input, output);

        for (std::size_t o = 0; o < O; ++o)
        {
            int32_t acc = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                acc += static_cast<int32_t>(weights[o * N + i]) *
                       (static_cast<int32_t>(input[i]) - 5);
            }
            const int32_t want = layer.requantizer.apply(acc);
            const int32_t got = output[o];
            ++g_cases;
            if (got != want)
            {
                fail("dense o", static_cast<uint32_t>(o), 0u, got, want);
            }
        }
    }

} // namespace

extern "C" int run_tests()
{
    const char* const backend = tinymind::simd::activeBackendName();
    semihost::write("arm_dsp_qemu_test: backend=");
    semihost::write(backend);
    semihost::write("\n");
    if (!sameString(backend, "arm_dsp"))
    {
        semihost::write("FAIL: dispatch did not resolve to arm_dsp\n");
        return 1;
    }

    testDotSweep();
    testDot2x2Sweep();
    testConvRows2x2();
    testPointwiseConv();
    testPointwiseWide();
    testPaddedConv();
    testDense();

    semihost::writeUnsigned(g_cases);
    semihost::write(" cases, ");
    semihost::writeUnsigned(g_failures);
    semihost::write(" mismatches\n");

    return (g_failures == 0) ? 0 : 1;
}
