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

// Bit-exactness test for the M-profile SIMD backends, run on emulated cores
// so their instructions actually execute. No CI host can run them: they are
// AArch32 M-profile only and the hosted suites build for x86-64 / AArch64.
//
//   TINYMIND_ENABLE_SIMD_ARM_DSP      SMLAD / SXTB16 / SSUB16, on QEMU
//                                     mps2-an386 (Cortex-M4).
//   TINYMIND_ENABLE_SIMD_HELIUM_MVE_I VMLADAVA on predicated VLDRB loads, on
//                                     QEMU mps3-an547 (Cortex-M55).
//   TINYMIND_ENABLE_SIMD_HELIUM_MVE_F the fp32 dot product, called directly
//                                     (nothing dispatches to it), on the
//                                     same Cortex-M55 image.
//
// Every integer case compares the dispatched dot product against the scalar
// reference. A per-lane sweep checks every (x, w) pair in each byte lane of
// one block (two SMLAD words, or one 16-lane MVE vector) with every zero
// point, so each lane's widening path is checked on every input. The length
// sweep covers every length 0..kMaxN (so every tail length of either
// backend), every pair of byte misalignments for x and w, zero points at
// both int8 extremes, and data sets chosen to hit the widest
// |w * (x - zp)| = 128 * 255. The 2x2 kernel (two inputs against two weight
// rows) gets the same sweep, and the convolution kernel a few shapes.
// QPointwiseConv2D (narrow and wide enough for the 2x2 path), padded and
// unpadded QConv2DPerChannel on both its patch and row paths, and QDense are
// then checked end to end against direct evaluation.
//
// The fp32 dot product is not bit-exact with a sequential sum by design, so
// it is checked bit-exactly against a model of its own arithmetic (four
// lane accumulators, fused multiply-add, a fixed reduction order), and
// against a double-precision sum within a rounding-error bound.

#include "include/tinymind_platform.hpp"

#if TINYMIND_ENABLE_SIMD_HELIUM_MVE_I
#define TM_EXPECTED_BACKEND "helium_mve_i"
#elif TINYMIND_ENABLE_SIMD_ARM_DSP
#define TM_EXPECTED_BACKEND "arm_dsp"
#else
#error "simd_qemu_test needs TINYMIND_ENABLE_SIMD_ARM_DSP=1 or TINYMIND_ENABLE_SIMD_HELIUM_MVE_I=1"
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
    // Bytes one block of the backend consumes: two 4-byte SMLAD words, or
    // one 16-lane MVE vector.
#if TINYMIND_ENABLE_SIMD_HELIUM_MVE_I
    constexpr std::size_t kBlockLanes = 16;
#else
    constexpr std::size_t kBlockLanes = 8;
#endif
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

    // Exhaustive per lane: every (x, w) pair in each byte lane of one block,
    // with every zero point, against the scalar reference. For SMLAD that is
    // two words: SXTB16 widens lanes 0 and 2 of a word and the ROR #8 form
    // lanes 1 and 3, so each extraction path sees every byte value with every
    // zero point. For MVE it is one 16-lane VMLADAVA. The other lanes hold
    // fixed non-zero values so the pairwise sums and the accumulator carry
    // are exercised too. The dot product is a sum of lane products, so this
    // checks every lane's contribution on every input.
    void testDotEveryLane()
    {
        constexpr std::size_t kLanes = kBlockLanes;
        alignas(4) static int8_t xbuf[kLanes];
        alignas(4) static int8_t wbuf[kLanes];
        for (std::size_t lane = 0; lane < kLanes; ++lane)
        {
            for (int zp = -128; zp <= 127; ++zp)
            {
                for (std::size_t k = 0; k < kLanes; ++k)
                {
                    xbuf[k] = static_cast<int8_t>(static_cast<int>((k * 37u) % 256u) - 128);
                    wbuf[k] = static_cast<int8_t>(static_cast<int>((k * 91u + 17u) % 256u) - 128);
                }
                for (int x = -128; x <= 127; ++x)
                {
                    xbuf[lane] = static_cast<int8_t>(x);
                    for (int w = -128; w <= 127; ++w)
                    {
                        wbuf[lane] = static_cast<int8_t>(w);
                        const int32_t want = tinymind::simd::int8DotWithZeroPointScalar(
                            xbuf, wbuf, kLanes, static_cast<int8_t>(zp));
                        const int32_t got = tinymind::simd::int8DotWithZeroPoint(
                            xbuf, wbuf, kLanes, static_cast<int8_t>(zp));
                        ++g_cases;
                        if (got != want)
                        {
                            fail("dot lane,x|w", static_cast<uint32_t>(lane),
                                 static_cast<uint32_t>(((x + 128) << 8) | (w + 128)), got, want);
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

#if TINYMIND_ENABLE_SIMD_HELIUM_MVE_F
    uint32_t floatBits(float f)
    {
        uint32_t u;
        __builtin_memcpy(&u, &f, sizeof(u));
        return u;
    }

    // One scalar fused multiply-add, acc + a * b with a single rounding.
    // Inline asm rather than __builtin_fmaf: Clang lowers the builtin to a
    // call to fmaf (math-errno), and these images link no libm.
    float fusedMultiplyAdd(float a, float b, float acc)
    {
        __asm__("vfma.f32 %0, %1, %2" : "+t"(acc) : "t"(a), "t"(b));
        return acc;
    }

    // Model of helium_mve_f::floatDotProduct: lane j of the accumulator
    // takes elements j, j + 4, ... by fused multiply-add (lanes the tail
    // predicate switches off add 0 * 0, which leaves them unchanged), and
    // the four lanes are then summed as (l0 + l1) + (l2 + l3).
    float floatDotModel(const float* a, const float* b, std::size_t n)
    {
        float lane[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (std::size_t i = 0; i < n; i += 4)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                const bool active = (i + j) < n;
                lane[j] = fusedMultiplyAdd(active ? a[i + j] : 0.0f, active ? b[i + j] : 0.0f, lane[j]);
            }
        }
        return (lane[0] + lane[1]) + (lane[2] + lane[3]);
    }

    enum class FloatFill { Small, Spread };

    float nextFloat(FloatFill mode, std::size_t i)
    {
        const float v = static_cast<float>(nextByte()) / 32.0f;
        if (mode == FloatFill::Small)
        {
            return v;
        }
        // Magnitudes from 2^-10 to 2^9, so large terms swamp small ones and
        // the reduction order is visible in the result.
        float scale = 1.0f / 1024.0f;
        for (std::size_t k = 0; k < (i * 7u) % 20u; ++k)
        {
            scale *= 2.0f;
        }
        return v * scale;
    }

    // The fp32 dot product for every length 0..kMaxN and every element
    // offset of both operands: bit-exact against the lane model, and within
    // (ceil(n / 4) + 2) roundings of the double-precision sum, so a model
    // that drifted along with the backend would still be caught.
    void testFloatDot()
    {
        static const FloatFill kFills[] = {FloatFill::Small, FloatFill::Spread};
        static float abuf[kMaxN + kSlack];
        static float bbuf[kMaxN + kSlack];

        for (const FloatFill mode : kFills)
        {
            for (std::size_t i = 0; i < kMaxN + kSlack; ++i)
            {
                abuf[i] = nextFloat(mode, i);
                bbuf[i] = nextFloat(mode, i + 3u);
            }
            for (std::size_t ao = 0; ao < kSlack; ++ao)
            {
                for (std::size_t bo = 0; bo < kSlack; ++bo)
                {
                    for (std::size_t n = 0; n <= kMaxN; ++n)
                    {
                        const float* const a = abuf + ao;
                        const float* const b = bbuf + bo;
                        const float got = tinymind::simd::helium_mve_f::floatDotProduct(a, b, n);
                        const float model = floatDotModel(a, b, n);

                        double exact = 0.0;
                        double magnitude = 0.0;
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            const double p = static_cast<double>(a[i]) * static_cast<double>(b[i]);
                            exact += p;
                            magnitude += (p < 0.0) ? -p : p;
                        }
                        const double roundings = static_cast<double>((n + 3u) / 4u + 2u);
                        const double bound = roundings * magnitude / 16777216.0; // 2^-24 per rounding
                        const double err = static_cast<double>(got) - exact;

                        ++g_cases;
                        if (floatBits(got) != floatBits(model))
                        {
                            fail("fp32 dot vs lane model n,offset (bits)", static_cast<uint32_t>(n),
                                 static_cast<uint32_t>(ao * kSlack + bo),
                                 static_cast<int32_t>(floatBits(got)), static_cast<int32_t>(floatBits(model)));
                        }
                        ++g_cases;
                        if (((err < 0.0) ? -err : err) > bound)
                        {
                            fail("fp32 dot vs double sum n,offset (bits)", static_cast<uint32_t>(n),
                                 static_cast<uint32_t>(ao * kSlack + bo),
                                 static_cast<int32_t>(floatBits(got)),
                                 static_cast<int32_t>(floatBits(static_cast<float>(exact))));
                        }
                    }
                }
            }
        }
    }
#endif

} // namespace

extern "C" int run_tests()
{
    const char* const backend = tinymind::simd::activeBackendName();
    semihost::write("simd_qemu_test: backend=");
    semihost::write(backend);
    semihost::write("\n");
    if (!sameString(backend, TM_EXPECTED_BACKEND))
    {
        semihost::write("FAIL: dispatch did not resolve to " TM_EXPECTED_BACKEND "\n");
        return 1;
    }

    testDotSweep();
    testDotEveryLane();
    testDot2x2Sweep();
    testConvRows2x2();
    testPointwiseConv();
    testPointwiseWide();
    testPaddedConv();
    testDense();
#if TINYMIND_ENABLE_SIMD_HELIUM_MVE_F
    testFloatDot();
#endif

    semihost::writeUnsigned(g_cases);
    semihost::write(" cases, ");
    semihost::writeUnsigned(g_failures);
    semihost::write(" mismatches\n");

    return (g_failures == 0) ? 0 : 1;
}
