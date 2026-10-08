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
// to hit the widest |w * (x - zp)| = 128 * 255. QPointwiseConv2D and QDense
// are then checked end to end through their dispatched reductions.

#include "include/tinymind_platform.hpp"

#if !TINYMIND_ENABLE_SIMD_ARM_DSP
#error "arm_dsp_qemu_test must be built with TINYMIND_ENABLE_SIMD_ARM_DSP=1"
#endif

#include "include/simd/simd_dispatch.hpp"
#include "qaffine.hpp"
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
    testPointwiseConv();
    testDense();

    semihost::writeUnsigned(g_cases);
    semihost::write(" cases, ");
    semihost::writeUnsigned(g_failures);
    semihost::write(" mismatches\n");

    return (g_failures == 0) ? 0 : 1;
}
