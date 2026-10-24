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

// exhaustive_unit_test.cpp : properties checked on every input of a domain.
//
// A fuzzer samples a domain; where the domain is small enough to enumerate,
// checking every element is a proof for that build, not a sample. Each case
// below states its domain. Where the full domain is too large for a PR run
// (all 2^32 int32 accumulators), the PR run covers the range real layers
// reach and TINYMIND_EXHAUSTIVE_FULL=1 (the nightly job, `make full`) covers
// the whole type.
//
//   - Fixed-point LUT sigmoid / tanh: every input of every supported 8- and
//     16-bit signed format, and every input in [-7, 7] of the 32-bit ones.
//     Monotone, in range, and within a derived error bound.
//   - Requantization: the one-rounding fast path equals the two-step
//     gemmlowp form, and FastRequantizer equals Requantizer, for every
//     accumulator in the range, every shift the fast path takes.
//   - int8 dot product: the scalar reference against an int64 sum for every
//     (x, w, zero point) triple; the active SIMD backend against the scalar
//     reference for every (x, w) in every vector lane.
//   - QSoftmax1D: every int8 row of 2 and of 3 features; exact rounding of
//     the LUT ratio, order, mass, accuracy against real softmax and shift
//     invariance.
//   - QLayerNorm's integer square root and Q30 inverse square root: every
//     input in the range a layer reaches.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "qaffine.hpp"
#include "qsoftmax.hpp"
#include "qlayernorm.hpp"
#include "include/qcalibration.hpp"
#include "include/simd/simd_dispatch.hpp"

#define BOOST_TEST_MODULE exhaustive
#include <boost/test/included/unit_test.hpp>

#if defined(_OPENMP)
#define TINYMIND_EXHAUSTIVE_PARALLEL _Pragma("omp parallel for reduction(+:failures) schedule(static)")
#else
#define TINYMIND_EXHAUSTIVE_PARALLEL
#endif

namespace {

bool fullSweep()
{
    const char* full = std::getenv("TINYMIND_EXHAUSTIVE_FULL");
    return (full != nullptr) && (full[0] == '1');
}

// ---------------------------------------------------------------------------
// Fixed-point LUT activations.
//
// The tables hold f at knots 1/8 apart over [-6, 6] and interpolate linearly
// between them; outside that range they clamp. The error bound is the sum of
//   tail:   |f(+/-inf) - f(+/-6)|, the most a clamped input can be off;
//   interp: h^2 / 8 * max|f''| with h = 1/8, linear interpolation's bound;
//   3 LSB:  the stored table values and the interpolation arithmetic.
// ---------------------------------------------------------------------------

const double kKnotSpacing = 0.125;
const double kInterpFactor = kKnotSpacing * kKnotSpacing / 8.0;
const double kSigmoidMaxCurvature = 1.0 / (6.0 * 1.7320508075688772);  // 1 / (6 sqrt 3)
const double kTanhMaxCurvature = 4.0 / (3.0 * 1.7320508075688772);     // 4 / (3 sqrt 3)

struct ActivationReport
{
    long long inputs;
    long long failures;
    double maxError;
};

template<unsigned I, unsigned F, bool Sigmoid>
ActivationReport sweepActivation()
{
    typedef tinymind::QValue<I, F, true> Q;
    typedef typename Q::FullWidthValueType Raw;

    long long lo = -(1LL << (I + F - 1));
    long long hi = (1LL << (I + F - 1)) - 1;
    if (I + F > 16)
    {
        // Every input in [-7, 7]: the interpolated range and a clamped
        // stretch either side of it.
        lo = -(7LL << F);
        hi = (7LL << F);
    }

    const double lsb = std::ldexp(1.0, -static_cast<int>(F));
    const double tail = Sigmoid ? (1.0 - 1.0 / (1.0 + std::exp(-6.0))) : (1.0 - std::tanh(6.0));
    const double curvature = Sigmoid ? kSigmoidMaxCurvature : kTanhMaxCurvature;
    const double bound = tail + kInterpFactor * curvature + 3.0 * lsb;
    const double floorValue = Sigmoid ? 0.0 : -1.0;

    ActivationReport report = { 0, 0, 0.0 };
    double previous = -2.0;
    for (long long raw = lo; raw <= hi; ++raw)
    {
        const Q q(static_cast<Raw>(raw));
        const Q y = Sigmoid ? tinymind::SigmoidActivationPolicy<Q>::activationFunction(q)
                            : tinymind::TanhActivationPolicy<Q>::activationFunction(q);
        const double x = static_cast<double>(raw) * lsb;
        const double got = static_cast<double>(y.getValue()) * lsb;
        const double want = Sigmoid ? 1.0 / (1.0 + std::exp(-x)) : std::tanh(x);
        const double error = std::fabs(got - want);

        bool ok = (error <= bound);
        ok = ok && (got >= floorValue) && (got <= 1.0);
        ok = ok && (got >= previous);
        if (!ok && report.failures < 5)
        {
            std::printf("  Q%u.%u %s(%.6f) = %.6f, want %.6f (bound %.6f, previous %.6f)\n",
                        I, F, Sigmoid ? "sigmoid" : "tanh", x, got, want, bound, previous);
        }
        report.failures += ok ? 0 : 1;
        report.maxError = std::fmax(report.maxError, error);
        report.inputs += 1;
        previous = got;
    }
    std::printf("  Q%u.%-2u %-7s %9lld inputs  max error %.6f (%.2f LSB, bound %.2f LSB)\n",
                I, F, Sigmoid ? "sigmoid" : "tanh", report.inputs, report.maxError,
                report.maxError / lsb, bound / lsb);
    return report;
}

template<unsigned I, unsigned F>
void checkActivations()
{
    const ActivationReport s = sweepActivation<I, F, true>();
    const ActivationReport t = sweepActivation<I, F, false>();
    BOOST_TEST(s.failures == 0);
    BOOST_TEST(t.failures == 0);
}

// ---------------------------------------------------------------------------
// int8 helpers.
// ---------------------------------------------------------------------------

// The accumulator range for the PR run: |acc| <= 2^22 covers an int8 dot
// product of 128 terms at the rails (128 * 255 * 128 < 2^22) plus a bias.
const int64_t kPrAccumulatorReach = static_cast<int64_t>(1) << 22;

void accumulatorRange(int64_t& lo, int64_t& hi)
{
    if (fullSweep())
    {
        lo = INT32_MIN;
        hi = INT32_MAX;
    }
    else
    {
        lo = -kPrAccumulatorReach;
        hi = kPrAccumulatorReach;
    }
}

} // namespace

BOOST_AUTO_TEST_SUITE(test_suite_exhaustive)

BOOST_AUTO_TEST_CASE(lut_activations_every_input)
{
    std::printf("LUT sigmoid / tanh, every input:\n");
    checkActivations<4, 4>();
    checkActivations<5, 3>();
    checkActivations<4, 12>();
    checkActivations<5, 11>();
    checkActivations<6, 10>();
    checkActivations<7, 9>();
    checkActivations<8, 8>();
    checkActivations<9, 7>();
    checkActivations<10, 6>();
    checkActivations<11, 5>();
    checkActivations<12, 4>();
    checkActivations<13, 3>();
    checkActivations<8, 24>();
    checkActivations<16, 16>();
    checkActivations<24, 8>();
}

BOOST_AUTO_TEST_CASE(requantize_fast_path_every_accumulator)
{
    // roundingHighMulDivideByPOT is documented to equal the two-step
    // roundingDivideByPOT(saturatingRoundingDoublingHighMul(x, m), e) bit for
    // bit for m >= 0 and 2 <= e <= 31. Requantizer::apply takes it in exactly
    // that regime.
    int64_t lo;
    int64_t hi;
    accumulatorRange(lo, hi);
    const int32_t multipliers[] = { 0x40000000, 0x5A82799A, 0x7FFFFFFF };
    long long failures = 0;
    for (int32_t shift = 2; shift <= 31; ++shift)
    {
        for (std::size_t m = 0; m < sizeof(multipliers) / sizeof(multipliers[0]); ++m)
        {
            const int32_t multiplier = multipliers[m];
            TINYMIND_EXHAUSTIVE_PARALLEL
            for (int64_t x = lo; x <= hi; ++x)
            {
                const int32_t acc = static_cast<int32_t>(x);
                const int32_t fast = tinymind::roundingHighMulDivideByPOT(acc, multiplier, shift);
                const int32_t twoStep = tinymind::roundingDivideByPOT(
                    tinymind::saturatingRoundingDoublingHighMul(acc, multiplier), shift);
                failures += (fast == twoStep) ? 0 : 1;
            }
        }
    }
    std::printf("requantize fast path: accumulators [%lld, %lld], shifts 2..31, 3 multipliers, %lld mismatches\n",
                static_cast<long long>(lo), static_cast<long long>(hi), failures);
    BOOST_TEST(failures == 0);
}

BOOST_AUTO_TEST_CASE(fast_requantizer_every_accumulator)
{
    // FastRequantizer folds the zero point into the rounding nudge and clamps
    // against constants; it must equal Requantizer::apply wherever init()
    // accepts the requantizer.
    int64_t lo;
    int64_t hi;
    accumulatorRange(lo, hi);
    if (!fullSweep())
    {
        lo /= 4;
        hi /= 4;
    }
    const int8_t zeroPoints[] = { -128, -1, 0, 1, 127 };
    const uint8_t zeroPointsU[] = { 0, 128, 255 };
    long long failures = 0;
    long long configs = 0;
    for (int32_t shift = 2; shift <= 31; ++shift)
    {
        for (std::size_t z = 0; z < sizeof(zeroPoints); ++z)
        {
            tinymind::Requantizer<int32_t, int8_t> rq;
            rq.multiplier = 0x5A82799A;
            rq.shift = shift;
            rq.zero_point = zeroPoints[z];
            rq.qmin = -128;
            rq.qmax = 127;
            tinymind::detail::FastRequantizer<int8_t> fast;
            if (!fast.init(rq))
            {
                continue;
            }
            ++configs;
            TINYMIND_EXHAUSTIVE_PARALLEL
            for (int64_t x = lo; x <= hi; ++x)
            {
                const int32_t acc = static_cast<int32_t>(x);
                failures += (fast.apply(acc) == rq.apply(acc)) ? 0 : 1;
            }
        }
        for (std::size_t z = 0; z < sizeof(zeroPointsU); ++z)
        {
            tinymind::Requantizer<int32_t, uint8_t> rq;
            rq.multiplier = 0x5A82799A;
            rq.shift = shift;
            rq.zero_point = zeroPointsU[z];
            rq.qmin = 0;
            rq.qmax = 255;
            tinymind::detail::FastRequantizer<uint8_t> fast;
            if (!fast.init(rq))
            {
                continue;
            }
            ++configs;
            TINYMIND_EXHAUSTIVE_PARALLEL
            for (int64_t x = lo; x <= hi; ++x)
            {
                const int32_t acc = static_cast<int32_t>(x);
                failures += (fast.apply(acc) == rq.apply(acc)) ? 0 : 1;
            }
        }
    }
    std::printf("FastRequantizer: accumulators [%lld, %lld], %lld accepted configurations, %lld mismatches\n",
                static_cast<long long>(lo), static_cast<long long>(hi), configs, failures);
    BOOST_TEST(configs > 100);
    BOOST_TEST(failures == 0);
}

BOOST_AUTO_TEST_CASE(int8_dot_scalar_every_triple)
{
    // Every (x, w, zero point) for one term, against an int64 sum.
    long long failures = 0;
    for (int zp = -128; zp <= 127; ++zp)
    {
        for (int x = -128; x <= 127; ++x)
        {
            for (int w = -128; w <= 127; ++w)
            {
                const int8_t xv = static_cast<int8_t>(x);
                const int8_t wv = static_cast<int8_t>(w);
                const int64_t want = static_cast<int64_t>(w) * (static_cast<int64_t>(x) - zp);
                const int32_t got = tinymind::simd::int8DotWithZeroPointScalar(
                    &xv, &wv, 1, static_cast<int8_t>(zp));
                failures += (static_cast<int64_t>(got) == want) ? 0 : 1;
            }
        }
    }
    BOOST_TEST(failures == 0);
}

BOOST_AUTO_TEST_CASE(int8_dot_backend_every_lane)
{
    // Every (x, w) pair in every lane of a 64-element dot product (wide
    // enough for the 512-bit backends' blocks, and every residue of the
    // narrower ones), against the scalar reference. The other lanes hold a
    // fixed non-zero background so lane sums and the zero-point term are
    // exercised. Lane 0 takes every zero point; the other lanes take the
    // rails and their neighbours.
    const std::size_t n = 64;
    int8_t xs[n];
    int8_t ws[n];
    long long failures = 0;
    long long calls = 0;
    const int edgeZeroPoints[] = { -128, -127, -1, 0, 1, 126, 127 };
    for (std::size_t lane = 0; lane < n; ++lane)
    {
        for (std::size_t k = 0; k < n; ++k)
        {
            xs[k] = static_cast<int8_t>(static_cast<int>((k * 37u) % 256u) - 128);
            ws[k] = static_cast<int8_t>(static_cast<int>((k * 91u + 17u) % 256u) - 128);
        }
        const std::size_t zpCount = (lane == 0) ? 256u : sizeof(edgeZeroPoints) / sizeof(edgeZeroPoints[0]);
        for (std::size_t zi = 0; zi < zpCount; ++zi)
        {
            const int zp = (lane == 0) ? static_cast<int>(zi) - 128 : edgeZeroPoints[zi];
            for (int x = -128; x <= 127; ++x)
            {
                xs[lane] = static_cast<int8_t>(x);
                for (int w = -128; w <= 127; ++w)
                {
                    ws[lane] = static_cast<int8_t>(w);
                    const int32_t got = tinymind::simd::int8DotWithZeroPoint(xs, ws, n, static_cast<int8_t>(zp));
                    const int32_t want = tinymind::simd::int8DotWithZeroPointScalar(xs, ws, n, static_cast<int8_t>(zp));
                    failures += (got == want) ? 0 : 1;
                    ++calls;
                }
            }
        }
    }
    std::printf("int8 dot, backend %s: %lld calls, %lld mismatches\n",
                tinymind::simd::activeBackendName(), calls, failures);
    BOOST_TEST(failures == 0);
}

namespace {

template<std::size_t F>
long long checkSoftmaxRow(const tinymind::QSoftmax1D<int8_t, int8_t, 1, F>& sm,
                          const int8_t* in, const int8_t* out, float scale)
{
    long long failures = 0;

    // Exact: each output is 256 * lut[i] / sum(lut) rounded to nearest (ties
    // up), computed here in double rather than the kernel's integer division.
    // A tie can round either way between the two formulations, so a
    // difference is accepted only there.
    int32_t rowMax = in[0];
    for (std::size_t i = 1; i < F; ++i)
    {
        rowMax = (in[i] > rowMax) ? in[i] : rowMax;
    }
    double lutSum = 0.0;
    for (std::size_t i = 0; i < F; ++i)
    {
        lutSum += static_cast<double>(sm.exp_lut[tinymind::qSoftmaxLUTIndex(in[i], rowMax)]);
    }
    for (std::size_t i = 0; i < F; ++i)
    {
        const double ratio = 256.0 * static_cast<double>(sm.exp_lut[tinymind::qSoftmaxLUTIndex(in[i], rowMax)]) / lutSum;
        double want = std::floor(ratio + 0.5);
        want = (want > 255.0) ? 255.0 : want;
        const double got = static_cast<double>(out[i]) + 128.0;
        const bool tie = std::fabs((ratio - std::floor(ratio)) - 0.5) < 1e-9;
        failures += ((got == want) || (tie && std::fabs(got - want) <= 1.0)) ? 0 : 1;
    }

    int32_t total = 0;
    int32_t maxQ = in[0];
    for (std::size_t i = 0; i < F; ++i)
    {
        total += static_cast<int32_t>(out[i]) + 128;
        if (in[i] > maxQ)
        {
            maxQ = in[i];
        }
        for (std::size_t j = 0; j < F; ++j)
        {
            if ((in[i] > in[j]) && (out[i] < out[j]))
            {
                ++failures;
            }
            if ((in[i] == in[j]) && (out[i] != out[j]))
            {
                ++failures;
            }
        }
    }
    const int32_t massError = (total > 256) ? (total - 256) : (256 - total);
    failures += (2 * massError <= static_cast<int32_t>(F) + 2) ? 0 : 1;

    double denom = 0.0;
    for (std::size_t i = 0; i < F; ++i)
    {
        denom += std::exp(static_cast<double>(in[i] - maxQ) * static_cast<double>(scale));
    }
    for (std::size_t i = 0; i < F; ++i)
    {
        const double p = std::exp(static_cast<double>(in[i] - maxQ) * static_cast<double>(scale)) / denom;
        double want = std::floor(256.0 * p + 0.5);
        if (want > 255.0)
        {
            want = 255.0;
        }
        failures += (std::fabs(static_cast<double>(out[i]) + 128.0 - want) <= 1.0) ? 0 : 1;
    }

    // Shift invariance: the row moved by +1 (where it fits) gives the same
    // output. Every row is visited, so every shift by any amount is covered
    // by chaining these.
    int8_t moved[F];
    bool fits = true;
    for (std::size_t i = 0; i < F; ++i)
    {
        fits = fits && (in[i] < 127);
        moved[i] = static_cast<int8_t>(fits ? in[i] + 1 : in[i]);
    }
    if (fits)
    {
        int8_t movedOut[F];
        sm.forward(moved, movedOut);
        for (std::size_t i = 0; i < F; ++i)
        {
            failures += (movedOut[i] == out[i]) ? 0 : 1;
        }
    }
    return failures;
}

template<std::size_t F>
void configureSoftmax(tinymind::QSoftmax1D<int8_t, int8_t, 1, F>& sm, const int32_t* lut)
{
    sm.exp_lut = lut;
    sm.output_zero_point = -128;
    sm.qmin = -128;
    sm.qmax = 127;
}

} // namespace

BOOST_AUTO_TEST_CASE(softmax_every_pair)
{
    const float scales[] = { 1.0f / 256.0f, 1.0f / 32.0f, 1.0f / 8.0f, 0.5f, 2.0f, 8.0f };
    long long failures = 0;
    long long rows = 0;
    for (std::size_t s = 0; s < sizeof(scales) / sizeof(scales[0]); ++s)
    {
        int32_t lut[tinymind::kQSoftmaxExpLUTSize];
        tinymind::buildQSoftmaxExpLUT(scales[s], lut);
        tinymind::QSoftmax1D<int8_t, int8_t, 1, 2> sm;
        configureSoftmax(sm, lut);
        for (int a = -128; a <= 127; ++a)
        {
            for (int b = -128; b <= 127; ++b)
            {
                const int8_t in[2] = { static_cast<int8_t>(a), static_cast<int8_t>(b) };
                int8_t out[2];
                sm.forward(in, out);
                failures += checkSoftmaxRow(sm, in, out, scales[s]);
                ++rows;
            }
        }
    }
    std::printf("softmax, 2 features: %lld rows, %lld failures\n", rows, failures);
    BOOST_TEST(failures == 0);
}

BOOST_AUTO_TEST_CASE(softmax_every_triple)
{
    const float scale = 1.0f / 16.0f;
    int32_t lut[tinymind::kQSoftmaxExpLUTSize];
    tinymind::buildQSoftmaxExpLUT(scale, lut);
    tinymind::QSoftmax1D<int8_t, int8_t, 1, 3> sm;
    configureSoftmax(sm, lut);
    long long failures = 0;
    long long rows = 0;
    for (int a = -128; a <= 127; ++a)
    {
        for (int b = -128; b <= 127; ++b)
        {
            for (int c = -128; c <= 127; ++c)
            {
                const int8_t in[3] = { static_cast<int8_t>(a), static_cast<int8_t>(b), static_cast<int8_t>(c) };
                int8_t out[3];
                sm.forward(in, out);
                failures += checkSoftmaxRow(sm, in, out, scale);
                ++rows;
            }
        }
    }
    std::printf("softmax, 3 features: %lld rows, %lld failures\n", rows, failures);
    BOOST_TEST(failures == 0);
}

BOOST_AUTO_TEST_CASE(layernorm_integer_sqrt_every_input)
{
    // qInvSqrtQ30 takes var_q + epsilon_q. An int8 row's variance is at most
    // 128^2 = 2^14, and epsilon_q is small, so the PR run covers every input
    // up to 2^24; the full run covers every uint32.
    const uint64_t top = fullSweep() ? UINT64_C(0xFFFFFFFF) : (UINT64_C(1) << 24);
    long long failures = 0;
    TINYMIND_EXHAUSTIVE_PARALLEL
    for (uint64_t x = 1; x <= top; ++x)
    {
        const uint64_t v = x << 28;
        const uint64_t s = tinymind::qIntegerSqrt64(v);
        // s is the integer square root: s^2 <= v < (s + 1)^2.
        bool ok = (s * s <= v) && ((s + 1) * (s + 1) > v);
        uint64_t inv = (UINT64_C(1) << 44) / s;
        if (inv > UINT64_C(0x7FFFFFFF))
        {
            inv = UINT64_C(0x7FFFFFFF);
        }
        ok = ok && (tinymind::qInvSqrtQ30(static_cast<uint32_t>(x)) == static_cast<uint32_t>(inv));
        // And it is 2^30 / sqrt(x) to within the floor of s and of the
        // division: a relative error under 2^-13.
        const double exact = std::ldexp(1.0, 30) / std::sqrt(static_cast<double>(x));
        ok = ok && (std::fabs(static_cast<double>(inv) - exact) <= exact * std::ldexp(1.0, -13) + 1.0);
        failures += ok ? 0 : 1;
    }
    std::printf("layernorm sqrt: every x in [1, %llu], %lld failures\n",
                static_cast<unsigned long long>(top), failures);
    BOOST_TEST(tinymind::qIntegerSqrt64(0) == 0u);
    BOOST_TEST(tinymind::qInvSqrtQ30(0) == 0x7FFFFFFFu);
    BOOST_TEST(failures == 0);
}

BOOST_AUTO_TEST_SUITE_END()
