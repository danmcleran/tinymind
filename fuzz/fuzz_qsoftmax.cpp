/*
 * libFuzzer harness for the int8 QSoftmax1D compute kernel.
 *
 * Rationale: like QLayerNorm, softmax is a per-row reduction whose integer
 * path has overflow/divide edge cases that fixed-input unit tests miss. The
 * high-risk spots are the int64 exp-sum accumulation, the
 * (exp << 8 + half_sum) / sum numerator, and the all-zero-LUT branch that
 * guards the divide-by-zero. Adversarial activation rows (a dominant class,
 * an all-equal row, a near-degenerate scale) drive each of those.
 *
 * The exp LUT is built through buildQSoftmaxExpLUT -- the same calibration
 * helper the importers use -- so findings are reachable in real deployments,
 * not artifacts of impossible LUT values.
 *
 * Properties checked on every input (a failure traps):
 *   1. Order: within a row, a larger input never gets a smaller output, and
 *      equal inputs get equal outputs.
 *   2. Mass: each row's probabilities, in units of 1/256, sum to 256 within
 *      the rounding of F elements (half a unit each) plus one unit for the
 *      clamp of a single saturated class at qmax.
 *   3. Accuracy: each output is within one unit of round(256 * softmax) of
 *      the dequantized row, computed in double.
 *   4. Shift invariance: adding a constant to a row leaves its output
 *      unchanged bit for bit (softmax depends only on differences).
 *   5. Equivariance: reversing a row reverses its output bit for bit.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include <cmath>

#include "qsoftmax.hpp"
#include "qcalibration.hpp"
#include "fuzz_check.hpp"

using tinymind::QSoftmax1D;
using tinymind::buildQSoftmaxExpLUT;
using tinymind::kQSoftmaxExpLUTSize;

namespace {

// Linear byte cursor. Reads past the end return 0, so short inputs exercise
// all-equal / zero rows rather than being rejected.
struct ByteReader
{
    const uint8_t* p;
    size_t n;
    size_t i;

    explicit ByteReader(const uint8_t* data, size_t size) : p(data), n(size), i(0) {}

    uint8_t u8() { return (i < n) ? p[i++] : static_cast<uint8_t>(0); }
    int8_t  i8() { return static_cast<int8_t>(u8()); }

    // Map a byte to a float in [lo, hi].
    float f(float lo, float hi) { return lo + (static_cast<float>(u8()) / 255.0f) * (hi - lo); }
};

template<std::size_t R, std::size_t F>
void run(ByteReader& br)
{
    int8_t  input[R * F];
    int8_t  output[R * F];

    for (std::size_t k = 0; k < R * F; ++k) { input[k] = br.i8(); output[k] = 0; }

    // input_scale spans calibrated-to-mis-calibrated: a tiny scale makes every
    // exp(q-max) collapse toward an all-zero LUT (the divide-by-zero guard); a
    // large scale saturates the dominant class.
    const float input_scale = br.f(1.0f / 4096.0f, 8.0f);

    int32_t exp_lut[kQSoftmaxExpLUTSize];
    buildQSoftmaxExpLUT(input_scale, exp_lut);

    QSoftmax1D<int8_t, int8_t, R, F> sm;
    sm.exp_lut = exp_lut;
    sm.output_zero_point = -128;
    sm.qmin = -128;
    sm.qmax = 127;

    sm.forward(input, output);
    tinymind_fuzz::emit(output, sizeof(output));

    // Probabilities in units of 1/256: output - zero point.
    const int32_t zp = static_cast<int32_t>(sm.output_zero_point);
    for (std::size_t r = 0; r < R; ++r)
    {
        const int8_t* in  = input  + r * F;
        const int8_t* out = output + r * F;

        int32_t total = 0;
        int32_t max_q = in[0];
        for (std::size_t i = 0; i < F; ++i)
        {
            total += static_cast<int32_t>(out[i]) - zp;
            if (in[i] > max_q) { max_q = in[i]; }
            for (std::size_t j = 0; j < F; ++j)
            {
                if (in[i] > in[j])
                {
                    TINYMIND_FUZZ_CHECK(out[i] >= out[j]);
                }
                if (in[i] == in[j])
                {
                    TINYMIND_FUZZ_CHECK(out[i] == out[j]);
                }
            }
        }
        const int32_t mass_error = (total > 256) ? (total - 256) : (256 - total);
        TINYMIND_FUZZ_CHECK(2 * mass_error <= static_cast<int32_t>(F) + 2);

        double denom = 0.0;
        for (std::size_t i = 0; i < F; ++i)
        {
            denom += std::exp(static_cast<double>(in[i] - max_q) * static_cast<double>(input_scale));
        }
        for (std::size_t i = 0; i < F; ++i)
        {
            const double p = std::exp(static_cast<double>(in[i] - max_q) *
                                      static_cast<double>(input_scale)) / denom;
            double want = std::floor(256.0 * p + 0.5);
            if (want > 255.0) { want = 255.0; }
            const double got = static_cast<double>(static_cast<int32_t>(out[i]) - zp);
            TINYMIND_FUZZ_CHECK(std::fabs(got - want) <= 1.0);
        }
    }

    // Shift: move every row by the same constant, as far as the row with
    // the widest headroom in the chosen direction allows for all rows.
    int8_t shifted_in[R * F];
    int8_t shifted_out[R * F];
    int32_t lo = 127;
    int32_t hi = -128;
    for (std::size_t k = 0; k < R * F; ++k)
    {
        if (input[k] < lo) { lo = input[k]; }
        if (input[k] > hi) { hi = input[k]; }
    }
    const int32_t up   = 127 - hi;
    const int32_t down = lo + 128;
    const int32_t raw  = static_cast<int32_t>(br.u8());
    const int32_t delta = (raw & 0x01) ? ((up > 0) ? (raw % (up + 1)) : 0)
                                       : -((down > 0) ? (raw % (down + 1)) : 0);
    for (std::size_t k = 0; k < R * F; ++k)
    {
        shifted_in[k] = static_cast<int8_t>(input[k] + delta);
    }
    sm.forward(shifted_in, shifted_out);
    for (std::size_t k = 0; k < R * F; ++k)
    {
        TINYMIND_FUZZ_CHECK(shifted_out[k] == output[k]);
    }

    int8_t reversed_in[R * F];
    int8_t reversed_out[R * F];
    for (std::size_t r = 0; r < R; ++r)
    {
        for (std::size_t i = 0; i < F; ++i)
        {
            reversed_in[r * F + i] = input[r * F + (F - 1 - i)];
        }
    }
    sm.forward(reversed_in, reversed_out);
    for (std::size_t r = 0; r < R; ++r)
    {
        for (std::size_t i = 0; i < F; ++i)
        {
            TINYMIND_FUZZ_CHECK(reversed_out[r * F + i] == output[r * F + (F - 1 - i)]);
        }
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // The first byte selects the shape so the corpus can steer it: single row
    // (reduction edge cases), small and wider feature counts.
    switch (br.u8() & 0x03)
    {
        case 0:  run<1, 4>(br);  break;
        case 1:  run<2, 8>(br);  break;
        case 2:  run<1, 16>(br); break;
        default: run<4, 8>(br);  break;
    }
    return 0;
}
