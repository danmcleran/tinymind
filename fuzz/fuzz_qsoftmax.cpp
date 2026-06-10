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
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qsoftmax.hpp"
#include "qcalibration.hpp"

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

    // Touch the output so the call cannot be optimized away.
    volatile int sink = 0;
    for (std::size_t k = 0; k < R * F; ++k) { sink += output[k]; }
    (void)sink;
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
