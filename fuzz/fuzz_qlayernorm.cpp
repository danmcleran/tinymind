/*
 * libFuzzer harness for the int8 QLayerNorm1D compute kernel.
 *
 * Rationale: TinyMind has no runtime deserialization surface -- weights are
 * baked in as const arrays by the offline importers, so there is no parser to
 * fuzz. The real input-dependent risk is *arithmetic* in the int8 layers fed
 * adversarial activation tensors and quantization parameters: the exact class
 * of bug that fixed-input unit tests miss (e.g. the int32 overflow in
 * QLayerNorm's gamma multiply that only a near-zero-variance row triggers).
 *
 * This harness drives QLayerNorm1D::forward with fuzz-derived input, gamma,
 * beta and output scale, under ASan+UBSan. Short/empty inputs decay to
 * constant (zero-variance) rows on purpose -- that is the pathological case
 * that inflates inv_stddev. Quantization parameters are built through the same
 * helpers the importers use, so findings are reachable in real deployments,
 * not artifacts of impossible parameter values.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qlayernorm.hpp"

using tinymind::QLayerNorm1D;
using tinymind::quantizeLayerNormGamma;
using tinymind::quantizeLayerNormBeta;
using tinymind::buildQLayerNormOutputParams;

namespace {

// Linear byte cursor. Reads past the end return 0, so short inputs exercise
// constant/zero-variance rows rather than being rejected.
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
    int16_t gamma_int[F];
    int32_t beta_int[F];
    float   gamma_f[F];
    float   beta_f[F];

    for (std::size_t k = 0; k < R * F; ++k) { input[k] = br.i8(); output[k] = 0; }
    for (std::size_t k = 0; k < F; ++k)     { gamma_f[k] = br.f(-4.0f, 4.0f); }
    for (std::size_t k = 0; k < F; ++k)     { beta_f[k]  = br.f(-2.0f, 2.0f); }

    // A small positive output scale spanning the realistic range.
    const float output_scale = br.f(1.0f / 4096.0f, 1.0f);

    quantizeLayerNormGamma(gamma_f, F, gamma_int);
    quantizeLayerNormBeta(beta_f, F, output_scale, beta_int);

    int32_t out_mult = 0;
    int32_t out_shift = 0;
    buildQLayerNormOutputParams(output_scale, out_mult, out_shift);

    QLayerNorm1D<int8_t, int8_t, R, F> ln;
    ln.gamma = gamma_int;
    ln.beta = beta_int;
    // epsilon down to 1 keeps inv-sqrt finite while letting near-zero variance
    // drive inv_stddev as large as the fixed-point domain allows.
    ln.epsilon_q = static_cast<int32_t>(br.u8()) + 1;
    ln.output_multiplier = out_mult;
    ln.output_shift = out_shift;
    ln.output_zero_point = 0;
    ln.qmin = -128;
    ln.qmax = 127;

    ln.forward(input, output);

    // Touch the output so the call cannot be optimized away.
    volatile int sink = 0;
    for (std::size_t k = 0; k < R * F; ++k) { sink += output[k]; }
    (void)sink;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // A few shapes: single row (variance edge cases), small and wider feature
    // counts. The first byte selects the shape so the corpus can steer it.
    switch (br.u8() & 0x03)
    {
        case 0:  run<1, 4>(br);  break;
        case 1:  run<2, 8>(br);  break;
        case 2:  run<1, 16>(br); break;
        default: run<4, 8>(br);  break;
    }
    return 0;
}
