/*
 * libFuzzer harness for the int8 QAttentionSoftmax1D kernel.
 *
 * Rationale: softmax attention chains three requantized MAC stages (Q/K/V
 * projections, the S x S score grid with 1/sqrt(P) folded into its
 * requantizer, and the final A @ V) around an inlined per-row TFLite-style
 * softmax. That inlined softmax duplicates QSoftmax1D's int64 exp-sum,
 * (exp << 8 + half_sum) / sum numerator and all-zero-LUT guard, so it gets
 * the same adversarial-row treatment fuzz_qsoftmax gives the standalone
 * kernel -- but here the rows are *produced* by the upstream integer
 * pipeline, so miscalibrated projection/score grids are part of the attack
 * surface. The optional bias pointer is fuzz-toggled to cover the nullptr
 * branch.
 *
 * Requantizers are built through quantizeMultiplier with output scales
 * derived from the upstream scales times a headroom factor, the same way
 * calibration derives them, so every (multiplier, shift) pair stays in the
 * importer-reachable band.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qattention_softmax.hpp"
#include "qcalibration.hpp"

using tinymind::QAttentionSoftmax1D;
using tinymind::Requantizer;
using tinymind::buildRequantizer;
using tinymind::buildQSoftmaxExpLUT;
using tinymind::kQSoftmaxExpLUTSize;
using tinymind::qAttentionInvSqrt;
using tinymind::quantizeMultiplier;

namespace {

// Linear byte cursor. Reads past the end return 0, so short inputs decay to
// zero tensors rather than being rejected.
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

template<std::size_t S, std::size_t E, std::size_t P>
void run(ByteReader& br)
{
    typedef QAttentionSoftmax1D<int8_t, int8_t, int32_t, int8_t, int8_t,
                                int8_t, int8_t, S, E, P> Att;

    int8_t input[Att::InputSize];
    int8_t weights[Att::TotalWeights];
    for (std::size_t k = 0; k < Att::InputSize; ++k)    { input[k] = br.i8(); }
    for (std::size_t k = 0; k < Att::TotalWeights; ++k) { weights[k] = br.i8(); }

    const float in_scale = br.f(1.0f / 256.0f, 2.0f);
    const float w_scale[3] = {
        br.f(1.0f / 256.0f, 1.0f),
        br.f(1.0f / 256.0f, 1.0f),
        br.f(1.0f / 256.0f, 1.0f),
    };

    // Output scales derive from the upstream scales times a headroom factor,
    // exactly as calibration picks them, which keeps each requantizer ratio
    // in [1/128, 2] -- the left-shift regime the primitive supports.
    const float q_scale = in_scale * w_scale[0] * br.f(0.5f, 128.0f);
    const float k_scale = in_scale * w_scale[1] * br.f(0.5f, 128.0f);
    const float v_scale = in_scale * w_scale[2] * br.f(0.5f, 128.0f);

    // Optional biases: int32 in the accumulator scale, derived from a real
    // band the way the importers quantize them. One bit covers the nullptr
    // (bias-free) branch.
    int32_t biases[Att::TotalBiases];
    const bool with_bias = (br.u8() & 0x01) != 0;
    for (std::size_t g = 0; g < 3; ++g)
    {
        for (std::size_t k = 0; k < P; ++k)
        {
            const float b_real = br.f(-8.0f, 8.0f);
            biases[g * P + k] = static_cast<int32_t>(
                b_real / (in_scale * w_scale[g]));
        }
    }

    Att att;
    att.weights          = weights;
    att.biases           = with_bias ? biases : static_cast<const int32_t*>(nullptr);
    att.input_zero_point = br.i8();
    att.q_zero_point     = br.i8();
    att.k_zero_point     = br.i8();
    att.v_zero_point     = br.i8();
    att.attn_zero_point  = -128;

    att.q_requantizer = buildRequantizer<int8_t>(
        in_scale, w_scale[0], q_scale, att.q_zero_point, -128, 127);
    att.k_requantizer = buildRequantizer<int8_t>(
        in_scale, w_scale[1], k_scale, att.k_zero_point, -128, 127);
    att.v_requantizer = buildRequantizer<int8_t>(
        in_scale, w_scale[2], v_scale, att.v_zero_point, -128, 127);

    // Score grid: ratio = q_scale * k_scale * (1/sqrt(P)) / score_scale.
    const double inv_sqrt_p  = qAttentionInvSqrt(P);
    const float  score_scale = static_cast<float>(
        static_cast<double>(q_scale) * static_cast<double>(k_scale) *
        inv_sqrt_p * static_cast<double>(br.f(0.5f, 128.0f)));
    int32_t score_mult = 0;
    int32_t score_shift = 0;
    quantizeMultiplier(static_cast<double>(q_scale) * static_cast<double>(k_scale) *
                       inv_sqrt_p / static_cast<double>(score_scale),
                       score_mult, score_shift);
    att.score_requantizer.multiplier = score_mult;
    att.score_requantizer.shift      = score_shift;
    att.score_requantizer.zero_point = br.i8();
    att.score_requantizer.qmin       = -128;
    att.score_requantizer.qmax       = 127;

    // Softmax exp LUT built from the score scale, same as the importers. A
    // tiny score scale collapses the LUT toward all-zero (the sum <= 0
    // guard); a large one saturates the dominant class.
    int32_t exp_lut[kQSoftmaxExpLUTSize];
    buildQSoftmaxExpLUT(score_scale, exp_lut);
    att.softmax_exp_lut = exp_lut;
    att.attn_qmin       = -128;
    att.attn_qmax       = 127;

    // A @ V: attention weights are on the fixed 1/256 grid.
    const float out_scale = (1.0f / 256.0f) * v_scale * br.f(0.5f, 128.0f);
    att.output_requantizer = buildRequantizer<int8_t>(
        1.0f / 256.0f, v_scale, out_scale, br.i8(), -128, 127);

    int8_t q_scratch[Att::QScratchSize];
    int8_t k_scratch[Att::KScratchSize];
    int8_t v_scratch[Att::VScratchSize];
    int8_t score_scratch[Att::ScoreScratchSize];
    int8_t attn_scratch[Att::AttnScratchSize];
    int8_t output[Att::OutputSize];

    att.forward(input, q_scratch, k_scratch, v_scratch,
                score_scratch, attn_scratch, output);

    // Touch the output so the call cannot be optimized away.
    volatile int sink = 0;
    for (std::size_t k = 0; k < Att::OutputSize; ++k) { sink += output[k]; }
    (void)sink;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // The first byte selects the (S, E, P) shape so the corpus can steer it:
    // single-row softmax edge cases, square, wide-projection, long-sequence.
    switch (br.u8() & 0x03)
    {
        case 0:  run<1, 8, 4>(br); break;
        case 1:  run<4, 4, 4>(br); break;
        case 2:  run<3, 2, 8>(br); break;
        default: run<8, 3, 2>(br); break;
    }
    return 0;
}
