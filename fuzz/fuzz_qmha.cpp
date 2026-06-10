/*
 * libFuzzer harness for QMultiHeadLinearAttention1D over int8 QAttention1D
 * heads.
 *
 * Rationale: the linear-attention head is softmax-free -- its risk is the
 * chain of four requantized MAC stages (Q/K/V projections with ReLU folded
 * into the Q/K requantizer rails, K'^T @ V into the P x P KV grid, then
 * Q' @ KV into the output grid) where each stage's output grid is the next
 * stage's input. The multi-head wrapper adds the per-head scratch reuse and
 * the strided output interleave; running it with independently miscalibrated
 * heads exercises both. The optional bias pointer is fuzz-toggled per run to
 * cover the nullptr branch.
 *
 * Requantizers are built through buildRequantizer with output scales derived
 * from the upstream scales times a headroom factor, the same way calibration
 * derives them, so every (multiplier, shift) pair stays in the
 * importer-reachable band.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qmha.hpp"
#include "qcalibration.hpp"

using tinymind::QMultiHeadLinearAttention1D;
using tinymind::buildRequantizer;

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

template<std::size_t S, std::size_t E, std::size_t D, std::size_t NH>
void run(ByteReader& br)
{
    typedef QMultiHeadLinearAttention1D<int8_t, int8_t, int32_t, int8_t,
                                        int8_t, int8_t, S, E, D, NH> Mha;
    typedef typename Mha::HeadType Head;

    int8_t input[S * E];
    for (std::size_t k = 0; k < S * E; ++k) { input[k] = br.i8(); }

    const float in_scale = br.f(1.0f / 256.0f, 2.0f);

    // Per-head weight and bias storage must stay alive across forward();
    // heads only hold pointers.
    int8_t  weights[NH][Head::TotalWeights];
    int32_t biases[NH][Head::TotalBiases];
    const bool with_bias = (br.u8() & 0x01) != 0;

    Mha mha;
    for (std::size_t h = 0; h < NH; ++h)
    {
        Head& head = mha.heads[h];

        for (std::size_t k = 0; k < Head::TotalWeights; ++k) { weights[h][k] = br.i8(); }

        const float w_scale[3] = {
            br.f(1.0f / 256.0f, 1.0f),
            br.f(1.0f / 256.0f, 1.0f),
            br.f(1.0f / 256.0f, 1.0f),
        };

        // Each grid's scale derives from its producers times a headroom
        // factor, keeping every requantizer ratio in [1/128, 2].
        const float q_scale  = in_scale * w_scale[0] * br.f(0.5f, 128.0f);
        const float k_scale  = in_scale * w_scale[1] * br.f(0.5f, 128.0f);
        const float v_scale  = in_scale * w_scale[2] * br.f(0.5f, 128.0f);
        const float kv_scale = k_scale * v_scale * br.f(0.5f, 128.0f);
        const float out_scale = q_scale * kv_scale * br.f(0.5f, 128.0f);

        for (std::size_t g = 0; g < 3; ++g)
        {
            for (std::size_t k = 0; k < D; ++k)
            {
                const float b_real = br.f(-8.0f, 8.0f);
                biases[h][g * D + k] = static_cast<int32_t>(
                    b_real / (in_scale * w_scale[g]));
            }
        }

        head.weights          = weights[h];
        head.biases           = with_bias ? biases[h]
                                          : static_cast<const int32_t*>(nullptr);
        head.input_zero_point = br.i8();
        head.q_zero_point     = br.i8();
        head.k_zero_point     = br.i8();
        head.v_zero_point     = br.i8();
        head.kv_zero_point    = br.i8();

        // ReLU on Q'/K' folds into the requantizer as qmin = zero_point.
        head.q_requantizer = buildRequantizer<int8_t>(
            in_scale, w_scale[0], q_scale, head.q_zero_point,
            head.q_zero_point, 127);
        head.k_requantizer = buildRequantizer<int8_t>(
            in_scale, w_scale[1], k_scale, head.k_zero_point,
            head.k_zero_point, 127);
        head.v_requantizer = buildRequantizer<int8_t>(
            in_scale, w_scale[2], v_scale, head.v_zero_point, -128, 127);
        head.kv_requantizer = buildRequantizer<int8_t>(
            k_scale, v_scale, kv_scale, head.kv_zero_point, -128, 127);
        head.output_requantizer = buildRequantizer<int8_t>(
            q_scale, kv_scale, out_scale, br.i8(), -128, 127);
    }

    int8_t q_scratch[Mha::QScratchSize];
    int8_t k_scratch[Mha::KScratchSize];
    int8_t v_scratch[Mha::VScratchSize];
    int8_t kv_scratch[Mha::KVScratchSize];
    int8_t head_out_scratch[Mha::HeadOutScratchSize];
    int8_t output[Mha::OutputSize];

    mha.forward(input, q_scratch, k_scratch, v_scratch,
                kv_scratch, head_out_scratch, output);

    // Touch the output so the call cannot be optimized away.
    volatile int sink = 0;
    for (std::size_t k = 0; k < Mha::OutputSize; ++k) { sink += output[k]; }
    (void)sink;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // The first byte selects the (S, E, HeadDim, NumHeads) shape so the
    // corpus can steer it.
    switch (br.u8() & 0x03)
    {
        case 0:  run<2, 4, 2, 2>(br); break;
        case 1:  run<3, 3, 4, 2>(br); break;
        case 2:  run<1, 6, 2, 3>(br); break;
        default: run<4, 2, 3, 2>(br); break;
    }
    return 0;
}
