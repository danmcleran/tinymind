/*
 * libFuzzer harness for the int8 QCfCCell continuous-time kernel.
 *
 * Rationale: the CfC cell layers five independent MAC->rescale->LUT paths
 * (backbone trunk, ff1/ff2 heads, time-gate A and B) and then interpolates
 * h = (1 - t) * ff1 + t * ff2 with the same sigmoid-grid identity QGRU uses.
 * The time-gate is the novel corner: the elapsed-time constant ts is folded
 * into the time-A requantizer and the combined bias q(b_A * ts + b_B), so a
 * large ts stretches that one rescale ratio far beyond what the other paths
 * see. The hidden state recurs through the backbone, so saturation rails
 * compound across steps; the harness runs 1-4 fuzz-driven time steps.
 *
 * All (multiplier, shift) pairs, LUTs and biases are built through the same
 * qcalibration.hpp helpers the importers use, with scales (and ts) drawn from
 * the band real calibration emits, so findings are reachable in real
 * deployments.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qcfc.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"

using tinymind::QCfCCell;
using tinymind::QCfCScales;
using tinymind::QCfCParams;
using tinymind::buildQCfCParams;
using tinymind::quantizeQCfCBias;
using tinymind::quantizeQCfCTimeBias;
using tinymind::buildQSigmoidLUT;
using tinymind::buildQTanhLUT;
using tinymind::kQActivationLUTSize;

namespace {

// Linear byte cursor. Reads past the end return 0, so short inputs decay to
// zero weights / states rather than being rejected.
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

template<std::size_t I, std::size_t H, std::size_t B>
void run(ByteReader& br)
{
    // Scales in the band real int8 calibration emits; the lut_input_scale
    // floor keeps the MAC->LUT rescale ratios within the left-shift regime
    // the requantization primitive supports. ts spans sub-sample to coarse
    // regular-sampling intervals.
    QCfCScales s;
    s.input_scale            = br.f(1.0f / 4096.0f, 1.0f);
    s.hidden_scale           = br.f(1.0f / 4096.0f, 1.0f);
    s.lut_input_scale        = br.f(1.0f / 1024.0f, 1.0f / 16.0f);
    s.w_backbone_input_scale = br.f(1.0f / 256.0f, 1.0f);
    s.w_backbone_hidden_scale= br.f(1.0f / 256.0f, 1.0f);
    s.w_ff1_scale            = br.f(1.0f / 256.0f, 1.0f);
    s.w_ff2_scale            = br.f(1.0f / 256.0f, 1.0f);
    s.w_time_a_scale         = br.f(1.0f / 256.0f, 1.0f);
    s.w_time_b_scale         = br.f(1.0f / 256.0f, 1.0f);
    s.ts                     = static_cast<double>(br.f(1.0f / 64.0f, 8.0f));

    QCfCParams p;
    buildQCfCParams(s, p);

    int8_t sigmoid_lut[kQActivationLUTSize];
    int8_t tanh_lut[kQActivationLUTSize];
    buildQSigmoidLUT(s.lut_input_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT(s.lut_input_scale, 0, 1.0f / 128.0f, 0, tanh_lut);

    float   bias_real[B > H ? B : H];
    int32_t b_backbone[B];
    int32_t b_ff1[H];
    int32_t b_ff2[H];
    int32_t b_time[H];
    float   b_a_real[H];
    float   b_b_real[H];

    for (std::size_t k = 0; k < B; ++k) { bias_real[k] = br.f(-4.0f, 4.0f); }
    quantizeQCfCBias(bias_real, B, s.lut_input_scale, b_backbone);
    for (std::size_t k = 0; k < H; ++k) { bias_real[k] = br.f(-4.0f, 4.0f); }
    quantizeQCfCBias(bias_real, H, s.lut_input_scale, b_ff1);
    for (std::size_t k = 0; k < H; ++k) { bias_real[k] = br.f(-4.0f, 4.0f); }
    quantizeQCfCBias(bias_real, H, s.lut_input_scale, b_ff2);
    for (std::size_t k = 0; k < H; ++k) { b_a_real[k] = br.f(-4.0f, 4.0f); }
    for (std::size_t k = 0; k < H; ++k) { b_b_real[k] = br.f(-4.0f, 4.0f); }
    quantizeQCfCTimeBias(b_a_real, b_b_real, H, s.ts, s.lut_input_scale, b_time);

    int8_t w_backbone_input[B * I];
    int8_t w_backbone_hidden[B * H];
    int8_t w_ff1[H * B];
    int8_t w_ff2[H * B];
    int8_t w_time_a[H * B];
    int8_t w_time_b[H * B];
    for (std::size_t k = 0; k < B * I; ++k) { w_backbone_input[k] = br.i8(); }
    for (std::size_t k = 0; k < B * H; ++k) { w_backbone_hidden[k] = br.i8(); }
    for (std::size_t k = 0; k < H * B; ++k) { w_ff1[k] = br.i8(); }
    for (std::size_t k = 0; k < H * B; ++k) { w_ff2[k] = br.i8(); }
    for (std::size_t k = 0; k < H * B; ++k) { w_time_a[k] = br.i8(); }
    for (std::size_t k = 0; k < H * B; ++k) { w_time_b[k] = br.i8(); }

    QCfCCell<int8_t, int8_t, int32_t, int8_t, I, H, B> cell;
    cell.w_backbone_input  = w_backbone_input;
    cell.w_backbone_hidden = w_backbone_hidden;
    cell.b_backbone        = b_backbone;
    cell.w_ff1             = w_ff1;
    cell.w_ff2             = w_ff2;
    cell.w_time_a          = w_time_a;
    cell.w_time_b          = w_time_b;
    cell.b_ff1             = b_ff1;
    cell.b_ff2             = b_ff2;
    cell.b_time            = b_time;
    cell.input_zero_point  = br.i8();
    cell.hidden_zero_point = br.i8();
    cell.backbone_input_multiplier  = p.backbone_input_multiplier;
    cell.backbone_input_shift       = p.backbone_input_shift;
    cell.backbone_hidden_multiplier = p.backbone_hidden_multiplier;
    cell.backbone_hidden_shift      = p.backbone_hidden_shift;
    cell.ff1_multiplier             = p.ff1_multiplier;
    cell.ff1_shift                  = p.ff1_shift;
    cell.ff2_multiplier             = p.ff2_multiplier;
    cell.ff2_shift                  = p.ff2_shift;
    cell.time_a_multiplier          = p.time_a_multiplier;
    cell.time_a_shift               = p.time_a_shift;
    cell.time_b_multiplier          = p.time_b_multiplier;
    cell.time_b_shift               = p.time_b_shift;
    cell.sigmoid_lut                = sigmoid_lut;
    cell.tanh_lut                   = tanh_lut;
    cell.one_minus_t_times_ff1_multiplier = p.one_minus_t_times_ff1_multiplier;
    cell.one_minus_t_times_ff1_shift      = p.one_minus_t_times_ff1_shift;
    cell.t_times_ff2_multiplier           = p.t_times_ff2_multiplier;
    cell.t_times_ff2_shift                = p.t_times_ff2_shift;
    cell.output_qmin                = -128;
    cell.output_qmax                = 127;

    int8_t h_state[H];
    for (std::size_t k = 0; k < H; ++k) { h_state[k] = br.i8(); }

    // Recurrence: carry state across fuzz-driven steps so saturation rails
    // feed back through the backbone's hidden MAC.
    const std::size_t steps = 1 + (br.u8() & 0x03);
    int8_t x[I];
    for (std::size_t t = 0; t < steps; ++t)
    {
        for (std::size_t j = 0; j < I; ++j) { x[j] = br.i8(); }
        cell.forward(x, h_state);
    }

    // Touch the state so the calls cannot be optimized away.
    volatile int sink = 0;
    for (std::size_t k = 0; k < H; ++k) { sink += h_state[k]; }
    (void)sink;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // The first byte selects the shape so the corpus can steer it.
    switch (br.u8() & 0x03)
    {
        case 0:  run<3, 4, 5>(br); break;
        case 1:  run<1, 2, 8>(br); break;
        case 2:  run<2, 6, 3>(br); break;
        default: run<4, 4, 4>(br); break;
    }
    return 0;
}
