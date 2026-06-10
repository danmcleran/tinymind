/*
 * libFuzzer harness for the int8 QGRUCell recurrent kernel.
 *
 * Rationale: the GRU cell has the same two-MAC-per-gate rescale structure as
 * QLSTM plus two paths QLSTM does not have: the reset product r_t * h_prev is
 * materialized back into the int8 hidden grid before the new-gate MAC consumes
 * it, and the final interpolation uses the "(1 - z) == 128 - z in the sigmoid
 * grid" identity. Both are exact-arithmetic corners where an off-by-one in
 * centering or a saturation rail feeds back into the next step's recurrent
 * MAC. The harness runs 1-4 fuzz-driven time steps with the hidden state
 * carried across steps.
 *
 * All (multiplier, shift) pairs, LUTs and biases are built through the same
 * qcalibration.hpp helpers the importers use, with scales drawn from the band
 * real calibration emits, so findings are reachable in real deployments.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qgru.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"

using tinymind::QGRUCell;
using tinymind::QGRUScales;
using tinymind::QGRUParams;
using tinymind::buildQGRUParams;
using tinymind::quantizeQGRUBiases;
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

template<std::size_t I, std::size_t H>
void run(ByteReader& br)
{
    // Scales in the band real int8 calibration emits; the lut_input_scale
    // floor keeps the MAC->LUT rescale ratios within the left-shift regime
    // the requantization primitive supports.
    QGRUScales s;
    s.input_scale     = br.f(1.0f / 4096.0f, 1.0f);
    s.hidden_scale    = br.f(1.0f / 4096.0f, 1.0f);
    s.lut_input_scale = br.f(1.0f / 1024.0f, 1.0f / 16.0f);
    for (std::size_t g = 0; g < 3; ++g)
    {
        s.w_input_scale[g]     = br.f(1.0f / 256.0f, 1.0f);
        s.w_recurrent_scale[g] = br.f(1.0f / 256.0f, 1.0f);
    }

    QGRUParams p;
    buildQGRUParams(s, p);

    int8_t sigmoid_lut[kQActivationLUTSize];
    int8_t tanh_lut[kQActivationLUTSize];
    buildQSigmoidLUT(s.lut_input_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT(s.lut_input_scale, 0, 1.0f / 128.0f, 0, tanh_lut);

    float   bias_real[3 * H];
    int32_t bias_int[3 * H];
    for (std::size_t k = 0; k < 3 * H; ++k) { bias_real[k] = br.f(-4.0f, 4.0f); }
    quantizeQGRUBiases(bias_real, H, s.lut_input_scale, bias_int);

    int8_t w_input[3 * H * I];
    int8_t w_recurrent[3 * H * H];
    for (std::size_t k = 0; k < 3 * H * I; ++k) { w_input[k] = br.i8(); }
    for (std::size_t k = 0; k < 3 * H * H; ++k) { w_recurrent[k] = br.i8(); }

    QGRUCell<int8_t, int8_t, int32_t, int8_t, int8_t, I, H> cell;
    cell.w_input           = w_input;
    cell.w_recurrent       = w_recurrent;
    cell.biases            = bias_int;
    cell.input_zero_point  = br.i8();
    cell.hidden_zero_point = br.i8();
    for (std::size_t g = 0; g < 3; ++g)
    {
        cell.input_to_lut_multiplier[g]     = p.input_to_lut_multiplier[g];
        cell.input_to_lut_shift[g]          = p.input_to_lut_shift[g];
        cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
        cell.recurrent_to_lut_shift[g]      = p.recurrent_to_lut_shift[g];
    }
    cell.sigmoid_lut                    = sigmoid_lut;
    cell.tanh_lut                       = tanh_lut;
    cell.r_times_h_multiplier           = p.r_times_h_multiplier;
    cell.r_times_h_shift                = p.r_times_h_shift;
    cell.one_minus_z_times_n_multiplier = p.one_minus_z_times_n_multiplier;
    cell.one_minus_z_times_n_shift      = p.one_minus_z_times_n_shift;
    cell.z_times_h_multiplier           = p.z_times_h_multiplier;
    cell.z_times_h_shift                = p.z_times_h_shift;
    cell.output_qmin                    = -128;
    cell.output_qmax                    = 127;

    int8_t h_state[H];
    for (std::size_t k = 0; k < H; ++k) { h_state[k] = br.i8(); }

    // Recurrence: carry state across fuzz-driven steps so saturation rails
    // feed back into the next step's MACs.
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
        case 0:  run<3, 4>(br); break;
        case 1:  run<1, 8>(br); break;
        case 2:  run<4, 2>(br); break;
        default: run<2, 6>(br); break;
    }
    return 0;
}
