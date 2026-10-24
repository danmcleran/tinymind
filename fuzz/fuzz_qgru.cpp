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
 * Property checked (a failure traps): permutation equivariance. Relabelling
 * the hidden units (rows of every gate's weights and biases, columns of the
 * recurrent weights, entries of h) and the input features (columns of the
 * input weights, entries of x) must relabel the final hidden state the same
 * way, bit for bit. Integer sums are order-independent, so any difference
 * is an indexing, stride or gate-offset error.
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
#include "fuzz_check.hpp"

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

    int8_t h_initial[H];
    for (std::size_t k = 0; k < H; ++k) { h_initial[k] = h_state[k]; }

    // Recurrence: carry state across fuzz-driven steps so saturation rails
    // feed back into the next step's MACs.
    const std::size_t steps = 1 + (br.u8() & 0x03);
    int8_t x[4][I];
    for (std::size_t t = 0; t < steps; ++t)
    {
        for (std::size_t j = 0; j < I; ++j) { x[t][j] = br.i8(); }
        cell.forward(x[t], h_state);
    }
    tinymind_fuzz::emit(h_state, sizeof(h_state));

    // The same cell with hidden units and input features relabelled.
    std::size_t hp[H];
    std::size_t ip[I];
    tinymind_fuzz::makePermutation(hp, H, br.u8());
    tinymind_fuzz::makePermutation(ip, I, br.u8());

    int8_t  pw_input[3 * H * I];
    int8_t  pw_recurrent[3 * H * H];
    int32_t pbias[3 * H];
    tinymind_fuzz::permuteMatrix(w_input, pw_input, 3, H, I, hp, ip);
    tinymind_fuzz::permuteMatrix(w_recurrent, pw_recurrent, 3, H, H, hp, hp);
    tinymind_fuzz::permuteMatrix(bias_int, pbias, 3, H, 1, hp, static_cast<const std::size_t*>(nullptr));

    QGRUCell<int8_t, int8_t, int32_t, int8_t, int8_t, I, H> pcell = cell;
    pcell.w_input     = pw_input;
    pcell.w_recurrent = pw_recurrent;
    pcell.biases      = pbias;

    int8_t ph[H];
    for (std::size_t k = 0; k < H; ++k) { ph[hp[k]] = h_initial[k]; }
    int8_t px[I];
    for (std::size_t t = 0; t < steps; ++t)
    {
        for (std::size_t j = 0; j < I; ++j) { px[ip[j]] = x[t][j]; }
        pcell.forward(px, ph);
    }
    for (std::size_t k = 0; k < H; ++k)
    {
        TINYMIND_FUZZ_CHECK(ph[hp[k]] == h_state[k]);
    }
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
