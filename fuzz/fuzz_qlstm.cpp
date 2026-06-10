/*
 * libFuzzer harness for the int8 QLSTMCell recurrent kernel.
 *
 * Rationale: the LSTM cell chains four gate MACs through
 * multiplyByQuantizedMultiplier into LUT lookups, then runs the cell-state
 * update (f*c_prev + i*g) and the hidden update (o * tanh(c)) through three
 * more rescales and a Requantizer. The risk is *recurrent*: a state value that
 * one step pushes onto a saturation rail becomes the next step's input, so
 * overflow corners compound across time in a way single-step unit tests never
 * see. The harness therefore runs 1-4 fuzz-driven time steps with the hidden
 * and cell state carried across steps, and covers both the int8 and the
 * Phase 12 wide int16 cell-state storage.
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

#include "qlstm.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"

using tinymind::QLSTMCell;
using tinymind::QLSTMScales;
using tinymind::QLSTMParams;
using tinymind::buildQLSTMParams;
using tinymind::quantizeQLSTMBiases;
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

template<typename CellT>
struct CellRails;
template<> struct CellRails<int8_t>  { static constexpr int32_t lo = -128;   static constexpr int32_t hi = 127;   };
template<> struct CellRails<int16_t> { static constexpr int32_t lo = -32768; static constexpr int32_t hi = 32767; };

template<std::size_t I, std::size_t H, typename CellT>
void run(ByteReader& br)
{
    // Scales in the band real int8 calibration emits. The lut_input_scale
    // floor keeps every MAC->LUT rescale ratio within the left-shift regime
    // the requantization primitive supports (same reasoning as
    // fuzz_requantize's shift band).
    QLSTMScales s;
    s.input_scale     = br.f(1.0f / 4096.0f, 1.0f);
    s.hidden_scale    = br.f(1.0f / 4096.0f, 1.0f);
    s.cell_scale      = br.f(1.0f / 4096.0f, 1.0f);
    s.lut_input_scale = br.f(1.0f / 1024.0f, 1.0f / 16.0f);
    for (std::size_t g = 0; g < 4; ++g)
    {
        s.w_input_scale[g]     = br.f(1.0f / 256.0f, 1.0f);
        s.w_recurrent_scale[g] = br.f(1.0f / 256.0f, 1.0f);
    }

    QLSTMParams p;
    buildQLSTMParams(s, p);

    int8_t sigmoid_lut[kQActivationLUTSize];
    int8_t tanh_lut[kQActivationLUTSize];
    buildQSigmoidLUT(s.lut_input_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT(s.lut_input_scale, 0, 1.0f / 128.0f, 0, tanh_lut);

    float   bias_real[4 * H];
    int32_t bias_int[4 * H];
    for (std::size_t k = 0; k < 4 * H; ++k) { bias_real[k] = br.f(-4.0f, 4.0f); }
    quantizeQLSTMBiases(bias_real, H, s.lut_input_scale, bias_int);

    int8_t w_input[4 * H * I];
    int8_t w_recurrent[4 * H * H];
    for (std::size_t k = 0; k < 4 * H * I; ++k) { w_input[k] = br.i8(); }
    for (std::size_t k = 0; k < 4 * H * H; ++k) { w_recurrent[k] = br.i8(); }

    QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, CellT, I, H> cell;
    cell.w_input           = w_input;
    cell.w_recurrent       = w_recurrent;
    cell.biases            = bias_int;
    cell.input_zero_point  = br.i8();
    cell.hidden_zero_point = br.i8();
    cell.cell_zero_point   = static_cast<CellT>(br.i8());
    for (std::size_t g = 0; g < 4; ++g)
    {
        cell.input_to_lut_multiplier[g]     = p.input_to_lut_multiplier[g];
        cell.input_to_lut_shift[g]          = p.input_to_lut_shift[g];
        cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
        cell.recurrent_to_lut_shift[g]      = p.recurrent_to_lut_shift[g];
    }
    cell.sigmoid_lut             = sigmoid_lut;
    cell.tanh_lut                = tanh_lut;
    cell.f_times_c_multiplier    = p.f_times_c_multiplier;
    cell.f_times_c_shift         = p.f_times_c_shift;
    cell.i_times_g_multiplier    = p.i_times_g_multiplier;
    cell.i_times_g_shift         = p.i_times_g_shift;
    cell.cell_qmin               = static_cast<CellT>(CellRails<CellT>::lo);
    cell.cell_qmax               = static_cast<CellT>(CellRails<CellT>::hi);
    cell.cell_to_tanh_multiplier = p.cell_to_tanh_multiplier;
    cell.cell_to_tanh_shift      = p.cell_to_tanh_shift;
    cell.tanh_cell_lut           = tanh_lut;
    cell.output_requantizer.multiplier = p.output_multiplier;
    cell.output_requantizer.shift      = p.output_shift;
    cell.output_requantizer.zero_point = cell.hidden_zero_point;
    cell.output_requantizer.qmin       = -128;
    cell.output_requantizer.qmax       = 127;

    int8_t h_state[H];
    CellT  c_state[H];
    for (std::size_t k = 0; k < H; ++k) { h_state[k] = br.i8(); }
    // Seed the cell state across its full storage range, not just int8: the
    // wide int16 cell exists precisely to hold values an int8 grid cannot.
    for (std::size_t k = 0; k < H; ++k)
    {
        const int32_t span = CellRails<CellT>::hi - CellRails<CellT>::lo;
        const int32_t raw  = static_cast<int32_t>(br.u8()) << 8 | br.u8();
        c_state[k] = static_cast<CellT>(CellRails<CellT>::lo + (raw % (span + 1)));
    }

    // Recurrence: carry state across fuzz-driven steps so saturation rails
    // feed back into the next step's MACs.
    const std::size_t steps = 1 + (br.u8() & 0x03);
    int8_t x[I];
    for (std::size_t t = 0; t < steps; ++t)
    {
        for (std::size_t j = 0; j < I; ++j) { x[j] = br.i8(); }
        cell.forward(x, h_state, c_state);
    }

    // Touch the state so the calls cannot be optimized away.
    volatile int sink = 0;
    for (std::size_t k = 0; k < H; ++k) { sink += h_state[k]; sink += static_cast<int>(c_state[k]); }
    (void)sink;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // The first byte selects shape and cell-state width so the corpus can
    // steer it; the int16 case locks the Phase 12 wide-cell corner.
    switch (br.u8() & 0x03)
    {
        case 0:  run<3, 4, int8_t>(br);  break;
        case 1:  run<1, 8, int8_t>(br);  break;
        case 2:  run<4, 2, int8_t>(br);  break;
        default: run<3, 4, int16_t>(br); break;
    }
    return 0;
}
