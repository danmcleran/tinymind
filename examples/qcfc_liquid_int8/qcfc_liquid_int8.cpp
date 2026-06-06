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

// int8 Closed-form Continuous-time (CfC) liquid cell -- end-to-end deployment
// exemplar. Calibrates a small CfC cell on the host (qcalibration.hpp), then
// runs a pure-integer recurrent forward (qcfc.hpp, QCfCCell) over a fixed input
// sequence with no <cmath> on the inference path.
//
//   make run     parity report: max-abs error of the int8 hidden state vs the
//                float CfC reference, over the sequence
//   make bench   CSV cycle/byte report (per-step forward cost + footprint)
//   make golden  stable int8 hidden-state byte stream (deterministic; for a
//                future integration-suite fixture)
//
// Regular-sampling deployable form: the elapsed time ts is a calibration
// constant folded into the time-gate-A requantizer (see qcfc.hpp).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>

#include "qcfc.hpp"
#include "qaffine.hpp"
#include "qactivations.hpp"
#include "include/qcalibration.hpp"
#include "include/bench/report.hpp"

namespace {

constexpr std::size_t I  = 3;    // inputs
constexpr std::size_t H  = 6;    // hidden / state width
constexpr std::size_t BB = 8;    // backbone width
constexpr std::size_t L  = 16;   // sequence length
constexpr double      TS = 1.0;  // elapsed time per step (regular sampling)

typedef tinymind::QCfCCell<int8_t, int8_t, int32_t, int8_t, I, H, BB> Cell;

// --- float weights (deterministic, small so tanh stays near-linear) ---------
float w_bin[BB * I], w_bh[BB * H];
float w_ff1[H * BB], w_ff2[H * BB], w_ta[H * BB], w_tb[H * BB];
float b_bb[BB], b_ff1[H], b_ff2[H], b_a[H], b_b[H];

float seqIn[L][I];

void initData()
{
    unsigned s = 0xC0FFEEu;
    auto nxt = [&s]() {
        s = s * 1103515245u + 12345u;
        return 0.30f * ((static_cast<float>((s >> 16) & 0x7fff) / 32767.0f) - 0.5f);
    };
    for (std::size_t i = 0; i < BB * I; ++i) w_bin[i] = nxt();
    for (std::size_t i = 0; i < BB * H; ++i) w_bh[i]  = nxt();
    for (std::size_t i = 0; i < H * BB; ++i) { w_ff1[i]=nxt(); w_ff2[i]=nxt(); w_ta[i]=nxt(); w_tb[i]=nxt(); }
    for (std::size_t i = 0; i < BB; ++i) b_bb[i] = nxt();
    for (std::size_t i = 0; i < H; ++i) { b_ff1[i]=nxt(); b_ff2[i]=nxt(); b_a[i]=nxt(); b_b[i]=nxt(); }
    for (std::size_t t = 0; t < L; ++t)
        for (std::size_t i = 0; i < I; ++i) seqIn[t][i] = nxt();
}

// Float CfC reference matching the int8 cell's exact computation graph.
void floatCfcStep(const float* x, float* h)
{
    auto sig = [](float v){ return 1.0f / (1.0f + std::exp(-v)); };
    float x1[BB];
    for (std::size_t u = 0; u < BB; ++u)
    {
        float z = b_bb[u];
        for (std::size_t j = 0; j < I; ++j) z += w_bin[u * I + j] * x[j];
        for (std::size_t k = 0; k < H; ++k) z += w_bh[u * H + k] * h[k];
        x1[u] = std::tanh(z);
    }
    float out[H];
    for (std::size_t i = 0; i < H; ++i)
    {
        float a1 = b_ff1[i], a2 = b_ff2[i], aA = b_a[i], aB = b_b[i];
        for (std::size_t u = 0; u < BB; ++u)
        {
            a1 += w_ff1[i * BB + u] * x1[u];
            a2 += w_ff2[i * BB + u] * x1[u];
            aA += w_ta [i * BB + u] * x1[u];
            aB += w_tb [i * BB + u] * x1[u];
        }
        const float t = sig(aA * static_cast<float>(TS) + aB);
        out[i] = (1.0f - t) * std::tanh(a1) + t * std::tanh(a2);
    }
    for (std::size_t i = 0; i < H; ++i) h[i] = out[i];
}

// --- quantized model (caller-owned buffers) ---------------------------------
int8_t  w_bin_q[BB * I], w_bh_q[BB * H];
int8_t  w_ff1_q[H * BB], w_ff2_q[H * BB], w_ta_q[H * BB], w_tb_q[H * BB];
int32_t b_bb_q[BB], b_ff1_q[H], b_ff2_q[H], b_time_q[H];
int8_t  sigmoid_lut[256], tanh_lut[256];
Cell    gCell;

float x_scale   = 1.0f / 127.0f;   // set by calibrateScales()
float h_scale   = 1.0f / 127.0f;   // set by calibrateScales()
float lut_scale = 8.0f / 127.0f;   // set by calibrateScales()

// Calibrate the input, hidden-state, and shared LUT-input scales from the float
// reference so the int8 grid actually resolves every signal. The hand-crafted
// weights here are small, so the activations and pre-activations are small
// (~0.03 hidden, ~0.2 pre-activation); the naive 1/127 and 8/127 scales would
// quantize them to a few LSB and collapse all dynamics. Observing the real
// ranges (a RangeObserver pass) is exactly what a deployment does.
void calibrateScales()
{
    float x_absmax = 1e-6f, h_absmax = 1e-6f, p_absmax = 1e-6f;
    float h[H] = {0};
    for (std::size_t t = 0; t < L; ++t)
    {
        const float* x = seqIn[t];
        for (std::size_t i = 0; i < I; ++i)
            x_absmax = std::fmax(x_absmax, std::fabs(x[i]));

        // Backbone trunk + its pre-activations.
        float x1[BB];
        for (std::size_t u = 0; u < BB; ++u)
        {
            float z = b_bb[u];
            for (std::size_t j = 0; j < I; ++j) z += w_bin[u * I + j] * x[j];
            for (std::size_t k = 0; k < H; ++k) z += w_bh[u * H + k] * h[k];
            p_absmax = std::fmax(p_absmax, std::fabs(z));
            x1[u] = std::tanh(z);
        }
        // Head + time-gate pre-activations.
        for (std::size_t i = 0; i < H; ++i)
        {
            float a1 = b_ff1[i], a2 = b_ff2[i], aA = b_a[i], aB = b_b[i];
            for (std::size_t u = 0; u < BB; ++u)
            {
                a1 += w_ff1[i * BB + u] * x1[u];
                a2 += w_ff2[i * BB + u] * x1[u];
                aA += w_ta [i * BB + u] * x1[u];
                aB += w_tb [i * BB + u] * x1[u];
            }
            p_absmax = std::fmax(p_absmax, std::fmax(std::fabs(a1), std::fabs(a2)));
            p_absmax = std::fmax(p_absmax, std::fabs(aA * static_cast<float>(TS) + aB));
        }

        floatCfcStep(x, h);   // advances the float reference state
        for (std::size_t i = 0; i < H; ++i)
            h_absmax = std::fmax(h_absmax, std::fabs(h[i]));
    }
    x_scale   = x_absmax / 127.0f;
    h_scale   = h_absmax / 127.0f;
    lut_scale = p_absmax / 127.0f;
}

float absmax(const float* p, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        const float a = std::fabs(p[i]);
        if (a > m) m = a;
    }
    return m;
}

void quantizeSymmetricWeights(const float* src, int8_t* dst, std::size_t n, float scale)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        long c = std::lround(static_cast<double>(src[i]) / static_cast<double>(scale));
        if (c < -127) c = -127;
        if (c >  127) c =  127;
        dst[i] = static_cast<int8_t>(c);
    }
}

void calibrate()
{
    using namespace tinymind;
    QCfCScales sc;
    sc.input_scale = x_scale; sc.hidden_scale = h_scale; sc.lut_input_scale = lut_scale;
    sc.w_backbone_input_scale  = absmax(w_bin, BB * I) / 127.0f;
    sc.w_backbone_hidden_scale = absmax(w_bh,  BB * H) / 127.0f;
    sc.w_ff1_scale = absmax(w_ff1, H * BB) / 127.0f;
    sc.w_ff2_scale = absmax(w_ff2, H * BB) / 127.0f;
    sc.w_time_a_scale = absmax(w_ta, H * BB) / 127.0f;
    sc.w_time_b_scale = absmax(w_tb, H * BB) / 127.0f;
    sc.ts = TS;

    QCfCParams p; buildQCfCParams(sc, p);
    quantizeQCfCBias(b_bb,  BB, lut_scale, b_bb_q);
    quantizeQCfCBias(b_ff1, H,  lut_scale, b_ff1_q);
    quantizeQCfCBias(b_ff2, H,  lut_scale, b_ff2_q);
    quantizeQCfCTimeBias(b_a, b_b, H, sc.ts, lut_scale, b_time_q);
    quantizeSymmetricWeights(w_bin, w_bin_q, BB * I, sc.w_backbone_input_scale);
    quantizeSymmetricWeights(w_bh,  w_bh_q,  BB * H, sc.w_backbone_hidden_scale);
    quantizeSymmetricWeights(w_ff1, w_ff1_q, H * BB, sc.w_ff1_scale);
    quantizeSymmetricWeights(w_ff2, w_ff2_q, H * BB, sc.w_ff2_scale);
    quantizeSymmetricWeights(w_ta,  w_ta_q,  H * BB, sc.w_time_a_scale);
    quantizeSymmetricWeights(w_tb,  w_tb_q,  H * BB, sc.w_time_b_scale);
    buildQSigmoidLUT(lut_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_lut);

    gCell.w_backbone_input = w_bin_q; gCell.w_backbone_hidden = w_bh_q; gCell.b_backbone = b_bb_q;
    gCell.w_ff1 = w_ff1_q; gCell.w_ff2 = w_ff2_q; gCell.w_time_a = w_ta_q; gCell.w_time_b = w_tb_q;
    gCell.b_ff1 = b_ff1_q; gCell.b_ff2 = b_ff2_q; gCell.b_time = b_time_q;
    gCell.input_zero_point = 0; gCell.hidden_zero_point = 0;
    gCell.backbone_input_multiplier  = p.backbone_input_multiplier;
    gCell.backbone_input_shift       = p.backbone_input_shift;
    gCell.backbone_hidden_multiplier = p.backbone_hidden_multiplier;
    gCell.backbone_hidden_shift      = p.backbone_hidden_shift;
    gCell.ff1_multiplier = p.ff1_multiplier; gCell.ff1_shift = p.ff1_shift;
    gCell.ff2_multiplier = p.ff2_multiplier; gCell.ff2_shift = p.ff2_shift;
    gCell.time_a_multiplier = p.time_a_multiplier; gCell.time_a_shift = p.time_a_shift;
    gCell.time_b_multiplier = p.time_b_multiplier; gCell.time_b_shift = p.time_b_shift;
    gCell.sigmoid_lut = sigmoid_lut; gCell.tanh_lut = tanh_lut;
    gCell.one_minus_t_times_ff1_multiplier = p.one_minus_t_times_ff1_multiplier;
    gCell.one_minus_t_times_ff1_shift      = p.one_minus_t_times_ff1_shift;
    gCell.t_times_ff2_multiplier = p.t_times_ff2_multiplier;
    gCell.t_times_ff2_shift      = p.t_times_ff2_shift;
    gCell.output_qmin = -127; gCell.output_qmax = 127;
}

void quantizeInput(const float* x, int8_t* q)
{
    tinymind::quantizeBuffer<int8_t>(x, q, I, x_scale, 0, -128, 127);
}

std::size_t weightBytes()
{
    return sizeof(w_bin_q) + sizeof(w_bh_q) + sizeof(w_ff1_q) + sizeof(w_ff2_q) +
           sizeof(w_ta_q)  + sizeof(w_tb_q) + sizeof(b_bb_q)  + sizeof(b_ff1_q) +
           sizeof(b_ff2_q) + sizeof(b_time_q);
}

int runParity()
{
    float  hf[H] = {0};
    int8_t hq[H] = {0};
    float  maxErr = 0.0f;

    // Parity-trajectory CSV: float vs int8 hidden unit 0, plus per-step worst
    // error across all units -- for plot.py.
    std::FILE* csv = std::fopen("qcfc_parity.csv", "w");
    std::fprintf(csv, "step,h0_float,h0_int8,max_abs_err\n");

    for (std::size_t t = 0; t < L; ++t)
    {
        floatCfcStep(seqIn[t], hf);
        int8_t xq[I]; quantizeInput(seqIn[t], xq);
        gCell.forward(xq, hq);
        float stepErr = 0.0f;
        for (std::size_t i = 0; i < H; ++i)
        {
            const float back = tinymind::dequantize<int8_t>(hq[i], h_scale, 0);
            const float e = std::fabs(back - hf[i]);
            stepErr = std::fmax(stepErr, e);
            maxErr = std::fmax(maxErr, e);
        }
        std::fprintf(csv, "%zu,%.6f,%.6f,%.6f\n", t, hf[0],
                     tinymind::dequantize<int8_t>(hq[0], h_scale, 0), stepErr);
    }
    std::fclose(csv);
    std::printf("qcfc_liquid_int8: CfC %zu->%zu (backbone %zu), %zu steps, ts=%.2f\n",
                I, H, BB, L, TS);
    std::printf("  weight bytes        : %zu (+ %zu LUT, shared)\n",
                weightBytes(), sizeof(sigmoid_lut) + sizeof(tanh_lut));
    const float h_range = h_scale * 255.0f;   // int8 hidden dynamic range
    std::printf("  calibrated scales: x=%.5f  h=%.5f  lut_in=%.5f\n",
                x_scale, h_scale, lut_scale);
    std::printf("  worst max-abs error vs float = %.6f  (%.1f%% of the %.4f hidden range)\n",
                maxErr, 100.0f * maxErr / h_range, h_range);
    const bool pass = (maxErr < 0.10f * h_range);   // within 10% of the hidden range
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int runBench()
{
    int8_t hq[H] = {0};
    int8_t xq[L][I];
    for (std::size_t t = 0; t < L; ++t) quantizeInput(seqIn[t], xq[t]);

    tinymind::bench::enableCycleCounter();
    const tinymind::bench::Cycles t0 = tinymind::bench::readCycleCounter();
    for (std::size_t t = 0; t < L; ++t) gCell.forward(xq[t], hq);
    const tinymind::bench::Cycles t1 = tinymind::bench::readCycleCounter();

    tinymind::bench::writeHeader(std::cout);
    const tinymind::bench::Cycles perCall =
        static_cast<tinymind::bench::Cycles>((t1 - t0) / L);
    tinymind::bench::writeRow(std::cout,
        { "qcfc_cell_step", weightBytes(), sizeof(hq), perCall });
    return 0;
}

int runGolden()
{
    int8_t hq[H] = {0};
    for (std::size_t t = 0; t < L; ++t)
    {
        int8_t xq[I]; quantizeInput(seqIn[t], xq);
        gCell.forward(xq, hq);
    }
    for (std::size_t i = 0; i < H; ++i)
        std::printf("%d%s", static_cast<int>(hq[i]), (i + 1 == H) ? "\n" : " ");
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    initData();
    calibrateScales();
    calibrate();
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)  return runBench();
    if (argc > 1 && std::strcmp(argv[1], "--golden") == 0) return runGolden();
    return runParity();
}
