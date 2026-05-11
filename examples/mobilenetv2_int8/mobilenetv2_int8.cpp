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

// Phase 16 exemplar: int8 MobileNetV2-shaped pipeline.
//
// Builds the canonical MNv2 unit (expand 1x1, depthwise 3x3, project 1x1)
// twice — once at stride 1 with a residual skip, once at stride 2 without
// — wired around a 3x3 stride-2 conv stem and a final GAP + dense head:
//
//   input  [16][16][4]
//      |
//   QPad2D pad=1                          -> [18][18][4]
//   QConv2DPerChannel 3x3 stride 2, F=8   -> [8][8][8]      (stem)
//   qrelu
//      |
//   ---- IR block 1 (stride 1, 8 -> 8, expand x4) -----------------
//   QPointwiseConv2D 8 -> 32              -> [8][8][32]     (expand)
//   qrelu
//   QPad2D pad=1                          -> [10][10][32]
//   QDepthwiseConv2D 3x3 stride 1, C=32   -> [8][8][32]
//   qrelu
//   QPointwiseConv2D 32 -> 8              -> [8][8][8]      (project, linear)
//   QAdd skip from stem-relu              -> [8][8][8]
//   ---------------------------------------------------------------
//      |
//   ---- IR block 2 (stride 2, 8 -> 16, expand x4) ----------------
//   QPointwiseConv2D 8 -> 32              -> [8][8][32]
//   qrelu
//   QPad2D pad=1                          -> [10][10][32]
//   QDepthwiseConv2D 3x3 stride 2, C=32   -> [4][4][32]
//   qrelu
//   QPointwiseConv2D 32 -> 16             -> [4][4][16]    (project, linear, no skip)
//   ---------------------------------------------------------------
//      |
//   QGlobalAvgPool2D                      -> [16]
//   QDense 16 -> 4                        -> [4] int8 logits
//
// MaxPool / qrelu / QGlobalAvgPool are pure pass-throughs on the int8
// affine grid (max, clamp, integer-mean), so consecutive layers reuse
// the upstream (scale, zero_point). The projection convolutions are
// kept linear (no qrelu) following MNv2's "linear bottleneck" rule.
//
// Three modes via argv[1]:
//
//   (default)  Per-tensor calibration report + max-abs error vs float.
//   --bench    CSV cycle/byte report (one row per layer).
//   --golden   int8 logit bytes for the bundled test set (stable text).

#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qpad.hpp"
#include "qpool2d.hpp"
#include "qadd.hpp"
#include "qdense.hpp"
#include "qactivations.hpp"
#include "include/qcalibration.hpp"
#include "bench/platform.hpp"
#include "bench/report.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

// ---- Spatial / channel shapes ----------------------------------------------
constexpr std::size_t H_IN = 16;
constexpr std::size_t W_IN = 16;
constexpr std::size_t C_IN = 4;

constexpr std::size_t STEM_F = 8;
constexpr std::size_t STEM_PAD = 1;
constexpr std::size_t STEM_K = 3;
constexpr std::size_t STEM_S = 2;

constexpr std::size_t EXPAND_RATIO = 4;
constexpr std::size_t IR1_OUT_C = STEM_F;            // 8
constexpr std::size_t IR1_EXP_C = STEM_F * EXPAND_RATIO; // 32

constexpr std::size_t IR2_OUT_C = 16;
constexpr std::size_t IR2_EXP_C = STEM_F * EXPAND_RATIO; // expand IR2 from IR1 out = 8 channels
constexpr std::size_t DW_PAD = 1;
constexpr std::size_t DW_K = 3;
constexpr std::size_t IR2_DW_S = 2;

constexpr std::size_t NUM_CLASSES = 4;
constexpr std::size_t NUM_INPUTS  = 4;
constexpr std::size_t NUM_CALIB   = 32;

// ---- Sizes -----------------------------------------------------------------
constexpr std::size_t IN_SIZE       = H_IN * W_IN * C_IN;
constexpr std::size_t STEM_PAD_H    = H_IN + 2 * STEM_PAD;        // 18
constexpr std::size_t STEM_PAD_W    = W_IN + 2 * STEM_PAD;        // 18
constexpr std::size_t STEM_PAD_SIZE = STEM_PAD_H * STEM_PAD_W * C_IN;
constexpr std::size_t STEM_OUT_H    = (STEM_PAD_H - STEM_K) / STEM_S + 1; // 8
constexpr std::size_t STEM_OUT_W    = (STEM_PAD_W - STEM_K) / STEM_S + 1; // 8
constexpr std::size_t STEM_OUT_SIZE = STEM_OUT_H * STEM_OUT_W * STEM_F;

constexpr std::size_t EXP1_SIZE = STEM_OUT_H * STEM_OUT_W * IR1_EXP_C;
constexpr std::size_t DW1_PAD_H = STEM_OUT_H + 2 * DW_PAD;
constexpr std::size_t DW1_PAD_W = STEM_OUT_W + 2 * DW_PAD;
constexpr std::size_t DW1_PAD_SIZE = DW1_PAD_H * DW1_PAD_W * IR1_EXP_C;
constexpr std::size_t DW1_OUT_H = (DW1_PAD_H - DW_K) / 1 + 1;     // 8
constexpr std::size_t DW1_OUT_W = (DW1_PAD_W - DW_K) / 1 + 1;     // 8
constexpr std::size_t DW1_OUT_SIZE = DW1_OUT_H * DW1_OUT_W * IR1_EXP_C;
constexpr std::size_t PROJ1_SIZE = STEM_OUT_H * STEM_OUT_W * IR1_OUT_C;
constexpr std::size_t IR1_ADD_SIZE = PROJ1_SIZE;

constexpr std::size_t EXP2_SIZE    = STEM_OUT_H * STEM_OUT_W * IR2_EXP_C;
constexpr std::size_t DW2_PAD_H    = STEM_OUT_H + 2 * DW_PAD;
constexpr std::size_t DW2_PAD_W    = STEM_OUT_W + 2 * DW_PAD;
constexpr std::size_t DW2_PAD_SIZE = DW2_PAD_H * DW2_PAD_W * IR2_EXP_C;
constexpr std::size_t DW2_OUT_H    = (DW2_PAD_H - DW_K) / IR2_DW_S + 1; // 4
constexpr std::size_t DW2_OUT_W    = (DW2_PAD_W - DW_K) / IR2_DW_S + 1; // 4
constexpr std::size_t DW2_OUT_SIZE = DW2_OUT_H * DW2_OUT_W * IR2_EXP_C;
constexpr std::size_t PROJ2_SIZE   = DW2_OUT_H * DW2_OUT_W * IR2_OUT_C;

// ---- Weight buffer sizes ---------------------------------------------------
constexpr std::size_t W_STEM = STEM_F   * STEM_K * STEM_K * C_IN;
constexpr std::size_t W_EXP1 = IR1_EXP_C * STEM_F;          // 1x1 expand
constexpr std::size_t W_DW1  = IR1_EXP_C * DW_K * DW_K;     // depthwise
constexpr std::size_t W_PRJ1 = IR1_OUT_C * IR1_EXP_C;       // 1x1 project
constexpr std::size_t W_EXP2 = IR2_EXP_C * STEM_F;
constexpr std::size_t W_DW2  = IR2_EXP_C * DW_K * DW_K;
constexpr std::size_t W_PRJ2 = IR2_OUT_C * IR2_EXP_C;
constexpr std::size_t W_DENSE = IR2_OUT_C * NUM_CLASSES;

// ---------------------------------------------------------------------------
// Float reference helpers (NHWC).
// ---------------------------------------------------------------------------

void fpad(const float* in, float* out,
          std::size_t hin, std::size_t win, std::size_t c, std::size_t pad)
{
    const std::size_t hout = hin + 2 * pad;
    const std::size_t wout = win + 2 * pad;
    for (std::size_t oh = 0; oh < hout; ++oh)
    {
        const bool ih_ok = (oh >= pad) && (oh < pad + hin);
        for (std::size_t ow = 0; ow < wout; ++ow)
        {
            const bool iw_ok = (ow >= pad) && (ow < pad + win);
            const std::size_t off_out = (oh * wout + ow) * c;
            if (ih_ok && iw_ok)
            {
                const std::size_t off_in = ((oh - pad) * win + (ow - pad)) * c;
                for (std::size_t ci = 0; ci < c; ++ci) out[off_out + ci] = in[off_in + ci];
            }
            else
            {
                for (std::size_t ci = 0; ci < c; ++ci) out[off_out + ci] = 0.0f;
            }
        }
    }
}

void fconv(const float* in, const float* w, const float* b, float* out,
           std::size_t hin, std::size_t win, std::size_t cin,
           std::size_t kh, std::size_t kw, std::size_t stride, std::size_t f)
{
    const std::size_t hout = (hin - kh) / stride + 1;
    const std::size_t wout = (win - kw) / stride + 1;
    for (std::size_t oh = 0; oh < hout; ++oh)
        for (std::size_t ow = 0; ow < wout; ++ow)
            for (std::size_t fi = 0; fi < f; ++fi)
            {
                float a = b[fi];
                for (std::size_t kh_i = 0; kh_i < kh; ++kh_i)
                    for (std::size_t kw_i = 0; kw_i < kw; ++kw_i)
                        for (std::size_t ci = 0; ci < cin; ++ci)
                        {
                            const float x = in[((oh * stride + kh_i) * win + ow * stride + kw_i) * cin + ci];
                            const float wv = w[((fi * kh + kh_i) * kw + kw_i) * cin + ci];
                            a += wv * x;
                        }
                out[(oh * wout + ow) * f + fi] = a;
            }
}

void fdw(const float* in, const float* w, const float* b, float* out,
         std::size_t hin, std::size_t win, std::size_t c,
         std::size_t kh, std::size_t kw, std::size_t stride)
{
    const std::size_t hout = (hin - kh) / stride + 1;
    const std::size_t wout = (win - kw) / stride + 1;
    for (std::size_t oh = 0; oh < hout; ++oh)
        for (std::size_t ow = 0; ow < wout; ++ow)
            for (std::size_t ci = 0; ci < c; ++ci)
            {
                float a = b[ci];
                for (std::size_t kh_i = 0; kh_i < kh; ++kh_i)
                    for (std::size_t kw_i = 0; kw_i < kw; ++kw_i)
                    {
                        const float x = in[((oh * stride + kh_i) * win + ow * stride + kw_i) * c + ci];
                        const float wv = w[(ci * kh + kh_i) * kw + kw_i];
                        a += wv * x;
                    }
                out[(oh * wout + ow) * c + ci] = a;
            }
}

void fpw(const float* in, const float* w, const float* b, float* out,
         std::size_t hin, std::size_t win, std::size_t cin, std::size_t f)
{
    for (std::size_t oh = 0; oh < hin; ++oh)
        for (std::size_t ow = 0; ow < win; ++ow)
            for (std::size_t fi = 0; fi < f; ++fi)
            {
                float a = b[fi];
                for (std::size_t ci = 0; ci < cin; ++ci)
                {
                    const float x = in[(oh * win + ow) * cin + ci];
                    const float wv = w[fi * cin + ci];
                    a += wv * x;
                }
                out[(oh * win + ow) * f + fi] = a;
            }
}

void frelu(float* x, std::size_t n) { for (std::size_t i = 0; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f; }
void fadd(const float* a, const float* b, float* y, std::size_t n) { for (std::size_t i = 0; i < n; ++i) y[i] = a[i] + b[i]; }

void fgap(const float* in, float* out, std::size_t hin, std::size_t win, std::size_t c)
{
    const float den = 1.0f / static_cast<float>(hin * win);
    for (std::size_t ci = 0; ci < c; ++ci)
    {
        float s = 0.0f;
        for (std::size_t h = 0; h < hin; ++h)
            for (std::size_t w = 0; w < win; ++w)
                s += in[(h * win + w) * c + ci];
        out[ci] = s * den;
    }
}

void fdense(const float* in, const float* w, const float* b, float* out,
            std::size_t in_dim, std::size_t out_dim)
{
    for (std::size_t o = 0; o < out_dim; ++o)
    {
        float a = b[o];
        for (std::size_t i = 0; i < in_dim; ++i)
            a += w[o * in_dim + i] * in[i];
        out[o] = a;
    }
}

float maxAbsDiff(const float* a, const float* b, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

void fillW(float* dst, std::size_t n, float amp, float freq, float phase)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        dst[i] = amp * std::sin(freq * t + phase);
    }
}

} // namespace

int main(int argc, char** argv)
{
    using tinymind::QConv2DPerChannel;
    using tinymind::QDepthwiseConv2D;
    using tinymind::QPointwiseConv2D;
    using tinymind::QPad2D;
    using tinymind::QGlobalAvgPool2D;
    using tinymind::QAdd;
    using tinymind::QDense;
    using tinymind::Requantizer;
    using tinymind::RangeObserver;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::computePerChannelSymmetricScales;
    using tinymind::buildRequantizer;
    using tinymind::buildQAddParams;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;
    using tinymind::qreluBuffer;

    const bool bench_mode  = (argc >= 2) && std::strcmp(argv[1], "--bench")  == 0;
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    // ----- Hand-crafted weights.
    static float w_stem[W_STEM], b_stem[STEM_F]                        = {};
    static float w_exp1[W_EXP1], b_exp1[IR1_EXP_C]                     = {};
    static float w_dw1 [W_DW1],  b_dw1 [IR1_EXP_C]                     = {};
    static float w_prj1[W_PRJ1], b_prj1[IR1_OUT_C]                     = {};
    static float w_exp2[W_EXP2], b_exp2[IR2_EXP_C]                     = {};
    static float w_dw2 [W_DW2],  b_dw2 [IR2_EXP_C]                     = {};
    static float w_prj2[W_PRJ2], b_prj2[IR2_OUT_C]                     = {};
    static float w_dense[W_DENSE], b_dense[NUM_CLASSES]                = {};

    fillW(w_stem,  W_STEM,  0.10f, 5.0f, 0.1f);
    fillW(w_exp1,  W_EXP1,  0.10f, 7.0f, 0.3f);
    fillW(w_dw1,   W_DW1,   0.10f, 9.0f, 0.5f);
    fillW(w_prj1,  W_PRJ1,  0.08f, 4.0f, 0.7f);
    fillW(w_exp2,  W_EXP2,  0.10f, 11.0f, 0.2f);
    fillW(w_dw2,   W_DW2,   0.10f, 13.0f, 0.4f);
    fillW(w_prj2,  W_PRJ2,  0.08f, 6.0f, 0.6f);
    fillW(w_dense, W_DENSE, 0.15f, 3.0f, 0.2f);

    // ----- Synthetic dataset.
    std::vector<std::vector<float>> inputs(NUM_CALIB, std::vector<float>(IN_SIZE));
    for (std::size_t s = 0; s < NUM_CALIB; ++s)
        for (std::size_t i = 0; i < IN_SIZE; ++i)
        {
            const float phase = static_cast<float>(s * 13 + i) * 0.11f;
            inputs[s][i] = 0.7f * std::sin(phase) + 0.3f * std::cos(0.4f * static_cast<float>(s + i));
        }

    // ----- Float forward + range observation.
    RangeObserver o_in, o_stem, o_exp1, o_dw1, o_prj1, o_ir1add;
    RangeObserver o_exp2, o_dw2, o_prj2, o_logits;

    std::vector<std::vector<float>> float_logits(NUM_CALIB, std::vector<float>(NUM_CLASSES));

    static float buf_pad_stem[STEM_PAD_SIZE], buf_stem[STEM_OUT_SIZE];
    static float buf_exp1[EXP1_SIZE], buf_dw1_pad[DW1_PAD_SIZE], buf_dw1[DW1_OUT_SIZE];
    static float buf_prj1[PROJ1_SIZE], buf_ir1[IR1_ADD_SIZE];
    static float buf_exp2[EXP2_SIZE], buf_dw2_pad[DW2_PAD_SIZE], buf_dw2[DW2_OUT_SIZE];
    static float buf_prj2[PROJ2_SIZE], buf_gap[IR2_OUT_C];

    for (std::size_t s = 0; s < NUM_CALIB; ++s)
    {
        fpad(inputs[s].data(), buf_pad_stem, H_IN, W_IN, C_IN, STEM_PAD);
        fconv(buf_pad_stem, w_stem, b_stem, buf_stem,
              STEM_PAD_H, STEM_PAD_W, C_IN, STEM_K, STEM_K, STEM_S, STEM_F);
        frelu(buf_stem, STEM_OUT_SIZE);

        // IR block 1.
        fpw(buf_stem, w_exp1, b_exp1, buf_exp1, STEM_OUT_H, STEM_OUT_W, STEM_F, IR1_EXP_C);
        frelu(buf_exp1, EXP1_SIZE);
        fpad(buf_exp1, buf_dw1_pad, STEM_OUT_H, STEM_OUT_W, IR1_EXP_C, DW_PAD);
        fdw(buf_dw1_pad, w_dw1, b_dw1, buf_dw1,
            DW1_PAD_H, DW1_PAD_W, IR1_EXP_C, DW_K, DW_K, 1);
        frelu(buf_dw1, DW1_OUT_SIZE);
        fpw(buf_dw1, w_prj1, b_prj1, buf_prj1, DW1_OUT_H, DW1_OUT_W, IR1_EXP_C, IR1_OUT_C);
        // Linear bottleneck — no ReLU after project.
        fadd(buf_prj1, buf_stem, buf_ir1, IR1_ADD_SIZE);

        // IR block 2.
        fpw(buf_ir1, w_exp2, b_exp2, buf_exp2, STEM_OUT_H, STEM_OUT_W, STEM_F, IR2_EXP_C);
        frelu(buf_exp2, EXP2_SIZE);
        fpad(buf_exp2, buf_dw2_pad, STEM_OUT_H, STEM_OUT_W, IR2_EXP_C, DW_PAD);
        fdw(buf_dw2_pad, w_dw2, b_dw2, buf_dw2,
            DW2_PAD_H, DW2_PAD_W, IR2_EXP_C, DW_K, DW_K, IR2_DW_S);
        frelu(buf_dw2, DW2_OUT_SIZE);
        fpw(buf_dw2, w_prj2, b_prj2, buf_prj2, DW2_OUT_H, DW2_OUT_W, IR2_EXP_C, IR2_OUT_C);

        fgap(buf_prj2, buf_gap, DW2_OUT_H, DW2_OUT_W, IR2_OUT_C);
        fdense(buf_gap, w_dense, b_dense, float_logits[s].data(), IR2_OUT_C, NUM_CLASSES);

        o_in    .observe(inputs[s].data(), IN_SIZE);
        o_stem  .observe(buf_stem,   STEM_OUT_SIZE);
        o_exp1  .observe(buf_exp1,   EXP1_SIZE);
        o_dw1   .observe(buf_dw1,    DW1_OUT_SIZE);
        o_prj1  .observe(buf_prj1,   PROJ1_SIZE);
        o_ir1add.observe(buf_ir1,    IR1_ADD_SIZE);
        o_exp2  .observe(buf_exp2,   EXP2_SIZE);
        o_dw2   .observe(buf_dw2,    DW2_OUT_SIZE);
        o_prj2  .observe(buf_prj2,   PROJ2_SIZE);
        o_logits.observe(float_logits[s].data(), NUM_CLASSES);
    }

    // ----- Affine params (one per *non-passthrough* tensor).
    const auto p_in    = computeAffineParamsAsymmetric(o_in.min_value,    o_in.max_value,    -128, 127);
    const auto p_stem  = computeAffineParamsAsymmetric(o_stem.min_value,  o_stem.max_value,  -128, 127);
    const auto p_exp1  = computeAffineParamsAsymmetric(o_exp1.min_value,  o_exp1.max_value,  -128, 127);
    const auto p_dw1   = computeAffineParamsAsymmetric(o_dw1.min_value,   o_dw1.max_value,   -128, 127);
    const auto p_prj1  = computeAffineParamsAsymmetric(o_prj1.min_value,  o_prj1.max_value,  -128, 127);
    const auto p_ir1   = computeAffineParamsAsymmetric(o_ir1add.min_value,o_ir1add.max_value,-128, 127);
    const auto p_exp2  = computeAffineParamsAsymmetric(o_exp2.min_value,  o_exp2.max_value,  -128, 127);
    const auto p_dw2   = computeAffineParamsAsymmetric(o_dw2.min_value,   o_dw2.max_value,   -128, 127);
    const auto p_prj2  = computeAffineParamsAsymmetric(o_prj2.min_value,  o_prj2.max_value,  -128, 127);
    const auto& p_gap  = p_prj2; // GAP is a pass-through.
    const auto p_logits= computeAffineParamsAsymmetric(o_logits.min_value,o_logits.max_value,-128, 127);

    // ----- Per-channel weight scales for conv / dw / pw.
    float ws_stem[STEM_F];
    computePerChannelSymmetricScales(w_stem, STEM_F, STEM_K * STEM_K * C_IN, 127, ws_stem);

    float ws_exp1[IR1_EXP_C], ws_dw1[IR1_EXP_C], ws_prj1[IR1_OUT_C];
    computePerChannelSymmetricScales(w_exp1, IR1_EXP_C, STEM_F,    127, ws_exp1);
    computePerChannelSymmetricScales(w_dw1,  IR1_EXP_C, DW_K * DW_K, 127, ws_dw1);
    computePerChannelSymmetricScales(w_prj1, IR1_OUT_C, IR1_EXP_C, 127, ws_prj1);

    float ws_exp2[IR2_EXP_C], ws_dw2[IR2_EXP_C], ws_prj2[IR2_OUT_C];
    computePerChannelSymmetricScales(w_exp2, IR2_EXP_C, STEM_F,    127, ws_exp2);
    computePerChannelSymmetricScales(w_dw2,  IR2_EXP_C, DW_K * DW_K, 127, ws_dw2);
    computePerChannelSymmetricScales(w_prj2, IR2_OUT_C, IR2_EXP_C, 127, ws_prj2);

    // Dense per-tensor symmetric.
    float dense_abs = 0.0f;
    for (std::size_t i = 0; i < W_DENSE; ++i) { const float a = std::fabs(w_dense[i]); if (a > dense_abs) dense_abs = a; }
    const float ws_dense = dense_abs / 127.0f;

    // ----- Quantize weights / biases.
    static int8_t qw_stem[W_STEM], qw_exp1[W_EXP1], qw_dw1[W_DW1], qw_prj1[W_PRJ1];
    static int8_t qw_exp2[W_EXP2], qw_dw2[W_DW2], qw_prj2[W_PRJ2], qw_dense[W_DENSE];
    static int32_t qb_stem[STEM_F]     = {};
    static int32_t qb_exp1[IR1_EXP_C]  = {};
    static int32_t qb_dw1[IR1_EXP_C]   = {};
    static int32_t qb_prj1[IR1_OUT_C]  = {};
    static int32_t qb_exp2[IR2_EXP_C]  = {};
    static int32_t qb_dw2[IR2_EXP_C]   = {};
    static int32_t qb_prj2[IR2_OUT_C]  = {};
    static int32_t qb_dense[NUM_CLASSES] = {};

    auto quantPerChannelW = [](const float* src, int8_t* dst, std::size_t F, std::size_t WPF, const float* ws)
    {
        for (std::size_t f = 0; f < F; ++f)
            quantizeBuffer<int8_t>(&src[f * WPF], &dst[f * WPF], WPF, ws[f], 0, -128, 127);
    };
    quantPerChannelW(w_stem, qw_stem, STEM_F,   STEM_K * STEM_K * C_IN, ws_stem);
    quantPerChannelW(w_exp1, qw_exp1, IR1_EXP_C, STEM_F,                ws_exp1);
    quantPerChannelW(w_dw1,  qw_dw1,  IR1_EXP_C, DW_K * DW_K,           ws_dw1);
    quantPerChannelW(w_prj1, qw_prj1, IR1_OUT_C, IR1_EXP_C,             ws_prj1);
    quantPerChannelW(w_exp2, qw_exp2, IR2_EXP_C, STEM_F,                ws_exp2);
    quantPerChannelW(w_dw2,  qw_dw2,  IR2_EXP_C, DW_K * DW_K,           ws_dw2);
    quantPerChannelW(w_prj2, qw_prj2, IR2_OUT_C, IR2_EXP_C,             ws_prj2);
    quantizeBuffer<int8_t>(w_dense, qw_dense, W_DENSE, ws_dense, 0, -128, 127);

    auto bias_q = [](const float* bf, int32_t* bq, std::size_t F, float in_scale, const float* ws)
    {
        for (std::size_t f = 0; f < F; ++f)
        {
            const double s = static_cast<double>(in_scale) * static_cast<double>(ws[f]);
            bq[f] = static_cast<int32_t>(std::lround(static_cast<double>(bf[f]) / s));
        }
    };
    bias_q(b_stem, qb_stem, STEM_F,   p_in.scale,   ws_stem);
    bias_q(b_exp1, qb_exp1, IR1_EXP_C, p_stem.scale, ws_exp1);
    bias_q(b_dw1,  qb_dw1,  IR1_EXP_C, p_exp1.scale, ws_dw1);
    bias_q(b_prj1, qb_prj1, IR1_OUT_C, p_dw1.scale,  ws_prj1);
    bias_q(b_exp2, qb_exp2, IR2_EXP_C, p_ir1.scale,  ws_exp2);
    bias_q(b_dw2,  qb_dw2,  IR2_EXP_C, p_exp2.scale, ws_dw2);
    bias_q(b_prj2, qb_prj2, IR2_OUT_C, p_dw2.scale,  ws_prj2);
    for (std::size_t o = 0; o < NUM_CLASSES; ++o)
    {
        const double s = static_cast<double>(p_gap.scale) * static_cast<double>(ws_dense);
        qb_dense[o] = static_cast<int32_t>(std::lround(static_cast<double>(b_dense[o]) / s));
    }

    // ----- Layer instances.
    QPad2D<int8_t, H_IN, W_IN, C_IN, STEM_PAD, STEM_PAD, STEM_PAD, STEM_PAD> stem_pad;
    stem_pad.pad_value = static_cast<int8_t>(p_in.zero_point);

    QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t,
                      STEM_PAD_H, STEM_PAD_W, C_IN,
                      STEM_K, STEM_K, STEM_S, STEM_S, STEM_F> stem_conv;
    Requantizer<int32_t, int8_t> rq_stem[STEM_F];
    for (std::size_t f = 0; f < STEM_F; ++f)
        rq_stem[f] = buildRequantizer<int8_t>(p_in.scale, ws_stem[f],
                                              p_stem.scale, p_stem.zero_point, -128, 127);
    stem_conv.weights = qw_stem; stem_conv.biases = qb_stem;
    stem_conv.input_zero_point = static_cast<int8_t>(p_in.zero_point);
    stem_conv.requantizers = rq_stem;

    // IR1 expand: pointwise per-channel.
    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                     STEM_OUT_H, STEM_OUT_W, STEM_F, IR1_EXP_C> ir1_expand;
    // QPointwiseConv2D uses one Requantizer (per-tensor). Use the average
    // weight scale as the per-tensor scale; the rounding error is absorbed
    // by the post-layer ReLU calibration window.
    {
        float avg = 0.0f;
        for (std::size_t f = 0; f < IR1_EXP_C; ++f) avg += ws_exp1[f];
        avg /= static_cast<float>(IR1_EXP_C);
        ir1_expand.weights = qw_exp1; ir1_expand.biases = qb_exp1;
        ir1_expand.input_zero_point = static_cast<int8_t>(p_stem.zero_point);
        ir1_expand.requantizer = buildRequantizer<int8_t>(p_stem.scale, avg,
                                                          p_exp1.scale, p_exp1.zero_point,
                                                          -128, 127);
    }

    QPad2D<int8_t, STEM_OUT_H, STEM_OUT_W, IR1_EXP_C, DW_PAD, DW_PAD, DW_PAD, DW_PAD> ir1_dw_pad;
    ir1_dw_pad.pad_value = static_cast<int8_t>(p_exp1.zero_point);

    QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                     DW1_PAD_H, DW1_PAD_W, IR1_EXP_C, DW_K, DW_K, 1, 1> ir1_dw;
    Requantizer<int32_t, int8_t> rq_ir1_dw[IR1_EXP_C];
    for (std::size_t c = 0; c < IR1_EXP_C; ++c)
        rq_ir1_dw[c] = buildRequantizer<int8_t>(p_exp1.scale, ws_dw1[c],
                                                p_dw1.scale, p_dw1.zero_point, -128, 127);
    ir1_dw.weights = qw_dw1; ir1_dw.biases = qb_dw1;
    ir1_dw.input_zero_point = static_cast<int8_t>(p_exp1.zero_point);
    ir1_dw.requantizers = rq_ir1_dw;

    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                     DW1_OUT_H, DW1_OUT_W, IR1_EXP_C, IR1_OUT_C> ir1_project;
    {
        float avg = 0.0f;
        for (std::size_t f = 0; f < IR1_OUT_C; ++f) avg += ws_prj1[f];
        avg /= static_cast<float>(IR1_OUT_C);
        ir1_project.weights = qw_prj1; ir1_project.biases = qb_prj1;
        ir1_project.input_zero_point = static_cast<int8_t>(p_dw1.zero_point);
        ir1_project.requantizer = buildRequantizer<int8_t>(p_dw1.scale, avg,
                                                           p_prj1.scale, p_prj1.zero_point,
                                                           -128, 127);
    }

    QAdd<int8_t, int8_t, int8_t, IR1_ADD_SIZE> ir1_add;
    {
        const auto addp = buildQAddParams(p_prj1.scale, p_stem.scale, p_ir1.scale);
        ir1_add.input_a_zero_point = static_cast<int8_t>(p_prj1.zero_point);
        ir1_add.input_b_zero_point = static_cast<int8_t>(p_stem.zero_point);
        ir1_add.left_shift = addp.left_shift;
        ir1_add.input_a_multiplier = addp.input_a_multiplier;
        ir1_add.input_a_shift = addp.input_a_shift;
        ir1_add.input_b_multiplier = addp.input_b_multiplier;
        ir1_add.input_b_shift = addp.input_b_shift;
        ir1_add.output_requantizer.multiplier = addp.output_multiplier;
        ir1_add.output_requantizer.shift = addp.output_shift;
        ir1_add.output_requantizer.zero_point = static_cast<int8_t>(p_ir1.zero_point);
        ir1_add.output_requantizer.qmin = -128;
        ir1_add.output_requantizer.qmax = 127;
    }

    // IR2.
    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                     STEM_OUT_H, STEM_OUT_W, STEM_F, IR2_EXP_C> ir2_expand;
    {
        float avg = 0.0f;
        for (std::size_t f = 0; f < IR2_EXP_C; ++f) avg += ws_exp2[f];
        avg /= static_cast<float>(IR2_EXP_C);
        ir2_expand.weights = qw_exp2; ir2_expand.biases = qb_exp2;
        ir2_expand.input_zero_point = static_cast<int8_t>(p_ir1.zero_point);
        ir2_expand.requantizer = buildRequantizer<int8_t>(p_ir1.scale, avg,
                                                          p_exp2.scale, p_exp2.zero_point,
                                                          -128, 127);
    }

    QPad2D<int8_t, STEM_OUT_H, STEM_OUT_W, IR2_EXP_C, DW_PAD, DW_PAD, DW_PAD, DW_PAD> ir2_dw_pad;
    ir2_dw_pad.pad_value = static_cast<int8_t>(p_exp2.zero_point);

    QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                     DW2_PAD_H, DW2_PAD_W, IR2_EXP_C, DW_K, DW_K, IR2_DW_S, IR2_DW_S> ir2_dw;
    Requantizer<int32_t, int8_t> rq_ir2_dw[IR2_EXP_C];
    for (std::size_t c = 0; c < IR2_EXP_C; ++c)
        rq_ir2_dw[c] = buildRequantizer<int8_t>(p_exp2.scale, ws_dw2[c],
                                                p_dw2.scale, p_dw2.zero_point, -128, 127);
    ir2_dw.weights = qw_dw2; ir2_dw.biases = qb_dw2;
    ir2_dw.input_zero_point = static_cast<int8_t>(p_exp2.zero_point);
    ir2_dw.requantizers = rq_ir2_dw;

    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                     DW2_OUT_H, DW2_OUT_W, IR2_EXP_C, IR2_OUT_C> ir2_project;
    {
        float avg = 0.0f;
        for (std::size_t f = 0; f < IR2_OUT_C; ++f) avg += ws_prj2[f];
        avg /= static_cast<float>(IR2_OUT_C);
        ir2_project.weights = qw_prj2; ir2_project.biases = qb_prj2;
        ir2_project.input_zero_point = static_cast<int8_t>(p_dw2.zero_point);
        ir2_project.requantizer = buildRequantizer<int8_t>(p_dw2.scale, avg,
                                                           p_prj2.scale, p_prj2.zero_point,
                                                           -128, 127);
    }

    QGlobalAvgPool2D<int8_t, int32_t, DW2_OUT_H, DW2_OUT_W, IR2_OUT_C> qgap;
    qgap.qmin = -128; qgap.qmax = 127;

    QDense<int8_t, int8_t, int32_t, int8_t, IR2_OUT_C, NUM_CLASSES> qdense;
    qdense.weights = qw_dense; qdense.biases = qb_dense;
    qdense.input_zero_point = static_cast<int8_t>(p_gap.zero_point);
    qdense.requantizer = buildRequantizer<int8_t>(p_gap.scale, ws_dense,
                                                  p_logits.scale, p_logits.zero_point,
                                                  -128, 127);

    // ----- Int8 forward pass.
    static int8_t q_in[IN_SIZE], q_pad_stem[STEM_PAD_SIZE], q_stem[STEM_OUT_SIZE];
    static int8_t q_exp1[EXP1_SIZE], q_dw1_pad[DW1_PAD_SIZE], q_dw1[DW1_OUT_SIZE];
    static int8_t q_prj1[PROJ1_SIZE], q_ir1[IR1_ADD_SIZE];
    static int8_t q_exp2[EXP2_SIZE], q_dw2_pad[DW2_PAD_SIZE], q_dw2[DW2_OUT_SIZE];
    static int8_t q_prj2[PROJ2_SIZE], q_gap[IR2_OUT_C], q_logits[NUM_CLASSES];
    float deq_logits[NUM_CLASSES];

    std::vector<std::array<int8_t, NUM_CLASSES>> all_q_logits(NUM_INPUTS);

    float worst_err = 0.0f;
    tinymind::bench::enableCycleCounter();

    using tinymind::bench::Cycles;
    Cycles c_stem_pad=0, c_stem_conv=0, c_stem_relu=0;
    Cycles c_ir1_exp=0, c_ir1_relu1=0, c_ir1_dwp=0, c_ir1_dw=0, c_ir1_relu2=0, c_ir1_prj=0, c_ir1_add=0;
    Cycles c_ir2_exp=0, c_ir2_relu1=0, c_ir2_dwp=0, c_ir2_dw=0, c_ir2_relu2=0, c_ir2_prj=0;
    Cycles c_gap=0, c_dense=0;

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        quantizeBuffer<int8_t>(inputs[s].data(), q_in, IN_SIZE,
                               p_in.scale, p_in.zero_point, -128, 127);

        Cycles t = tinymind::bench::readCycleCounter();
        stem_pad.forward(q_in, q_pad_stem);
        Cycles t2 = tinymind::bench::readCycleCounter(); c_stem_pad += t2 - t;
        stem_conv.forward(q_pad_stem, q_stem);
        t = tinymind::bench::readCycleCounter();         c_stem_conv += t - t2;
        qreluBuffer<int8_t>(q_stem, STEM_OUT_SIZE, static_cast<int8_t>(p_stem.zero_point));
        t2 = tinymind::bench::readCycleCounter();        c_stem_relu += t2 - t;

        // IR1.
        ir1_expand.forward(q_stem, q_exp1);
        t = tinymind::bench::readCycleCounter();         c_ir1_exp += t - t2;
        qreluBuffer<int8_t>(q_exp1, EXP1_SIZE, static_cast<int8_t>(p_exp1.zero_point));
        t2 = tinymind::bench::readCycleCounter();        c_ir1_relu1 += t2 - t;
        ir1_dw_pad.forward(q_exp1, q_dw1_pad);
        t = tinymind::bench::readCycleCounter();         c_ir1_dwp += t - t2;
        ir1_dw.forward(q_dw1_pad, q_dw1);
        t2 = tinymind::bench::readCycleCounter();        c_ir1_dw += t2 - t;
        qreluBuffer<int8_t>(q_dw1, DW1_OUT_SIZE, static_cast<int8_t>(p_dw1.zero_point));
        t = tinymind::bench::readCycleCounter();         c_ir1_relu2 += t - t2;
        ir1_project.forward(q_dw1, q_prj1);
        t2 = tinymind::bench::readCycleCounter();        c_ir1_prj += t2 - t;
        ir1_add.forward(q_prj1, q_stem, q_ir1);
        t = tinymind::bench::readCycleCounter();         c_ir1_add += t - t2;

        // IR2.
        ir2_expand.forward(q_ir1, q_exp2);
        t2 = tinymind::bench::readCycleCounter();        c_ir2_exp += t2 - t;
        qreluBuffer<int8_t>(q_exp2, EXP2_SIZE, static_cast<int8_t>(p_exp2.zero_point));
        t = tinymind::bench::readCycleCounter();         c_ir2_relu1 += t - t2;
        ir2_dw_pad.forward(q_exp2, q_dw2_pad);
        t2 = tinymind::bench::readCycleCounter();        c_ir2_dwp += t2 - t;
        ir2_dw.forward(q_dw2_pad, q_dw2);
        t = tinymind::bench::readCycleCounter();         c_ir2_dw += t - t2;
        qreluBuffer<int8_t>(q_dw2, DW2_OUT_SIZE, static_cast<int8_t>(p_dw2.zero_point));
        t2 = tinymind::bench::readCycleCounter();        c_ir2_relu2 += t2 - t;
        ir2_project.forward(q_dw2, q_prj2);
        t = tinymind::bench::readCycleCounter();         c_ir2_prj += t - t2;

        qgap.forward(q_prj2, q_gap);
        t2 = tinymind::bench::readCycleCounter();        c_gap += t2 - t;
        qdense.forward(q_gap, q_logits);
        t = tinymind::bench::readCycleCounter();         c_dense += t - t2;

        for (std::size_t i = 0; i < NUM_CLASSES; ++i) all_q_logits[s][i] = q_logits[i];

        dequantizeBuffer<int8_t>(q_logits, deq_logits, NUM_CLASSES,
                                 p_logits.scale, p_logits.zero_point);
        const float err = maxAbsDiff(deq_logits, float_logits[s].data(), NUM_CLASSES);
        if (err > worst_err) worst_err = err;
    }

    if (golden_mode)
    {
        std::printf("# mobilenetv2_int8 golden output\n");
        std::printf("# samples=%zu classes=%zu\n", NUM_INPUTS, NUM_CLASSES);
        for (std::size_t s = 0; s < NUM_INPUTS; ++s)
        {
            std::printf("sample %zu:", s);
            for (std::size_t i = 0; i < NUM_CLASSES; ++i)
                std::printf(" %d", static_cast<int>(all_q_logits[s][i]));
            std::printf("\n");
        }
        return 0;
    }

    if (bench_mode)
    {
        tinymind::bench::writeHeader(std::cout);
        const std::size_t stemBytes = sizeof(stem_conv) + sizeof(qw_stem) + sizeof(qb_stem) + sizeof(rq_stem);
        const std::size_t exp1Bytes = sizeof(ir1_expand) + sizeof(qw_exp1) + sizeof(qb_exp1);
        const std::size_t dw1Bytes  = sizeof(ir1_dw) + sizeof(qw_dw1) + sizeof(qb_dw1) + sizeof(rq_ir1_dw);
        const std::size_t prj1Bytes = sizeof(ir1_project) + sizeof(qw_prj1) + sizeof(qb_prj1);
        const std::size_t exp2Bytes = sizeof(ir2_expand) + sizeof(qw_exp2) + sizeof(qb_exp2);
        const std::size_t dw2Bytes  = sizeof(ir2_dw) + sizeof(qw_dw2) + sizeof(qb_dw2) + sizeof(rq_ir2_dw);
        const std::size_t prj2Bytes = sizeof(ir2_project) + sizeof(qw_prj2) + sizeof(qb_prj2);
        const std::size_t denseBytes = sizeof(qdense) + sizeof(qw_dense) + sizeof(qb_dense);
        tinymind::bench::writeRow(std::cout, {"stem_pad",       sizeof(stem_pad),    STEM_PAD_SIZE,  c_stem_pad});
        tinymind::bench::writeRow(std::cout, {"stem_conv3x3_s2", stemBytes,          STEM_OUT_SIZE,  c_stem_conv});
        tinymind::bench::writeRow(std::cout, {"stem_relu",      0,                   STEM_OUT_SIZE,  c_stem_relu});
        tinymind::bench::writeRow(std::cout, {"ir1_expand_pw",  exp1Bytes,           EXP1_SIZE,      c_ir1_exp});
        tinymind::bench::writeRow(std::cout, {"ir1_expand_relu", 0,                  EXP1_SIZE,      c_ir1_relu1});
        tinymind::bench::writeRow(std::cout, {"ir1_dw_pad",     sizeof(ir1_dw_pad),  DW1_PAD_SIZE,   c_ir1_dwp});
        tinymind::bench::writeRow(std::cout, {"ir1_dw3x3",      dw1Bytes,            DW1_OUT_SIZE,   c_ir1_dw});
        tinymind::bench::writeRow(std::cout, {"ir1_dw_relu",    0,                   DW1_OUT_SIZE,   c_ir1_relu2});
        tinymind::bench::writeRow(std::cout, {"ir1_project_pw", prj1Bytes,           PROJ1_SIZE,     c_ir1_prj});
        tinymind::bench::writeRow(std::cout, {"ir1_add",        sizeof(ir1_add),     IR1_ADD_SIZE,   c_ir1_add});
        tinymind::bench::writeRow(std::cout, {"ir2_expand_pw",  exp2Bytes,           EXP2_SIZE,      c_ir2_exp});
        tinymind::bench::writeRow(std::cout, {"ir2_expand_relu", 0,                  EXP2_SIZE,      c_ir2_relu1});
        tinymind::bench::writeRow(std::cout, {"ir2_dw_pad",     sizeof(ir2_dw_pad),  DW2_PAD_SIZE,   c_ir2_dwp});
        tinymind::bench::writeRow(std::cout, {"ir2_dw3x3_s2",   dw2Bytes,            DW2_OUT_SIZE,   c_ir2_dw});
        tinymind::bench::writeRow(std::cout, {"ir2_dw_relu",    0,                   DW2_OUT_SIZE,   c_ir2_relu2});
        tinymind::bench::writeRow(std::cout, {"ir2_project_pw", prj2Bytes,           PROJ2_SIZE,     c_ir2_prj});
        tinymind::bench::writeRow(std::cout, {"global_avgpool", sizeof(qgap),        IR2_OUT_C,      c_gap});
        tinymind::bench::writeRow(std::cout, {"dense_16x4",     denseBytes,          NUM_CLASSES,    c_dense});
        return 0;
    }

    const float lrange = o_logits.max_value - o_logits.min_value;
    std::printf("MobileNetV2 int8 vs float reference\n");
    std::printf("  input  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_in.min_value,    o_in.max_value,    p_in.scale,    p_in.zero_point);
    std::printf("  stem   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_stem.min_value,  o_stem.max_value,  p_stem.scale,  p_stem.zero_point);
    std::printf("  IR1.exp range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_exp1.min_value,  o_exp1.max_value,  p_exp1.scale,  p_exp1.zero_point);
    std::printf("  IR1.dw  range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_dw1.min_value,   o_dw1.max_value,   p_dw1.scale,   p_dw1.zero_point);
    std::printf("  IR1.prj range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_prj1.min_value,  o_prj1.max_value,  p_prj1.scale,  p_prj1.zero_point);
    std::printf("  IR1.add range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_ir1add.min_value,o_ir1add.max_value,p_ir1.scale,   p_ir1.zero_point);
    std::printf("  IR2.exp range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_exp2.min_value,  o_exp2.max_value,  p_exp2.scale,  p_exp2.zero_point);
    std::printf("  IR2.dw  range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_dw2.min_value,   o_dw2.max_value,   p_dw2.scale,   p_dw2.zero_point);
    std::printf("  IR2.prj range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_prj2.min_value,  o_prj2.max_value,  p_prj2.scale,  p_prj2.zero_point);
    std::printf("  logits  range:[%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_logits.min_value,o_logits.max_value,p_logits.scale,p_logits.zero_point);
    std::printf("  worst logits max-abs err: %.5f  (%.1f%% of logits range)\n",
                worst_err, 100.0f * worst_err / (lrange + 1e-6f));

    const float tol = 0.50f * lrange;
    if (worst_err > tol)
    {
        std::printf("FAIL: error %.5f > tolerance %.5f\n", worst_err, tol);
        return 1;
    }
    std::printf("PASS (tolerance %.5f, 50%% of range)\n", tol);
    return 0;
}
