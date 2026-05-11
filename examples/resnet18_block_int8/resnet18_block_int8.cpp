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

// Phase 16 exemplar: int8 ResNet-18-shaped stem + 1 stage.
//
// Pipeline (NHWC):
//
//   input  [16][16][3]
//      |
//   QPad2D pad=3                          -> [22][22][3]
//   QConv2DPerChannel 7x7 stride 2, F=8   -> [8][8][8]
//   qrelu in conv1 domain
//   QMaxPool2D 2x2 stride 2               -> [4][4][8]
//      |
//   ---- basic block (channels=8, SAME padding) ---------------------
//   QPad2D pad=1                          -> [6][6][8]
//   QConv2DPerChannel 3x3 stride 1, F=8   -> [4][4][8]
//   qrelu
//   QPad2D pad=1                          -> [6][6][8]
//   QConv2DPerChannel 3x3 stride 1, F=8   -> [4][4][8]
//   QAdd skip from post-pool path
//   qrelu                                 -> [4][4][8]
//   -----------------------------------------------------------------
//      |
//   QGlobalAvgPool2D                      -> [8]
//   QDense 8 -> NUM_CLASSES (4)           -> [4]   int8 logits
//
// Exercises Phase 10 ops (QPad2D, QConv2DPerChannel, QAdd) plus the
// existing dense/global-avg-pool primitives, at a depth and dimension
// that resembles a real ResNet stem + first stage. The driver runs the
// same pipeline in float as a reference, calibrates each activation
// tensor on a small synthetic dataset, builds the int8 layers, runs
// them on a fixed 4-sample test set, and verifies parity within a
// loose deployment tolerance.
//
// Three modes via argv[1]:
//
//   (default)  Print calibration report + max-abs error vs float; exit 0 on PASS.
//   --bench    Emit CSV cycle/byte report (one row per layer) to stdout.
//   --golden   Emit the int8 output bytes for the bundled test set to
//              stdout in a stable text format; consumed by
//              unit_test/integration to lock the Phase 16 regression.

#include "qaffine.hpp"
#include "qconv2d.hpp"
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

constexpr std::size_t H = 16;
constexpr std::size_t W = 16;
constexpr std::size_t Cin = 3;

constexpr std::size_t STEM_PAD = 3;
constexpr std::size_t STEM_KH = 7;
constexpr std::size_t STEM_KW = 7;
constexpr std::size_t STEM_STRIDE = 2;
constexpr std::size_t F_STEM = 8;

constexpr std::size_t POOL_K = 2;
constexpr std::size_t POOL_S = 2;

constexpr std::size_t BLOCK_PAD = 1;
constexpr std::size_t BLOCK_KH = 3;
constexpr std::size_t BLOCK_KW = 3;
constexpr std::size_t F_BLOCK = F_STEM;

constexpr std::size_t NUM_CLASSES = 4;
constexpr std::size_t NUM_INPUTS = 4;    // golden test set
constexpr std::size_t NUM_CALIB  = 32;   // calibration sweep is wider

constexpr std::size_t IN_SIZE = H * W * Cin;
constexpr std::size_t STEM_PAD_H = H + 2 * STEM_PAD;
constexpr std::size_t STEM_PAD_W = W + 2 * STEM_PAD;
constexpr std::size_t STEM_PAD_SIZE = STEM_PAD_H * STEM_PAD_W * Cin;
constexpr std::size_t STEM_OUT_H = (STEM_PAD_H - STEM_KH) / STEM_STRIDE + 1; // 8
constexpr std::size_t STEM_OUT_W = (STEM_PAD_W - STEM_KW) / STEM_STRIDE + 1; // 8
constexpr std::size_t STEM_OUT_SIZE = STEM_OUT_H * STEM_OUT_W * F_STEM;

constexpr std::size_t POOL_OUT_H = (STEM_OUT_H - POOL_K) / POOL_S + 1; // 4
constexpr std::size_t POOL_OUT_W = (STEM_OUT_W - POOL_K) / POOL_S + 1; // 4
constexpr std::size_t POOL_OUT_SIZE = POOL_OUT_H * POOL_OUT_W * F_BLOCK;

constexpr std::size_t BLOCK_PAD_H = POOL_OUT_H + 2 * BLOCK_PAD; // 6
constexpr std::size_t BLOCK_PAD_W = POOL_OUT_W + 2 * BLOCK_PAD; // 6
constexpr std::size_t BLOCK_PAD_SIZE = BLOCK_PAD_H * BLOCK_PAD_W * F_BLOCK;
constexpr std::size_t BLOCK_OUT_H = POOL_OUT_H; // 4
constexpr std::size_t BLOCK_OUT_W = POOL_OUT_W; // 4
constexpr std::size_t BLOCK_OUT_SIZE = POOL_OUT_SIZE;

constexpr std::size_t W_STEM = F_STEM * STEM_KH * STEM_KW * Cin;
constexpr std::size_t W_BLOCK = F_BLOCK * BLOCK_KH * BLOCK_KW * F_BLOCK;

// ---------------------------------------------------------------------------
// Float reference helpers (NHWC).
// ---------------------------------------------------------------------------

void floatPad2D(const float* in, float* out,
                std::size_t hin, std::size_t win, std::size_t c,
                std::size_t pad)
{
    const std::size_t hout = hin + 2 * pad;
    const std::size_t wout = win + 2 * pad;
    for (std::size_t oh = 0; oh < hout; ++oh)
    {
        const bool inside_h = (oh >= pad) && (oh < pad + hin);
        for (std::size_t ow = 0; ow < wout; ++ow)
        {
            const bool inside_w = (ow >= pad) && (ow < pad + win);
            const std::size_t out_off = (oh * wout + ow) * c;
            if (inside_h && inside_w)
            {
                const std::size_t ih = oh - pad;
                const std::size_t iw = ow - pad;
                const std::size_t in_off = (ih * win + iw) * c;
                for (std::size_t ci = 0; ci < c; ++ci) out[out_off + ci] = in[in_off + ci];
            }
            else
            {
                for (std::size_t ci = 0; ci < c; ++ci) out[out_off + ci] = 0.0f;
            }
        }
    }
}

void floatConv2D(const float* in, const float* w, const float* b, float* out,
                 std::size_t hin, std::size_t win, std::size_t cin,
                 std::size_t kh, std::size_t kw, std::size_t stride,
                 std::size_t f)
{
    const std::size_t hout = (hin - kh) / stride + 1;
    const std::size_t wout = (win - kw) / stride + 1;
    for (std::size_t oh = 0; oh < hout; ++oh)
    {
        for (std::size_t ow = 0; ow < wout; ++ow)
        {
            for (std::size_t fi = 0; fi < f; ++fi)
            {
                float acc = b[fi];
                for (std::size_t ki = 0; ki < kh; ++ki)
                {
                    for (std::size_t kj = 0; kj < kw; ++kj)
                    {
                        for (std::size_t ci = 0; ci < cin; ++ci)
                        {
                            const std::size_t ih = oh * stride + ki;
                            const std::size_t iw = ow * stride + kj;
                            const float x = in[(ih * win + iw) * cin + ci];
                            const float wv = w[((fi * kh + ki) * kw + kj) * cin + ci];
                            acc += wv * x;
                        }
                    }
                }
                out[(oh * wout + ow) * f + fi] = acc;
            }
        }
    }
}

void floatRelu(float* x, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

void floatAdd(const float* a, const float* b, float* y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) y[i] = a[i] + b[i];
}

void floatMaxPool2D(const float* in, float* out,
                    std::size_t hin, std::size_t win, std::size_t c,
                    std::size_t k, std::size_t stride)
{
    const std::size_t hout = (hin - k) / stride + 1;
    const std::size_t wout = (win - k) / stride + 1;
    for (std::size_t oh = 0; oh < hout; ++oh)
    {
        for (std::size_t ow = 0; ow < wout; ++ow)
        {
            for (std::size_t ci = 0; ci < c; ++ci)
            {
                float m = in[(oh * stride * win + ow * stride) * c + ci];
                for (std::size_t ki = 0; ki < k; ++ki)
                {
                    for (std::size_t kj = 0; kj < k; ++kj)
                    {
                        const float v = in[((oh * stride + ki) * win + ow * stride + kj) * c + ci];
                        if (v > m) m = v;
                    }
                }
                out[(oh * wout + ow) * c + ci] = m;
            }
        }
    }
}

void floatGlobalAvgPool(const float* in, float* out,
                        std::size_t hin, std::size_t win, std::size_t c)
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

void floatDense(const float* in, const float* w, const float* b, float* out,
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

void floatPipeline(const float* input,
                   const float* w_stem, const float* b_stem,
                   const float* w_b1,   const float* b_b1,
                   const float* w_b2,   const float* b_b2,
                   const float* w_dense, const float* b_dense,
                   float* logits,
                   float* sp, float* sc, float* sr, float* pool,
                   float* bp1, float* bc1, float* br1,
                   float* bp2, float* bc2, float* badd,
                   float* gap)
{
    floatPad2D(input, sp, H, W, Cin, STEM_PAD);
    floatConv2D(sp, w_stem, b_stem, sc,
                STEM_PAD_H, STEM_PAD_W, Cin,
                STEM_KH, STEM_KW, STEM_STRIDE, F_STEM);
    std::memcpy(sr, sc, STEM_OUT_SIZE * sizeof(float));
    floatRelu(sr, STEM_OUT_SIZE);
    floatMaxPool2D(sr, pool, STEM_OUT_H, STEM_OUT_W, F_STEM, POOL_K, POOL_S);

    floatPad2D(pool, bp1, POOL_OUT_H, POOL_OUT_W, F_BLOCK, BLOCK_PAD);
    floatConv2D(bp1, w_b1, b_b1, bc1,
                BLOCK_PAD_H, BLOCK_PAD_W, F_BLOCK,
                BLOCK_KH, BLOCK_KW, 1, F_BLOCK);
    std::memcpy(br1, bc1, BLOCK_OUT_SIZE * sizeof(float));
    floatRelu(br1, BLOCK_OUT_SIZE);
    floatPad2D(br1, bp2, BLOCK_OUT_H, BLOCK_OUT_W, F_BLOCK, BLOCK_PAD);
    floatConv2D(bp2, w_b2, b_b2, bc2,
                BLOCK_PAD_H, BLOCK_PAD_W, F_BLOCK,
                BLOCK_KH, BLOCK_KW, 1, F_BLOCK);
    floatAdd(bc2, pool, badd, BLOCK_OUT_SIZE);
    floatRelu(badd, BLOCK_OUT_SIZE);

    floatGlobalAvgPool(badd, gap, BLOCK_OUT_H, BLOCK_OUT_W, F_BLOCK);
    floatDense(gap, w_dense, b_dense, logits, F_BLOCK, NUM_CLASSES);
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

void fillWeights(float* dst, std::size_t n, float amp, float freq, float phase)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        dst[i] = amp * std::sin(freq * t + phase);
    }
}

void fillInputs(std::vector<std::vector<float>>& xs)
{
    for (std::size_t s = 0; s < xs.size(); ++s)
    {
        for (std::size_t i = 0; i < xs[s].size(); ++i)
        {
            const float phase = static_cast<float>(s * 17 + i) * 0.13f;
            xs[s][i] = 0.7f * std::sin(phase) +
                       0.3f * std::cos(0.4f * static_cast<float>(s + i));
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    using tinymind::QConv2DPerChannel;
    using tinymind::QPad2D;
    using tinymind::QMaxPool2D;
    using tinymind::QAdd;
    using tinymind::QGlobalAvgPool2D;
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
    const bool quiet = bench_mode || golden_mode;

    // ----- Hand-crafted weights so the example is deterministic.
    float w_stem[W_STEM], b_stem[F_STEM];
    float w_b1[W_BLOCK], b_b1[F_BLOCK];
    float w_b2[W_BLOCK], b_b2[F_BLOCK];
    float w_dense[F_BLOCK * NUM_CLASSES], b_dense[NUM_CLASSES];

    // Amplitudes deliberately tame so intermediate activations stay inside
    // a sensible int8 dynamic range without QAT — Phase 16 is a deployment
    // exemplar, not a trained model.
    fillWeights(w_stem,  W_STEM,             0.10f, 5.0f, 0.1f);
    fillWeights(w_b1,    W_BLOCK,            0.06f, 7.0f, 0.5f);
    fillWeights(w_b2,    W_BLOCK,            0.06f, 9.0f, 1.1f);
    fillWeights(w_dense, F_BLOCK * NUM_CLASSES, 0.15f, 3.0f, 0.2f);

    for (std::size_t i = 0; i < F_STEM;     ++i) b_stem[i]  = 0.0f;
    for (std::size_t i = 0; i < F_BLOCK;    ++i) b_b1[i]    = 0.0f;
    for (std::size_t i = 0; i < F_BLOCK;    ++i) b_b2[i]    = 0.0f;
    for (std::size_t i = 0; i < NUM_CLASSES;++i) b_dense[i] = 0.0f;

    // ----- Synthetic dataset. The first NUM_INPUTS samples are the golden
    // test set; the full NUM_CALIB sweep drives calibration so observers
    // see a representative spread.
    std::vector<std::vector<float>> inputs(NUM_CALIB, std::vector<float>(IN_SIZE));
    fillInputs(inputs);

    // ----- Float forward + range observation.
    // QMaxPool2D and qreluBuffer are pure pass-throughs on the int8 affine
    // grid (max and clamp don't rescale), so the post-pool grid is just
    // the post-stem grid and the post-relu grid is the conv grid. We
    // observe them in float for reporting but don't compute distinct
    // affine params for them.
    RangeObserver obs_in, obs_stem, obs_relu_stem, obs_pool;
    RangeObserver obs_bc1, obs_br1, obs_bc2, obs_badd, obs_gap, obs_logits;

    std::vector<std::vector<float>> float_logits(NUM_CALIB, std::vector<float>(NUM_CLASSES));
    std::vector<float> sp(STEM_PAD_SIZE), sc(STEM_OUT_SIZE), sr(STEM_OUT_SIZE);
    std::vector<float> pool(POOL_OUT_SIZE);
    std::vector<float> bp1(BLOCK_PAD_SIZE), bc1(BLOCK_OUT_SIZE), br1(BLOCK_OUT_SIZE);
    std::vector<float> bp2(BLOCK_PAD_SIZE), bc2(BLOCK_OUT_SIZE), badd(BLOCK_OUT_SIZE);
    std::vector<float> gap(F_BLOCK);

    for (std::size_t s = 0; s < NUM_CALIB; ++s)
    {
        floatPipeline(inputs[s].data(),
                      w_stem, b_stem, w_b1, b_b1, w_b2, b_b2,
                      w_dense, b_dense, float_logits[s].data(),
                      sp.data(), sc.data(), sr.data(), pool.data(),
                      bp1.data(), bc1.data(), br1.data(),
                      bp2.data(), bc2.data(), badd.data(), gap.data());

        obs_in       .observe(inputs[s].data(), IN_SIZE);
        obs_stem     .observe(sc.data(),   STEM_OUT_SIZE);
        obs_relu_stem.observe(sr.data(),   STEM_OUT_SIZE);
        obs_pool     .observe(pool.data(), POOL_OUT_SIZE);
        obs_bc1      .observe(bc1.data(),  BLOCK_OUT_SIZE);
        obs_br1      .observe(br1.data(),  BLOCK_OUT_SIZE);
        obs_bc2      .observe(bc2.data(),  BLOCK_OUT_SIZE);
        obs_badd     .observe(badd.data(), BLOCK_OUT_SIZE);
        obs_gap      .observe(gap.data(),  F_BLOCK);
        obs_logits   .observe(float_logits[s].data(), NUM_CLASSES);
    }

    // ----- Affine params.
    const auto p_in       = computeAffineParamsAsymmetric(obs_in.min_value,       obs_in.max_value,       -128, 127);
    const auto p_stem     = computeAffineParamsAsymmetric(obs_stem.min_value,     obs_stem.max_value,     -128, 127);
    const auto p_bc1      = computeAffineParamsAsymmetric(obs_bc1.min_value,      obs_bc1.max_value,      -128, 127);
    const auto p_bc2      = computeAffineParamsAsymmetric(obs_bc2.min_value,      obs_bc2.max_value,      -128, 127);
    const auto p_badd     = computeAffineParamsAsymmetric(obs_badd.min_value,     obs_badd.max_value,     -128, 127);
    // QGlobalAvgPool2D is a pass-through on (scale, zero_point) — gap output sits on p_badd's grid.
    const auto& p_gap = p_badd;
    const auto p_logits   = computeAffineParamsAsymmetric(obs_logits.min_value,   obs_logits.max_value,   -128, 127);
    (void)obs_relu_stem; // kept for diagnostics only
    (void)obs_pool;
    (void)obs_br1;
    (void)obs_gap;

    // ----- Per-channel weight scales for the three convs.
    float ws_stem[F_STEM];
    float ws_b1[F_BLOCK], ws_b2[F_BLOCK];
    computePerChannelSymmetricScales(w_stem, F_STEM,  STEM_KH * STEM_KW * Cin,    127, ws_stem);
    computePerChannelSymmetricScales(w_b1,   F_BLOCK, BLOCK_KH * BLOCK_KW * F_BLOCK, 127, ws_b1);
    computePerChannelSymmetricScales(w_b2,   F_BLOCK, BLOCK_KH * BLOCK_KW * F_BLOCK, 127, ws_b2);

    // Dense weight scale (per-tensor symmetric).
    float w_dense_abs = 0.0f;
    for (std::size_t i = 0; i < F_BLOCK * NUM_CLASSES; ++i)
    {
        const float a = std::fabs(w_dense[i]);
        if (a > w_dense_abs) w_dense_abs = a;
    }
    const float ws_dense = w_dense_abs / 127.0f;

    // ----- Quantize weights / biases.
    int8_t qw_stem[W_STEM], qw_b1[W_BLOCK], qw_b2[W_BLOCK];
    int8_t qw_dense[F_BLOCK * NUM_CLASSES];
    int32_t qb_stem[F_STEM] = {0};
    int32_t qb_b1[F_BLOCK] = {0};
    int32_t qb_b2[F_BLOCK] = {0};
    int32_t qb_dense[NUM_CLASSES] = {0};

    for (std::size_t f = 0; f < F_STEM; ++f)
    {
        const std::size_t off = f * STEM_KH * STEM_KW * Cin;
        quantizeBuffer<int8_t>(&w_stem[off], &qw_stem[off],
                               STEM_KH * STEM_KW * Cin, ws_stem[f], 0, -128, 127);
        const float s = p_in.scale * ws_stem[f];
        qb_stem[f] = static_cast<int32_t>(std::lround(
            static_cast<double>(b_stem[f]) / static_cast<double>(s)));
    }
    for (std::size_t f = 0; f < F_BLOCK; ++f)
    {
        const std::size_t off = f * BLOCK_KH * BLOCK_KW * F_BLOCK;
        quantizeBuffer<int8_t>(&w_b1[off], &qw_b1[off],
                               BLOCK_KH * BLOCK_KW * F_BLOCK, ws_b1[f], 0, -128, 127);
        // block_conv1 reads pool output, which is the post-stem int8 grid.
        const float s = p_stem.scale * ws_b1[f];
        qb_b1[f] = static_cast<int32_t>(std::lround(
            static_cast<double>(b_b1[f]) / static_cast<double>(s)));

        quantizeBuffer<int8_t>(&w_b2[off], &qw_b2[off],
                               BLOCK_KH * BLOCK_KW * F_BLOCK, ws_b2[f], 0, -128, 127);
        // block_conv2 reads the post-bconv1 ReLU output, which is on the p_bc1 grid.
        const float s2 = p_bc1.scale * ws_b2[f];
        qb_b2[f] = static_cast<int32_t>(std::lround(
            static_cast<double>(b_b2[f]) / static_cast<double>(s2)));
    }

    quantizeBuffer<int8_t>(w_dense, qw_dense,
                           F_BLOCK * NUM_CLASSES, ws_dense, 0, -128, 127);
    for (std::size_t o = 0; o < NUM_CLASSES; ++o)
    {
        const float s = p_gap.scale * ws_dense;
        qb_dense[o] = static_cast<int32_t>(std::lround(
            static_cast<double>(b_dense[o]) / static_cast<double>(s)));
    }

    // ----- Build layer instances.
    QPad2D<int8_t, H, W, Cin, STEM_PAD, STEM_PAD, STEM_PAD, STEM_PAD> stem_pad;
    stem_pad.pad_value = static_cast<int8_t>(p_in.zero_point);

    QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t,
                      STEM_PAD_H, STEM_PAD_W, Cin,
                      STEM_KH, STEM_KW, STEM_STRIDE, STEM_STRIDE, F_STEM> stem_conv;
    Requantizer<int32_t, int8_t> rq_stem[F_STEM];
    for (std::size_t f = 0; f < F_STEM; ++f)
    {
        rq_stem[f] = buildRequantizer<int8_t>(p_in.scale, ws_stem[f],
                                              p_stem.scale, p_stem.zero_point,
                                              -128, 127);
    }
    stem_conv.weights = qw_stem;
    stem_conv.biases  = qb_stem;
    stem_conv.input_zero_point = static_cast<int8_t>(p_in.zero_point);
    stem_conv.requantizers = rq_stem;

    QMaxPool2D<int8_t, STEM_OUT_H, STEM_OUT_W, F_STEM, POOL_K, POOL_K, POOL_S, POOL_S> stem_pool;

    QPad2D<int8_t, POOL_OUT_H, POOL_OUT_W, F_BLOCK, BLOCK_PAD, BLOCK_PAD, BLOCK_PAD, BLOCK_PAD>
        block_pad1, block_pad2;
    // Pool/relu are pass-throughs — pad value matches the upstream affine grid.
    block_pad1.pad_value = static_cast<int8_t>(p_stem.zero_point);
    block_pad2.pad_value = static_cast<int8_t>(p_bc1.zero_point);

    QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t,
                      BLOCK_PAD_H, BLOCK_PAD_W, F_BLOCK,
                      BLOCK_KH, BLOCK_KW, 1, 1, F_BLOCK> block_conv1, block_conv2;
    Requantizer<int32_t, int8_t> rq_b1[F_BLOCK], rq_b2[F_BLOCK];
    for (std::size_t f = 0; f < F_BLOCK; ++f)
    {
        rq_b1[f] = buildRequantizer<int8_t>(p_stem.scale, ws_b1[f],
                                            p_bc1.scale, p_bc1.zero_point,
                                            -128, 127);
        rq_b2[f] = buildRequantizer<int8_t>(p_bc1.scale, ws_b2[f],
                                            p_bc2.scale, p_bc2.zero_point,
                                            -128, 127);
    }
    block_conv1.weights = qw_b1; block_conv1.biases = qb_b1;
    block_conv1.input_zero_point = static_cast<int8_t>(p_stem.zero_point);
    block_conv1.requantizers = rq_b1;
    block_conv2.weights = qw_b2; block_conv2.biases = qb_b2;
    block_conv2.input_zero_point = static_cast<int8_t>(p_bc1.zero_point);
    block_conv2.requantizers = rq_b2;

    QAdd<int8_t, int8_t, int8_t, BLOCK_OUT_SIZE> block_add;
    {
        const auto addp = buildQAddParams(p_bc2.scale, p_stem.scale, p_badd.scale);
        block_add.input_a_zero_point = static_cast<int8_t>(p_bc2.zero_point);
        block_add.input_b_zero_point = static_cast<int8_t>(p_stem.zero_point);
        block_add.left_shift = addp.left_shift;
        block_add.input_a_multiplier = addp.input_a_multiplier;
        block_add.input_a_shift = addp.input_a_shift;
        block_add.input_b_multiplier = addp.input_b_multiplier;
        block_add.input_b_shift = addp.input_b_shift;
        block_add.output_requantizer.multiplier = addp.output_multiplier;
        block_add.output_requantizer.shift = addp.output_shift;
        block_add.output_requantizer.zero_point = static_cast<int8_t>(p_badd.zero_point);
        block_add.output_requantizer.qmin = -128;
        block_add.output_requantizer.qmax = 127;
    }

    QGlobalAvgPool2D<int8_t, int32_t, BLOCK_OUT_H, BLOCK_OUT_W, F_BLOCK> qgap;
    qgap.qmin = -128;
    qgap.qmax = 127;

    QDense<int8_t, int8_t, int32_t, int8_t, F_BLOCK, NUM_CLASSES> qdense;
    qdense.weights = qw_dense;
    qdense.biases  = qb_dense;
    qdense.input_zero_point = static_cast<int8_t>(p_gap.zero_point);
    qdense.requantizer = buildRequantizer<int8_t>(p_gap.scale, ws_dense,
                                                  p_logits.scale, p_logits.zero_point,
                                                  -128, 127);

    // ----- int8 forward pass.
    int8_t q_in[IN_SIZE], q_sp[STEM_PAD_SIZE], q_sc[STEM_OUT_SIZE], q_sr[STEM_OUT_SIZE];
    int8_t q_pool[POOL_OUT_SIZE];
    int8_t q_bp1[BLOCK_PAD_SIZE], q_bc1[BLOCK_OUT_SIZE], q_br1[BLOCK_OUT_SIZE];
    int8_t q_bp2[BLOCK_PAD_SIZE], q_bc2[BLOCK_OUT_SIZE], q_badd[BLOCK_OUT_SIZE], q_brelu[BLOCK_OUT_SIZE];
    int8_t q_gap[F_BLOCK], q_logits[NUM_CLASSES];
    float deq_logits[NUM_CLASSES];

    std::vector<std::array<int8_t, NUM_CLASSES>> all_q_logits(NUM_INPUTS);

    float worst_err = 0.0f;
    tinymind::bench::enableCycleCounter();

    using tinymind::bench::Cycles;
    Cycles c_pad1 = 0, c_conv1 = 0, c_relu1 = 0, c_pool = 0;
    Cycles c_pad2 = 0, c_bc1 = 0, c_brelu1 = 0;
    Cycles c_pad3 = 0, c_bc2 = 0, c_add = 0, c_brelu2 = 0;
    Cycles c_gap = 0, c_dense = 0;

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        quantizeBuffer<int8_t>(inputs[s].data(), q_in, IN_SIZE,
                               p_in.scale, p_in.zero_point, -128, 127);

        Cycles t = tinymind::bench::readCycleCounter();
        stem_pad.forward(q_in, q_sp);
        Cycles t2 = tinymind::bench::readCycleCounter();   c_pad1 += t2 - t;
        stem_conv.forward(q_sp, q_sc);
        t = tinymind::bench::readCycleCounter();           c_conv1 += t - t2;
        // qrelu clamps in-place at the conv output's zero_point.
        std::memcpy(q_sr, q_sc, STEM_OUT_SIZE);
        qreluBuffer<int8_t>(q_sr, STEM_OUT_SIZE, static_cast<int8_t>(p_stem.zero_point));
        t2 = tinymind::bench::readCycleCounter();          c_relu1 += t2 - t;
        // q_pool ends up on the same int8 grid as q_sr (max is pass-through).
        stem_pool.forward(q_sr, q_pool);
        t = tinymind::bench::readCycleCounter();           c_pool += t - t2;

        block_pad1.forward(q_pool, q_bp1);
        t2 = tinymind::bench::readCycleCounter();          c_pad2 += t2 - t;
        block_conv1.forward(q_bp1, q_bc1);
        t = tinymind::bench::readCycleCounter();           c_bc1 += t - t2;
        std::memcpy(q_br1, q_bc1, BLOCK_OUT_SIZE);
        qreluBuffer<int8_t>(q_br1, BLOCK_OUT_SIZE, static_cast<int8_t>(p_bc1.zero_point));
        t2 = tinymind::bench::readCycleCounter();          c_brelu1 += t2 - t;
        block_pad2.forward(q_br1, q_bp2);
        t = tinymind::bench::readCycleCounter();           c_pad3 += t - t2;
        block_conv2.forward(q_bp2, q_bc2);
        t2 = tinymind::bench::readCycleCounter();          c_bc2 += t2 - t;
        block_add.forward(q_bc2, q_pool, q_badd);
        t = tinymind::bench::readCycleCounter();           c_add += t - t2;
        std::memcpy(q_brelu, q_badd, BLOCK_OUT_SIZE);
        qreluBuffer<int8_t>(q_brelu, BLOCK_OUT_SIZE, static_cast<int8_t>(p_badd.zero_point));
        t2 = tinymind::bench::readCycleCounter();          c_brelu2 += t2 - t;

        qgap.forward(q_brelu, q_gap);
        t = tinymind::bench::readCycleCounter();           c_gap += t - t2;
        qdense.forward(q_gap, q_logits);
        t2 = tinymind::bench::readCycleCounter();          c_dense += t2 - t;

        for (std::size_t i = 0; i < NUM_CLASSES; ++i) all_q_logits[s][i] = q_logits[i];

        dequantizeBuffer<int8_t>(q_logits, deq_logits, NUM_CLASSES,
                                 p_logits.scale, p_logits.zero_point);
        const float err = maxAbsDiff(deq_logits, float_logits[s].data(), NUM_CLASSES);
        if (err > worst_err) worst_err = err;
    }

    if (golden_mode)
    {
        std::printf("# resnet18_block_int8 golden output\n");
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
        const std::size_t conv1Bytes = sizeof(stem_conv)  + sizeof(qw_stem) + sizeof(qb_stem)  + sizeof(rq_stem);
        const std::size_t bc1Bytes   = sizeof(block_conv1) + sizeof(qw_b1)  + sizeof(qb_b1)    + sizeof(rq_b1);
        const std::size_t bc2Bytes   = sizeof(block_conv2) + sizeof(qw_b2)  + sizeof(qb_b2)    + sizeof(rq_b2);
        const std::size_t denseBytes = sizeof(qdense)      + sizeof(qw_dense) + sizeof(qb_dense);
        tinymind::bench::writeRow(std::cout, {"stem_pad",   sizeof(stem_pad),   STEM_PAD_SIZE,   c_pad1});
        tinymind::bench::writeRow(std::cout, {"stem_conv7x7_s2", conv1Bytes,    STEM_OUT_SIZE,   c_conv1});
        tinymind::bench::writeRow(std::cout, {"stem_relu",  0,                  STEM_OUT_SIZE,   c_relu1});
        tinymind::bench::writeRow(std::cout, {"stem_maxpool2x2", sizeof(stem_pool), POOL_OUT_SIZE, c_pool});
        tinymind::bench::writeRow(std::cout, {"block_pad1", sizeof(block_pad1), BLOCK_PAD_SIZE,  c_pad2});
        tinymind::bench::writeRow(std::cout, {"block_conv3x3_a", bc1Bytes,      BLOCK_OUT_SIZE,  c_bc1});
        tinymind::bench::writeRow(std::cout, {"block_relu_a", 0,                BLOCK_OUT_SIZE,  c_brelu1});
        tinymind::bench::writeRow(std::cout, {"block_pad2", sizeof(block_pad2), BLOCK_PAD_SIZE,  c_pad3});
        tinymind::bench::writeRow(std::cout, {"block_conv3x3_b", bc2Bytes,      BLOCK_OUT_SIZE,  c_bc2});
        tinymind::bench::writeRow(std::cout, {"block_add",  sizeof(block_add),  BLOCK_OUT_SIZE,  c_add});
        tinymind::bench::writeRow(std::cout, {"block_relu_b", 0,                BLOCK_OUT_SIZE,  c_brelu2});
        tinymind::bench::writeRow(std::cout, {"global_avgpool", sizeof(qgap),   F_BLOCK,         c_gap});
        tinymind::bench::writeRow(std::cout, {"dense_8x4",  denseBytes,         NUM_CLASSES,     c_dense});
        return 0;
    }

    if (quiet) return 0;

    const float lrange = obs_logits.max_value - obs_logits.min_value;
    std::printf("ResNet18 stem+stage int8 vs float reference\n");
    std::printf("  input    range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_in.min_value, obs_in.max_value, p_in.scale, p_in.zero_point);
    std::printf("  stem     range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_stem.min_value, obs_stem.max_value, p_stem.scale, p_stem.zero_point);
    std::printf("  pool float range (on p_stem grid): [%+.3f, %+.3f]\n",
                obs_pool.min_value, obs_pool.max_value);
    std::printf("  bconv1   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_bc1.min_value, obs_bc1.max_value, p_bc1.scale, p_bc1.zero_point);
    std::printf("  bconv2   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_bc2.min_value, obs_bc2.max_value, p_bc2.scale, p_bc2.zero_point);
    std::printf("  add+relu range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_badd.min_value, obs_badd.max_value, p_badd.scale, p_badd.zero_point);
    std::printf("  logits   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_logits.min_value, obs_logits.max_value, p_logits.scale, p_logits.zero_point);
    std::printf("  worst logits max-abs err: %.5f  (%.1f%% of logits range)\n",
                worst_err, 100.0f * worst_err / (lrange + 1e-6f));

    const float tol = 0.40f * lrange;
    if (worst_err > tol)
    {
        std::printf("FAIL: error %.5f > tolerance %.5f\n", worst_err, tol);
        return 1;
    }
    std::printf("PASS (tolerance %.5f, 40%% of range)\n", tol);
    return 0;
}
