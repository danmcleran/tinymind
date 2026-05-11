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

// Phase 13 demonstration: int8 transformer encoder block.
//
// Block shape (S = SequenceLength, E = EmbeddingDim):
//
//   input (S, E)
//       |
//   [LayerNorm1] ---> [QAttention1D (linear)] ---> [QAdd skip=input]
//                                                       |
//                                                  [LayerNorm2]
//                                                       |
//                                               [QDense F1] -> [qrelu]
//                                                       |
//                                                  [QDense F2]
//                                                       |
//                                                  [QAdd skip=lyrnorm1-out]
//                                                       |
//                                                  output (S, E)
//
// Uses linear (ReLU-kernel) attention to keep the example freestanding-
// portable across MCU targets without the softmax LUT footprint. The
// softmax-attention variant slots in by replacing QAttention1D with
// QAttentionSoftmax1D and adding the exp LUT pointer.
//
// The driver runs the same block in float as a reference, calibrates each
// activation tensor on a small synthetic dataset, builds the int8 layers,
// runs them, and prints the max-abs error after dequantization. This is
// the Phase-13 counterpart of examples/resnet_block_int8.

#include "qaffine.hpp"
#include "qattention1d.hpp"
#include "qlayernorm.hpp"
#include "qdense.hpp"
#include "qadd.hpp"
#include "qactivations.hpp"
#include "include/qcalibration.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t S = 4;   // sequence length
constexpr std::size_t E = 8;   // embedding dim (= attention projection dim)
constexpr std::size_t F = 16;  // FFN inner dim

constexpr std::size_t XE_SIZE   = S * E;
constexpr std::size_t XF_SIZE   = S * F;
constexpr std::size_t W_ATTN_SZ = 3 * E * E;
constexpr std::size_t B_ATTN_SZ = 3 * E;
constexpr std::size_t W1_SZ     = F * E;   // FFN W1: E -> F
constexpr std::size_t W2_SZ     = E * F;   // FFN W2: F -> E

// ---------------------------------------------------------------------------
// Float reference forward pass.
// ---------------------------------------------------------------------------

void floatLayerNorm(const float* in, const float* gamma, const float* beta,
                    float eps, float* out)
{
    for (std::size_t t = 0; t < S; ++t)
    {
        const float* row = in + t * E;
        float* out_row = out + t * E;
        float m = 0.0f;
        for (std::size_t i = 0; i < E; ++i) m += row[i];
        m /= static_cast<float>(E);
        float v = 0.0f;
        for (std::size_t i = 0; i < E; ++i)
        {
            const float d = row[i] - m;
            v += d * d;
        }
        v /= static_cast<float>(E);
        const float inv = 1.0f / std::sqrt(v + eps);
        for (std::size_t i = 0; i < E; ++i)
        {
            out_row[i] = gamma[i] * (row[i] - m) * inv + beta[i];
        }
    }
}

void floatLinearAttention(const float* x, const float* w, const float* b,
                          float* y)
{
    float q[S * E], k[S * E], v[S * E], kv[E * E];
    const float* wq = w;
    const float* wk = w + E * E;
    const float* wv = w + 2 * E * E;
    const float* bq = b;
    const float* bk = b + E;
    const float* bv = b + 2 * E;

    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < E; ++p)
        {
            float aq = bq[p], ak = bk[p], av = bv[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += wq[e * E + p] * x[t * E + e];
                ak += wk[e * E + p] * x[t * E + e];
                av += wv[e * E + p] * x[t * E + e];
            }
            q[t * E + p] = (aq < 0.0f) ? 0.0f : aq;
            k[t * E + p] = (ak < 0.0f) ? 0.0f : ak;
            v[t * E + p] = av;
        }
    }
    for (std::size_t i = 0; i < E; ++i)
    {
        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t t = 0; t < S; ++t)
            {
                a += k[t * E + i] * v[t * E + j];
            }
            kv[i * E + j] = a;
        }
    }
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < E; ++i)
            {
                a += q[t * E + i] * kv[i * E + j];
            }
            y[t * E + j] = a;
        }
    }
}

void floatDense(const float* in, const float* w, const float* b,
                std::size_t in_dim, std::size_t out_dim, float* out)
{
    // w is (out_dim, in_dim) row-major.
    const std::size_t rows = S;
    for (std::size_t t = 0; t < rows; ++t)
    {
        for (std::size_t o = 0; o < out_dim; ++o)
        {
            float a = b[o];
            for (std::size_t i = 0; i < in_dim; ++i)
            {
                a += w[o * in_dim + i] * in[t * in_dim + i];
            }
            out[t * out_dim + o] = a;
        }
    }
}

void floatRelu(float* x, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void floatAdd(const float* a, const float* b, float* y, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) y[i] = a[i] + b[i];
}

void floatEncoder(const float* input,
                  const float* gamma1, const float* beta1,
                  const float* w_attn, const float* b_attn,
                  const float* gamma2, const float* beta2,
                  const float* w_ff1, const float* b_ff1,
                  const float* w_ff2, const float* b_ff2,
                  float eps,
                  float* output,
                  float* ln1_out, float* attn_out,
                  float* skip1, float* ln2_out,
                  float* ff1_out, float* ff2_out)
{
    floatLayerNorm(input, gamma1, beta1, eps, ln1_out);
    floatLinearAttention(ln1_out, w_attn, b_attn, attn_out);
    floatAdd(attn_out, input, skip1, XE_SIZE);
    floatLayerNorm(skip1, gamma2, beta2, eps, ln2_out);
    floatDense(ln2_out, w_ff1, b_ff1, E, F, ff1_out);
    floatRelu(ff1_out, XF_SIZE);
    floatDense(ff1_out, w_ff2, b_ff2, F, E, ff2_out);
    floatAdd(skip1, ff2_out, output, XE_SIZE);
}

// ---------------------------------------------------------------------------
// Small utilities.
// ---------------------------------------------------------------------------

float absmaxBuf(const float* buf, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        const float a = (buf[i] < 0.0f) ? -buf[i] : buf[i];
        if (a > m) m = a;
    }
    return m;
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

void quantizeSymToI8(const float* src, std::size_t n,
                     float scale, int8_t* dst)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        long q = std::lround(static_cast<double>(src[i]) /
                             static_cast<double>(scale));
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        dst[i] = static_cast<int8_t>(q);
    }
}

} // namespace

int main()
{
    using tinymind::QAttention1D;
    using tinymind::QLayerNorm1D;
    using tinymind::QDense;
    using tinymind::QAdd;
    using tinymind::Requantizer;
    using tinymind::RangeObserver;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::buildRequantizer;
    using tinymind::buildQAddParams;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;
    using tinymind::dequantize;
    using tinymind::qreluBuffer;
    using tinymind::quantizeLayerNormGamma;
    using tinymind::quantizeLayerNormBeta;
    using tinymind::buildQLayerNormOutputParams;
    using tinymind::quantizeLayerNormEpsilon;

    // ----- Hand-crafted weights so the example runs deterministically.
    float w_attn[W_ATTN_SZ];
    float b_attn[B_ATTN_SZ];
    for (std::size_t i = 0; i < W_ATTN_SZ; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(W_ATTN_SZ);
        w_attn[i] = 0.18f * std::sin(7.0f * t);
    }
    for (std::size_t i = 0; i < B_ATTN_SZ; ++i)
    {
        b_attn[i] = 0.0f;
    }

    float w_ff1[W1_SZ], b_ff1[F];
    float w_ff2[W2_SZ], b_ff2[E];
    for (std::size_t i = 0; i < W1_SZ; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(W1_SZ);
        w_ff1[i] = 0.22f * std::cos(5.0f * t);
    }
    for (std::size_t i = 0; i < W2_SZ; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(W2_SZ);
        w_ff2[i] = 0.20f * std::sin(11.0f * t);
    }
    for (std::size_t i = 0; i < F; ++i) b_ff1[i] = 0.0f;
    for (std::size_t i = 0; i < E; ++i) b_ff2[i] = 0.0f;

    // gamma=1, beta=0 -> plain LayerNorm with no affine reshape.
    float gamma1[E], beta1[E], gamma2[E], beta2[E];
    for (std::size_t i = 0; i < E; ++i)
    {
        gamma1[i] = 1.0f; beta1[i] = 0.0f;
        gamma2[i] = 1.0f; beta2[i] = 0.0f;
    }
    const float ln_eps = 1e-3f;

    // ----- Synthetic dataset (8 inputs) for calibration + parity check.
    constexpr std::size_t NUM_INPUTS = 8;
    std::vector<std::vector<float>> inputs(NUM_INPUTS, std::vector<float>(XE_SIZE));
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        for (std::size_t i = 0; i < XE_SIZE; ++i)
        {
            const float phase = static_cast<float>(s * XE_SIZE + i) * 0.21f;
            inputs[s][i] = 0.8f * std::sin(phase) +
                           0.2f * static_cast<float>(static_cast<int>(i % 5) - 2);
        }
    }

    // Float forward over the dataset; collect activation ranges per tensor.
    RangeObserver obs_in, obs_ln1, obs_attn, obs_skip1, obs_ln2,
                  obs_ff1, obs_ff2, obs_out;
    std::vector<std::vector<float>> float_outputs(NUM_INPUTS,
                                                  std::vector<float>(XE_SIZE));

    float ln1_buf[XE_SIZE], attn_buf[XE_SIZE], skip1_buf[XE_SIZE];
    float ln2_buf[XE_SIZE], ff1_buf[XF_SIZE], ff2_buf[XE_SIZE];

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        floatEncoder(inputs[s].data(),
                     gamma1, beta1, w_attn, b_attn,
                     gamma2, beta2, w_ff1, b_ff1, w_ff2, b_ff2,
                     ln_eps, float_outputs[s].data(),
                     ln1_buf, attn_buf, skip1_buf, ln2_buf, ff1_buf, ff2_buf);

        obs_in   .observe(inputs[s].data(), XE_SIZE);
        obs_ln1  .observe(ln1_buf,   XE_SIZE);
        obs_attn .observe(attn_buf,  XE_SIZE);
        obs_skip1.observe(skip1_buf, XE_SIZE);
        obs_ln2  .observe(ln2_buf,   XE_SIZE);
        obs_ff1  .observe(ff1_buf,   XF_SIZE);
        obs_ff2  .observe(ff2_buf,   XE_SIZE);
        obs_out  .observe(float_outputs[s].data(), XE_SIZE);
    }

    // ----- Affine params for every activation tensor in the chain.
    const auto p_in    = computeAffineParamsAsymmetric(obs_in.min_value,    obs_in.max_value,    -128, 127);
    const auto p_ln1   = computeAffineParamsAsymmetric(obs_ln1.min_value,   obs_ln1.max_value,   -128, 127);
    const auto p_attn  = computeAffineParamsAsymmetric(obs_attn.min_value,  obs_attn.max_value,  -128, 127);
    const auto p_skip1 = computeAffineParamsAsymmetric(obs_skip1.min_value, obs_skip1.max_value, -128, 127);
    const auto p_ln2   = computeAffineParamsAsymmetric(obs_ln2.min_value,   obs_ln2.max_value,   -128, 127);
    const auto p_ff1   = computeAffineParamsAsymmetric(obs_ff1.min_value,   obs_ff1.max_value,   -128, 127);
    const auto p_ff2   = computeAffineParamsAsymmetric(obs_ff2.min_value,   obs_ff2.max_value,   -128, 127);
    const auto p_out   = computeAffineParamsAsymmetric(obs_out.min_value,   obs_out.max_value,   -128, 127);

    // ----- Per-tensor symmetric weight scales (attn has three slabs).
    float wq_scale = absmaxBuf(w_attn,             E * E) / 127.0f;
    float wk_scale = absmaxBuf(w_attn + E * E,     E * E) / 127.0f;
    float wv_scale = absmaxBuf(w_attn + 2 * E * E, E * E) / 127.0f;
    float w1_scale = absmaxBuf(w_ff1, W1_SZ) / 127.0f;
    float w2_scale = absmaxBuf(w_ff2, W2_SZ) / 127.0f;

    // Intermediate Q'/K'/V'/KV ranges for the attention layer's inner scales.
    // We re-run the layer over the calibration set to collect them.
    RangeObserver obs_q, obs_k, obs_v, obs_kv;
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        floatLayerNorm(inputs[s].data(), gamma1, beta1, ln_eps, ln1_buf);
        float q_b[XE_SIZE], k_b[XE_SIZE], v_b[XE_SIZE], kv_b[E * E];
        const float* wq = w_attn;
        const float* wk = w_attn + E * E;
        const float* wv = w_attn + 2 * E * E;
        const float* bq = b_attn;
        const float* bk = b_attn + E;
        const float* bv = b_attn + 2 * E;
        for (std::size_t t = 0; t < S; ++t)
        {
            for (std::size_t p = 0; p < E; ++p)
            {
                float aq = bq[p], ak = bk[p], av = bv[p];
                for (std::size_t e = 0; e < E; ++e)
                {
                    aq += wq[e * E + p] * ln1_buf[t * E + e];
                    ak += wk[e * E + p] * ln1_buf[t * E + e];
                    av += wv[e * E + p] * ln1_buf[t * E + e];
                }
                q_b[t * E + p] = (aq < 0.0f) ? 0.0f : aq;
                k_b[t * E + p] = (ak < 0.0f) ? 0.0f : ak;
                v_b[t * E + p] = av;
            }
        }
        for (std::size_t i = 0; i < E; ++i)
        {
            for (std::size_t j = 0; j < E; ++j)
            {
                float a = 0.0f;
                for (std::size_t t = 0; t < S; ++t)
                {
                    a += k_b[t * E + i] * v_b[t * E + j];
                }
                kv_b[i * E + j] = a;
            }
        }
        obs_q .observe(q_b,  XE_SIZE);
        obs_k .observe(k_b,  XE_SIZE);
        obs_v .observe(v_b,  XE_SIZE);
        obs_kv.observe(kv_b, E * E);
    }
    const auto p_q  = computeAffineParamsAsymmetric(0.0f,             obs_q.max_value, -128, 127); // ReLU
    const auto p_k  = computeAffineParamsAsymmetric(0.0f,             obs_k.max_value, -128, 127); // ReLU
    const auto p_v  = computeAffineParamsAsymmetric(obs_v.min_value,  obs_v.max_value, -128, 127);
    const auto p_kv = computeAffineParamsAsymmetric(obs_kv.min_value, obs_kv.max_value, -128, 127);

    // ----- Quantize weights / biases.
    int8_t qw_attn[W_ATTN_SZ];
    quantizeSymToI8(w_attn,             E * E, wq_scale, qw_attn);
    quantizeSymToI8(w_attn + E * E,     E * E, wk_scale, qw_attn + E * E);
    quantizeSymToI8(w_attn + 2 * E * E, E * E, wv_scale, qw_attn + 2 * E * E);

    int32_t qb_attn[B_ATTN_SZ] = {0};
    for (std::size_t p = 0; p < E; ++p)
    {
        qb_attn[p]         = static_cast<int32_t>(std::lround(
            static_cast<double>(b_attn[p]) /
            (static_cast<double>(p_ln1.scale) * static_cast<double>(wq_scale))));
        qb_attn[E + p]     = static_cast<int32_t>(std::lround(
            static_cast<double>(b_attn[E + p]) /
            (static_cast<double>(p_ln1.scale) * static_cast<double>(wk_scale))));
        qb_attn[2 * E + p] = static_cast<int32_t>(std::lround(
            static_cast<double>(b_attn[2 * E + p]) /
            (static_cast<double>(p_ln1.scale) * static_cast<double>(wv_scale))));
    }

    int8_t qw_ff1[W1_SZ], qw_ff2[W2_SZ];
    quantizeSymToI8(w_ff1, W1_SZ, w1_scale, qw_ff1);
    quantizeSymToI8(w_ff2, W2_SZ, w2_scale, qw_ff2);

    int32_t qb_ff1[F] = {0}, qb_ff2[E] = {0};

    // ----- Build layer instances.

    // LayerNorm 1: input -> ln1.
    int16_t gamma1_q[E];
    int32_t beta1_q[E];
    quantizeLayerNormGamma(gamma1, E, gamma1_q);
    quantizeLayerNormBeta(beta1, E, p_ln1.scale, beta1_q);
    int32_t ln1_out_mult = 0, ln1_out_shift = 0;
    buildQLayerNormOutputParams(p_ln1.scale, ln1_out_mult, ln1_out_shift);

    QLayerNorm1D<int8_t, int8_t, S, E> qln1;
    qln1.gamma = gamma1_q;
    qln1.beta = beta1_q;
    qln1.epsilon_q = quantizeLayerNormEpsilon(ln_eps, p_in.scale);
    qln1.output_multiplier = ln1_out_mult;
    qln1.output_shift = ln1_out_shift;
    qln1.output_zero_point = static_cast<int8_t>(p_ln1.zero_point);
    qln1.qmin = -128;
    qln1.qmax = 127;

    // LayerNorm 2: skip1 -> ln2.
    int16_t gamma2_q[E];
    int32_t beta2_q[E];
    quantizeLayerNormGamma(gamma2, E, gamma2_q);
    quantizeLayerNormBeta(beta2, E, p_ln2.scale, beta2_q);
    int32_t ln2_out_mult = 0, ln2_out_shift = 0;
    buildQLayerNormOutputParams(p_ln2.scale, ln2_out_mult, ln2_out_shift);

    QLayerNorm1D<int8_t, int8_t, S, E> qln2;
    qln2.gamma = gamma2_q;
    qln2.beta = beta2_q;
    qln2.epsilon_q = quantizeLayerNormEpsilon(ln_eps, p_skip1.scale);
    qln2.output_multiplier = ln2_out_mult;
    qln2.output_shift = ln2_out_shift;
    qln2.output_zero_point = static_cast<int8_t>(p_ln2.zero_point);
    qln2.qmin = -128;
    qln2.qmax = 127;

    // Attention.
    QAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                 S, E, E> qattn;
    qattn.weights = qw_attn;
    qattn.biases = qb_attn;
    qattn.input_zero_point = static_cast<int8_t>(p_ln1.zero_point);
    qattn.q_zero_point     = static_cast<int8_t>(p_q.zero_point);
    qattn.k_zero_point     = static_cast<int8_t>(p_k.zero_point);
    qattn.v_zero_point     = static_cast<int8_t>(p_v.zero_point);
    qattn.kv_zero_point    = static_cast<int8_t>(p_kv.zero_point);
    qattn.q_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wq_scale, p_q.scale,
                                                   p_q.zero_point,
                                                   p_q.zero_point, 127);
    qattn.k_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wk_scale, p_k.scale,
                                                   p_k.zero_point,
                                                   p_k.zero_point, 127);
    qattn.v_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wv_scale, p_v.scale,
                                                   p_v.zero_point, -128, 127);
    qattn.kv_requantizer = buildRequantizer<int8_t>(p_k.scale, p_v.scale, p_kv.scale,
                                                    p_kv.zero_point, -128, 127);
    qattn.output_requantizer = buildRequantizer<int8_t>(p_q.scale, p_kv.scale,
                                                        p_attn.scale,
                                                        p_attn.zero_point, -128, 127);

    // QAdd #1: attn + input -> skip1. Both inputs must share a single
    // (scale, zp) -- we approximate by treating the input grid as p_in and
    // the attention grid as p_attn; QAdd's per-input rescalers reconcile.
    QAdd<int8_t, int8_t, int8_t, XE_SIZE> qadd1;
    {
        const auto addp = buildQAddParams(p_attn.scale, p_in.scale, p_skip1.scale);
        qadd1.input_a_zero_point = static_cast<int8_t>(p_attn.zero_point);
        qadd1.input_b_zero_point = static_cast<int8_t>(p_in.zero_point);
        qadd1.left_shift = addp.left_shift;
        qadd1.input_a_multiplier = addp.input_a_multiplier;
        qadd1.input_a_shift = addp.input_a_shift;
        qadd1.input_b_multiplier = addp.input_b_multiplier;
        qadd1.input_b_shift = addp.input_b_shift;
        qadd1.output_requantizer.multiplier = addp.output_multiplier;
        qadd1.output_requantizer.shift = addp.output_shift;
        qadd1.output_requantizer.zero_point = static_cast<int8_t>(p_skip1.zero_point);
        qadd1.output_requantizer.qmin = -128;
        qadd1.output_requantizer.qmax = 127;
    }

    // FFN dense 1: ln2 -> ff1 (with ReLU folded via qmin = zp).
    QDense<int8_t, int8_t, int32_t, int8_t, E, F> qff1;
    qff1.weights = qw_ff1;
    qff1.biases  = qb_ff1;
    qff1.input_zero_point = static_cast<int8_t>(p_ln2.zero_point);
    qff1.requantizer = buildRequantizer<int8_t>(p_ln2.scale, w1_scale, p_ff1.scale,
                                                p_ff1.zero_point,
                                                p_ff1.zero_point, 127);

    // FFN dense 2: ff1 -> ff2.
    QDense<int8_t, int8_t, int32_t, int8_t, F, E> qff2;
    qff2.weights = qw_ff2;
    qff2.biases  = qb_ff2;
    qff2.input_zero_point = static_cast<int8_t>(p_ff1.zero_point);
    qff2.requantizer = buildRequantizer<int8_t>(p_ff1.scale, w2_scale, p_ff2.scale,
                                                p_ff2.zero_point, -128, 127);

    // QAdd #2: skip1 + ff2 -> out.
    QAdd<int8_t, int8_t, int8_t, XE_SIZE> qadd2;
    {
        const auto addp = buildQAddParams(p_skip1.scale, p_ff2.scale, p_out.scale);
        qadd2.input_a_zero_point = static_cast<int8_t>(p_skip1.zero_point);
        qadd2.input_b_zero_point = static_cast<int8_t>(p_ff2.zero_point);
        qadd2.left_shift = addp.left_shift;
        qadd2.input_a_multiplier = addp.input_a_multiplier;
        qadd2.input_a_shift = addp.input_a_shift;
        qadd2.input_b_multiplier = addp.input_b_multiplier;
        qadd2.input_b_shift = addp.input_b_shift;
        qadd2.output_requantizer.multiplier = addp.output_multiplier;
        qadd2.output_requantizer.shift = addp.output_shift;
        qadd2.output_requantizer.zero_point = static_cast<int8_t>(p_out.zero_point);
        qadd2.output_requantizer.qmin = -128;
        qadd2.output_requantizer.qmax = 127;
    }

    // ----- int8 forward pass over the dataset.
    int8_t q_in[XE_SIZE], q_ln1[XE_SIZE], q_attn_out[XE_SIZE], q_skip1[XE_SIZE];
    int8_t q_ln2[XE_SIZE], q_ff1[XF_SIZE], q_ff2[XE_SIZE], q_out[XE_SIZE];
    int8_t q_scr[XE_SIZE], k_scr[XE_SIZE], v_scr[XE_SIZE], kv_scr[E * E];
    float deq_out[XE_SIZE];

    float worst_err = 0.0f;
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        quantizeBuffer<int8_t>(inputs[s].data(), q_in, XE_SIZE,
                               p_in.scale, p_in.zero_point, -128, 127);

        qln1.forward(q_in, q_ln1);
        qattn.forward(q_ln1, q_scr, k_scr, v_scr, kv_scr, q_attn_out);
        qadd1.forward(q_attn_out, q_in, q_skip1);

        qln2.forward(q_skip1, q_ln2);
        // QDense operates on a single row; drive it per-sequence-step.
        for (std::size_t t = 0; t < S; ++t)
        {
            qff1.forward(q_ln2 + t * E, q_ff1 + t * F);
        }
        qreluBuffer<int8_t>(q_ff1, XF_SIZE,
                            static_cast<int8_t>(p_ff1.zero_point));
        for (std::size_t t = 0; t < S; ++t)
        {
            qff2.forward(q_ff1 + t * F, q_ff2 + t * E);
        }

        qadd2.forward(q_skip1, q_ff2, q_out);

        dequantizeBuffer<int8_t>(q_out, deq_out, XE_SIZE, p_out.scale, p_out.zero_point);
        const float err = maxAbsDiff(deq_out, float_outputs[s].data(), XE_SIZE);
        if (err > worst_err) worst_err = err;
    }

    const float out_range = obs_out.max_value - obs_out.min_value;

    std::printf("Transformer encoder int8 vs float reference\n");
    std::printf("  input   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_in.min_value, obs_in.max_value, p_in.scale, p_in.zero_point);
    std::printf("  ln1     range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_ln1.min_value, obs_ln1.max_value, p_ln1.scale, p_ln1.zero_point);
    std::printf("  attn    range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_attn.min_value, obs_attn.max_value, p_attn.scale, p_attn.zero_point);
    std::printf("  skip1   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_skip1.min_value, obs_skip1.max_value, p_skip1.scale, p_skip1.zero_point);
    std::printf("  ln2     range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_ln2.min_value, obs_ln2.max_value, p_ln2.scale, p_ln2.zero_point);
    std::printf("  ff1     range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_ff1.min_value, obs_ff1.max_value, p_ff1.scale, p_ff1.zero_point);
    std::printf("  ff2     range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_ff2.min_value, obs_ff2.max_value, p_ff2.scale, p_ff2.zero_point);
    std::printf("  output  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_out.min_value, obs_out.max_value, p_out.scale, p_out.zero_point);
    std::printf("  worst block max-abs err: %.5f   (%.1f%% of output range)\n",
                worst_err,
                100.0f * worst_err / (out_range + 1e-6f));

    // Loose tolerance: this is a six-stage int8 chain (LN, Attn, Add, LN,
    // FFN, Add) with no QAT and no cross-layer equalization. Real
    // deployments tighten this through Phase 15 importer tooling.
    const float tol = 0.40f * out_range;
    if (worst_err > tol)
    {
        std::printf("FAIL: error %.5f > tolerance %.5f\n", worst_err, tol);
        return 1;
    }
    std::printf("PASS (tolerance %.5f, 40%% of range)\n", tol);
    return 0;
}
