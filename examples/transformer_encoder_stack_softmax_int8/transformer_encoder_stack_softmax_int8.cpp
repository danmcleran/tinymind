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

// int8 transformer encoder stack -- SOFTMAX-ATTENTION variant.
//
// Identical pipeline to examples/transformer_encoder_stack_int8, but each
// block uses standard (softmax) self-attention instead of linear
// (ReLU-kernel) attention:
//
//   token ids (S)
//       |
//   [QEmbedding]            int8 gather from [Vocab, E] table   (qembedding.hpp)
//       |
//   [QPositionalEncoding]   add fixed sinusoidal table          (qpositional.hpp)
//       |
//   [EncoderBlock 0..N-1]   LN -> SoftmaxAttention -> Add ;
//                           LN -> Dense->ReLU->Dense -> Add
//       |
//   output (S, E)
//
// Softmax attention follows the TFLite int8 convention: scores are formed as
// (Q . K^T) / sqrt(P), requantized onto an int8 score grid, then per row the
// kernel subtracts the row max, looks exp up in a 256-entry int32 LUT, and
// normalizes to the 1/256 probability grid at zero_point -128. The exp LUT is
// the extra footprint this variant pays over the linear-attention stack; on a
// freestanding target it lives in flash.
//
// The driver builds the same stack in float, calibrates every activation
// tensor on a small synthetic token dataset, builds the int8 layers, runs
// them, and reports the worst max-abs error after dequantization.

#include "qaffine.hpp"
#include "qembedding.hpp"
#include "qpositional.hpp"
#include "qattention_softmax.hpp"
#include "qlayernorm.hpp"
#include "qdense.hpp"
#include "qadd.hpp"
#include "qactivations.hpp"
#include "include/qcalibration.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t VOCAB       = 16;  // token vocabulary size
constexpr std::size_t S           = 8;   // sequence length
constexpr std::size_t E           = 8;   // embedding dim (= attn proj dim P)
constexpr std::size_t F           = 16;  // FFN inner dim
constexpr std::size_t NUM_BLOCKS  = 2;   // stacked encoder blocks

constexpr std::size_t XE_SIZE   = S * E;
constexpr std::size_t XF_SIZE   = S * F;
constexpr std::size_t SS_SIZE   = S * S;
constexpr std::size_t W_ATTN_SZ = 3 * E * E;
constexpr std::size_t B_ATTN_SZ = 3 * E;
constexpr std::size_t W1_SZ     = F * E;   // FFN W1: E -> F
constexpr std::size_t W2_SZ     = E * F;   // FFN W2: F -> E

constexpr float LN_EPS = 1e-3f;

// Probability grid produced by the int8 softmax: 1/256 scale, zp -128.
constexpr float ATTN_PROB_SCALE = 1.0f / 256.0f;

// ---------------------------------------------------------------------------
// Per-block float weights.
// ---------------------------------------------------------------------------

struct BlockWeights
{
    float gamma1[E], beta1[E], gamma2[E], beta2[E];
    float w_attn[W_ATTN_SZ], b_attn[B_ATTN_SZ];
    float w_ff1[W1_SZ], b_ff1[F];
    float w_ff2[W2_SZ], b_ff2[E];
};

// ---------------------------------------------------------------------------
// Float reference ops.
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

// Standard (softmax) self-attention. Projections carry NO ReLU. Also returns
// the intermediate q/k/v and pre-softmax scores for calibration when the
// observer pointers are non-null.
void floatSoftmaxAttention(const float* x, const float* w, const float* b,
                           float* y,
                           float* q_obs, float* k_obs, float* v_obs,
                           float* score_obs)
{
    float q[XE_SIZE], k[XE_SIZE], v[XE_SIZE];
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
            q[t * E + p] = aq;
            k[t * E + p] = ak;
            v[t * E + p] = av;
        }
    }

    const float inv_sqrt_p = 1.0f / std::sqrt(static_cast<float>(E));
    float scores[SS_SIZE];
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t u = 0; u < S; ++u)
        {
            float a = 0.0f;
            for (std::size_t p = 0; p < E; ++p)
            {
                a += q[t * E + p] * k[u * E + p];
            }
            scores[t * S + u] = a * inv_sqrt_p;
        }
    }

    // Row softmax -> attention output.
    for (std::size_t t = 0; t < S; ++t)
    {
        float mx = scores[t * S];
        for (std::size_t u = 1; u < S; ++u)
        {
            if (scores[t * S + u] > mx) mx = scores[t * S + u];
        }
        float denom = 0.0f;
        float probs[S];
        for (std::size_t u = 0; u < S; ++u)
        {
            probs[u] = std::exp(scores[t * S + u] - mx);
            denom += probs[u];
        }
        for (std::size_t u = 0; u < S; ++u) probs[u] /= denom;

        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t u = 0; u < S; ++u)
            {
                a += probs[u] * v[u * E + j];
            }
            y[t * E + j] = a;
        }
    }

    if (q_obs)     std::memcpy(q_obs,     q,      sizeof(float) * XE_SIZE);
    if (k_obs)     std::memcpy(k_obs,     k,      sizeof(float) * XE_SIZE);
    if (v_obs)     std::memcpy(v_obs,     v,      sizeof(float) * XE_SIZE);
    if (score_obs) std::memcpy(score_obs, scores, sizeof(float) * SS_SIZE);
}

void floatDense(const float* in, const float* w, const float* b,
                std::size_t in_dim, std::size_t out_dim, float* out)
{
    for (std::size_t t = 0; t < S; ++t)
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

void floatBlock(const float* input, const BlockWeights& W,
                float* output,
                float* ln1_out, float* attn_out, float* skip1,
                float* ln2_out, float* ff1_out, float* ff2_out)
{
    floatLayerNorm(input, W.gamma1, W.beta1, LN_EPS, ln1_out);
    floatSoftmaxAttention(ln1_out, W.w_attn, W.b_attn, attn_out,
                          nullptr, nullptr, nullptr, nullptr);
    floatAdd(attn_out, input, skip1, XE_SIZE);
    floatLayerNorm(skip1, W.gamma2, W.beta2, LN_EPS, ln2_out);
    floatDense(ln2_out, W.w_ff1, W.b_ff1, E, F, ff1_out);
    floatRelu(ff1_out, XF_SIZE);
    floatDense(ff1_out, W.w_ff2, W.b_ff2, F, E, ff2_out);
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

void quantizeSymToI8(const float* src, std::size_t n, float scale, int8_t* dst)
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

// Symmetric scale (zp 0) from an observed absmax, with an epsilon floor so a
// degenerate all-zero tensor still yields a finite multiplier.
float symScale(float absmax)
{
    const float a = (absmax > 1e-6f) ? absmax : 1e-6f;
    return a / 127.0f;
}

using tinymind::AffineParams;
using tinymind::Requantizer;

struct BlockQuant
{
    int8_t  qw_attn[W_ATTN_SZ];
    int32_t qb_attn[B_ATTN_SZ];
    int8_t  qw_ff1[W1_SZ];
    int32_t qb_ff1[F];
    int8_t  qw_ff2[W2_SZ];
    int32_t qb_ff2[E];
    int16_t gamma1_q[E];
    int32_t beta1_q[E];
    int16_t gamma2_q[E];
    int32_t beta2_q[E];
    int32_t exp_lut[256];

    tinymind::QLayerNorm1D<int8_t, int8_t, S, E> qln1, qln2;
    tinymind::QAttentionSoftmax1D<int8_t, int8_t, int32_t,
                                  int8_t, int8_t, int8_t, int8_t,
                                  S, E, E> qattn;
    tinymind::QAdd<int8_t, int8_t, int8_t, XE_SIZE> qadd1, qadd2;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, F> qff1;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, F, E> qff2;

    AffineParams p_in, p_out;
};

void runBlockQ(const BlockQuant& B, const int8_t* q_in, int8_t* q_out,
               int8_t* q_ln1, int8_t* q_attn_out, int8_t* q_skip1,
               int8_t* q_ln2, int8_t* q_ff1, int8_t* q_ff2,
               int8_t* q_scr, int8_t* k_scr, int8_t* v_scr,
               int8_t* score_scr, int8_t* attn_scr,
               int8_t ff1_zp)
{
    B.qln1.forward(q_in, q_ln1);
    B.qattn.forward(q_ln1, q_scr, k_scr, v_scr, score_scr, attn_scr, q_attn_out);
    B.qadd1.forward(q_attn_out, q_in, q_skip1);
    B.qln2.forward(q_skip1, q_ln2);
    for (std::size_t t = 0; t < S; ++t)
    {
        B.qff1.forward(q_ln2 + t * E, q_ff1 + t * F);
    }
    tinymind::qreluBuffer<int8_t>(q_ff1, XF_SIZE, ff1_zp);
    for (std::size_t t = 0; t < S; ++t)
    {
        B.qff2.forward(q_ff1 + t * F, q_ff2 + t * E);
    }
    B.qadd2.forward(q_skip1, q_ff2, q_out);
}

} // namespace

int main(int argc, char** argv)
{
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    using tinymind::RangeObserver;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::buildRequantizer;
    using tinymind::buildQAddParams;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;
    using tinymind::quantizeMultiplier;
    using tinymind::quantizeLayerNormGamma;
    using tinymind::quantizeLayerNormBeta;
    using tinymind::buildQLayerNormOutputParams;
    using tinymind::quantizeLayerNormEpsilon;
    using tinymind::buildQSoftmaxExpLUT;
    using tinymind::qAttentionInvSqrt;
    using tinymind::QEmbedding;
    using tinymind::QPositionalEncoding1D;
    using tinymind::sinusoidalPositionalTable;

    // ----- Embedding table [VOCAB, E].
    float emb_table[VOCAB * E];
    for (std::size_t i = 0; i < VOCAB * E; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(VOCAB * E);
        emb_table[i] = 0.6f * std::sin(9.0f * t) + 0.1f * std::cos(2.0f * t);
    }

    // ----- Fixed sinusoidal positional table [S, E].
    float pos_table[XE_SIZE];
    sinusoidalPositionalTable(S, E, pos_table);

    // ----- Per-block weights.
    std::vector<BlockWeights> blocks(NUM_BLOCKS);
    for (std::size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        BlockWeights& W = blocks[b];
        const float seed = 1.0f + 0.5f * static_cast<float>(b);
        for (std::size_t i = 0; i < W_ATTN_SZ; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W_ATTN_SZ);
            W.w_attn[i] = 0.18f * std::sin((7.0f + seed) * t);
        }
        for (std::size_t i = 0; i < B_ATTN_SZ; ++i) W.b_attn[i] = 0.0f;
        for (std::size_t i = 0; i < W1_SZ; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W1_SZ);
            W.w_ff1[i] = 0.22f * std::cos((5.0f + seed) * t);
        }
        for (std::size_t i = 0; i < W2_SZ; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W2_SZ);
            W.w_ff2[i] = 0.20f * std::sin((11.0f + seed) * t);
        }
        for (std::size_t i = 0; i < F; ++i) W.b_ff1[i] = 0.0f;
        for (std::size_t i = 0; i < E; ++i) W.b_ff2[i] = 0.0f;
        for (std::size_t i = 0; i < E; ++i)
        {
            W.gamma1[i] = 1.0f; W.beta1[i] = 0.0f;
            W.gamma2[i] = 1.0f; W.beta2[i] = 0.0f;
        }
    }

    // ----- Synthetic token dataset.
    constexpr std::size_t NUM_INPUTS = 8;
    std::vector<std::vector<int32_t>> tokens(NUM_INPUTS, std::vector<int32_t>(S));
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        for (std::size_t t = 0; t < S; ++t)
        {
            tokens[s][t] = static_cast<int32_t>((s * S + t * 3 + 1) % VOCAB);
        }
    }

    // ----- Float forward + calibration observers.
    RangeObserver obs_emb, obs_pos, obs_x0;
    std::vector<RangeObserver> obs_ln1(NUM_BLOCKS), obs_attn(NUM_BLOCKS),
        obs_skip1(NUM_BLOCKS), obs_ln2(NUM_BLOCKS), obs_ff1(NUM_BLOCKS),
        obs_ff2(NUM_BLOCKS), obs_blkout(NUM_BLOCKS);
    std::vector<RangeObserver> obs_q(NUM_BLOCKS), obs_k(NUM_BLOCKS),
        obs_v(NUM_BLOCKS), obs_score(NUM_BLOCKS);

    obs_pos.observe(pos_table, XE_SIZE);

    std::vector<std::vector<float>> float_outputs(NUM_INPUTS,
                                                  std::vector<float>(XE_SIZE));

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        float emb_buf[XE_SIZE], x[XE_SIZE];
        for (std::size_t t = 0; t < S; ++t)
        {
            const float* row = emb_table + tokens[s][t] * E;
            for (std::size_t e = 0; e < E; ++e) emb_buf[t * E + e] = row[e];
        }
        obs_emb.observe(emb_buf, XE_SIZE);
        floatAdd(emb_buf, pos_table, x, XE_SIZE);
        obs_x0.observe(x, XE_SIZE);

        for (std::size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            float ln1[XE_SIZE], attn[XE_SIZE], skip1[XE_SIZE];
            float ln2[XE_SIZE], ff1[XF_SIZE], ff2[XE_SIZE], out[XE_SIZE];
            floatBlock(x, blocks[b], out, ln1, attn, skip1, ln2, ff1, ff2);

            obs_ln1[b].observe(ln1, XE_SIZE);
            obs_attn[b].observe(attn, XE_SIZE);
            obs_skip1[b].observe(skip1, XE_SIZE);
            obs_ln2[b].observe(ln2, XE_SIZE);
            obs_ff1[b].observe(ff1, XF_SIZE);
            obs_ff2[b].observe(ff2, XE_SIZE);
            obs_blkout[b].observe(out, XE_SIZE);

            // Inner attention tensors (q/k/v, pre-softmax scores).
            {
                float q_b[XE_SIZE], k_b[XE_SIZE], v_b[XE_SIZE], score_b[SS_SIZE];
                float attn_tmp[XE_SIZE];
                floatLayerNorm(x, blocks[b].gamma1, blocks[b].beta1, LN_EPS, ln1);
                floatSoftmaxAttention(ln1, blocks[b].w_attn, blocks[b].b_attn,
                                      attn_tmp, q_b, k_b, v_b, score_b);
                obs_q[b].observe(q_b, XE_SIZE);
                obs_k[b].observe(k_b, XE_SIZE);
                obs_v[b].observe(v_b, XE_SIZE);
                obs_score[b].observe(score_b, SS_SIZE);
            }

            std::memcpy(x, out, sizeof(float) * XE_SIZE);
        }
        std::memcpy(float_outputs[s].data(), x, sizeof(float) * XE_SIZE);
    }

    // ----- Embedding / positional / stack-input grids.
    const auto p_emb = computeAffineParamsAsymmetric(obs_emb.min_value, obs_emb.max_value, -128, 127);
    const auto p_pos = computeAffineParamsAsymmetric(obs_pos.min_value, obs_pos.max_value, -128, 127);
    const auto p_x0  = computeAffineParamsAsymmetric(obs_x0.min_value,  obs_x0.max_value,  -128, 127);

    int8_t qemb_table[VOCAB * E];
    quantizeBuffer<int8_t>(emb_table, qemb_table, VOCAB * E,
                           p_emb.scale, p_emb.zero_point, -128, 127);
    QEmbedding<int8_t, int8_t, VOCAB, E> qembed;
    qembed.table = qemb_table;
    qembed.requantizer = nullptr;

    int8_t qpos_table[XE_SIZE];
    quantizeBuffer<int8_t>(pos_table, qpos_table, XE_SIZE,
                           p_pos.scale, p_pos.zero_point, -128, 127);
    QPositionalEncoding1D<int8_t, int8_t, int8_t, S, E> qpos;
    qpos.table = qpos_table;
    {
        const auto addp = buildQAddParams(p_emb.scale, p_pos.scale, p_x0.scale);
        qpos.adder.input_a_zero_point = static_cast<int8_t>(p_emb.zero_point);
        qpos.adder.input_b_zero_point = static_cast<int8_t>(p_pos.zero_point);
        qpos.adder.left_shift = addp.left_shift;
        qpos.adder.input_a_multiplier = addp.input_a_multiplier;
        qpos.adder.input_a_shift = addp.input_a_shift;
        qpos.adder.input_b_multiplier = addp.input_b_multiplier;
        qpos.adder.input_b_shift = addp.input_b_shift;
        qpos.adder.output_requantizer.multiplier = addp.output_multiplier;
        qpos.adder.output_requantizer.shift = addp.output_shift;
        qpos.adder.output_requantizer.zero_point = static_cast<int8_t>(p_x0.zero_point);
        qpos.adder.output_requantizer.qmin = -128;
        qpos.adder.output_requantizer.qmax = 127;
    }

    // ----- Build int8 encoder blocks.
    std::vector<BlockQuant> qblocks(NUM_BLOCKS);
    for (std::size_t b = 0; b < NUM_BLOCKS; ++b)
    {
        BlockQuant& B = qblocks[b];
        const BlockWeights& W = blocks[b];

        const AffineParams p_in = (b == 0) ? p_x0 : qblocks[b - 1].p_out;
        B.p_in = p_in;

        const auto p_ln1   = computeAffineParamsAsymmetric(obs_ln1[b].min_value,   obs_ln1[b].max_value,   -128, 127);
        const auto p_attn  = computeAffineParamsAsymmetric(obs_attn[b].min_value,  obs_attn[b].max_value,  -128, 127);
        const auto p_skip1 = computeAffineParamsAsymmetric(obs_skip1[b].min_value, obs_skip1[b].max_value, -128, 127);
        const auto p_ln2   = computeAffineParamsAsymmetric(obs_ln2[b].min_value,   obs_ln2[b].max_value,   -128, 127);
        const auto p_ff1   = computeAffineParamsAsymmetric(obs_ff1[b].min_value,   obs_ff1[b].max_value,   -128, 127);
        const auto p_ff2   = computeAffineParamsAsymmetric(obs_ff2[b].min_value,   obs_ff2[b].max_value,   -128, 127);
        const auto p_out   = computeAffineParamsAsymmetric(obs_blkout[b].min_value, obs_blkout[b].max_value, -128, 127);
        B.p_out = p_out;

        // Symmetric (zp 0) grids for q / k / v / score, matching the proven
        // int8 softmax-attention reference in the quantization unit test.
        const float qs = symScale((obs_q[b].max_value > -obs_q[b].min_value)
                                  ? obs_q[b].max_value : -obs_q[b].min_value);
        const float ks = symScale((obs_k[b].max_value > -obs_k[b].min_value)
                                  ? obs_k[b].max_value : -obs_k[b].min_value);
        const float vs = symScale((obs_v[b].max_value > -obs_v[b].min_value)
                                  ? obs_v[b].max_value : -obs_v[b].min_value);
        const float score_scale = symScale((obs_score[b].max_value > -obs_score[b].min_value)
                                  ? obs_score[b].max_value : -obs_score[b].min_value);

        // Weight scales.
        const float wq_scale = absmaxBuf(W.w_attn,             E * E) / 127.0f;
        const float wk_scale = absmaxBuf(W.w_attn + E * E,     E * E) / 127.0f;
        const float wv_scale = absmaxBuf(W.w_attn + 2 * E * E, E * E) / 127.0f;
        const float w1_scale = absmaxBuf(W.w_ff1, W1_SZ) / 127.0f;
        const float w2_scale = absmaxBuf(W.w_ff2, W2_SZ) / 127.0f;

        quantizeSymToI8(W.w_attn,             E * E, wq_scale, B.qw_attn);
        quantizeSymToI8(W.w_attn + E * E,     E * E, wk_scale, B.qw_attn + E * E);
        quantizeSymToI8(W.w_attn + 2 * E * E, E * E, wv_scale, B.qw_attn + 2 * E * E);
        for (std::size_t i = 0; i < B_ATTN_SZ; ++i) B.qb_attn[i] = 0;
        quantizeSymToI8(W.w_ff1, W1_SZ, w1_scale, B.qw_ff1);
        quantizeSymToI8(W.w_ff2, W2_SZ, w2_scale, B.qw_ff2);
        for (std::size_t i = 0; i < F; ++i) B.qb_ff1[i] = 0;
        for (std::size_t i = 0; i < E; ++i) B.qb_ff2[i] = 0;

        // LayerNorm 1.
        quantizeLayerNormGamma(W.gamma1, E, B.gamma1_q);
        quantizeLayerNormBeta(W.beta1, E, p_ln1.scale, B.beta1_q);
        int32_t ln1_mult = 0, ln1_shift = 0;
        buildQLayerNormOutputParams(p_ln1.scale, ln1_mult, ln1_shift);
        B.qln1.gamma = B.gamma1_q;
        B.qln1.beta = B.beta1_q;
        B.qln1.epsilon_q = quantizeLayerNormEpsilon(LN_EPS, p_in.scale);
        B.qln1.output_multiplier = ln1_mult;
        B.qln1.output_shift = ln1_shift;
        B.qln1.output_zero_point = static_cast<int8_t>(p_ln1.zero_point);
        B.qln1.qmin = -128;
        B.qln1.qmax = 127;

        // LayerNorm 2.
        quantizeLayerNormGamma(W.gamma2, E, B.gamma2_q);
        quantizeLayerNormBeta(W.beta2, E, p_ln2.scale, B.beta2_q);
        int32_t ln2_mult = 0, ln2_shift = 0;
        buildQLayerNormOutputParams(p_ln2.scale, ln2_mult, ln2_shift);
        B.qln2.gamma = B.gamma2_q;
        B.qln2.beta = B.beta2_q;
        B.qln2.epsilon_q = quantizeLayerNormEpsilon(LN_EPS, p_skip1.scale);
        B.qln2.output_multiplier = ln2_mult;
        B.qln2.output_shift = ln2_shift;
        B.qln2.output_zero_point = static_cast<int8_t>(p_ln2.zero_point);
        B.qln2.qmin = -128;
        B.qln2.qmax = 127;

        // Softmax attention.
        buildQSoftmaxExpLUT(score_scale, B.exp_lut);
        B.qattn.weights = B.qw_attn;
        B.qattn.biases  = B.qb_attn;
        B.qattn.input_zero_point = static_cast<int8_t>(p_ln1.zero_point);
        B.qattn.q_zero_point = 0;
        B.qattn.k_zero_point = 0;
        B.qattn.v_zero_point = 0;
        B.qattn.attn_zero_point = -128;
        B.qattn.q_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wq_scale, qs, 0, -128, 127);
        B.qattn.k_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wk_scale, ks, 0, -128, 127);
        B.qattn.v_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wv_scale, vs, 0, -128, 127);
        {
            Requantizer<int32_t, int8_t> r;
            const double ratio =
                (static_cast<double>(qs) * static_cast<double>(ks)) /
                static_cast<double>(score_scale) * qAttentionInvSqrt(E);
            int32_t mult = 0, shft = 0;
            quantizeMultiplier(ratio, mult, shft);
            r.multiplier = mult;
            r.shift = shft;
            r.zero_point = 0;
            r.qmin = -128;
            r.qmax = 127;
            B.qattn.score_requantizer = r;
        }
        B.qattn.softmax_exp_lut = B.exp_lut;
        B.qattn.attn_qmin = -128;
        B.qattn.attn_qmax = 127;
        B.qattn.output_requantizer = buildRequantizer<int8_t>(ATTN_PROB_SCALE, vs,
                                                             p_attn.scale,
                                                             p_attn.zero_point,
                                                             -128, 127);

        // QAdd 1: attn + block-input -> skip1.
        {
            const auto addp = buildQAddParams(p_attn.scale, p_in.scale, p_skip1.scale);
            B.qadd1.input_a_zero_point = static_cast<int8_t>(p_attn.zero_point);
            B.qadd1.input_b_zero_point = static_cast<int8_t>(p_in.zero_point);
            B.qadd1.left_shift = addp.left_shift;
            B.qadd1.input_a_multiplier = addp.input_a_multiplier;
            B.qadd1.input_a_shift = addp.input_a_shift;
            B.qadd1.input_b_multiplier = addp.input_b_multiplier;
            B.qadd1.input_b_shift = addp.input_b_shift;
            B.qadd1.output_requantizer.multiplier = addp.output_multiplier;
            B.qadd1.output_requantizer.shift = addp.output_shift;
            B.qadd1.output_requantizer.zero_point = static_cast<int8_t>(p_skip1.zero_point);
            B.qadd1.output_requantizer.qmin = -128;
            B.qadd1.output_requantizer.qmax = 127;
        }

        // FFN.
        B.qff1.weights = B.qw_ff1;
        B.qff1.biases  = B.qb_ff1;
        B.qff1.input_zero_point = static_cast<int8_t>(p_ln2.zero_point);
        B.qff1.requantizer = buildRequantizer<int8_t>(p_ln2.scale, w1_scale, p_ff1.scale,
                                                      p_ff1.zero_point, p_ff1.zero_point, 127);
        B.qff2.weights = B.qw_ff2;
        B.qff2.biases  = B.qb_ff2;
        B.qff2.input_zero_point = static_cast<int8_t>(p_ff1.zero_point);
        B.qff2.requantizer = buildRequantizer<int8_t>(p_ff1.scale, w2_scale, p_ff2.scale,
                                                      p_ff2.zero_point, -128, 127);

        // QAdd 2: skip1 + ff2 -> block output.
        {
            const auto addp = buildQAddParams(p_skip1.scale, p_ff2.scale, p_out.scale);
            B.qadd2.input_a_zero_point = static_cast<int8_t>(p_skip1.zero_point);
            B.qadd2.input_b_zero_point = static_cast<int8_t>(p_ff2.zero_point);
            B.qadd2.left_shift = addp.left_shift;
            B.qadd2.input_a_multiplier = addp.input_a_multiplier;
            B.qadd2.input_a_shift = addp.input_a_shift;
            B.qadd2.input_b_multiplier = addp.input_b_multiplier;
            B.qadd2.input_b_shift = addp.input_b_shift;
            B.qadd2.output_requantizer.multiplier = addp.output_multiplier;
            B.qadd2.output_requantizer.shift = addp.output_shift;
            B.qadd2.output_requantizer.zero_point = static_cast<int8_t>(p_out.zero_point);
            B.qadd2.output_requantizer.qmin = -128;
            B.qadd2.output_requantizer.qmax = 127;
        }
    }

    const AffineParams p_final = qblocks[NUM_BLOCKS - 1].p_out;

    // ----- int8 forward + parity check.
    int8_t q_emb[XE_SIZE], q_x[XE_SIZE], q_next[XE_SIZE];
    int8_t q_ln1[XE_SIZE], q_attn_out[XE_SIZE], q_skip1[XE_SIZE];
    int8_t q_ln2[XE_SIZE], q_ff1[XF_SIZE], q_ff2[XE_SIZE];
    int8_t q_scr[XE_SIZE], k_scr[XE_SIZE], v_scr[XE_SIZE];
    int8_t score_scr[SS_SIZE], attn_scr[SS_SIZE];
    float deq_out[XE_SIZE];

    std::vector<std::vector<int8_t>> all_q_out(NUM_INPUTS,
                                               std::vector<int8_t>(XE_SIZE));
    float worst_err = 0.0f;
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        qembed.forward(tokens[s].data(), S, q_emb);
        qpos.forward(q_emb, q_x);

        for (std::size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            const int8_t ff1_zp = qblocks[b].qff1.requantizer.zero_point;
            runBlockQ(qblocks[b], q_x, q_next,
                      q_ln1, q_attn_out, q_skip1, q_ln2, q_ff1, q_ff2,
                      q_scr, k_scr, v_scr, score_scr, attn_scr, ff1_zp);
            std::memcpy(q_x, q_next, sizeof(int8_t) * XE_SIZE);
        }

        for (std::size_t i = 0; i < XE_SIZE; ++i) all_q_out[s][i] = q_x[i];

        dequantizeBuffer<int8_t>(q_x, deq_out, XE_SIZE,
                                 p_final.scale, p_final.zero_point);
        const float err = maxAbsDiff(deq_out, float_outputs[s].data(), XE_SIZE);
        if (err > worst_err) worst_err = err;

        if (s == 0)
        {
            std::FILE* csv = std::fopen("transformer_encoder_stack_softmax_int8.csv", "w");
            std::fprintf(csv, "index,float,int8\n");
            for (std::size_t i = 0; i < XE_SIZE; ++i)
                std::fprintf(csv, "%zu,%.6f,%.6f\n", i,
                             float_outputs[0][i], deq_out[i]);
            std::fclose(csv);
        }
    }

    if (golden_mode)
    {
        std::printf("# transformer_encoder_stack_softmax_int8 golden output\n");
        std::printf("# blocks=%zu samples=%zu cells=%zu\n",
                    static_cast<size_t>(NUM_BLOCKS),
                    static_cast<size_t>(NUM_INPUTS),
                    static_cast<size_t>(XE_SIZE));
        for (std::size_t s = 0; s < NUM_INPUTS; ++s)
        {
            std::printf("sample %zu:", s);
            for (std::size_t i = 0; i < XE_SIZE; ++i)
                std::printf(" %d", static_cast<int>(all_q_out[s][i]));
            std::printf("\n");
        }
        return 0;
    }

    const float out_range = obs_blkout[NUM_BLOCKS - 1].max_value -
                            obs_blkout[NUM_BLOCKS - 1].min_value;

    std::printf("Transformer encoder STACK (softmax attention) int8 vs float\n");
    std::printf("  vocab=%zu  seq=%zu  embed=%zu  ffn=%zu  blocks=%zu\n",
                static_cast<size_t>(VOCAB), static_cast<size_t>(S),
                static_cast<size_t>(E), static_cast<size_t>(F),
                static_cast<size_t>(NUM_BLOCKS));
    std::printf("  embed   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_emb.min_value, obs_emb.max_value, p_emb.scale, p_emb.zero_point);
    std::printf("  stk in  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_x0.min_value, obs_x0.max_value, p_x0.scale, p_x0.zero_point);
    std::printf("  output  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                obs_blkout[NUM_BLOCKS - 1].min_value,
                obs_blkout[NUM_BLOCKS - 1].max_value,
                p_final.scale, p_final.zero_point);
    std::printf("  worst stack max-abs err: %.5f   (%.1f%% of output range)\n",
                worst_err, 100.0f * worst_err / (out_range + 1e-6f));

    // Softmax attention compounds more int8 stages (score grid + exp LUT +
    // normalize) than linear attention, so the noise floor is looser.
    const float tol = 0.50f * out_range;
    if (worst_err > tol)
    {
        std::printf("FAIL: error %.5f > tolerance %.5f\n", worst_err, tol);
        return 1;
    }
    std::printf("PASS (tolerance %.5f, 50%% of range)\n", tol);
    return 0;
}
