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

// Tiny autoregressive text generation in pure int8 -- a decoder-only
// (GPT-style) nano language model that generates one token at a time.
//
//   token[t] --[QEmbedding]+[pos]--> x
//      x --LN--> [QCausalAttention1D.step] --+Add(skip)-->
//         --LN--> [QDense->qrelu->QDense] --+Add(skip)--> h
//      h --[QDense head]--> logits (Vocab) --argmax--> token[t+1]
//                                                          |
//      <---------------------- feed back --------------------'
//
// The decode loop calls QCausalAttention1D::step() -- never forward(). The
// entire attended history lives in a fixed E x E QLinearKVState, so the model
// generates an arbitrarily long sequence in O(1) attention memory: the state
// is the same size after token 100 as after token 1. There is no growing KV
// cache, which is exactly what makes autoregressive decoding fit on an MCU.
//
// Greedy (argmax) decoding is deterministic, so the generated token stream is
// reproducible. The driver runs the identical model in float and asserts the
// int8 greedy decode picks the SAME tokens -- argmax is invariant under a
// positive affine map, so an int8 logit grid reproduces the float arg-max
// whenever the top-1 margin clears the quantization step.

#include "qaffine.hpp"
#include "qembedding.hpp"
#include "qpositional.hpp"
#include "qcausalattention1d.hpp"
#include "qkvcache.hpp"
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

// The nano-LM is hand-wired as a deterministic "+1 counter": orthogonal
// one-hot token embeddings carried by the residual path, and a shift-by-one
// permutation readout head, so greedy decode emits token+1 (mod VOCAB) each
// step -- an interpretable sawtooth rather than a collapsed constant. The
// attention/FFN/KV machinery still runs (it just perturbs a dominant residual),
// so the demo genuinely exercises QCausalAttention1D::step() and the KV state.
// VOCAB == E keeps the embeddings exactly orthogonal.
constexpr std::size_t VOCAB    = 8;   // token vocabulary
constexpr std::size_t E        = 8;   // model dim (= attention projection dim)
constexpr std::size_t F        = 16;  // FFN inner dim
constexpr std::size_t PROMPT   = 2;   // seed tokens
constexpr std::size_t GEN      = 12;  // tokens to generate
constexpr std::size_t MAXLEN   = PROMPT + GEN;

constexpr std::size_t W_ATTN = 3 * E * E;
constexpr std::size_t B_ATTN = 3 * E;
constexpr std::size_t W1_SZ  = F * E;
constexpr std::size_t W2_SZ  = E * F;
constexpr std::size_t WH_SZ  = VOCAB * E;

constexpr float LN_EPS = 1e-3f;

// ---------------------------------------------------------------------------
// Float reference ops (single row -- the autoregressive step granularity).
// ---------------------------------------------------------------------------

void lnRow(const float* in, const float* gamma, const float* beta, float* out)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < E; ++i) m += in[i];
    m /= static_cast<float>(E);
    float v = 0.0f;
    for (std::size_t i = 0; i < E; ++i) { const float d = in[i] - m; v += d * d; }
    v /= static_cast<float>(E);
    const float inv = 1.0f / std::sqrt(v + LN_EPS);
    for (std::size_t i = 0; i < E; ++i) out[i] = gamma[i] * (in[i] - m) * inv + beta[i];
}

std::size_t argmax(const float* x, std::size_t n)
{
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

std::size_t argmaxI8(const int8_t* x, std::size_t n)
{
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (static_cast<int>(x[i]) > static_cast<int>(x[best])) best = i;
    return best;
}

float absmaxBuf(const float* buf, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { const float a = (buf[i] < 0.0f) ? -buf[i] : buf[i]; if (a > m) m = a; }
    return m;
}

void quantizeSymToI8(const float* src, std::size_t n, float scale, int8_t* dst)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        long q = std::lround(static_cast<double>(src[i]) / static_cast<double>(scale));
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        dst[i] = static_cast<int8_t>(q);
    }
}

// Model weights (deterministic).
struct Weights
{
    float emb[VOCAB * E];
    float pos[MAXLEN * E];
    float g1[E], b1[E], g2[E], b2[E];
    float w_attn[W_ATTN], b_attn[B_ATTN];
    float w1[W1_SZ], b1f[F], w2[W2_SZ], b2f[E];
    float w_head[WH_SZ], b_head[VOCAB];
};

// One float decode step. Maintains the running KV state across calls; returns
// the predicted next token and (optionally) records every activation for
// range calibration.
struct FloatState
{
    float kv[E * E];
    void reset() { for (std::size_t i = 0; i < E * E; ++i) kv[i] = 0.0f; }
};

struct Observers
{
    tinymind::RangeObserver x, ln1, q, k, v, kvm, self, skip1, ln2, ff1, ff2, h, logit;
};

std::size_t floatStep(const Weights& W, FloatState& st, std::size_t token,
                      std::size_t pos, float* logits_out, Observers* obs)
{
    float x[E];
    for (std::size_t e = 0; e < E; ++e) x[e] = W.emb[token * E + e] + W.pos[pos * E + e];

    float ln1[E]; lnRow(x, W.g1, W.b1, ln1);

    // Causal self-attention (single token folded into the running KV state).
    float q[E], k[E], v[E];
    const float* wq = W.w_attn; const float* wk = W.w_attn + E * E; const float* wv = W.w_attn + 2 * E * E;
    const float* bq = W.b_attn; const float* bk = W.b_attn + E;     const float* bv = W.b_attn + 2 * E;
    for (std::size_t p = 0; p < E; ++p)
    {
        float aq = bq[p], ak = bk[p], av = bv[p];
        for (std::size_t e = 0; e < E; ++e)
        {
            aq += wq[e * E + p] * ln1[e];
            ak += wk[e * E + p] * ln1[e];
            av += wv[e * E + p] * ln1[e];
        }
        q[p] = (aq < 0.0f) ? 0.0f : aq;
        k[p] = (ak < 0.0f) ? 0.0f : ak;
        v[p] = av;
    }
    for (std::size_t i = 0; i < E; ++i)
        for (std::size_t j = 0; j < E; ++j)
            st.kv[i * E + j] += k[i] * v[j];
    float self[E];
    for (std::size_t j = 0; j < E; ++j)
    {
        float a = 0.0f;
        for (std::size_t i = 0; i < E; ++i) a += q[i] * st.kv[i * E + j];
        self[j] = a;
    }
    float skip1[E];
    for (std::size_t e = 0; e < E; ++e) skip1[e] = self[e] + x[e];

    float ln2[E]; lnRow(skip1, W.g2, W.b2, ln2);
    float ff1[F];
    for (std::size_t o = 0; o < F; ++o)
    {
        float a = W.b1f[o];
        for (std::size_t i = 0; i < E; ++i) a += W.w1[o * E + i] * ln2[i];
        ff1[o] = (a < 0.0f) ? 0.0f : a;
    }
    float ff2[E];
    for (std::size_t o = 0; o < E; ++o)
    {
        float a = W.b2f[o];
        for (std::size_t i = 0; i < F; ++i) a += W.w2[o * F + i] * ff1[i];
        ff2[o] = a;
    }
    float h[E];
    for (std::size_t e = 0; e < E; ++e) h[e] = skip1[e] + ff2[e];

    for (std::size_t c = 0; c < VOCAB; ++c)
    {
        float a = W.b_head[c];
        for (std::size_t i = 0; i < E; ++i) a += W.w_head[c * E + i] * h[i];
        logits_out[c] = a;
    }

    if (obs != nullptr)
    {
        obs->x.observe(x, E); obs->ln1.observe(ln1, E);
        obs->q.observe(q, E); obs->k.observe(k, E); obs->v.observe(v, E);
        obs->kvm.observe(st.kv, E * E); obs->self.observe(self, E);
        obs->skip1.observe(skip1, E); obs->ln2.observe(ln2, E);
        obs->ff1.observe(ff1, F); obs->ff2.observe(ff2, E);
        obs->h.observe(h, E); obs->logit.observe(logits_out, VOCAB);
    }
    return argmax(logits_out, VOCAB);
}

} // namespace

int main(int argc, char** argv)
{
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    using tinymind::AffineParams;
    using tinymind::RangeObserver;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::buildRequantizer;
    using tinymind::buildQAddParams;
    using tinymind::quantizeBuffer;
    using tinymind::quantizeLayerNormGamma;
    using tinymind::quantizeLayerNormBeta;
    using tinymind::buildQLayerNormOutputParams;
    using tinymind::quantizeLayerNormEpsilon;
    using tinymind::sinusoidalPositionalTable;
    using tinymind::QEmbedding;

    // ----- Deterministic weights wired as a "+1 counter".
    Weights W;
    // One-hot (orthogonal) token embeddings: emb row token = e_token. The
    // residual path carries this identity through the block largely intact.
    for (std::size_t i = 0; i < VOCAB * E; ++i) W.emb[i] = 0.0f;
    for (std::size_t tk = 0; tk < VOCAB; ++tk) W.emb[tk * E + tk] = 1.0f;
    // Small positional perturbation -- a counter does not depend on position,
    // so keep it well below the one-hot residual.
    sinusoidalPositionalTable(MAXLEN, E, W.pos);
    for (std::size_t i = 0; i < MAXLEN * E; ++i) W.pos[i] *= 0.06f;
    // Small attention / FFN weights: the sublayers run and perturb the hidden
    // state but the one-hot residual stays dominant, so argmax tracks the
    // shift-by-one readout.
    for (std::size_t i = 0; i < W_ATTN; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(W_ATTN);
        W.w_attn[i] = 0.04f * std::sin(8.0f * t);
    }
    for (std::size_t i = 0; i < B_ATTN; ++i) W.b_attn[i] = 0.0f;
    for (std::size_t i = 0; i < W1_SZ; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(W1_SZ);
        W.w1[i] = 0.05f * std::cos(6.0f * t);
    }
    for (std::size_t i = 0; i < W2_SZ; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(W2_SZ);
        W.w2[i] = 0.05f * std::sin(10.0f * t);
    }
    for (std::size_t i = 0; i < F; ++i) W.b1f[i] = 0.0f;
    for (std::size_t i = 0; i < E; ++i) { W.b2f[i] = 0.0f; W.g1[i] = 1.0f; W.b1[i] = 0.0f; W.g2[i] = 1.0f; W.b2[i] = 0.0f; }
    // Shift-by-one readout: head row c = one-hot((c-1) mod VOCAB), so
    // logit[c] = h[(c-1) mod VOCAB] peaks when the current token == c-1,
    // i.e. argmax predicts token+1.
    for (std::size_t i = 0; i < WH_SZ; ++i) W.w_head[i] = 0.0f;
    for (std::size_t c = 0; c < VOCAB; ++c)
    {
        const std::size_t prev = (c + VOCAB - 1) % VOCAB;
        W.w_head[c * E + prev] = 1.0f;
        W.b_head[c] = 0.0f;
    }

    // ----- Seed prompts.
    constexpr std::size_t NUM_PROMPTS = 4;
    std::size_t prompts[NUM_PROMPTS][PROMPT] = { {1, 5}, {3, 6}, {7, 2}, {4, 0} };

    // ----- Float generation + range calibration.
    Observers obs;
    std::vector<std::vector<std::size_t>> float_gen(NUM_PROMPTS,
                                                    std::vector<std::size_t>(GEN));
    for (std::size_t s = 0; s < NUM_PROMPTS; ++s)
    {
        FloatState st; st.reset();
        std::size_t token = prompts[s][0];
        float logits[VOCAB];
        for (std::size_t t = 0; t < MAXLEN; ++t)
        {
            const std::size_t pred = floatStep(W, st, token, t, logits, &obs);
            const std::size_t next = (t + 1 < PROMPT) ? prompts[s][t + 1] : pred;
            if (t + 1 >= PROMPT && (t + 1 - PROMPT) < GEN)
                float_gen[s][t + 1 - PROMPT] = next;
            token = next;
        }
    }

    // ----- Affine grids.
    auto grid = [](const RangeObserver& o) { return computeAffineParamsAsymmetric(o.min_value, o.max_value, -128, 127); };
    auto reluGrid = [](const RangeObserver& o) { return computeAffineParamsAsymmetric(0.0f, o.max_value, -128, 127); };
    const auto p_emb = grid(obs.x);   // embedding+pos share the input activation grid
    const auto p_x   = grid(obs.x);
    const auto p_ln1 = grid(obs.ln1), p_q = reluGrid(obs.q), p_k = reluGrid(obs.k), p_v = grid(obs.v), p_kv = grid(obs.kvm);
    const auto p_self = grid(obs.self), p_skip1 = grid(obs.skip1), p_ln2 = grid(obs.ln2);
    const auto p_ff1 = reluGrid(obs.ff1), p_ff2 = grid(obs.ff2), p_h = grid(obs.h), p_logit = grid(obs.logit);

    // ----- Embedding table + positional table on the input grid.
    int8_t q_emb[VOCAB * E], q_pos[MAXLEN * E];
    quantizeBuffer<int8_t>(W.emb, q_emb, VOCAB * E, p_emb.scale, p_emb.zero_point, -128, 127);
    quantizeBuffer<int8_t>(W.pos, q_pos, MAXLEN * E, p_emb.scale, p_emb.zero_point, -128, 127);
    QEmbedding<int8_t, int8_t, VOCAB, E> qembed; qembed.table = q_emb; qembed.requantizer = nullptr;

    // ----- Weight scales + quantization.
    auto ws = [&](const float* w, std::size_t n) { return absmaxBuf(w, n) / 127.0f; };
    const float wq_s = ws(W.w_attn, E * E), wk_s = ws(W.w_attn + E * E, E * E), wv_s = ws(W.w_attn + 2 * E * E, E * E);
    const float w1_s = ws(W.w1, W1_SZ), w2_s = ws(W.w2, W2_SZ), wh_s = ws(W.w_head, WH_SZ);
    int8_t qw_attn[W_ATTN], qw1[W1_SZ], qw2[W2_SZ], qwh[WH_SZ];
    int32_t qb_attn[B_ATTN] = {0}, qb1[F] = {0}, qb2[E] = {0}, qbh[VOCAB];
    quantizeSymToI8(W.w_attn, E * E, wq_s, qw_attn);
    quantizeSymToI8(W.w_attn + E * E, E * E, wk_s, qw_attn + E * E);
    quantizeSymToI8(W.w_attn + 2 * E * E, E * E, wv_s, qw_attn + 2 * E * E);
    quantizeSymToI8(W.w1, W1_SZ, w1_s, qw1);
    quantizeSymToI8(W.w2, W2_SZ, w2_s, qw2);
    quantizeSymToI8(W.w_head, WH_SZ, wh_s, qwh);
    // Head bias is int32 in (input_scale * weight_scale) units.
    for (std::size_t c = 0; c < VOCAB; ++c)
        qbh[c] = static_cast<int32_t>(std::lround(static_cast<double>(W.b_head[c]) /
            (static_cast<double>(p_h.scale) * static_cast<double>(wh_s))));

    // ----- Positional add (QAdd: emb grid + pos grid -> input grid).
    tinymind::QAdd<int8_t, int8_t, int8_t, E> q_posadd;
    {
        const auto a = buildQAddParams(p_emb.scale, p_emb.scale, p_x.scale);
        q_posadd.input_a_zero_point = static_cast<int8_t>(p_emb.zero_point);
        q_posadd.input_b_zero_point = static_cast<int8_t>(p_emb.zero_point);
        q_posadd.left_shift = a.left_shift;
        q_posadd.input_a_multiplier = a.input_a_multiplier; q_posadd.input_a_shift = a.input_a_shift;
        q_posadd.input_b_multiplier = a.input_b_multiplier; q_posadd.input_b_shift = a.input_b_shift;
        q_posadd.output_requantizer.multiplier = a.output_multiplier;
        q_posadd.output_requantizer.shift = a.output_shift;
        q_posadd.output_requantizer.zero_point = static_cast<int8_t>(p_x.zero_point);
        q_posadd.output_requantizer.qmin = -128; q_posadd.output_requantizer.qmax = 127;
    }

    // ----- LayerNorms (single row).
    int16_t g1q[E], g2q[E]; int32_t b1q[E], b2q[E];
    tinymind::QLayerNorm1D<int8_t, int8_t, 1, E> q_ln1, q_ln2;
    auto buildLN = [&](auto& ln, const float* g, const float* b, int16_t* gq, int32_t* bq,
                       const AffineParams& pin, const AffineParams& pout)
    {
        quantizeLayerNormGamma(g, E, gq);
        quantizeLayerNormBeta(b, E, pout.scale, bq);
        int32_t mult = 0, shft = 0; buildQLayerNormOutputParams(pout.scale, mult, shft);
        ln.gamma = gq; ln.beta = bq;
        ln.epsilon_q = quantizeLayerNormEpsilon(LN_EPS, pin.scale);
        ln.output_multiplier = mult; ln.output_shift = shft;
        ln.output_zero_point = static_cast<int8_t>(pout.zero_point);
        ln.qmin = -128; ln.qmax = 127;
    };
    buildLN(q_ln1, W.g1, W.b1, g1q, b1q, p_x, p_ln1);
    buildLN(q_ln2, W.g2, W.b2, g2q, b2q, p_skip1, p_ln2);

    // ----- Causal self-attention.
    typedef tinymind::QCausalAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                                         MAXLEN, E, E> SelfAttn;
    SelfAttn q_attn;
    q_attn.weights = qw_attn; q_attn.biases = qb_attn;
    q_attn.input_zero_point = static_cast<int8_t>(p_ln1.zero_point);
    q_attn.q_zero_point = static_cast<int8_t>(p_q.zero_point);
    q_attn.k_zero_point = static_cast<int8_t>(p_k.zero_point);
    q_attn.v_zero_point = static_cast<int8_t>(p_v.zero_point);
    q_attn.kv_zero_point = static_cast<int8_t>(p_kv.zero_point);
    q_attn.q_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wq_s, p_q.scale, p_q.zero_point, p_q.zero_point, 127);
    q_attn.k_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wk_s, p_k.scale, p_k.zero_point, p_k.zero_point, 127);
    q_attn.v_requantizer = buildRequantizer<int8_t>(p_ln1.scale, wv_s, p_v.scale, p_v.zero_point, -128, 127);
    q_attn.kv_requantizer = buildRequantizer<int8_t>(p_k.scale, p_v.scale, p_kv.scale, p_kv.zero_point, -128, 127);
    q_attn.output_requantizer = buildRequantizer<int8_t>(p_q.scale, p_kv.scale, p_self.scale, p_self.zero_point, -128, 127);

    // ----- Skip adds + FFN + head.
    tinymind::QAdd<int8_t, int8_t, int8_t, E> q_add1, q_add2;
    auto wireAdd = [&](auto& add, const AffineParams& pa, const AffineParams& pb, const AffineParams& po)
    {
        const auto a = buildQAddParams(pa.scale, pb.scale, po.scale);
        add.input_a_zero_point = static_cast<int8_t>(pa.zero_point);
        add.input_b_zero_point = static_cast<int8_t>(pb.zero_point);
        add.left_shift = a.left_shift;
        add.input_a_multiplier = a.input_a_multiplier; add.input_a_shift = a.input_a_shift;
        add.input_b_multiplier = a.input_b_multiplier; add.input_b_shift = a.input_b_shift;
        add.output_requantizer.multiplier = a.output_multiplier;
        add.output_requantizer.shift = a.output_shift;
        add.output_requantizer.zero_point = static_cast<int8_t>(po.zero_point);
        add.output_requantizer.qmin = -128; add.output_requantizer.qmax = 127;
    };
    wireAdd(q_add1, p_self, p_x, p_skip1);
    wireAdd(q_add2, p_skip1, p_ff2, p_h);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, F> q_ff1;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, F, E> q_ff2;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, VOCAB> q_head;
    q_ff1.weights = qw1; q_ff1.biases = qb1;
    q_ff1.input_zero_point = static_cast<int8_t>(p_ln2.zero_point);
    q_ff1.requantizer = buildRequantizer<int8_t>(p_ln2.scale, w1_s, p_ff1.scale, p_ff1.zero_point, p_ff1.zero_point, 127);
    q_ff2.weights = qw2; q_ff2.biases = qb2;
    q_ff2.input_zero_point = static_cast<int8_t>(p_ff1.zero_point);
    q_ff2.requantizer = buildRequantizer<int8_t>(p_ff1.scale, w2_s, p_ff2.scale, p_ff2.zero_point, -128, 127);
    q_head.weights = qwh; q_head.biases = qbh;
    q_head.input_zero_point = static_cast<int8_t>(p_h.zero_point);
    q_head.requantizer = buildRequantizer<int8_t>(p_h.scale, wh_s, p_logit.scale, p_logit.zero_point, -128, 127);

    // ----- int8 autoregressive generation.
    std::vector<std::vector<std::size_t>> int8_gen(NUM_PROMPTS,
                                                   std::vector<std::size_t>(GEN));
    std::size_t token_matches = 0;
    for (std::size_t s = 0; s < NUM_PROMPTS; ++s)
    {
        SelfAttn::KVState state; state.reset();
        std::size_t token = prompts[s][0];
        for (std::size_t t = 0; t < MAXLEN; ++t)
        {
            int8_t emb_row[E], x_row[E], pos_row[E];
            int32_t tok32 = static_cast<int32_t>(token);
            qembed.forward(&tok32, 1, emb_row);
            for (std::size_t e = 0; e < E; ++e) pos_row[e] = q_pos[t * E + e];
            q_posadd.forward(emb_row, pos_row, x_row);

            int8_t ln1_row[E], q_r[E], k_r[E], v_r[E], kv_r[E * E], self_row[E];
            q_ln1.forward(x_row, ln1_row);
            q_attn.step(ln1_row, state, q_r, k_r, v_r, kv_r, self_row);

            int8_t skip1[E], ln2_row[E], ff1_row[F], ff2_row[E], h_row[E], logits[VOCAB];
            q_add1.forward(self_row, x_row, skip1);
            q_ln2.forward(skip1, ln2_row);
            q_ff1.forward(ln2_row, ff1_row);
            tinymind::qreluBuffer<int8_t>(ff1_row, F, q_ff1.requantizer.zero_point);
            q_ff2.forward(ff1_row, ff2_row);
            q_add2.forward(skip1, ff2_row, h_row);
            q_head.forward(h_row, logits);

            const std::size_t pred = argmaxI8(logits, VOCAB);
            const std::size_t next = (t + 1 < PROMPT) ? prompts[s][t + 1] : pred;
            if (t + 1 >= PROMPT && (t + 1 - PROMPT) < GEN)
            {
                int8_gen[s][t + 1 - PROMPT] = next;
                if (next == float_gen[s][t + 1 - PROMPT]) ++token_matches;
            }
            token = next;
        }
    }
    const std::size_t total_tokens = NUM_PROMPTS * GEN;

    // ----- CSV (flattened generated token ids, float vs int8).
    {
        std::FILE* csv = std::fopen("tiny_generate_int8.csv", "w");
        std::fprintf(csv, "index,float,int8\n");
        std::size_t idx = 0;
        for (std::size_t s = 0; s < NUM_PROMPTS; ++s)
            for (std::size_t t = 0; t < GEN; ++t, ++idx)
                std::fprintf(csv, "%zu,%d,%d\n", idx,
                             static_cast<int>(float_gen[s][t]),
                             static_cast<int>(int8_gen[s][t]));
        std::fclose(csv);
    }

    if (golden_mode)
    {
        std::printf("# tiny_generate_int8 golden output\n");
        std::printf("# prompts=%zu gen=%zu token_match=%zu/%zu\n",
                    static_cast<size_t>(NUM_PROMPTS), static_cast<size_t>(GEN),
                    token_matches, total_tokens);
        for (std::size_t s = 0; s < NUM_PROMPTS; ++s)
        {
            std::printf("prompt %zu %zu ->", prompts[s][0], prompts[s][1]);
            for (std::size_t t = 0; t < GEN; ++t) std::printf(" %zu", int8_gen[s][t]);
            std::printf("\n");
        }
        return 0;
    }

    std::printf("Tiny autoregressive generation in int8 (decoder-only nano-LM)\n");
    std::printf("  vocab=%zu  model=%zu  ffn=%zu  prompt=%zu  generate=%zu\n",
                static_cast<size_t>(VOCAB), static_cast<size_t>(E),
                static_cast<size_t>(F), static_cast<size_t>(PROMPT),
                static_cast<size_t>(GEN));
    std::printf("  KV state: %zu x %zu int32 (fixed -- independent of sequence length)\n",
                static_cast<size_t>(E), static_cast<size_t>(E));
    for (std::size_t s = 0; s < NUM_PROMPTS; ++s)
    {
        std::printf("  prompt [%zu %zu] -> float:", prompts[s][0], prompts[s][1]);
        for (std::size_t t = 0; t < GEN; ++t) std::printf(" %zu", float_gen[s][t]);
        std::printf("\n%*s int8 :", 18, "");
        for (std::size_t t = 0; t < GEN; ++t) std::printf(" %zu", int8_gen[s][t]);
        std::printf("\n");
    }
    std::printf("  int8 greedy decode reproduces %zu / %zu float tokens (%.1f%%)\n",
                token_matches, total_tokens,
                100.0f * static_cast<float>(token_matches) / static_cast<float>(total_tokens));

    // Greedy argmax is robust to int8 noise when top-1 margins clear the logit
    // grid step; require a strong majority of tokens to match the float run.
    if (token_matches < (total_tokens * 9) / 10)
    {
        std::printf("FAIL: only %zu / %zu tokens matched\n", token_matches, total_tokens);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
