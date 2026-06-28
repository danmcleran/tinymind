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

// Int8 encoder-decoder (seq2seq) transformer. Finishes the transformer arc:
// the encoder family (examples/transformer_encoder_*) produced a memory
// tensor; this example consumes it from a decoder stack.
//
//   source ids (SEnc)                 target ids (SDec)
//        |                                  |
//   [QEmbedding]+[QPositional]        [QEmbedding]+[QPositional]
//        |                                  |
//   ENCODER block:                    DECODER block:
//     LN -> LinAttn -> Add              LN -> CausalLinAttn -> Add   (self)
//     LN -> FFN -> Add                  LN -> CrossLinAttn  -> Add   (to memory)
//        |                                LN -> FFN -> Add
//   memory (SEnc, E) ------------------>  |
//                                       output (SDec, E)
//
// Linear (ReLU-kernel) attention everywhere keeps the whole pipeline
// freestanding-portable -- no softmax/exp LUT. The decoder self-attention is
// CAUSAL: position t sees only s <= t. With linear attention that causal mask
// collapses into a fixed E x E running KV state (QLinearKVState), so the
// autoregressive decode is O(1) memory per token -- there is no growing cache.
//
// The headline check is that the int8 causal self-attention produces the SAME
// int8 stream whether it is run as one full-sequence forward() (training-time
// teacher forcing) or token-by-token via step() with the running KV state
// (inference-time autoregressive decode). That equivalence is what lets a
// model trained on full sequences deploy as a streaming decoder.

#include "qaffine.hpp"
#include "qembedding.hpp"
#include "qpositional.hpp"
#include "qattention1d.hpp"
#include "qcausalattention1d.hpp"
#include "qcrossattention.hpp"
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

constexpr std::size_t VOCAB = 16;  // shared source/target vocabulary
constexpr std::size_t SENC  = 6;   // source (encoder) sequence length
constexpr std::size_t SDEC  = 6;   // target (decoder) sequence length
constexpr std::size_t E     = 8;   // model dim (= attention projection dim)
constexpr std::size_t F     = 16;  // FFN inner dim

constexpr std::size_t ENC_SZ  = SENC * E;
constexpr std::size_t DEC_SZ  = SDEC * E;
constexpr std::size_t W_ATTN  = 3 * E * E;
constexpr std::size_t B_ATTN  = 3 * E;
constexpr std::size_t W1_SZ   = F * E;
constexpr std::size_t W2_SZ   = E * F;

constexpr float LN_EPS = 1e-3f;

// ---------------------------------------------------------------------------
// Float reference ops.
// ---------------------------------------------------------------------------

void floatLayerNorm(const float* in, std::size_t S, const float* gamma,
                    const float* beta, float* out)
{
    for (std::size_t t = 0; t < S; ++t)
    {
        const float* row = in + t * E;
        float* o = out + t * E;
        float m = 0.0f;
        for (std::size_t i = 0; i < E; ++i) m += row[i];
        m /= static_cast<float>(E);
        float v = 0.0f;
        for (std::size_t i = 0; i < E; ++i) { const float d = row[i] - m; v += d * d; }
        v /= static_cast<float>(E);
        const float inv = 1.0f / std::sqrt(v + LN_EPS);
        for (std::size_t i = 0; i < E; ++i) o[i] = gamma[i] * (row[i] - m) * inv + beta[i];
    }
}

void floatDense(const float* in, std::size_t S, const float* w, const float* b,
                std::size_t in_dim, std::size_t out_dim, float* out)
{
    for (std::size_t t = 0; t < S; ++t)
        for (std::size_t o = 0; o < out_dim; ++o)
        {
            float a = b[o];
            for (std::size_t i = 0; i < in_dim; ++i) a += w[o * in_dim + i] * in[t * in_dim + i];
            out[t * out_dim + o] = a;
        }
}

void floatRelu(float* x, std::size_t n) { for (std::size_t i = 0; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f; }
void floatAdd(const float* a, const float* b, float* y, std::size_t n) { for (std::size_t i = 0; i < n; ++i) y[i] = a[i] + b[i]; }

// Bidirectional linear (ReLU-kernel) attention -- the encoder self-attention.
void floatLinearAttention(const float* x, std::size_t S, const float* w,
                          const float* b, float* y)
{
    std::vector<float> q(S * E), k(S * E), v(S * E);
    float kv[E * E];
    const float* wq = w; const float* wk = w + E * E; const float* wv = w + 2 * E * E;
    const float* bq = b; const float* bk = b + E;     const float* bv = b + 2 * E;
    for (std::size_t t = 0; t < S; ++t)
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
    for (std::size_t i = 0; i < E; ++i)
        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t t = 0; t < S; ++t) a += k[t * E + i] * v[t * E + j];
            kv[i * E + j] = a;
        }
    for (std::size_t t = 0; t < S; ++t)
        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < E; ++i) a += q[t * E + i] * kv[i * E + j];
            y[t * E + j] = a;
        }
}

// Causal linear attention -- the decoder self-attention. y[t] attends to s<=t.
void floatCausalLinearAttention(const float* x, std::size_t S, const float* w,
                                const float* b, float* y)
{
    std::vector<float> q(S * E), k(S * E), v(S * E);
    const float* wq = w; const float* wk = w + E * E; const float* wv = w + 2 * E * E;
    const float* bq = b; const float* bk = b + E;     const float* bv = b + 2 * E;
    for (std::size_t t = 0; t < S; ++t)
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
    float kv[E * E];
    for (std::size_t i = 0; i < E * E; ++i) kv[i] = 0.0f;
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t i = 0; i < E; ++i)
            for (std::size_t j = 0; j < E; ++j)
                kv[i * E + j] += k[t * E + i] * v[t * E + j];
        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < E; ++i) a += q[t * E + i] * kv[i * E + j];
            y[t * E + j] = a;
        }
    }
}

// Linear cross-attention -- Q' from decoder rows, K'/V from encoder memory.
void floatCrossLinearAttention(const float* dec, const float* mem,
                               const float* w, const float* b, float* y)
{
    const float* wq = w; const float* wk = w + E * E; const float* wv = w + 2 * E * E;
    const float* bq = b; const float* bk = b + E;     const float* bv = b + 2 * E;
    float k[SENC * E], v[SENC * E], kv[E * E];
    for (std::size_t s = 0; s < SENC; ++s)
        for (std::size_t p = 0; p < E; ++p)
        {
            float ak = bk[p], av = bv[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                ak += wk[e * E + p] * mem[s * E + e];
                av += wv[e * E + p] * mem[s * E + e];
            }
            k[s * E + p] = (ak < 0.0f) ? 0.0f : ak;
            v[s * E + p] = av;
        }
    for (std::size_t i = 0; i < E * E; ++i) kv[i] = 0.0f;
    for (std::size_t i = 0; i < E; ++i)
        for (std::size_t j = 0; j < E; ++j)
            for (std::size_t s = 0; s < SENC; ++s)
                kv[i * E + j] += k[s * E + i] * v[s * E + j];
    for (std::size_t t = 0; t < SDEC; ++t)
    {
        float q[E];
        for (std::size_t p = 0; p < E; ++p)
        {
            float aq = bq[p];
            for (std::size_t e = 0; e < E; ++e) aq += wq[e * E + p] * dec[t * E + e];
            q[p] = (aq < 0.0f) ? 0.0f : aq;
        }
        for (std::size_t j = 0; j < E; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < E; ++i) a += q[i] * kv[i * E + j];
            y[t * E + j] = a;
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities.
// ---------------------------------------------------------------------------

float absmaxBuf(const float* buf, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { const float a = (buf[i] < 0.0f) ? -buf[i] : buf[i]; if (a > m) m = a; }
    return m;
}

float maxAbsDiff(const float* a, const float* b, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { const float d = std::fabs(a[i] - b[i]); if (d > m) m = d; }
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

} // namespace

int main(int argc, char** argv)
{
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    using tinymind::AffineParams;
    using tinymind::RangeObserver;
    using tinymind::Requantizer;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::buildRequantizer;
    using tinymind::buildQAddParams;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;
    using tinymind::quantizeLayerNormGamma;
    using tinymind::quantizeLayerNormBeta;
    using tinymind::buildQLayerNormOutputParams;
    using tinymind::quantizeLayerNormEpsilon;
    using tinymind::QEmbedding;
    using tinymind::QPositionalEncoding1D;
    using tinymind::sinusoidalPositionalTable;

    // ----- Shared embedding table [VOCAB, E].
    float emb_table[VOCAB * E];
    for (std::size_t i = 0; i < VOCAB * E; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(VOCAB * E);
        emb_table[i] = 0.6f * std::sin(9.0f * t) + 0.1f * std::cos(2.0f * t);
    }

    // ----- Positional tables (source / target).
    float pos_enc[ENC_SZ], pos_dec[DEC_SZ];
    sinusoidalPositionalTable(SENC, E, pos_enc);
    sinusoidalPositionalTable(SDEC, E, pos_dec);

    // ----- Encoder block weights.
    float e_gamma1[E], e_beta1[E], e_gamma2[E], e_beta2[E];
    float e_wattn[W_ATTN], e_battn[B_ATTN];
    float e_wff1[W1_SZ], e_bff1[F], e_wff2[W2_SZ], e_bff2[E];
    // ----- Decoder block weights (self, cross, ffn).
    float d_g1[E], d_b1[E], d_g2[E], d_b2[E], d_g3[E], d_b3[E];
    float d_wself[W_ATTN], d_bself[B_ATTN];
    float d_wcross[W_ATTN], d_bcross[B_ATTN];
    float d_wff1[W1_SZ], d_bff1[F], d_wff2[W2_SZ], d_bff2[E];
    {
        for (std::size_t i = 0; i < W_ATTN; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W_ATTN);
            e_wattn[i]  = 0.18f * std::sin(7.0f * t);
            d_wself[i]  = 0.17f * std::sin(8.0f * t);
            d_wcross[i] = 0.16f * std::cos(6.0f * t);
        }
        for (std::size_t i = 0; i < B_ATTN; ++i) { e_battn[i] = 0.0f; d_bself[i] = 0.0f; d_bcross[i] = 0.0f; }
        for (std::size_t i = 0; i < W1_SZ; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W1_SZ);
            e_wff1[i] = 0.22f * std::cos(5.0f * t);
            d_wff1[i] = 0.21f * std::cos(7.0f * t);
        }
        for (std::size_t i = 0; i < W2_SZ; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W2_SZ);
            e_wff2[i] = 0.20f * std::sin(11.0f * t);
            d_wff2[i] = 0.19f * std::sin(9.0f * t);
        }
        for (std::size_t i = 0; i < F; ++i) { e_bff1[i] = 0.0f; d_bff1[i] = 0.0f; }
        for (std::size_t i = 0; i < E; ++i)
        {
            e_bff2[i] = 0.0f; d_bff2[i] = 0.0f;
            e_gamma1[i] = 1.0f; e_beta1[i] = 0.0f; e_gamma2[i] = 1.0f; e_beta2[i] = 0.0f;
            d_g1[i] = 1.0f; d_b1[i] = 0.0f; d_g2[i] = 1.0f; d_b2[i] = 0.0f; d_g3[i] = 1.0f; d_b3[i] = 0.0f;
        }
    }

    // ----- Synthetic source/target token dataset.
    constexpr std::size_t NUM_INPUTS = 6;
    std::vector<std::vector<int32_t>> src(NUM_INPUTS, std::vector<int32_t>(SENC));
    std::vector<std::vector<int32_t>> tgt(NUM_INPUTS, std::vector<int32_t>(SDEC));
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        for (std::size_t t = 0; t < SENC; ++t) src[s][t] = static_cast<int32_t>((s * SENC + t * 3 + 1) % VOCAB);
        for (std::size_t t = 0; t < SDEC; ++t) tgt[s][t] = static_cast<int32_t>((s * SDEC + t * 5 + 2) % VOCAB);
    }

    // ----- Float forward over the dataset, collecting activation ranges.
    RangeObserver o_emb, o_pos_e, o_pos_d, o_xe, o_xd;
    RangeObserver o_eln1, o_eq, o_ek, o_ev, o_ekv, o_eatt, o_eskip, o_eln2, o_eff1, o_eff2, o_mem;
    RangeObserver o_dln1, o_dsq, o_dsk, o_dsv, o_dskv, o_dself, o_dskip1;
    RangeObserver o_dln2, o_dcq, o_dck, o_dcv, o_dckv, o_dcross, o_dskip2;
    RangeObserver o_dln3, o_dff1, o_dff2, o_dout;
    o_pos_e.observe(pos_enc, ENC_SZ);
    o_pos_d.observe(pos_dec, DEC_SZ);

    std::vector<std::vector<float>> float_out(NUM_INPUTS, std::vector<float>(DEC_SZ));

    auto embedAdd = [&](const std::vector<int32_t>& ids, std::size_t S,
                        const float* pos, float* out)
    {
        for (std::size_t t = 0; t < S; ++t)
        {
            const float* row = emb_table + ids[t] * E;
            for (std::size_t e = 0; e < E; ++e) out[t * E + e] = row[e] + pos[t * E + e];
        }
    };
    auto observeProj = [&](const float* x, std::size_t S, const float* w, const float* b,
                           bool relu_v, RangeObserver& oq, RangeObserver& ok,
                           RangeObserver& ov, RangeObserver& okv)
    {
        std::vector<float> q(S * E), k(S * E), v(S * E);
        float kv[E * E];
        const float* wq = w; const float* wk = w + E * E; const float* wv = w + 2 * E * E;
        const float* bq = b; const float* bk = b + E;     const float* bv = b + 2 * E;
        for (std::size_t t = 0; t < S; ++t)
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
                v[t * E + p] = relu_v ? ((av < 0.0f) ? 0.0f : av) : av;
            }
        for (std::size_t i = 0; i < E; ++i)
            for (std::size_t j = 0; j < E; ++j)
            {
                float a = 0.0f;
                for (std::size_t t = 0; t < S; ++t) a += k[t * E + i] * v[t * E + j];
                kv[i * E + j] = a;
            }
        oq.observe(q.data(), S * E); ok.observe(k.data(), S * E);
        ov.observe(v.data(), S * E); okv.observe(kv, E * E);
    };

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        float xe[ENC_SZ], xd[DEC_SZ];
        embedAdd(src[s], SENC, pos_enc, xe);
        embedAdd(tgt[s], SDEC, pos_dec, xd);
        for (std::size_t t = 0; t < SENC; ++t)
            for (std::size_t e = 0; e < E; ++e) o_emb.observe(emb_table[src[s][t] * E + e]);
        o_xe.observe(xe, ENC_SZ); o_xd.observe(xd, DEC_SZ);

        // Encoder block -> memory.
        float eln1[ENC_SZ], eatt[ENC_SZ], eskip[ENC_SZ], eln2[ENC_SZ];
        float eff1[SENC * F], eff2[ENC_SZ], mem[ENC_SZ];
        floatLayerNorm(xe, SENC, e_gamma1, e_beta1, eln1);
        observeProj(eln1, SENC, e_wattn, e_battn, false, o_eq, o_ek, o_ev, o_ekv);
        floatLinearAttention(eln1, SENC, e_wattn, e_battn, eatt);
        floatAdd(eatt, xe, eskip, ENC_SZ);
        floatLayerNorm(eskip, SENC, e_gamma2, e_beta2, eln2);
        floatDense(eln2, SENC, e_wff1, e_bff1, E, F, eff1);
        floatRelu(eff1, SENC * F);
        floatDense(eff1, SENC, e_wff2, e_bff2, F, E, eff2);
        floatAdd(eskip, eff2, mem, ENC_SZ);
        o_eln1.observe(eln1, ENC_SZ); o_eatt.observe(eatt, ENC_SZ);
        o_eskip.observe(eskip, ENC_SZ); o_eln2.observe(eln2, ENC_SZ);
        o_eff1.observe(eff1, SENC * F); o_eff2.observe(eff2, ENC_SZ);
        o_mem.observe(mem, ENC_SZ);

        // Decoder block.
        float dln1[DEC_SZ], dself[DEC_SZ], dskip1[DEC_SZ];
        float dln2[DEC_SZ], dcross[DEC_SZ], dskip2[DEC_SZ];
        float dln3[DEC_SZ], dff1[SDEC * F], dff2[DEC_SZ], dout[DEC_SZ];
        floatLayerNorm(xd, SDEC, d_g1, d_b1, dln1);
        observeProj(dln1, SDEC, d_wself, d_bself, false, o_dsq, o_dsk, o_dsv, o_dskv);
        floatCausalLinearAttention(dln1, SDEC, d_wself, d_bself, dself);
        floatAdd(dself, xd, dskip1, DEC_SZ);
        floatLayerNorm(dskip1, SDEC, d_g2, d_b2, dln2);
        // Cross projection ranges: Q from dln2, K/V from memory.
        {
            std::vector<float> q(SDEC * E);
            const float* wq = d_wcross; const float* bq = d_bcross;
            for (std::size_t t = 0; t < SDEC; ++t)
                for (std::size_t p = 0; p < E; ++p)
                {
                    float aq = bq[p];
                    for (std::size_t e = 0; e < E; ++e) aq += wq[e * E + p] * dln2[t * E + e];
                    q[t * E + p] = (aq < 0.0f) ? 0.0f : aq;
                }
            o_dcq.observe(q.data(), SDEC * E);
            float k[ENC_SZ], v[ENC_SZ], kv[E * E];
            const float* wk = d_wcross + E * E; const float* wv = d_wcross + 2 * E * E;
            const float* bk = d_bcross + E;     const float* bv = d_bcross + 2 * E;
            for (std::size_t sidx = 0; sidx < SENC; ++sidx)
                for (std::size_t p = 0; p < E; ++p)
                {
                    float ak = bk[p], av = bv[p];
                    for (std::size_t e = 0; e < E; ++e)
                    {
                        ak += wk[e * E + p] * mem[sidx * E + e];
                        av += wv[e * E + p] * mem[sidx * E + e];
                    }
                    k[sidx * E + p] = (ak < 0.0f) ? 0.0f : ak;
                    v[sidx * E + p] = av;
                }
            for (std::size_t i = 0; i < E; ++i)
                for (std::size_t j = 0; j < E; ++j)
                {
                    float a = 0.0f;
                    for (std::size_t sidx = 0; sidx < SENC; ++sidx) a += k[sidx * E + i] * v[sidx * E + j];
                    kv[i * E + j] = a;
                }
            o_dck.observe(k, ENC_SZ); o_dcv.observe(v, ENC_SZ); o_dckv.observe(kv, E * E);
        }
        floatCrossLinearAttention(dln2, mem, d_wcross, d_bcross, dcross);
        floatAdd(dcross, dskip1, dskip2, DEC_SZ);
        floatLayerNorm(dskip2, SDEC, d_g3, d_b3, dln3);
        floatDense(dln3, SDEC, d_wff1, d_bff1, E, F, dff1);
        floatRelu(dff1, SDEC * F);
        floatDense(dff1, SDEC, d_wff2, d_bff2, F, E, dff2);
        floatAdd(dskip2, dff2, dout, DEC_SZ);
        o_dln1.observe(dln1, DEC_SZ); o_dself.observe(dself, DEC_SZ); o_dskip1.observe(dskip1, DEC_SZ);
        o_dln2.observe(dln2, DEC_SZ); o_dcross.observe(dcross, DEC_SZ); o_dskip2.observe(dskip2, DEC_SZ);
        o_dln3.observe(dln3, DEC_SZ); o_dff1.observe(dff1, SDEC * F); o_dff2.observe(dff2, DEC_SZ);
        o_dout.observe(dout, DEC_SZ);

        std::memcpy(float_out[s].data(), dout, sizeof(float) * DEC_SZ);
    }

    // ----- Affine grids.
    auto grid = [](const RangeObserver& o) { return computeAffineParamsAsymmetric(o.min_value, o.max_value, -128, 127); };
    auto reluGrid = [](const RangeObserver& o) { return computeAffineParamsAsymmetric(0.0f, o.max_value, -128, 127); };
    const auto p_emb = grid(o_emb);
    const auto p_pe = grid(o_pos_e), p_pd = grid(o_pos_d);
    const auto p_xe = grid(o_xe), p_xd = grid(o_xd);
    const auto p_eln1 = grid(o_eln1), p_eq = reluGrid(o_eq), p_ek = reluGrid(o_ek), p_ev = grid(o_ev), p_ekv = grid(o_ekv);
    const auto p_eatt = grid(o_eatt), p_eskip = grid(o_eskip), p_eln2 = grid(o_eln2), p_eff1 = reluGrid(o_eff1), p_eff2 = grid(o_eff2);
    const auto p_mem = grid(o_mem);
    const auto p_dln1 = grid(o_dln1), p_dsq = reluGrid(o_dsq), p_dsk = reluGrid(o_dsk), p_dsv = grid(o_dsv), p_dskv = grid(o_dskv);
    const auto p_dself = grid(o_dself), p_dskip1 = grid(o_dskip1);
    const auto p_dln2 = grid(o_dln2), p_dcq = reluGrid(o_dcq), p_dck = reluGrid(o_dck), p_dcv = grid(o_dcv), p_dckv = grid(o_dckv);
    const auto p_dcross = grid(o_dcross), p_dskip2 = grid(o_dskip2);
    const auto p_dln3 = grid(o_dln3), p_dff1 = reluGrid(o_dff1), p_dff2 = grid(o_dff2), p_dout = grid(o_dout);

    // ----- Quantize embedding + positional layers.
    int8_t qemb_table[VOCAB * E];
    quantizeBuffer<int8_t>(emb_table, qemb_table, VOCAB * E, p_emb.scale, p_emb.zero_point, -128, 127);
    QEmbedding<int8_t, int8_t, VOCAB, E> qembed; qembed.table = qemb_table; qembed.requantizer = nullptr;

    int8_t qpos_enc[ENC_SZ], qpos_dec[DEC_SZ];
    quantizeBuffer<int8_t>(pos_enc, qpos_enc, ENC_SZ, p_pe.scale, p_pe.zero_point, -128, 127);
    quantizeBuffer<int8_t>(pos_dec, qpos_dec, DEC_SZ, p_pd.scale, p_pd.zero_point, -128, 127);

    QPositionalEncoding1D<int8_t, int8_t, int8_t, SENC, E> qpos_e; qpos_e.table = qpos_enc;
    QPositionalEncoding1D<int8_t, int8_t, int8_t, SDEC, E> qpos_d; qpos_d.table = qpos_dec;
    auto wirePos = [&](auto& qp, const AffineParams& pa, const AffineParams& pb, const AffineParams& po)
    {
        const auto a = buildQAddParams(pa.scale, pb.scale, po.scale);
        qp.adder.input_a_zero_point = static_cast<int8_t>(pa.zero_point);
        qp.adder.input_b_zero_point = static_cast<int8_t>(pb.zero_point);
        qp.adder.left_shift = a.left_shift;
        qp.adder.input_a_multiplier = a.input_a_multiplier; qp.adder.input_a_shift = a.input_a_shift;
        qp.adder.input_b_multiplier = a.input_b_multiplier; qp.adder.input_b_shift = a.input_b_shift;
        qp.adder.output_requantizer.multiplier = a.output_multiplier;
        qp.adder.output_requantizer.shift = a.output_shift;
        qp.adder.output_requantizer.zero_point = static_cast<int8_t>(po.zero_point);
        qp.adder.output_requantizer.qmin = -128; qp.adder.output_requantizer.qmax = 127;
    };
    wirePos(qpos_e, p_emb, p_pe, p_xe);
    wirePos(qpos_d, p_emb, p_pd, p_xd);

    // ----- Weight scales.
    auto wscale = [&](const float* w, std::size_t n) { return absmaxBuf(w, n) / 127.0f; };
    const float e_wq_s = wscale(e_wattn, E * E), e_wk_s = wscale(e_wattn + E * E, E * E), e_wv_s = wscale(e_wattn + 2 * E * E, E * E);
    const float e_w1_s = wscale(e_wff1, W1_SZ), e_w2_s = wscale(e_wff2, W2_SZ);
    const float ds_wq_s = wscale(d_wself, E * E), ds_wk_s = wscale(d_wself + E * E, E * E), ds_wv_s = wscale(d_wself + 2 * E * E, E * E);
    const float dc_wq_s = wscale(d_wcross, E * E), dc_wk_s = wscale(d_wcross + E * E, E * E), dc_wv_s = wscale(d_wcross + 2 * E * E, E * E);
    const float d_w1_s = wscale(d_wff1, W1_SZ), d_w2_s = wscale(d_wff2, W2_SZ);

    // ----- Quantized weights (persistent storage).
    int8_t qe_wattn[W_ATTN], qe_wff1[W1_SZ], qe_wff2[W2_SZ];
    int32_t qe_battn[B_ATTN] = {0}, qe_bff1[F] = {0}, qe_bff2[E] = {0};
    quantizeSymToI8(e_wattn, E * E, e_wq_s, qe_wattn);
    quantizeSymToI8(e_wattn + E * E, E * E, e_wk_s, qe_wattn + E * E);
    quantizeSymToI8(e_wattn + 2 * E * E, E * E, e_wv_s, qe_wattn + 2 * E * E);
    quantizeSymToI8(e_wff1, W1_SZ, e_w1_s, qe_wff1);
    quantizeSymToI8(e_wff2, W2_SZ, e_w2_s, qe_wff2);

    int8_t qds_w[W_ATTN], qdc_w[W_ATTN], qd_wff1[W1_SZ], qd_wff2[W2_SZ];
    int32_t qds_b[B_ATTN] = {0}, qdc_b[B_ATTN] = {0}, qd_bff1[F] = {0}, qd_bff2[E] = {0};
    quantizeSymToI8(d_wself, E * E, ds_wq_s, qds_w);
    quantizeSymToI8(d_wself + E * E, E * E, ds_wk_s, qds_w + E * E);
    quantizeSymToI8(d_wself + 2 * E * E, E * E, ds_wv_s, qds_w + 2 * E * E);
    quantizeSymToI8(d_wcross, E * E, dc_wq_s, qdc_w);
    quantizeSymToI8(d_wcross + E * E, E * E, dc_wk_s, qdc_w + E * E);
    quantizeSymToI8(d_wcross + 2 * E * E, E * E, dc_wv_s, qdc_w + 2 * E * E);
    quantizeSymToI8(d_wff1, W1_SZ, d_w1_s, qd_wff1);
    quantizeSymToI8(d_wff2, W2_SZ, d_w2_s, qd_wff2);

    // ----- LayerNorm quantized params (gamma=1, beta=0).
    int16_t e_g1q[E], e_g2q[E], d_g1q[E], d_g2q[E], d_g3q[E];
    int32_t e_b1q[E], e_b2q[E], d_b1q[E], d_b2q[E], d_b3q[E];
    auto buildLN = [&](auto& ln, const float* gamma, const float* beta, int16_t* gq, int32_t* bq,
                       const AffineParams& p_in, const AffineParams& p_outln)
    {
        quantizeLayerNormGamma(gamma, E, gq);
        quantizeLayerNormBeta(beta, E, p_outln.scale, bq);
        int32_t mult = 0, shft = 0;
        buildQLayerNormOutputParams(p_outln.scale, mult, shft);
        ln.gamma = gq; ln.beta = bq;
        ln.epsilon_q = quantizeLayerNormEpsilon(LN_EPS, p_in.scale);
        ln.output_multiplier = mult; ln.output_shift = shft;
        ln.output_zero_point = static_cast<int8_t>(p_outln.zero_point);
        ln.qmin = -128; ln.qmax = 127;
    };

    tinymind::QLayerNorm1D<int8_t, int8_t, SENC, E> q_eln1, q_eln2;
    tinymind::QLayerNorm1D<int8_t, int8_t, SDEC, E> q_dln1, q_dln2, q_dln3;
    buildLN(q_eln1, e_gamma1, e_beta1, e_g1q, e_b1q, p_xe, p_eln1);
    buildLN(q_eln2, e_gamma2, e_beta2, e_g2q, e_b2q, p_eskip, p_eln2);
    buildLN(q_dln1, d_g1, d_b1, d_g1q, d_b1q, p_xd, p_dln1);
    buildLN(q_dln2, d_g2, d_b2, d_g2q, d_b2q, p_dskip1, p_dln2);
    buildLN(q_dln3, d_g3, d_b3, d_g3q, d_b3q, p_dskip2, p_dln3);

    // ----- Encoder attention + FFN + adds.
    tinymind::QAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, SENC, E, E> q_eattn;
    q_eattn.weights = qe_wattn; q_eattn.biases = qe_battn;
    q_eattn.input_zero_point = static_cast<int8_t>(p_eln1.zero_point);
    q_eattn.q_zero_point = static_cast<int8_t>(p_eq.zero_point);
    q_eattn.k_zero_point = static_cast<int8_t>(p_ek.zero_point);
    q_eattn.v_zero_point = static_cast<int8_t>(p_ev.zero_point);
    q_eattn.kv_zero_point = static_cast<int8_t>(p_ekv.zero_point);
    q_eattn.q_requantizer = buildRequantizer<int8_t>(p_eln1.scale, e_wq_s, p_eq.scale, p_eq.zero_point, p_eq.zero_point, 127);
    q_eattn.k_requantizer = buildRequantizer<int8_t>(p_eln1.scale, e_wk_s, p_ek.scale, p_ek.zero_point, p_ek.zero_point, 127);
    q_eattn.v_requantizer = buildRequantizer<int8_t>(p_eln1.scale, e_wv_s, p_ev.scale, p_ev.zero_point, -128, 127);
    q_eattn.kv_requantizer = buildRequantizer<int8_t>(p_ek.scale, p_ev.scale, p_ekv.scale, p_ekv.zero_point, -128, 127);
    q_eattn.output_requantizer = buildRequantizer<int8_t>(p_eq.scale, p_ekv.scale, p_eatt.scale, p_eatt.zero_point, -128, 127);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, F> q_eff1;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, F, E> q_eff2;
    q_eff1.weights = qe_wff1; q_eff1.biases = qe_bff1;
    q_eff1.input_zero_point = static_cast<int8_t>(p_eln2.zero_point);
    q_eff1.requantizer = buildRequantizer<int8_t>(p_eln2.scale, e_w1_s, p_eff1.scale, p_eff1.zero_point, p_eff1.zero_point, 127);
    q_eff2.weights = qe_wff2; q_eff2.biases = qe_bff2;
    q_eff2.input_zero_point = static_cast<int8_t>(p_eff1.zero_point);
    q_eff2.requantizer = buildRequantizer<int8_t>(p_eff1.scale, e_w2_s, p_eff2.scale, p_eff2.zero_point, -128, 127);

    tinymind::QAdd<int8_t, int8_t, int8_t, ENC_SZ> q_eadd1, q_eadd2;
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
    wireAdd(q_eadd1, p_eatt, p_xe, p_eskip);
    wireAdd(q_eadd2, p_eskip, p_eff2, p_mem);

    // ----- Decoder causal self-attention.
    tinymind::QCausalAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, SDEC, E, E> q_dself;
    q_dself.weights = qds_w; q_dself.biases = qds_b;
    q_dself.input_zero_point = static_cast<int8_t>(p_dln1.zero_point);
    q_dself.q_zero_point = static_cast<int8_t>(p_dsq.zero_point);
    q_dself.k_zero_point = static_cast<int8_t>(p_dsk.zero_point);
    q_dself.v_zero_point = static_cast<int8_t>(p_dsv.zero_point);
    q_dself.kv_zero_point = static_cast<int8_t>(p_dskv.zero_point);
    q_dself.q_requantizer = buildRequantizer<int8_t>(p_dln1.scale, ds_wq_s, p_dsq.scale, p_dsq.zero_point, p_dsq.zero_point, 127);
    q_dself.k_requantizer = buildRequantizer<int8_t>(p_dln1.scale, ds_wk_s, p_dsk.scale, p_dsk.zero_point, p_dsk.zero_point, 127);
    q_dself.v_requantizer = buildRequantizer<int8_t>(p_dln1.scale, ds_wv_s, p_dsv.scale, p_dsv.zero_point, -128, 127);
    q_dself.kv_requantizer = buildRequantizer<int8_t>(p_dsk.scale, p_dsv.scale, p_dskv.scale, p_dskv.zero_point, -128, 127);
    q_dself.output_requantizer = buildRequantizer<int8_t>(p_dsq.scale, p_dskv.scale, p_dself.scale, p_dself.zero_point, -128, 127);

    // ----- Decoder cross-attention (Q from dln2 grid, K/V from memory grid).
    tinymind::QCrossAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, SDEC, SENC, E, E> q_dcross;
    q_dcross.weights = qdc_w; q_dcross.biases = qdc_b;
    q_dcross.q_input_zero_point = static_cast<int8_t>(p_dln2.zero_point);
    q_dcross.kv_input_zero_point = static_cast<int8_t>(p_mem.zero_point);
    q_dcross.q_zero_point = static_cast<int8_t>(p_dcq.zero_point);
    q_dcross.k_zero_point = static_cast<int8_t>(p_dck.zero_point);
    q_dcross.v_zero_point = static_cast<int8_t>(p_dcv.zero_point);
    q_dcross.kv_zero_point = static_cast<int8_t>(p_dckv.zero_point);
    q_dcross.q_requantizer = buildRequantizer<int8_t>(p_dln2.scale, dc_wq_s, p_dcq.scale, p_dcq.zero_point, p_dcq.zero_point, 127);
    q_dcross.k_requantizer = buildRequantizer<int8_t>(p_mem.scale, dc_wk_s, p_dck.scale, p_dck.zero_point, p_dck.zero_point, 127);
    q_dcross.v_requantizer = buildRequantizer<int8_t>(p_mem.scale, dc_wv_s, p_dcv.scale, p_dcv.zero_point, -128, 127);
    q_dcross.kv_requantizer = buildRequantizer<int8_t>(p_dck.scale, p_dcv.scale, p_dckv.scale, p_dckv.zero_point, -128, 127);
    q_dcross.output_requantizer = buildRequantizer<int8_t>(p_dcq.scale, p_dckv.scale, p_dcross.scale, p_dcross.zero_point, -128, 127);

    // ----- Decoder FFN + adds.
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, F> q_dff1;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, F, E> q_dff2;
    q_dff1.weights = qd_wff1; q_dff1.biases = qd_bff1;
    q_dff1.input_zero_point = static_cast<int8_t>(p_dln3.zero_point);
    q_dff1.requantizer = buildRequantizer<int8_t>(p_dln3.scale, d_w1_s, p_dff1.scale, p_dff1.zero_point, p_dff1.zero_point, 127);
    q_dff2.weights = qd_wff2; q_dff2.biases = qd_bff2;
    q_dff2.input_zero_point = static_cast<int8_t>(p_dff1.zero_point);
    q_dff2.requantizer = buildRequantizer<int8_t>(p_dff1.scale, d_w2_s, p_dff2.scale, p_dff2.zero_point, -128, 127);

    tinymind::QAdd<int8_t, int8_t, int8_t, DEC_SZ> q_dadd1, q_dadd2, q_dadd3;
    auto wireAddD = [&](auto& add, const AffineParams& pa, const AffineParams& pb, const AffineParams& po)
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
    wireAddD(q_dadd1, p_dself, p_xd, p_dskip1);
    wireAddD(q_dadd2, p_dcross, p_dskip1, p_dskip2);
    wireAddD(q_dadd3, p_dskip2, p_dff2, p_dout);

    // ----- int8 forward over the dataset.
    float worst_err = 0.0f;
    bool cache_equiv = true;
    std::size_t mismatch_count = 0;
    std::vector<std::vector<int8_t>> all_out(NUM_INPUTS, std::vector<int8_t>(DEC_SZ));

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        // Encoder.
        int8_t q_xe[ENC_SZ], q_emb_e[ENC_SZ];
        int8_t q_eln1b[ENC_SZ], q_eatt[ENC_SZ], q_eskip[ENC_SZ], q_eln2b[ENC_SZ];
        int8_t q_eff1b[SENC * F], q_eff2b[ENC_SZ], q_mem[ENC_SZ];
        int8_t e_qs[ENC_SZ], e_ks[ENC_SZ], e_vs[ENC_SZ], e_kvs[E * E];
        qembed.forward(src[s].data(), SENC, q_emb_e);
        qpos_e.forward(q_emb_e, q_xe);
        q_eln1.forward(q_xe, q_eln1b);
        q_eattn.forward(q_eln1b, e_qs, e_ks, e_vs, e_kvs, q_eatt);
        q_eadd1.forward(q_eatt, q_xe, q_eskip);
        q_eln2.forward(q_eskip, q_eln2b);
        for (std::size_t t = 0; t < SENC; ++t) q_eff1.forward(q_eln2b + t * E, q_eff1b + t * F);
        tinymind::qreluBuffer<int8_t>(q_eff1b, SENC * F, q_eff1.requantizer.zero_point);
        for (std::size_t t = 0; t < SENC; ++t) q_eff2.forward(q_eff1b + t * F, q_eff2b + t * E);
        q_eadd2.forward(q_eskip, q_eff2b, q_mem);

        // Decoder.
        int8_t q_xd[DEC_SZ], q_emb_d[DEC_SZ];
        int8_t q_dln1b[DEC_SZ], q_dself_full[DEC_SZ], q_dself_incr[DEC_SZ];
        int8_t q_dskip1[DEC_SZ], q_dln2b[DEC_SZ], q_dcross_out[DEC_SZ], q_dskip2[DEC_SZ];
        int8_t q_dln3b[DEC_SZ], q_dff1b[SDEC * F], q_dff2b[DEC_SZ], q_dout[DEC_SZ];
        int8_t d_qs[DEC_SZ], d_ks[DEC_SZ], d_vs[DEC_SZ], d_kvs[E * E];
        int8_t c_ks[ENC_SZ], c_vs[ENC_SZ], c_kvs[E * E], c_qs[DEC_SZ];

        qembed.forward(tgt[s].data(), SDEC, q_emb_d);
        qpos_d.forward(q_emb_d, q_xd);
        q_dln1.forward(q_xd, q_dln1b);

        // Causal self-attention, full-sequence pass (teacher forcing).
        tinymind::QCausalAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                                     SDEC, E, E>::KVState state_full;
        q_dself.forward(q_dln1b, state_full, d_qs, d_ks, d_vs, d_kvs, q_dself_full);

        // Causal self-attention, autoregressive decode (one token at a time).
        // The fixed E x E KV state is the entire decode memory.
        decltype(state_full) state_incr;
        state_incr.reset();
        int8_t q_r[E], k_r[E], v_r[E], kv_r[E * E];
        for (std::size_t t = 0; t < SDEC; ++t)
        {
            q_dself.step(q_dln1b + t * E, state_incr, q_r, k_r, v_r, kv_r, q_dself_incr + t * E);
        }
        for (std::size_t i = 0; i < DEC_SZ; ++i)
            if (q_dself_full[i] != q_dself_incr[i]) { cache_equiv = false; ++mismatch_count; }

        // Continue the decoder from the (identical) self-attention output.
        q_dadd1.forward(q_dself_full, q_xd, q_dskip1);
        q_dln2.forward(q_dskip1, q_dln2b);
        q_dcross.forward(q_dln2b, q_mem, c_ks, c_vs, c_kvs, c_qs, q_dcross_out);
        q_dadd2.forward(q_dcross_out, q_dskip1, q_dskip2);
        q_dln3.forward(q_dskip2, q_dln3b);
        for (std::size_t t = 0; t < SDEC; ++t) q_dff1.forward(q_dln3b + t * E, q_dff1b + t * F);
        tinymind::qreluBuffer<int8_t>(q_dff1b, SDEC * F, q_dff1.requantizer.zero_point);
        for (std::size_t t = 0; t < SDEC; ++t) q_dff2.forward(q_dff1b + t * F, q_dff2b + t * E);
        q_dadd3.forward(q_dskip2, q_dff2b, q_dout);

        for (std::size_t i = 0; i < DEC_SZ; ++i) all_out[s][i] = q_dout[i];

        float deq[DEC_SZ];
        dequantizeBuffer<int8_t>(q_dout, deq, DEC_SZ, p_dout.scale, p_dout.zero_point);
        const float err = maxAbsDiff(deq, float_out[s].data(), DEC_SZ);
        if (err > worst_err) worst_err = err;

        if (s == 0)
        {
            std::FILE* csv = std::fopen("seq2seq_int8.csv", "w");
            std::fprintf(csv, "index,float,int8\n");
            for (std::size_t i = 0; i < DEC_SZ; ++i)
                std::fprintf(csv, "%zu,%.6f,%.6f\n", i, float_out[0][i], deq[i]);
            std::fclose(csv);
        }
    }

    if (golden_mode)
    {
        std::printf("# seq2seq_int8 golden output\n");
        std::printf("# samples=%zu cells=%zu cache_equiv=%d\n",
                    static_cast<size_t>(NUM_INPUTS), static_cast<size_t>(DEC_SZ),
                    cache_equiv ? 1 : 0);
        for (std::size_t s = 0; s < NUM_INPUTS; ++s)
        {
            std::printf("sample %zu:", s);
            for (std::size_t i = 0; i < DEC_SZ; ++i) std::printf(" %d", static_cast<int>(all_out[s][i]));
            std::printf("\n");
        }
        return 0;
    }

    const float out_range = o_dout.max_value - o_dout.min_value;
    std::printf("Int8 encoder-decoder (seq2seq) transformer vs float reference\n");
    std::printf("  vocab=%zu  src_seq=%zu  tgt_seq=%zu  model=%zu  ffn=%zu\n",
                static_cast<size_t>(VOCAB), static_cast<size_t>(SENC),
                static_cast<size_t>(SDEC), static_cast<size_t>(E), static_cast<size_t>(F));
    std::printf("  memory  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                o_mem.min_value, o_mem.max_value, p_mem.scale, p_mem.zero_point);
    std::printf("  output  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                o_dout.min_value, o_dout.max_value, p_dout.scale, p_dout.zero_point);
    std::printf("  KV-cache incremental decode == full-sequence: %s",
                cache_equiv ? "YES (byte-identical)\n" : "NO\n");
    if (!cache_equiv)
        std::printf("  FAIL: %zu / %zu self-attention cells diverged\n",
                    mismatch_count, static_cast<size_t>(NUM_INPUTS * DEC_SZ));
    std::printf("  worst seq2seq max-abs err: %.5f   (%.1f%% of output range)\n",
                worst_err, 100.0f * worst_err / (out_range + 1e-6f));

    const float tol = 0.40f * out_range;
    if (!cache_equiv || worst_err > tol)
    {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS (tolerance %.5f, 40%% of range)\n", tol);
    return 0;
}
