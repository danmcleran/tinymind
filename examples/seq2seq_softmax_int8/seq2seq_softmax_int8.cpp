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

// Int8 encoder-decoder (seq2seq) transformer with STANDARD SOFTMAX attention
// in the decoder -- the softmax counterpart of examples/seq2seq_int8 (which
// uses linear ReLU-kernel attention). Same shape:
//
//   source ids (SEnc)                 target ids (SDec)
//        |                                  |
//   [QEmbedding]+[QPositional]        [QEmbedding]+[QPositional]
//        |                                  |
//   ENCODER block (linear attn):      DECODER block (softmax attn):
//     LN -> LinAttn -> Add              LN -> CausalSoftmaxAttn -> Add  (self)
//     LN -> FFN -> Add                  LN -> CrossSoftmaxAttn  -> Add  (to memory)
//        |                                LN -> FFN -> Add
//   memory (SEnc, E) ------------------>  |
//                                       output (SDec, E)
//
// The decoder self-attention (QCausalAttentionSoftmax1D) keeps a GROWING KV
// cache: softmax cannot collapse the attended history into a fixed state the
// way linear attention does, so each emitted token appends one (K, V) row.
// The causal mask is the cache length -- the score loop only runs over the
// rows that exist. Cross-attention (QCrossAttentionSoftmax1D) prefills the
// encoder K/V once and scores every decoder query against all of them.
//
// Both softmax layers follow the TFLite int8 convention: per-row max subtract,
// 256-entry exp LUT (buildQSoftmaxExpLUT), 1/256 probability grid. The score
// requantizer folds the 1/sqrt(d_k) factor via qAttentionInvSqrt().
//
// As in the linear variant, the int8 causal self-attention is run two ways --
// one full-sequence forward() and step() looped token-by-token -- and asserted
// byte-identical.

#include "qaffine.hpp"
#include "qembedding.hpp"
#include "qpositional.hpp"
#include "qattention1d.hpp"
#include "qcausalattention_softmax.hpp"
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

constexpr std::size_t VOCAB = 16;
constexpr std::size_t SENC  = 6;
constexpr std::size_t SDEC  = 6;
constexpr std::size_t E     = 8;
constexpr std::size_t F     = 16;

constexpr std::size_t ENC_SZ = SENC * E;
constexpr std::size_t DEC_SZ = SDEC * E;
constexpr std::size_t W_ATTN = 3 * E * E;
constexpr std::size_t B_ATTN = 3 * E;
constexpr std::size_t W1_SZ  = F * E;
constexpr std::size_t W2_SZ  = E * F;

constexpr float LN_EPS = 1e-3f;

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

// Encoder self-attention (bidirectional linear, ReLU kernel).
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
    using tinymind::quantizeMultiplier;
    using tinymind::qAttentionInvSqrt;
    using tinymind::buildQSoftmaxExpLUT;
    using tinymind::quantizeLayerNormGamma;
    using tinymind::quantizeLayerNormBeta;
    using tinymind::buildQLayerNormOutputParams;
    using tinymind::quantizeLayerNormEpsilon;
    using tinymind::QEmbedding;
    using tinymind::QPositionalEncoding1D;
    using tinymind::sinusoidalPositionalTable;

    // ----- Embedding + positional tables.
    float emb_table[VOCAB * E];
    for (std::size_t i = 0; i < VOCAB * E; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(VOCAB * E);
        emb_table[i] = 0.6f * std::sin(9.0f * t) + 0.1f * std::cos(2.0f * t);
    }
    float pos_enc[ENC_SZ], pos_dec[DEC_SZ];
    sinusoidalPositionalTable(SENC, E, pos_enc);
    sinusoidalPositionalTable(SDEC, E, pos_dec);

    // ----- Weights.
    float e_g1[E], e_b1[E], e_g2[E], e_b2[E];
    float e_wattn[W_ATTN], e_battn[B_ATTN], e_wff1[W1_SZ], e_bff1[F], e_wff2[W2_SZ], e_bff2[E];
    float d_g1[E], d_b1[E], d_g2[E], d_b2[E], d_g3[E], d_b3[E];
    float d_wself[W_ATTN], d_wcross[W_ATTN], d_wff1[W1_SZ], d_bff1[F], d_wff2[W2_SZ], d_bff2[E];
    // Non-zero attention biases so the softmax-weighted values do not cancel.
    float d_bself[B_ATTN], d_bcross[B_ATTN];
    {
        for (std::size_t i = 0; i < W_ATTN; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(W_ATTN);
            e_wattn[i]  = 0.18f * std::sin(7.0f * t);
            d_wself[i]  = 0.17f * std::sin(8.0f * t);
            d_wcross[i] = 0.16f * std::cos(6.0f * t);
        }
        for (std::size_t i = 0; i < B_ATTN; ++i)
        {
            e_battn[i]  = 0.0f;
            d_bself[i]  = 0.03f * static_cast<float>((static_cast<int>(i) % 3) - 1);
            d_bcross[i] = 0.04f * static_cast<float>((static_cast<int>(i) % 3) + 1);
        }
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
            e_g1[i] = 1.0f; e_b1[i] = 0.0f; e_g2[i] = 1.0f; e_b2[i] = 0.0f;
            d_g1[i] = 1.0f; d_b1[i] = 0.0f; d_g2[i] = 1.0f; d_b2[i] = 0.0f; d_g3[i] = 1.0f; d_b3[i] = 0.0f;
        }
    }

    // ----- Dataset.
    constexpr std::size_t NUM_INPUTS = 6;
    std::vector<std::vector<int32_t>> src(NUM_INPUTS, std::vector<int32_t>(SENC));
    std::vector<std::vector<int32_t>> tgt(NUM_INPUTS, std::vector<int32_t>(SDEC));
    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        for (std::size_t t = 0; t < SENC; ++t) src[s][t] = static_cast<int32_t>((s * SENC + t * 3 + 1) % VOCAB);
        for (std::size_t t = 0; t < SDEC; ++t) tgt[s][t] = static_cast<int32_t>((s * SDEC + t * 5 + 2) % VOCAB);
    }

    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(E));

    // ----- Observers.
    RangeObserver o_emb, o_pe, o_pd, o_xe, o_xd;
    RangeObserver o_eln1, o_eq, o_ek, o_ev, o_ekv, o_eatt, o_eskip, o_eln2, o_eff1, o_eff2, o_mem;
    RangeObserver o_dln1, o_dsq, o_dsk, o_dsv, o_dssc, o_dself, o_dskip1;
    RangeObserver o_dln2, o_dcq, o_dck, o_dcv, o_dcsc, o_dcross, o_dskip2;
    RangeObserver o_dln3, o_dff1, o_dff2, o_dout;
    o_pe.observe(pos_enc, ENC_SZ);
    o_pd.observe(pos_dec, DEC_SZ);

    std::vector<std::vector<float>> float_out(NUM_INPUTS, std::vector<float>(DEC_SZ));

    auto embedAdd = [&](const std::vector<int32_t>& ids, std::size_t S, const float* pos, float* out)
    {
        for (std::size_t t = 0; t < S; ++t)
        {
            const float* row = emb_table + ids[t] * E;
            for (std::size_t e = 0; e < E; ++e) out[t * E + e] = row[e] + pos[t * E + e];
        }
    };

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        float xe[ENC_SZ], xd[DEC_SZ];
        embedAdd(src[s], SENC, pos_enc, xe);
        embedAdd(tgt[s], SDEC, pos_dec, xd);
        for (std::size_t t = 0; t < SENC; ++t)
            for (std::size_t e = 0; e < E; ++e) o_emb.observe(emb_table[src[s][t] * E + e]);
        o_xe.observe(xe, ENC_SZ); o_xd.observe(xd, DEC_SZ);

        // Encoder block (linear attention) -> memory.
        float eln1[ENC_SZ], eatt[ENC_SZ], eskip[ENC_SZ], eln2[ENC_SZ];
        float eff1[SENC * F], eff2[ENC_SZ], mem[ENC_SZ];
        floatLayerNorm(xe, SENC, e_g1, e_b1, eln1);
        {
            std::vector<float> q(ENC_SZ), k(ENC_SZ), v(ENC_SZ); float kv[E * E];
            const float* wq = e_wattn; const float* wk = e_wattn + E * E; const float* wv = e_wattn + 2 * E * E;
            for (std::size_t t = 0; t < SENC; ++t)
                for (std::size_t p = 0; p < E; ++p)
                {
                    float aq = 0, ak = 0, av = 0;
                    for (std::size_t e = 0; e < E; ++e)
                    { aq += wq[e * E + p] * eln1[t * E + e]; ak += wk[e * E + p] * eln1[t * E + e]; av += wv[e * E + p] * eln1[t * E + e]; }
                    q[t * E + p] = (aq < 0) ? 0 : aq; k[t * E + p] = (ak < 0) ? 0 : ak; v[t * E + p] = av;
                }
            for (std::size_t i = 0; i < E; ++i) for (std::size_t j = 0; j < E; ++j)
            { float a = 0; for (std::size_t t = 0; t < SENC; ++t) a += k[t * E + i] * v[t * E + j]; kv[i * E + j] = a; }
            o_eq.observe(q.data(), ENC_SZ); o_ek.observe(k.data(), ENC_SZ); o_ev.observe(v.data(), ENC_SZ); o_ekv.observe(kv, E * E);
        }
        floatLinearAttention(eln1, SENC, e_wattn, e_battn, eatt);
        floatAdd(eatt, xe, eskip, ENC_SZ);
        floatLayerNorm(eskip, SENC, e_g2, e_b2, eln2);
        floatDense(eln2, SENC, e_wff1, e_bff1, E, F, eff1);
        floatRelu(eff1, SENC * F);
        floatDense(eff1, SENC, e_wff2, e_bff2, F, E, eff2);
        floatAdd(eskip, eff2, mem, ENC_SZ);
        o_eln1.observe(eln1, ENC_SZ); o_eatt.observe(eatt, ENC_SZ); o_eskip.observe(eskip, ENC_SZ);
        o_eln2.observe(eln2, ENC_SZ); o_eff1.observe(eff1, SENC * F); o_eff2.observe(eff2, ENC_SZ); o_mem.observe(mem, ENC_SZ);

        // Decoder block (softmax attention).
        float dln1[DEC_SZ], dself[DEC_SZ], dskip1[DEC_SZ];
        float dln2[DEC_SZ], dcross[DEC_SZ], dskip2[DEC_SZ];
        float dln3[DEC_SZ], dff1[SDEC * F], dff2[DEC_SZ], dout[DEC_SZ];

        floatLayerNorm(xd, SDEC, d_g1, d_b1, dln1);
        // Causal softmax self-attention.
        {
            std::vector<float> q(DEC_SZ), k(DEC_SZ), v(DEC_SZ);
            const float* wq = d_wself; const float* wk = d_wself + E * E; const float* wv = d_wself + 2 * E * E;
            const float* bq = d_bself; const float* bk = d_bself + E;     const float* bv = d_bself + 2 * E;
            for (std::size_t t = 0; t < SDEC; ++t)
                for (std::size_t p = 0; p < E; ++p)
                {
                    float aq = bq[p], ak = bk[p], av = bv[p];
                    for (std::size_t e = 0; e < E; ++e)
                    { aq += wq[e * E + p] * dln1[t * E + e]; ak += wk[e * E + p] * dln1[t * E + e]; av += wv[e * E + p] * dln1[t * E + e]; }
                    q[t * E + p] = aq; k[t * E + p] = ak; v[t * E + p] = av;
                }
            o_dsq.observe(q.data(), DEC_SZ); o_dsk.observe(k.data(), DEC_SZ); o_dsv.observe(v.data(), DEC_SZ);
            for (std::size_t t = 0; t < SDEC; ++t)
            {
                float sc[SDEC]; float m = -1e30f;
                for (std::size_t j = 0; j <= t; ++j)
                {
                    float a = 0; for (std::size_t p = 0; p < E; ++p) a += q[t * E + p] * k[j * E + p];
                    sc[j] = a * inv_sqrt; o_dssc.observe(sc[j]); if (sc[j] > m) m = sc[j];
                }
                float sum = 0; for (std::size_t j = 0; j <= t; ++j) { sc[j] = std::exp(sc[j] - m); sum += sc[j]; }
                for (std::size_t p = 0; p < E; ++p)
                { float a = 0; for (std::size_t j = 0; j <= t; ++j) a += (sc[j] / sum) * v[j * E + p]; dself[t * E + p] = a; }
            }
        }
        floatAdd(dself, xd, dskip1, DEC_SZ);
        floatLayerNorm(dskip1, SDEC, d_g2, d_b2, dln2);
        // Softmax cross-attention (Q from dln2, K/V from memory).
        {
            std::vector<float> q(DEC_SZ), k(ENC_SZ), v(ENC_SZ);
            const float* wq = d_wcross; const float* wk = d_wcross + E * E; const float* wv = d_wcross + 2 * E * E;
            const float* bq = d_bcross; const float* bk = d_bcross + E;     const float* bv = d_bcross + 2 * E;
            for (std::size_t t = 0; t < SDEC; ++t)
                for (std::size_t p = 0; p < E; ++p)
                { float aq = bq[p]; for (std::size_t e = 0; e < E; ++e) aq += wq[e * E + p] * dln2[t * E + e]; q[t * E + p] = aq; }
            for (std::size_t sidx = 0; sidx < SENC; ++sidx)
                for (std::size_t p = 0; p < E; ++p)
                {
                    float ak = bk[p], av = bv[p];
                    for (std::size_t e = 0; e < E; ++e) { ak += wk[e * E + p] * mem[sidx * E + e]; av += wv[e * E + p] * mem[sidx * E + e]; }
                    k[sidx * E + p] = ak; v[sidx * E + p] = av;
                }
            o_dcq.observe(q.data(), DEC_SZ); o_dck.observe(k.data(), ENC_SZ); o_dcv.observe(v.data(), ENC_SZ);
            for (std::size_t t = 0; t < SDEC; ++t)
            {
                float sc[SENC]; float m = -1e30f;
                for (std::size_t j = 0; j < SENC; ++j)
                { float a = 0; for (std::size_t p = 0; p < E; ++p) a += q[t * E + p] * k[j * E + p]; sc[j] = a * inv_sqrt; o_dcsc.observe(sc[j]); if (sc[j] > m) m = sc[j]; }
                float sum = 0; for (std::size_t j = 0; j < SENC; ++j) { sc[j] = std::exp(sc[j] - m); sum += sc[j]; }
                for (std::size_t p = 0; p < E; ++p)
                { float a = 0; for (std::size_t j = 0; j < SENC; ++j) a += (sc[j] / sum) * v[j * E + p]; dcross[t * E + p] = a; }
            }
        }
        floatAdd(dcross, dskip1, dskip2, DEC_SZ);
        floatLayerNorm(dskip2, SDEC, d_g3, d_b3, dln3);
        floatDense(dln3, SDEC, d_wff1, d_bff1, E, F, dff1);
        floatRelu(dff1, SDEC * F);
        floatDense(dff1, SDEC, d_wff2, d_bff2, F, E, dff2);
        floatAdd(dskip2, dff2, dout, DEC_SZ);
        o_dln1.observe(dln1, DEC_SZ); o_dself.observe(dself, DEC_SZ); o_dskip1.observe(dskip1, DEC_SZ);
        o_dln2.observe(dln2, DEC_SZ); o_dcross.observe(dcross, DEC_SZ); o_dskip2.observe(dskip2, DEC_SZ);
        o_dln3.observe(dln3, DEC_SZ); o_dff1.observe(dff1, SDEC * F); o_dff2.observe(dff2, DEC_SZ); o_dout.observe(dout, DEC_SZ);

        std::memcpy(float_out[s].data(), dout, sizeof(float) * DEC_SZ);
    }

    // ----- Grids.
    auto grid = [](const RangeObserver& o) { return computeAffineParamsAsymmetric(o.min_value, o.max_value, -128, 127); };
    auto reluGrid = [](const RangeObserver& o) { return computeAffineParamsAsymmetric(0.0f, o.max_value, -128, 127); };
    const auto p_emb = grid(o_emb), p_pe = grid(o_pe), p_pd = grid(o_pd), p_xe = grid(o_xe), p_xd = grid(o_xd);
    const auto p_eln1 = grid(o_eln1), p_eq = reluGrid(o_eq), p_ek = reluGrid(o_ek), p_ev = grid(o_ev), p_ekv = grid(o_ekv);
    const auto p_eatt = grid(o_eatt), p_eskip = grid(o_eskip), p_eln2 = grid(o_eln2), p_eff1 = reluGrid(o_eff1), p_eff2 = grid(o_eff2), p_mem = grid(o_mem);
    const auto p_dln1 = grid(o_dln1), p_dsq = grid(o_dsq), p_dsk = grid(o_dsk), p_dsv = grid(o_dsv), p_dssc = grid(o_dssc);
    const auto p_dself = grid(o_dself), p_dskip1 = grid(o_dskip1);
    const auto p_dln2 = grid(o_dln2), p_dcq = grid(o_dcq), p_dck = grid(o_dck), p_dcv = grid(o_dcv), p_dcsc = grid(o_dcsc);
    const auto p_dcross = grid(o_dcross), p_dskip2 = grid(o_dskip2);
    const auto p_dln3 = grid(o_dln3), p_dff1 = reluGrid(o_dff1), p_dff2 = grid(o_dff2), p_dout = grid(o_dout);
    const float attn_scale = 1.0f / 256.0f;

    // ----- Embedding + positional.
    int8_t qemb[VOCAB * E];
    quantizeBuffer<int8_t>(emb_table, qemb, VOCAB * E, p_emb.scale, p_emb.zero_point, -128, 127);
    QEmbedding<int8_t, int8_t, VOCAB, E> qembed; qembed.table = qemb; qembed.requantizer = nullptr;
    int8_t qpos_e[ENC_SZ], qpos_d[DEC_SZ];
    quantizeBuffer<int8_t>(pos_enc, qpos_e, ENC_SZ, p_pe.scale, p_pe.zero_point, -128, 127);
    quantizeBuffer<int8_t>(pos_dec, qpos_d, DEC_SZ, p_pd.scale, p_pd.zero_point, -128, 127);
    QPositionalEncoding1D<int8_t, int8_t, int8_t, SENC, E> qpe; qpe.table = qpos_e;
    QPositionalEncoding1D<int8_t, int8_t, int8_t, SDEC, E> qpd; qpd.table = qpos_d;
    auto wirePos = [&](auto& qp, const AffineParams& pa, const AffineParams& pb, const AffineParams& po)
    {
        const auto a = buildQAddParams(pa.scale, pb.scale, po.scale);
        qp.adder.input_a_zero_point = static_cast<int8_t>(pa.zero_point);
        qp.adder.input_b_zero_point = static_cast<int8_t>(pb.zero_point);
        qp.adder.left_shift = a.left_shift;
        qp.adder.input_a_multiplier = a.input_a_multiplier; qp.adder.input_a_shift = a.input_a_shift;
        qp.adder.input_b_multiplier = a.input_b_multiplier; qp.adder.input_b_shift = a.input_b_shift;
        qp.adder.output_requantizer.multiplier = a.output_multiplier; qp.adder.output_requantizer.shift = a.output_shift;
        qp.adder.output_requantizer.zero_point = static_cast<int8_t>(po.zero_point);
        qp.adder.output_requantizer.qmin = -128; qp.adder.output_requantizer.qmax = 127;
    };
    wirePos(qpe, p_emb, p_pe, p_xe);
    wirePos(qpd, p_emb, p_pd, p_xd);

    // ----- Weight scales + quantization.
    auto ws = [&](const float* w, std::size_t n) { return absmaxBuf(w, n) / 127.0f; };
    const float e_wq = ws(e_wattn, E * E), e_wk = ws(e_wattn + E * E, E * E), e_wv = ws(e_wattn + 2 * E * E, E * E);
    const float e_w1 = ws(e_wff1, W1_SZ), e_w2 = ws(e_wff2, W2_SZ);
    const float ds_wq = ws(d_wself, E * E), ds_wk = ws(d_wself + E * E, E * E), ds_wv = ws(d_wself + 2 * E * E, E * E);
    const float dc_wq = ws(d_wcross, E * E), dc_wk = ws(d_wcross + E * E, E * E), dc_wv = ws(d_wcross + 2 * E * E, E * E);
    const float d_w1 = ws(d_wff1, W1_SZ), d_w2 = ws(d_wff2, W2_SZ);

    int8_t qe_wattn[W_ATTN], qe_wff1[W1_SZ], qe_wff2[W2_SZ];
    int32_t qe_battn[B_ATTN] = {0}, qe_bff1[F] = {0}, qe_bff2[E] = {0};
    quantizeSymToI8(e_wattn, E * E, e_wq, qe_wattn);
    quantizeSymToI8(e_wattn + E * E, E * E, e_wk, qe_wattn + E * E);
    quantizeSymToI8(e_wattn + 2 * E * E, E * E, e_wv, qe_wattn + 2 * E * E);
    quantizeSymToI8(e_wff1, W1_SZ, e_w1, qe_wff1);
    quantizeSymToI8(e_wff2, W2_SZ, e_w2, qe_wff2);

    int8_t qds_w[W_ATTN], qdc_w[W_ATTN], qd_wff1[W1_SZ], qd_wff2[W2_SZ];
    int32_t qds_b[B_ATTN], qdc_b[B_ATTN], qd_bff1[F] = {0}, qd_bff2[E] = {0};
    quantizeSymToI8(d_wself, E * E, ds_wq, qds_w);
    quantizeSymToI8(d_wself + E * E, E * E, ds_wk, qds_w + E * E);
    quantizeSymToI8(d_wself + 2 * E * E, E * E, ds_wv, qds_w + 2 * E * E);
    quantizeSymToI8(d_wcross, E * E, dc_wq, qdc_w);
    quantizeSymToI8(d_wcross + E * E, E * E, dc_wk, qdc_w + E * E);
    quantizeSymToI8(d_wcross + 2 * E * E, E * E, dc_wv, qdc_w + 2 * E * E);
    quantizeSymToI8(d_wff1, W1_SZ, d_w1, qd_wff1);
    quantizeSymToI8(d_wff2, W2_SZ, d_w2, qd_wff2);
    // Self-attn biases (q,k,v) in (in_scale * w_scale) units, in_scale = p_dln1.
    for (std::size_t p = 0; p < E; ++p)
    {
        qds_b[p]         = static_cast<int32_t>(std::lround(d_bself[p] / (p_dln1.scale * ds_wq)));
        qds_b[E + p]     = static_cast<int32_t>(std::lround(d_bself[E + p] / (p_dln1.scale * ds_wk)));
        qds_b[2 * E + p] = static_cast<int32_t>(std::lround(d_bself[2 * E + p] / (p_dln1.scale * ds_wv)));
        // Cross-attn: Q from p_dln2, K/V from p_mem.
        qdc_b[p]         = static_cast<int32_t>(std::lround(d_bcross[p] / (p_dln2.scale * dc_wq)));
        qdc_b[E + p]     = static_cast<int32_t>(std::lround(d_bcross[E + p] / (p_mem.scale * dc_wk)));
        qdc_b[2 * E + p] = static_cast<int32_t>(std::lround(d_bcross[2 * E + p] / (p_mem.scale * dc_wv)));
    }

    // ----- Exp LUTs (one per score grid).
    int32_t self_lut[256], cross_lut[256];
    buildQSoftmaxExpLUT(p_dssc.scale, self_lut);
    buildQSoftmaxExpLUT(p_dcsc.scale, cross_lut);

    // ----- LayerNorms.
    int16_t e_g1q[E], e_g2q[E], d_g1q[E], d_g2q[E], d_g3q[E];
    int32_t e_b1q[E], e_b2q[E], d_b1q[E], d_b2q[E], d_b3q[E];
    auto buildLN = [&](auto& ln, const float* g, const float* b, int16_t* gq, int32_t* bq, const AffineParams& pin, const AffineParams& pout)
    {
        quantizeLayerNormGamma(g, E, gq); quantizeLayerNormBeta(b, E, pout.scale, bq);
        int32_t mult = 0, shft = 0; buildQLayerNormOutputParams(pout.scale, mult, shft);
        ln.gamma = gq; ln.beta = bq; ln.epsilon_q = quantizeLayerNormEpsilon(LN_EPS, pin.scale);
        ln.output_multiplier = mult; ln.output_shift = shft; ln.output_zero_point = static_cast<int8_t>(pout.zero_point);
        ln.qmin = -128; ln.qmax = 127;
    };
    tinymind::QLayerNorm1D<int8_t, int8_t, SENC, E> q_eln1, q_eln2;
    tinymind::QLayerNorm1D<int8_t, int8_t, SDEC, E> q_dln1, q_dln2, q_dln3;
    buildLN(q_eln1, e_g1, e_b1, e_g1q, e_b1q, p_xe, p_eln1);
    buildLN(q_eln2, e_g2, e_b2, e_g2q, e_b2q, p_eskip, p_eln2);
    buildLN(q_dln1, d_g1, d_b1, d_g1q, d_b1q, p_xd, p_dln1);
    buildLN(q_dln2, d_g2, d_b2, d_g2q, d_b2q, p_dskip1, p_dln2);
    buildLN(q_dln3, d_g3, d_b3, d_g3q, d_b3q, p_dskip2, p_dln3);

    // ----- Encoder attention (linear) + FFN + adds.
    tinymind::QAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, SENC, E, E> q_eattn;
    q_eattn.weights = qe_wattn; q_eattn.biases = qe_battn;
    q_eattn.input_zero_point = static_cast<int8_t>(p_eln1.zero_point);
    q_eattn.q_zero_point = static_cast<int8_t>(p_eq.zero_point); q_eattn.k_zero_point = static_cast<int8_t>(p_ek.zero_point);
    q_eattn.v_zero_point = static_cast<int8_t>(p_ev.zero_point); q_eattn.kv_zero_point = static_cast<int8_t>(p_ekv.zero_point);
    q_eattn.q_requantizer = buildRequantizer<int8_t>(p_eln1.scale, e_wq, p_eq.scale, p_eq.zero_point, p_eq.zero_point, 127);
    q_eattn.k_requantizer = buildRequantizer<int8_t>(p_eln1.scale, e_wk, p_ek.scale, p_ek.zero_point, p_ek.zero_point, 127);
    q_eattn.v_requantizer = buildRequantizer<int8_t>(p_eln1.scale, e_wv, p_ev.scale, p_ev.zero_point, -128, 127);
    q_eattn.kv_requantizer = buildRequantizer<int8_t>(p_ek.scale, p_ev.scale, p_ekv.scale, p_ekv.zero_point, -128, 127);
    q_eattn.output_requantizer = buildRequantizer<int8_t>(p_eq.scale, p_ekv.scale, p_eatt.scale, p_eatt.zero_point, -128, 127);

    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, F> q_eff1;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, F, E> q_eff2;
    q_eff1.weights = qe_wff1; q_eff1.biases = qe_bff1; q_eff1.input_zero_point = static_cast<int8_t>(p_eln2.zero_point);
    q_eff1.requantizer = buildRequantizer<int8_t>(p_eln2.scale, e_w1, p_eff1.scale, p_eff1.zero_point, p_eff1.zero_point, 127);
    q_eff2.weights = qe_wff2; q_eff2.biases = qe_bff2; q_eff2.input_zero_point = static_cast<int8_t>(p_eff1.zero_point);
    q_eff2.requantizer = buildRequantizer<int8_t>(p_eff1.scale, e_w2, p_eff2.scale, p_eff2.zero_point, -128, 127);

    auto wireAdd = [&](auto& add, const AffineParams& pa, const AffineParams& pb, const AffineParams& po)
    {
        const auto a = buildQAddParams(pa.scale, pb.scale, po.scale);
        add.input_a_zero_point = static_cast<int8_t>(pa.zero_point); add.input_b_zero_point = static_cast<int8_t>(pb.zero_point);
        add.left_shift = a.left_shift;
        add.input_a_multiplier = a.input_a_multiplier; add.input_a_shift = a.input_a_shift;
        add.input_b_multiplier = a.input_b_multiplier; add.input_b_shift = a.input_b_shift;
        add.output_requantizer.multiplier = a.output_multiplier; add.output_requantizer.shift = a.output_shift;
        add.output_requantizer.zero_point = static_cast<int8_t>(po.zero_point);
        add.output_requantizer.qmin = -128; add.output_requantizer.qmax = 127;
    };
    tinymind::QAdd<int8_t, int8_t, int8_t, ENC_SZ> q_eadd1, q_eadd2;
    wireAdd(q_eadd1, p_eatt, p_xe, p_eskip);
    wireAdd(q_eadd2, p_eskip, p_eff2, p_mem);

    // ----- Decoder causal softmax self-attention.
    typedef tinymind::QCausalAttentionSoftmax1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, int8_t, SDEC, E, E> SelfAttn;
    SelfAttn q_dself;
    q_dself.weights = qds_w; q_dself.biases = qds_b;
    q_dself.input_zero_point = static_cast<int8_t>(p_dln1.zero_point);
    q_dself.q_zero_point = static_cast<int8_t>(p_dsq.zero_point); q_dself.k_zero_point = static_cast<int8_t>(p_dsk.zero_point);
    q_dself.v_zero_point = static_cast<int8_t>(p_dsv.zero_point); q_dself.attn_zero_point = -128;
    q_dself.q_requantizer = buildRequantizer<int8_t>(p_dln1.scale, ds_wq, p_dsq.scale, p_dsq.zero_point, -128, 127);
    q_dself.k_requantizer = buildRequantizer<int8_t>(p_dln1.scale, ds_wk, p_dsk.scale, p_dsk.zero_point, -128, 127);
    q_dself.v_requantizer = buildRequantizer<int8_t>(p_dln1.scale, ds_wv, p_dsv.scale, p_dsv.zero_point, -128, 127);
    {
        Requantizer<int32_t, int8_t> r;
        const double ratio = (static_cast<double>(p_dsq.scale) * static_cast<double>(p_dsk.scale)) / static_cast<double>(p_dssc.scale) * qAttentionInvSqrt(E);
        int32_t mu = 0, sh = 0; quantizeMultiplier(ratio, mu, sh);
        r.multiplier = mu; r.shift = sh; r.zero_point = 0; r.qmin = -128; r.qmax = 127;
        q_dself.score_requantizer = r;
    }
    q_dself.softmax_exp_lut = self_lut; q_dself.attn_qmin = -128; q_dself.attn_qmax = 127;
    q_dself.output_requantizer = buildRequantizer<int8_t>(attn_scale, p_dsv.scale, p_dself.scale, p_dself.zero_point, -128, 127);

    // ----- Decoder softmax cross-attention.
    typedef tinymind::QCrossAttentionSoftmax1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, int8_t, SDEC, SENC, E, E> CrossAttn;
    CrossAttn q_dcross;
    q_dcross.weights = qdc_w; q_dcross.biases = qdc_b;
    q_dcross.q_input_zero_point = static_cast<int8_t>(p_dln2.zero_point);
    q_dcross.kv_input_zero_point = static_cast<int8_t>(p_mem.zero_point);
    q_dcross.q_zero_point = static_cast<int8_t>(p_dcq.zero_point); q_dcross.k_zero_point = static_cast<int8_t>(p_dck.zero_point);
    q_dcross.v_zero_point = static_cast<int8_t>(p_dcv.zero_point); q_dcross.attn_zero_point = -128;
    q_dcross.q_requantizer = buildRequantizer<int8_t>(p_dln2.scale, dc_wq, p_dcq.scale, p_dcq.zero_point, -128, 127);
    q_dcross.k_requantizer = buildRequantizer<int8_t>(p_mem.scale, dc_wk, p_dck.scale, p_dck.zero_point, -128, 127);
    q_dcross.v_requantizer = buildRequantizer<int8_t>(p_mem.scale, dc_wv, p_dcv.scale, p_dcv.zero_point, -128, 127);
    {
        Requantizer<int32_t, int8_t> r;
        const double ratio = (static_cast<double>(p_dcq.scale) * static_cast<double>(p_dck.scale)) / static_cast<double>(p_dcsc.scale) * qAttentionInvSqrt(E);
        int32_t mu = 0, sh = 0; quantizeMultiplier(ratio, mu, sh);
        r.multiplier = mu; r.shift = sh; r.zero_point = 0; r.qmin = -128; r.qmax = 127;
        q_dcross.score_requantizer = r;
    }
    q_dcross.softmax_exp_lut = cross_lut; q_dcross.attn_qmin = -128; q_dcross.attn_qmax = 127;
    q_dcross.output_requantizer = buildRequantizer<int8_t>(attn_scale, p_dcv.scale, p_dcross.scale, p_dcross.zero_point, -128, 127);

    // ----- Decoder FFN + adds.
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, E, F> q_dff1;
    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, F, E> q_dff2;
    q_dff1.weights = qd_wff1; q_dff1.biases = qd_bff1; q_dff1.input_zero_point = static_cast<int8_t>(p_dln3.zero_point);
    q_dff1.requantizer = buildRequantizer<int8_t>(p_dln3.scale, d_w1, p_dff1.scale, p_dff1.zero_point, p_dff1.zero_point, 127);
    q_dff2.weights = qd_wff2; q_dff2.biases = qd_bff2; q_dff2.input_zero_point = static_cast<int8_t>(p_dff1.zero_point);
    q_dff2.requantizer = buildRequantizer<int8_t>(p_dff1.scale, d_w2, p_dff2.scale, p_dff2.zero_point, -128, 127);

    tinymind::QAdd<int8_t, int8_t, int8_t, DEC_SZ> q_dadd1, q_dadd2, q_dadd3;
    wireAdd(q_dadd1, p_dself, p_xd, p_dskip1);
    wireAdd(q_dadd2, p_dcross, p_dskip1, p_dskip2);
    wireAdd(q_dadd3, p_dskip2, p_dff2, p_dout);

    // ----- int8 forward.
    float worst_err = 0.0f;
    bool cache_equiv = true;
    std::size_t mismatch = 0;
    std::vector<std::vector<int8_t>> all_out(NUM_INPUTS, std::vector<int8_t>(DEC_SZ));

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        int8_t q_xe[ENC_SZ], q_emb_e[ENC_SZ], q_eln1b[ENC_SZ], q_eatt[ENC_SZ], q_eskip[ENC_SZ], q_eln2b[ENC_SZ];
        int8_t q_eff1b[SENC * F], q_eff2b[ENC_SZ], q_mem[ENC_SZ];
        int8_t e_qs[ENC_SZ], e_ks[ENC_SZ], e_vs[ENC_SZ], e_kvs[E * E];
        qembed.forward(src[s].data(), SENC, q_emb_e);
        qpe.forward(q_emb_e, q_xe);
        q_eln1.forward(q_xe, q_eln1b);
        q_eattn.forward(q_eln1b, e_qs, e_ks, e_vs, e_kvs, q_eatt);
        q_eadd1.forward(q_eatt, q_xe, q_eskip);
        q_eln2.forward(q_eskip, q_eln2b);
        for (std::size_t t = 0; t < SENC; ++t) q_eff1.forward(q_eln2b + t * E, q_eff1b + t * F);
        tinymind::qreluBuffer<int8_t>(q_eff1b, SENC * F, q_eff1.requantizer.zero_point);
        for (std::size_t t = 0; t < SENC; ++t) q_eff2.forward(q_eff1b + t * F, q_eff2b + t * E);
        q_eadd2.forward(q_eskip, q_eff2b, q_mem);

        int8_t q_xd[DEC_SZ], q_emb_d[DEC_SZ], q_dln1b[DEC_SZ];
        int8_t q_dself_full[DEC_SZ], q_dself_incr[DEC_SZ];
        int8_t q_dskip1[DEC_SZ], q_dln2b[DEC_SZ], q_dcross_out[DEC_SZ], q_dskip2[DEC_SZ];
        int8_t q_dln3b[DEC_SZ], q_dff1b[SDEC * F], q_dff2b[DEC_SZ], q_dout[DEC_SZ];
        int8_t ds_qs[DEC_SZ], ds_ks[DEC_SZ], ds_vs[DEC_SZ], ds_sc[SDEC], ds_at[SDEC];
        int8_t cq[DEC_SZ], csc[SENC], cat[SENC];

        qembed.forward(tgt[s].data(), SDEC, q_emb_d);
        qpd.forward(q_emb_d, q_xd);
        q_dln1.forward(q_xd, q_dln1b);

        // Full-sequence causal softmax self-attention.
        SelfAttn::KVCache cache_full;
        q_dself.forward(q_dln1b, cache_full, ds_qs, ds_ks, ds_vs, ds_sc, ds_at, q_dself_full);

        // Token-by-token decode (growing cache).
        SelfAttn::KVCache cache_incr; cache_incr.reset();
        int8_t q_r[E], k_r[E], v_r[E];
        for (std::size_t t = 0; t < SDEC; ++t)
            q_dself.step(q_dln1b + t * E, cache_incr, q_r, k_r, v_r, ds_sc, ds_at, q_dself_incr + t * E);
        for (std::size_t i = 0; i < DEC_SZ; ++i)
            if (q_dself_full[i] != q_dself_incr[i]) { cache_equiv = false; ++mismatch; }

        q_dadd1.forward(q_dself_full, q_xd, q_dskip1);
        q_dln2.forward(q_dskip1, q_dln2b);
        CrossAttn::KVCache cross_cache;
        q_dcross.forward(q_dln2b, q_mem, cross_cache, cq, csc, cat, q_dcross_out);
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
            std::FILE* csv = std::fopen("seq2seq_softmax_int8.csv", "w");
            std::fprintf(csv, "index,float,int8\n");
            for (std::size_t i = 0; i < DEC_SZ; ++i) std::fprintf(csv, "%zu,%.6f,%.6f\n", i, float_out[0][i], deq[i]);
            std::fclose(csv);
        }
    }

    if (golden_mode)
    {
        std::printf("# seq2seq_softmax_int8 golden output\n");
        std::printf("# samples=%zu cells=%zu cache_equiv=%d\n",
                    static_cast<size_t>(NUM_INPUTS), static_cast<size_t>(DEC_SZ), cache_equiv ? 1 : 0);
        for (std::size_t s = 0; s < NUM_INPUTS; ++s)
        {
            std::printf("sample %zu:", s);
            for (std::size_t i = 0; i < DEC_SZ; ++i) std::printf(" %d", static_cast<int>(all_out[s][i]));
            std::printf("\n");
        }
        return 0;
    }

    const float out_range = o_dout.max_value - o_dout.min_value;
    std::printf("Int8 seq2seq transformer with SOFTMAX decoder attention vs float\n");
    std::printf("  vocab=%zu  src=%zu  tgt=%zu  model=%zu  ffn=%zu\n",
                static_cast<size_t>(VOCAB), static_cast<size_t>(SENC), static_cast<size_t>(SDEC),
                static_cast<size_t>(E), static_cast<size_t>(F));
    std::printf("  output range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n",
                o_dout.min_value, o_dout.max_value, p_dout.scale, p_dout.zero_point);
    std::printf("  growing-cache incremental decode == full-sequence: %s",
                cache_equiv ? "YES (byte-identical)\n" : "NO\n");
    if (!cache_equiv) std::printf("  FAIL: %zu / %zu self-attention cells diverged\n", mismatch, static_cast<size_t>(NUM_INPUTS * DEC_SZ));
    std::printf("  worst seq2seq max-abs err: %.5f   (%.1f%% of output range)\n",
                worst_err, 100.0f * worst_err / (out_range + 1e-6f));

    const float tol = 0.50f * out_range;  // softmax compounds more int8 stages than the linear variant
    if (!cache_equiv || worst_err > tol) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS (tolerance %.5f, 50%% of range)\n", tol);
    return 0;
}
