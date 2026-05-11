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

// Phase 16 mixed-precision exemplar:
//
//   int8 CNN feature extractor  ->  fp16 attention head  ->  int8 dense classifier
//
// Pipeline shape (S = sequence length, E = embedding dim):
//
//   input  [S=8][E=8]   float
//      |
//   ----[ int8 frontend ]------------------------------------------------
//   QDense  E -> E (shared per step, calibrated input scale)
//   qrelu                                          -> [S][E] int8
//   ----[ Phase 9 bridge: affineI8ToFp16Buffer ]--------------------------
//   fp16 buffer                                    -> [S][E] fp16
//   ----[ fp16 attention head ]------------------------------------------
//   Linear self-attention (ReLU-kernel), all arithmetic in float promoted
//   from fp16, results stored as fp16 between sub-steps. Output is the
//   per-token attention result mean-pooled over S, producing one fp16
//   summary vector of length E.
//   ----[ Phase 9 bridge: fp16ToAffineI8Buffer ]-------------------------
//   int8 buffer length E
//   ----[ int8 classifier ]----------------------------------------------
//   QDense E -> NUM_CLASSES
//   int8 logits
//
// Demonstrates: an int8 frontend and an int8 classifier *bracket* an
// fp16 (storage) attention head, with the Phase 9 affineI8<->fp16
// converters running scalar at the layer boundary. Inner arithmetic of
// the attention head goes through float promote on each MAC — the
// promotes themselves are free if the target has fp16-vector arithmetic
// (Phase 14 SIMD_NEON_FP16 / SIMD_AVX512F bf16 etc.), and scalar
// otherwise. Library compiles the storage tier on any toolchain.

#include "qaffine.hpp"
#include "qdense.hpp"
#include "qactivations.hpp"
#include "qbridge.hpp"
#include "include/qcalibration.hpp"
#include "include/tinymind_fp16.hpp"
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

constexpr std::size_t S = 8;     // sequence length
constexpr std::size_t E = 8;     // embedding dim
constexpr std::size_t NUM_CLASSES = 4;
constexpr std::size_t NUM_INPUTS  = 4;
constexpr std::size_t NUM_CALIB   = 32;

constexpr std::size_t IN_SIZE = S * E;
constexpr std::size_t W_FRONT = E * E;
constexpr std::size_t W_HEAD  = 3 * E * E;   // Wq, Wk, Wv (linear-attn projections)
constexpr std::size_t B_HEAD  = 3 * E;
constexpr std::size_t W_BACK  = E * NUM_CLASSES;

// ---------------------------------------------------------------------------
// Float helpers.
// ---------------------------------------------------------------------------
void fdense_row(const float* in, const float* w, const float* b, float* out,
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

void frelu(float* x, std::size_t n) { for (std::size_t i = 0; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f; }

void flinearAttention(const float* x, const float* w, const float* b, float* y)
{
    // x is (S, E); y is (S, E). w packs (Wq, Wk, Wv) each (E, E) row-major
    // with input dim then output dim.
    const float* wq = w;
    const float* wk = w + E * E;
    const float* wv = w + 2 * E * E;
    const float* bq = b;
    const float* bk = b + E;
    const float* bv = b + 2 * E;

    float q[S * E], k[S * E], v[S * E], kv[E * E];
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

void fmeanPool(const float* x, float* y)
{
    const float den = 1.0f / static_cast<float>(S);
    for (std::size_t e = 0; e < E; ++e)
    {
        float s = 0.0f;
        for (std::size_t t = 0; t < S; ++t) s += x[t * E + e];
        y[e] = s * den;
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
    using tinymind::QDense;
    using tinymind::Requantizer;
    using tinymind::RangeObserver;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::buildRequantizer;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;
    using tinymind::qreluBuffer;
    using tinymind::affineI8ToFp16;
    using tinymind::fp16ToAffineI8;
    using tinymind::fp16_t;
    using tinymind::fp16ToFloat;
    using tinymind::floatToFp16;

    const bool bench_mode  = (argc >= 2) && std::strcmp(argv[1], "--bench")  == 0;
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    // ----- Hand-crafted weights.
    float w_front[W_FRONT], b_front[E] = {};
    float w_head[W_HEAD], b_head[B_HEAD] = {};
    float w_back[W_BACK], b_back[NUM_CLASSES] = {};

    fillW(w_front, W_FRONT, 0.15f, 5.0f, 0.2f);
    fillW(w_head,  W_HEAD,  0.10f, 7.0f, 0.4f);
    fillW(w_back,  W_BACK,  0.20f, 3.0f, 0.6f);

    // ----- Synthetic dataset.
    std::vector<std::vector<float>> inputs(NUM_CALIB, std::vector<float>(IN_SIZE));
    for (std::size_t s = 0; s < NUM_CALIB; ++s)
        for (std::size_t i = 0; i < IN_SIZE; ++i)
        {
            const float phase = static_cast<float>(s * 11 + i) * 0.15f;
            inputs[s][i] = 0.6f * std::sin(phase) +
                           0.3f * std::cos(0.3f * static_cast<float>(s + i));
        }

    // ----- Float reference forward + observation.
    RangeObserver o_in, o_feat, o_relu, o_attn, o_pool, o_logits;
    std::vector<std::vector<float>> float_logits(NUM_CALIB, std::vector<float>(NUM_CLASSES));

    float feat[S * E], post_relu[S * E], attn_out[S * E], pool[E];
    for (std::size_t s = 0; s < NUM_CALIB; ++s)
    {
        for (std::size_t t = 0; t < S; ++t)
            fdense_row(&inputs[s][t * E], w_front, b_front, &feat[t * E], E, E);
        std::memcpy(post_relu, feat, sizeof(feat));
        frelu(post_relu, S * E);
        flinearAttention(post_relu, w_head, b_head, attn_out);
        // Residual skip from post-ReLU into the attention output keeps the
        // pooled summary off the zero mean that a tiny synthetic dataset
        // would otherwise force on a linear-attention head.
        for (std::size_t i = 0; i < S * E; ++i) attn_out[i] += post_relu[i];
        fmeanPool(attn_out, pool);
        fdense_row(pool, w_back, b_back, float_logits[s].data(), E, NUM_CLASSES);

        o_in    .observe(inputs[s].data(), IN_SIZE);
        o_feat  .observe(feat,    S * E);
        o_relu  .observe(post_relu, S * E);
        o_attn  .observe(attn_out, S * E);
        o_pool  .observe(pool, E);
        o_logits.observe(float_logits[s].data(), NUM_CLASSES);
    }

    // ----- Affine params for the int8 frontend and tail.
    const auto p_in    = computeAffineParamsAsymmetric(o_in.min_value,   o_in.max_value,   -128, 127);
    const auto p_feat  = computeAffineParamsAsymmetric(o_feat.min_value, o_feat.max_value, -128, 127);
    // post-ReLU sits on p_feat grid (clamp); represent it as p_feat for the
    // bridge to fp16 since the bridge converts from int8 affine directly.
    const auto& p_relu = p_feat;
    // Pool output range / scale for the int8 classifier's input grid.
    const auto p_pool  = computeAffineParamsAsymmetric(o_pool.min_value, o_pool.max_value, -128, 127);
    const auto p_logits= computeAffineParamsAsymmetric(o_logits.min_value, o_logits.max_value, -128, 127);
    (void)o_attn;

    // ----- Per-tensor symmetric weight scales for the two int8 dense layers.
    float front_abs = 0.0f;
    for (std::size_t i = 0; i < W_FRONT; ++i) { const float a = std::fabs(w_front[i]); if (a > front_abs) front_abs = a; }
    const float ws_front = front_abs / 127.0f;

    float back_abs = 0.0f;
    for (std::size_t i = 0; i < W_BACK; ++i) { const float a = std::fabs(w_back[i]); if (a > back_abs) back_abs = a; }
    const float ws_back = back_abs / 127.0f;

    int8_t qw_front[W_FRONT], qw_back[W_BACK];
    int32_t qb_front[E] = {}, qb_back[NUM_CLASSES] = {};
    quantizeBuffer<int8_t>(w_front, qw_front, W_FRONT, ws_front, 0, -128, 127);
    quantizeBuffer<int8_t>(w_back,  qw_back,  W_BACK,  ws_back,  0, -128, 127);
    {
        const double sf = static_cast<double>(p_in.scale) * static_cast<double>(ws_front);
        for (std::size_t o = 0; o < E; ++o)
            qb_front[o] = static_cast<int32_t>(std::lround(static_cast<double>(b_front[o]) / sf));
        const double sb = static_cast<double>(p_pool.scale) * static_cast<double>(ws_back);
        for (std::size_t o = 0; o < NUM_CLASSES; ++o)
            qb_back[o] = static_cast<int32_t>(std::lround(static_cast<double>(b_back[o]) / sb));
    }

    // ----- Layer instances.
    QDense<int8_t, int8_t, int32_t, int8_t, E, E> qfront;
    qfront.weights = qw_front;
    qfront.biases  = qb_front;
    qfront.input_zero_point = static_cast<int8_t>(p_in.zero_point);
    qfront.requantizer = buildRequantizer<int8_t>(p_in.scale, ws_front,
                                                  p_feat.scale, p_feat.zero_point,
                                                  -128, 127);

    QDense<int8_t, int8_t, int32_t, int8_t, E, NUM_CLASSES> qback;
    qback.weights = qw_back;
    qback.biases  = qb_back;
    qback.input_zero_point = static_cast<int8_t>(p_pool.zero_point);
    qback.requantizer = buildRequantizer<int8_t>(p_pool.scale, ws_back,
                                                 p_logits.scale, p_logits.zero_point,
                                                 -128, 127);

    // ----- Int8 / fp16 mixed forward pass.
    int8_t q_in[IN_SIZE], q_feat[S * E];
    fp16_t fp_feat[S * E], fp_attn[S * E], fp_pool[E];
    int8_t q_pool[E], q_logits[NUM_CLASSES];
    float deq_logits[NUM_CLASSES];

    std::vector<std::array<int8_t, NUM_CLASSES>> all_q_logits(NUM_INPUTS);

    tinymind::bench::enableCycleCounter();
    using tinymind::bench::Cycles;
    Cycles c_front = 0, c_relu = 0, c_to_fp16 = 0, c_attn = 0, c_pool = 0, c_to_i8 = 0, c_back = 0;

    float worst_err = 0.0f;

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        quantizeBuffer<int8_t>(inputs[s].data(), q_in, IN_SIZE,
                               p_in.scale, p_in.zero_point, -128, 127);

        Cycles t = tinymind::bench::readCycleCounter();
        for (std::size_t step = 0; step < S; ++step)
            qfront.forward(q_in + step * E, q_feat + step * E);
        Cycles t2 = tinymind::bench::readCycleCounter(); c_front += t2 - t;
        qreluBuffer<int8_t>(q_feat, S * E, static_cast<int8_t>(p_feat.zero_point));
        t = tinymind::bench::readCycleCounter();          c_relu += t - t2;

        // ---- Phase 9 bridge: int8 affine -> fp16.
        for (std::size_t i = 0; i < S * E; ++i)
            fp_feat[i] = affineI8ToFp16(q_feat[i], p_relu.scale, p_relu.zero_point);
        t2 = tinymind::bench::readCycleCounter();        c_to_fp16 += t2 - t;

        // ---- fp16 attention head. Inner arithmetic in float; storage fp16.
        {
            float xf[S * E];
            for (std::size_t i = 0; i < S * E; ++i) xf[i] = fp16ToFloat(fp_feat[i]);
            float yf[S * E];
            flinearAttention(xf, w_head, b_head, yf);
            // Residual skip from fp16 feature buffer keeps the pooled summary
            // off the zero mean a tiny linear-attention head would otherwise
            // force on the bundled synthetic inputs.
            for (std::size_t i = 0; i < S * E; ++i) yf[i] += xf[i];
            for (std::size_t i = 0; i < S * E; ++i) fp_attn[i] = floatToFp16(yf[i]);
        }
        t = tinymind::bench::readCycleCounter();          c_attn += t - t2;

        // Mean-pool over S in fp16 (promote to float for the sum, store fp16).
        for (std::size_t e = 0; e < E; ++e)
        {
            float sum = 0.0f;
            for (std::size_t step = 0; step < S; ++step)
                sum += fp16ToFloat(fp_attn[step * E + e]);
            fp_pool[e] = floatToFp16(sum / static_cast<float>(S));
        }
        t2 = tinymind::bench::readCycleCounter();        c_pool += t2 - t;

        // ---- Phase 9 bridge: fp16 -> int8 affine.
        for (std::size_t e = 0; e < E; ++e)
            q_pool[e] = fp16ToAffineI8(fp_pool[e], p_pool.scale, p_pool.zero_point,
                                       -128, 127);
        t = tinymind::bench::readCycleCounter();          c_to_i8 += t - t2;

        // ---- int8 classifier.
        qback.forward(q_pool, q_logits);
        t2 = tinymind::bench::readCycleCounter();        c_back += t2 - t;

        for (std::size_t i = 0; i < NUM_CLASSES; ++i) all_q_logits[s][i] = q_logits[i];

        dequantizeBuffer<int8_t>(q_logits, deq_logits, NUM_CLASSES,
                                 p_logits.scale, p_logits.zero_point);
        const float err = maxAbsDiff(deq_logits, float_logits[s].data(), NUM_CLASSES);
        if (err > worst_err) worst_err = err;
    }

    if (golden_mode)
    {
        std::printf("# mixed_precision_kws golden output\n");
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
        const std::size_t frontBytes = sizeof(qfront) + sizeof(qw_front) + sizeof(qb_front);
        const std::size_t backBytes  = sizeof(qback)  + sizeof(qw_back)  + sizeof(qb_back);
        const std::size_t headBytes  = sizeof(w_head) + sizeof(b_head); // fp32-host weights, fp16-stored at runtime
        tinymind::bench::writeRow(std::cout, {"int8_front_dense", frontBytes, S * E,          c_front});
        tinymind::bench::writeRow(std::cout, {"int8_front_relu",  0,          S * E,          c_relu});
        tinymind::bench::writeRow(std::cout, {"bridge_i8_to_fp16",0,          (S * E) * 2,    c_to_fp16});
        tinymind::bench::writeRow(std::cout, {"fp16_linear_attn", headBytes,  (S * E) * 2,    c_attn});
        tinymind::bench::writeRow(std::cout, {"fp16_mean_pool",   0,          E * 2,          c_pool});
        tinymind::bench::writeRow(std::cout, {"bridge_fp16_to_i8",0,          E,              c_to_i8});
        tinymind::bench::writeRow(std::cout, {"int8_back_dense",  backBytes,  NUM_CLASSES,    c_back});
        return 0;
    }

    const float lrange = o_logits.max_value - o_logits.min_value;
    std::printf("Mixed-precision KWS (int8 front -> fp16 attn -> int8 back)\n");
    std::printf("  input  range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_in.min_value,    o_in.max_value,    p_in.scale,    p_in.zero_point);
    std::printf("  feat   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_feat.min_value,  o_feat.max_value,  p_feat.scale,  p_feat.zero_point);
    std::printf("  pool   range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_pool.min_value,  o_pool.max_value,  p_pool.scale,  p_pool.zero_point);
    std::printf("  logits range: [%+.3f, %+.3f]  scale=%.5f zp=%d\n", o_logits.min_value,o_logits.max_value,p_logits.scale,p_logits.zero_point);
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
