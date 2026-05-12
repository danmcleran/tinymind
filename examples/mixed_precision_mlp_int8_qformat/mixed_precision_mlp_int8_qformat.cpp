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

// Phase 17 hybrid mixed-precision exemplar:
//
//   int8 QDense  ->  qrelu  ->  affine->QValue bridge  ->  Q-format dense
//                            ->  QValue->affine bridge  ->  int8 QDense classifier
//
// Demonstrates the offline-training -> embedded-inference path for a model
// that wants the int8 affine grid at the boundaries (matches the PyTorch /
// TFLite weight-export story) but a Q-format middle tier (matches the
// existing TinyMind QValue pipeline, e.g. for code already written against
// QValue<8,8>). The Phase 17 pure-integer qbridge functions span the gap
// so the inference path never touches <cmath> at runtime.
//
// Host-side, FLOAT && STD: calibration drives RangeObserver +
// computeAffineParamsAsymmetric for the two int8 layers, plus
// buildAffineToQValueIntParams / buildQValueToAffineIntParams for the two
// precision boundaries. The deployable target consumes the resulting
// integer triples and runs pure-integer.

#include "qaffine.hpp"
#include "qbridge.hpp"
#include "qdense.hpp"
#include "qactivations.hpp"
#include "qformat.hpp"
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

constexpr std::size_t E_IN     = 6;
constexpr std::size_t E_HIDDEN = 8;
constexpr std::size_t E_OUT    = 4;
constexpr std::size_t NUM_INPUTS = 4;
constexpr std::size_t NUM_CALIB  = 32;

typedef tinymind::QValue<8, 8, true> Q88;

// ---------------------------------------------------------------------------
// Float helpers: reference forward + calibration data generator.
// ---------------------------------------------------------------------------
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

void frelu(float* x, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
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

// ---------------------------------------------------------------------------
// Pure-integer Q-format dense (matvec) + per-output tanh via QValue's
// built-in transfer function. Mirrors a NeuralNet<> hidden layer carrying
// Q8.8 weights, but is single-step inference and caller-owned.
// ---------------------------------------------------------------------------
void qFormatDenseTanh(const Q88* in,
                      const Q88* w, const Q88* b,
                      Q88* out,
                      std::size_t in_dim, std::size_t out_dim)
{
    // MAC in int32 (Q16.16): weight raw int16 * input raw int16 = int32.
    // Bias is added pre-narrow as (b_raw << 8) so it sits in Q16.16.
    // Result narrowed by an arithmetic right-shift of 8 back to Q8.8 with
    // round-to-nearest, then saturated to the int16 raw range.
    const int32_t lo = -32768;
    const int32_t hi = 32767;
    for (std::size_t o = 0; o < out_dim; ++o)
    {
        int32_t acc = static_cast<int32_t>(b[o].getValue()) << 8;
        for (std::size_t i = 0; i < in_dim; ++i)
        {
            acc += static_cast<int32_t>(w[o * in_dim + i].getValue())
                 * static_cast<int32_t>(in[i].getValue());
        }
        // round-to-nearest right-shift by 8.
        const int32_t rounded = (acc >= 0) ? acc + (1 << 7) : acc - (1 << 7);
        int32_t narrowed = rounded >> 8;
        if (narrowed < lo) narrowed = lo;
        if (narrowed > hi) narrowed = hi;
        Q88 pre_act(static_cast<Q88::FullWidthValueType>(narrowed));
        // Activation: identity here (the integer bridge handles the tanh-free
        // case more transparently). A real model would call into QValue's
        // tanh transfer function on this raw value; that path uses the
        // existing lookupTables.cpp data and is independent of Phase 17.
        out[o] = pre_act;
    }
}

} // namespace

int main(int argc, char** argv)
{
    using tinymind::QDense;
    using tinymind::RangeObserver;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::buildRequantizer;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;
    using tinymind::qreluBuffer;
    using tinymind::AffineToQValueIntParams;
    using tinymind::QValueToAffineIntParams;
    using tinymind::buildAffineToQValueIntParams;
    using tinymind::buildQValueToAffineIntParams;
    using tinymind::affineToQValueIntBuffer;
    using tinymind::qValueToAffineIntBuffer;
    using tinymind::floatToQValue;
    using tinymind::qValueToFloat;

    const bool bench_mode  = (argc >= 2) && std::strcmp(argv[1], "--bench")  == 0;
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    // ----- Hand-crafted weights for the two int8 layers + the Q-format mid.
    constexpr std::size_t W_FRONT = E_HIDDEN * E_IN;
    constexpr std::size_t W_MID   = E_HIDDEN * E_HIDDEN;
    constexpr std::size_t W_BACK  = E_OUT    * E_HIDDEN;

    float w_front[W_FRONT], b_front[E_HIDDEN] = {};
    float w_mid  [W_MID],   b_mid  [E_HIDDEN] = {};
    float w_back [W_BACK],  b_back [E_OUT]    = {};
    fillW(w_front, W_FRONT, 0.30f, 5.0f, 0.2f);
    fillW(w_mid,   W_MID,   0.25f, 4.0f, 0.5f);
    fillW(w_back,  W_BACK,  0.40f, 3.0f, 0.7f);

    // ----- Synthetic calibration dataset.
    std::vector<std::vector<float>> inputs(NUM_CALIB, std::vector<float>(E_IN));
    for (std::size_t s = 0; s < NUM_CALIB; ++s)
        for (std::size_t i = 0; i < E_IN; ++i)
        {
            const float phase = static_cast<float>(s * 9 + i) * 0.17f;
            inputs[s][i] = 0.7f * std::sin(phase) +
                           0.2f * std::cos(0.4f * static_cast<float>(s + i));
        }

    // ----- Float reference forward + range observation.
    RangeObserver o_in, o_h1, o_h1_relu, o_h2, o_out;
    std::vector<std::vector<float>> float_logits(NUM_CALIB, std::vector<float>(E_OUT));

    float h1[E_HIDDEN], h1_relu[E_HIDDEN], h2[E_HIDDEN], h2_act[E_HIDDEN];
    for (std::size_t s = 0; s < NUM_CALIB; ++s)
    {
        fdense(inputs[s].data(), w_front, b_front, h1, E_IN, E_HIDDEN);
        std::memcpy(h1_relu, h1, sizeof(h1));
        frelu(h1_relu, E_HIDDEN);

        fdense(h1_relu, w_mid, b_mid, h2, E_HIDDEN, E_HIDDEN);
        std::memcpy(h2_act, h2, sizeof(h2));
        // Skip the tanh in float ref to match the integer Q-format middle
        // layer's identity activation (exemplar focuses on the precision
        // bridges; activation parity belongs to the existing QValue tests).

        fdense(h2_act, w_back, b_back, float_logits[s].data(), E_HIDDEN, E_OUT);

        o_in     .observe(inputs[s].data(), E_IN);
        o_h1     .observe(h1, E_HIDDEN);
        o_h1_relu.observe(h1_relu, E_HIDDEN);
        o_h2     .observe(h2_act, E_HIDDEN);
        o_out    .observe(float_logits[s].data(), E_OUT);
    }

    // ----- Affine params for the two int8 dense layers' boundaries.
    const auto p_in  = computeAffineParamsAsymmetric(o_in.min_value,
                                                     o_in.max_value, -128, 127);
    const auto p_h1  = computeAffineParamsAsymmetric(o_h1.min_value,
                                                     o_h1.max_value, -128, 127);
    // post-ReLU lives on p_h1 grid (clamp at zero_point).
    const auto& p_h1_relu = p_h1;
    const auto p_h2  = computeAffineParamsAsymmetric(o_h2.min_value,
                                                     o_h2.max_value, -128, 127);
    const auto p_out = computeAffineParamsAsymmetric(o_out.min_value,
                                                     o_out.max_value, -128, 127);

    // ----- Phase 17 integer bridge params, built host-side.
    //   bridge_to_q  : (int8 affine, p_h1_relu) -> Q88
    //   bridge_to_i8 : Q88 -> (int8 affine, p_h2)
    const AffineToQValueIntParams<Q88> bridge_to_q =
        buildAffineToQValueIntParams<Q88>(p_h1_relu.scale, p_h1_relu.zero_point);
    const QValueToAffineIntParams<Q88> bridge_to_i8 =
        buildQValueToAffineIntParams<Q88>(p_h2.scale, p_h2.zero_point, -128, 127);

    // ----- Symmetric int8 weights + int32 biases for the two QDense layers.
    float front_abs = 0.0f;
    for (std::size_t i = 0; i < W_FRONT; ++i)
    { const float a = std::fabs(w_front[i]); if (a > front_abs) front_abs = a; }
    const float ws_front = (front_abs > 0.0f) ? front_abs / 127.0f : 1.0f;

    float back_abs = 0.0f;
    for (std::size_t i = 0; i < W_BACK; ++i)
    { const float a = std::fabs(w_back[i]); if (a > back_abs) back_abs = a; }
    const float ws_back = (back_abs > 0.0f) ? back_abs / 127.0f : 1.0f;

    int8_t qw_front[W_FRONT], qw_back[W_BACK];
    int32_t qb_front[E_HIDDEN] = {}, qb_back[E_OUT] = {};
    quantizeBuffer<int8_t>(w_front, qw_front, W_FRONT, ws_front, 0, -128, 127);
    quantizeBuffer<int8_t>(w_back,  qw_back,  W_BACK,  ws_back,  0, -128, 127);
    {
        const double sf = static_cast<double>(p_in.scale)
                        * static_cast<double>(ws_front);
        for (std::size_t o = 0; o < E_HIDDEN; ++o)
            qb_front[o] = static_cast<int32_t>(std::lround(
                static_cast<double>(b_front[o]) / sf));
        const double sb = static_cast<double>(p_h2.scale)
                        * static_cast<double>(ws_back);
        for (std::size_t o = 0; o < E_OUT; ++o)
            qb_back[o] = static_cast<int32_t>(std::lround(
                static_cast<double>(b_back[o]) / sb));
    }

    // ----- Q-format weights for the middle dense layer.
    Q88 qw_mid[W_MID], qb_mid[E_HIDDEN];
    for (std::size_t i = 0; i < W_MID;   ++i) qw_mid[i] = floatToQValue<Q88>(w_mid[i]);
    for (std::size_t i = 0; i < E_HIDDEN; ++i) qb_mid[i] = floatToQValue<Q88>(b_mid[i]);

    // ----- Layer instances.
    QDense<int8_t, int8_t, int32_t, int8_t, E_IN, E_HIDDEN> qfront;
    qfront.weights = qw_front;
    qfront.biases  = qb_front;
    qfront.input_zero_point = static_cast<int8_t>(p_in.zero_point);
    qfront.requantizer = buildRequantizer<int8_t>(
        p_in.scale, ws_front, p_h1.scale, p_h1.zero_point, -128, 127);

    QDense<int8_t, int8_t, int32_t, int8_t, E_HIDDEN, E_OUT> qback;
    qback.weights = qw_back;
    qback.biases  = qb_back;
    qback.input_zero_point = static_cast<int8_t>(p_h2.zero_point);
    qback.requantizer = buildRequantizer<int8_t>(
        p_h2.scale, ws_back, p_out.scale, p_out.zero_point, -128, 127);

    // ----- Mixed-precision forward (int8 frontend -> Q-format mid -> int8 back).
    int8_t q_in[E_IN], q_h1[E_HIDDEN], q_h2[E_HIDDEN], q_logits[E_OUT];
    Q88    qf_h1[E_HIDDEN], qf_h2[E_HIDDEN];
    float  deq_logits[E_OUT];

    std::vector<std::array<int8_t, E_OUT>> all_q_logits(NUM_INPUTS);

    tinymind::bench::enableCycleCounter();
    using tinymind::bench::Cycles;
    Cycles c_front = 0, c_relu = 0, c_to_q = 0, c_mid = 0, c_to_i8 = 0, c_back = 0;

    float worst_err = 0.0f;

    for (std::size_t s = 0; s < NUM_INPUTS; ++s)
    {
        quantizeBuffer<int8_t>(inputs[s].data(), q_in, E_IN,
                               p_in.scale, p_in.zero_point, -128, 127);

        Cycles t = tinymind::bench::readCycleCounter();
        qfront.forward(q_in, q_h1);
        Cycles t2 = tinymind::bench::readCycleCounter(); c_front += t2 - t;

        qreluBuffer<int8_t>(q_h1, E_HIDDEN, static_cast<int8_t>(p_h1.zero_point));
        t = tinymind::bench::readCycleCounter();          c_relu += t - t2;

        // ---- Phase 17 bridge: int8 affine -> Q88 (pure integer).
        affineToQValueIntBuffer<Q88, int8_t>(q_h1, qf_h1, E_HIDDEN, bridge_to_q);
        t2 = tinymind::bench::readCycleCounter();        c_to_q += t2 - t;

        // ---- Q-format dense (Q88 matvec, int32 accumulator).
        qFormatDenseTanh(qf_h1, qw_mid, qb_mid, qf_h2, E_HIDDEN, E_HIDDEN);
        t = tinymind::bench::readCycleCounter();          c_mid += t - t2;

        // ---- Phase 17 bridge: Q88 -> int8 affine (pure integer).
        qValueToAffineIntBuffer<Q88, int8_t>(qf_h2, q_h2, E_HIDDEN, bridge_to_i8);
        t2 = tinymind::bench::readCycleCounter();        c_to_i8 += t2 - t;

        // ---- int8 classifier.
        qback.forward(q_h2, q_logits);
        t = tinymind::bench::readCycleCounter();          c_back += t - t2;

        for (std::size_t i = 0; i < E_OUT; ++i) all_q_logits[s][i] = q_logits[i];

        dequantizeBuffer<int8_t>(q_logits, deq_logits, E_OUT,
                                 p_out.scale, p_out.zero_point);
        const float err = maxAbsDiff(deq_logits, float_logits[s].data(), E_OUT);
        if (err > worst_err) worst_err = err;
    }

    if (golden_mode)
    {
        std::printf("# mixed_precision_mlp_int8_qformat golden output\n");
        std::printf("# samples=%zu classes=%zu\n", NUM_INPUTS, E_OUT);
        for (std::size_t s = 0; s < NUM_INPUTS; ++s)
        {
            std::printf("sample %zu:", s);
            for (std::size_t i = 0; i < E_OUT; ++i)
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
        const std::size_t midBytes   = sizeof(qw_mid) + sizeof(qb_mid);
        tinymind::bench::writeRow(std::cout, {"int8_front_dense", frontBytes, E_HIDDEN,        c_front});
        tinymind::bench::writeRow(std::cout, {"int8_front_relu",  0,          E_HIDDEN,        c_relu});
        tinymind::bench::writeRow(std::cout, {"bridge_i8_to_q88", 0,          E_HIDDEN * sizeof(Q88), c_to_q});
        tinymind::bench::writeRow(std::cout, {"q88_mid_dense",    midBytes,   E_HIDDEN * sizeof(Q88), c_mid});
        tinymind::bench::writeRow(std::cout, {"bridge_q88_to_i8", 0,          E_HIDDEN,        c_to_i8});
        tinymind::bench::writeRow(std::cout, {"int8_back_dense",  backBytes,  E_OUT,           c_back});
        return 0;
    }

    const float lrange = o_out.max_value - o_out.min_value;
    const float tol = (lrange > 0.0f) ? 0.6f * lrange : 1.0f;
    std::cout << "mixed_precision_mlp_int8_qformat: worst max-abs error vs float "
              << "= " << worst_err
              << " (tolerance " << tol
              << ", logits range " << lrange << ")\n";
    if (worst_err <= tol)
    {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL\n";
    return 1;
}
