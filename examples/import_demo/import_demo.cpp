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

// Phase 15 importer demo.
//
// Demonstrates the full host-side import path end-to-end without a
// PyTorch dependency:
//
//   * A small MLP (3 -> 8 -> 4 -> 2 sigmoid) carries deterministic float
//     weights baked into this binary.
//   * A 64-sample calibration dataset drives a forward pass; per-tensor
//     activation ranges are observed with a mix of RangeObserver,
//     PercentileObserver, and KLDivergenceObserver -- the three
//     Phase 15 calibration upgrades.
//   * Symmetric per-tensor weight scales fit, int8 weights + int32
//     biases emitted, per-layer Requantizers built via
//     buildRequantizer / buildQSigmoidLUT.
//   * Optional Cross-Layer Equalization pass between fc1 / fc2 to
//     show the Phase 15 weight-balancing transform.
//   * Pure-integer forward over a held-out test set.
//   * Max-abs error between the float reference and the int8 output
//     reported; a sanity tolerance gates exit status.
//
// In a production flow the float weights and calibration ranges come
// out of apps/import_pytorch/tinymind_import (which consumes a
// torch.state_dict and emits a TinyMind-format weights.hpp). This
// binary stands alone so `make check` can run it without torch.

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "qaffine.hpp"
#include "qcalibration.hpp"
#include "qactivations.hpp"
#include "qdense.hpp"

namespace {

constexpr std::size_t kIn   = 3;
constexpr std::size_t kH1   = 8;
constexpr std::size_t kH2   = 4;
constexpr std::size_t kOut  = 2;
constexpr int32_t    kQmin = -128;
constexpr int32_t    kQmax =  127;

// Float reference forward (sigmoid output).
struct FloatMLP
{
    float fc1_w[kH1 * kIn];
    float fc1_b[kH1];
    float fc2_w[kH2 * kH1];
    float fc2_b[kH2];
    float fc3_w[kOut * kH2];
    float fc3_b[kOut];

    void forward(const float* x, float* h1, float* h2,
                 float* logit, float* y) const
    {
        for (std::size_t i = 0; i < kH1; ++i)
        {
            float acc = fc1_b[i];
            for (std::size_t k = 0; k < kIn; ++k)
            {
                acc += fc1_w[i * kIn + k] * x[k];
            }
            h1[i] = (acc > 0.0f) ? acc : 0.0f;
        }
        for (std::size_t i = 0; i < kH2; ++i)
        {
            float acc = fc2_b[i];
            for (std::size_t k = 0; k < kH1; ++k)
            {
                acc += fc2_w[i * kH1 + k] * h1[k];
            }
            h2[i] = (acc > 0.0f) ? acc : 0.0f;
        }
        for (std::size_t i = 0; i < kOut; ++i)
        {
            float acc = fc3_b[i];
            for (std::size_t k = 0; k < kH2; ++k)
            {
                acc += fc3_w[i * kH2 + k] * h2[k];
            }
            logit[i] = acc;
            y[i] = 1.0f / (1.0f + std::exp(-acc));
        }
    }
};

void initWeights(FloatMLP& m, std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-0.75f, 0.75f);
    auto sample = [&](float* dst, std::size_t n, float bias_shift) {
        for (std::size_t i = 0; i < n; ++i) dst[i] = u(rng);
        // Bias channel imbalance to exercise CLE — channel 0 of fc1 gets
        // unusually small weights, channel 1 unusually large.
        (void)bias_shift;
    };
    sample(m.fc1_w, kH1 * kIn, 0.0f);
    sample(m.fc1_b, kH1, 0.0f);
    // Engineered per-channel imbalance on fc1 output / fc2 input.
    for (std::size_t k = 0; k < kIn; ++k) m.fc1_w[0 * kIn + k] *= 0.05f;
    for (std::size_t k = 0; k < kIn; ++k) m.fc1_w[1 * kIn + k] *= 6.0f;
    m.fc1_b[0] *= 0.05f;
    m.fc1_b[1] *= 6.0f;
    sample(m.fc2_w, kH2 * kH1, 0.0f);
    // Compensate on fc2 input side so the model still produces useful values.
    for (std::size_t r = 0; r < kH2; ++r) m.fc2_w[r * kH1 + 0] *= 6.0f;
    for (std::size_t r = 0; r < kH2; ++r) m.fc2_w[r * kH1 + 1] *= 0.05f;
    sample(m.fc2_b, kH2, 0.0f);
    sample(m.fc3_w, kOut * kH2, 0.0f);
    sample(m.fc3_b, kOut, 0.0f);
}

void calibrationDataset(std::vector<std::array<float, kIn>>& out,
                        std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    out.resize(64);
    for (auto& row : out)
    {
        for (std::size_t k = 0; k < kIn; ++k) row[k] = u(rng);
    }
}

} // namespace

int main()
{
    std::mt19937 rng(1337u);
    FloatMLP m_orig;
    initWeights(m_orig, rng);
    FloatMLP m = m_orig;

    std::vector<std::array<float, kIn>> cal;
    calibrationDataset(cal, rng);

    // ---- Optional: Cross-Layer Equalize fc1 <-> fc2 weights pre-quant. ----
    // Verify floats outputs unchanged afterward.
    float sample_x[kIn] = {0.3f, -0.4f, 0.6f};
    float h1a[kH1]; float h2a[kH2]; float lga[kOut]; float ya[kOut];
    m.forward(sample_x, h1a, h2a, lga, ya);
    tinymind::crossLayerEqualizeDense(m.fc1_w, m.fc1_b, m.fc2_w,
                                      kIn, kH1, kH2);
    float h1b[kH1]; float h2b[kH2]; float lgb[kOut]; float yb[kOut];
    m.forward(sample_x, h1b, h2b, lgb, yb);
    float cle_drift = 0.0f;
    for (std::size_t i = 0; i < kOut; ++i)
    {
        cle_drift = std::max(cle_drift, std::abs(ya[i] - yb[i]));
    }
    std::cout << "CLE float drift (ReLU model): "
              << std::scientific << std::setprecision(3) << cle_drift << "\n";

    // ---- Calibration pass over the float forward.
    tinymind::RangeObserver       in_obs;
    tinymind::RangeObserver       h1_obs;
    tinymind::PercentileObserver  h2_obs;     // percentile clip
    tinymind::KLDivergenceObserver lg_obs;    // KL clip for the logit
    h2_obs.reserve(cal.size() * kH2);

    // KL observer needs absmax first; pass over the dataset twice.
    {
        float h1[kH1]; float h2[kH2]; float logit[kOut]; float y[kOut];
        for (const auto& row : cal)
        {
            in_obs.observe(row.data(), kIn);
            m.forward(row.data(), h1, h2, logit, y);
            h1_obs.observe(h1, kH1);
            h2_obs.observe(h2, kH2);
            for (std::size_t k = 0; k < kOut; ++k) lg_obs.observeAbsRange(logit[k]);
        }
    }
    {
        float h1[kH1]; float h2[kH2]; float logit[kOut]; float y[kOut];
        for (const auto& row : cal)
        {
            m.forward(row.data(), h1, h2, logit, y);
            for (std::size_t k = 0; k < kOut; ++k) lg_obs.observeHistogram(logit[k]);
        }
    }

    const auto in_ap = tinymind::computeAffineParamsAsymmetric(
        in_obs.min_value, in_obs.max_value, kQmin, kQmax);
    const auto h1_ap = tinymind::computeAffineParamsAsymmetric(
        h1_obs.min_value, h1_obs.max_value, kQmin, kQmax);
    float h2_lo = 0.0f; float h2_hi = 0.0f;
    h2_obs.rangeAtPercentile(0.05f, 99.95f, h2_lo, h2_hi);
    const auto h2_ap = tinymind::computeAffineParamsAsymmetric(
        h2_lo, h2_hi, kQmin, kQmax);
    const float lg_threshold = lg_obs.computeThreshold();
    const auto lg_ap = tinymind::computeAffineParamsAsymmetric(
        -lg_threshold, lg_threshold, kQmin, kQmax);
    constexpr float sigmoid_scale = 1.0f / 256.0f;
    constexpr int32_t sigmoid_zp = -128;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ranges:\n"
              << "  input  scale=" << in_ap.scale << " zp=" << in_ap.zero_point << "\n"
              << "  h1     scale=" << h1_ap.scale << " zp=" << h1_ap.zero_point << "\n"
              << "  h2     scale=" << h2_ap.scale << " zp=" << h2_ap.zero_point
              << " (percentile [" << h2_lo << ", " << h2_hi << "])\n"
              << "  logit  scale=" << lg_ap.scale << " zp=" << lg_ap.zero_point
              << " (kl threshold=" << lg_threshold << ")\n";

    // ---- Weight quantization (symmetric per-tensor, qmax_signed = 127).
    const float fc1_w_scale = [&] {
        float ax = 0.0f;
        for (std::size_t i = 0; i < kH1 * kIn; ++i)
            ax = std::max(ax, std::abs(m.fc1_w[i]));
        return (ax > 0.0f) ? ax / 127.0f : 1.0f;
    }();
    const float fc2_w_scale = [&] {
        float ax = 0.0f;
        for (std::size_t i = 0; i < kH2 * kH1; ++i)
            ax = std::max(ax, std::abs(m.fc2_w[i]));
        return (ax > 0.0f) ? ax / 127.0f : 1.0f;
    }();
    const float fc3_w_scale = [&] {
        float ax = 0.0f;
        for (std::size_t i = 0; i < kOut * kH2; ++i)
            ax = std::max(ax, std::abs(m.fc3_w[i]));
        return (ax > 0.0f) ? ax / 127.0f : 1.0f;
    }();

    std::vector<int8_t>  q_fc1_w(kH1 * kIn);
    std::vector<int32_t> q_fc1_b(kH1);
    std::vector<int8_t>  q_fc2_w(kH2 * kH1);
    std::vector<int32_t> q_fc2_b(kH2);
    std::vector<int8_t>  q_fc3_w(kOut * kH2);
    std::vector<int32_t> q_fc3_b(kOut);

    for (std::size_t i = 0; i < kH1 * kIn; ++i)
    {
        q_fc1_w[i] = tinymind::quantize<int8_t>(
            m.fc1_w[i], fc1_w_scale, 0, -127, 127);
    }
    for (std::size_t i = 0; i < kH2 * kH1; ++i)
    {
        q_fc2_w[i] = tinymind::quantize<int8_t>(
            m.fc2_w[i], fc2_w_scale, 0, -127, 127);
    }
    for (std::size_t i = 0; i < kOut * kH2; ++i)
    {
        q_fc3_w[i] = tinymind::quantize<int8_t>(
            m.fc3_w[i], fc3_w_scale, 0, -127, 127);
    }
    auto quant_bias = [&](const float* b, std::size_t n, float scale, int32_t* dst) {
        for (std::size_t i = 0; i < n; ++i)
        {
            const long q = std::lround(static_cast<double>(b[i]) /
                                       static_cast<double>(scale));
            const long qclip = std::max<long>(static_cast<long>(INT32_MIN),
                                              std::min<long>(static_cast<long>(INT32_MAX), q));
            dst[i] = static_cast<int32_t>(qclip);
        }
    };
    quant_bias(m.fc1_b, kH1,  in_ap.scale * fc1_w_scale, q_fc1_b.data());
    quant_bias(m.fc2_b, kH2,  h1_ap.scale * fc2_w_scale, q_fc2_b.data());
    quant_bias(m.fc3_b, kOut, h2_ap.scale * fc3_w_scale, q_fc3_b.data());

    // ---- Layer wiring.
    typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, kIn,  kH1>  QFc1;
    typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, kH1, kH2>   QFc2;
    typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, kH2, kOut>  QFc3;

    QFc1 fc1; QFc2 fc2; QFc3 fc3;
    fc1.weights = q_fc1_w.data();
    fc1.biases  = q_fc1_b.data();
    fc1.input_zero_point = static_cast<int8_t>(in_ap.zero_point);
    fc1.requantizer = tinymind::buildRequantizer<int8_t>(
        in_ap.scale, fc1_w_scale, h1_ap.scale, h1_ap.zero_point, kQmin, kQmax);
    fc2.weights = q_fc2_w.data();
    fc2.biases  = q_fc2_b.data();
    fc2.input_zero_point = static_cast<int8_t>(h1_ap.zero_point);
    fc2.requantizer = tinymind::buildRequantizer<int8_t>(
        h1_ap.scale, fc2_w_scale, h2_ap.scale, h2_ap.zero_point, kQmin, kQmax);
    fc3.weights = q_fc3_w.data();
    fc3.biases  = q_fc3_b.data();
    fc3.input_zero_point = static_cast<int8_t>(h2_ap.zero_point);
    fc3.requantizer = tinymind::buildRequantizer<int8_t>(
        h2_ap.scale, fc3_w_scale, lg_ap.scale, lg_ap.zero_point, kQmin, kQmax);

    int8_t sigmoid_lut[tinymind::kQActivationLUTSize];
    tinymind::buildQSigmoidLUT(lg_ap.scale, lg_ap.zero_point,
                               sigmoid_scale, sigmoid_zp, sigmoid_lut);

    // ---- Parity check over held-out test set.
    std::vector<std::array<float, kIn>> tests;
    {
        std::mt19937 trng(99u);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        tests.resize(16);
        for (auto& row : tests)
        {
            for (std::size_t k = 0; k < kIn; ++k) row[k] = u(trng);
        }
    }

    float max_abs_err = 0.0f;
    for (const auto& row : tests)
    {
        // Float reference (on the post-CLE weights -- output identical
        // to the pre-CLE forward by construction).
        float h1[kH1]; float h2[kH2]; float lg[kOut]; float yf[kOut];
        m.forward(row.data(), h1, h2, lg, yf);

        // int8 path.
        int8_t qx[kIn];
        for (std::size_t k = 0; k < kIn; ++k)
        {
            qx[k] = tinymind::quantize<int8_t>(
                row[k], in_ap.scale, in_ap.zero_point, kQmin, kQmax);
        }
        int8_t qh1[kH1]; int8_t qh2[kH2]; int8_t qlg[kOut]; int8_t qy[kOut];
        fc1.forward(qx, qh1);
        tinymind::qreluBuffer(qh1, kH1, static_cast<int8_t>(h1_ap.zero_point));
        fc2.forward(qh1, qh2);
        tinymind::qreluBuffer(qh2, kH2, static_cast<int8_t>(h2_ap.zero_point));
        fc3.forward(qh2, qlg);
        for (std::size_t i = 0; i < kOut; ++i)
        {
            qy[i] = tinymind::qApplyLUT(qlg[i], sigmoid_lut);
        }
        for (std::size_t i = 0; i < kOut; ++i)
        {
            const float yq = tinymind::dequantize<int8_t>(
                qy[i], sigmoid_scale, sigmoid_zp);
            const float err = std::abs(yq - yf[i]);
            if (err > max_abs_err) max_abs_err = err;
        }
    }

    std::cout << "\nimport_demo parity test (16 samples)\n"
              << "  max |y_int8 - y_float| = " << max_abs_err << "\n";

    // The sigmoid output spans roughly [0, 1]; 1 LSB of the 1/256 grid is
    // ~0.004, and accumulated layer-by-layer rounding lifts that. 0.08
    // (~20 LSBs) is a comfortable tolerance for a 3-layer MLP calibrated
    // off a 64-sample dataset.
    constexpr float kTolerance = 0.08f;
    const bool ok = (max_abs_err <= kTolerance);
    std::cout << "  tolerance " << kTolerance << " : "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
