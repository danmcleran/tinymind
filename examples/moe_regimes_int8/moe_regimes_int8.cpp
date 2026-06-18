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

// int8 Mixture-of-Experts regime-routing exemplar.
//
// A heterogeneous 1D regression: the target f(x) = 0.6*sin(1.5x) + 0.25*x is
// approximated by a top-1 MoE of three LINEAR experts. A tiny linear router
// partitions the input domain into three regimes (x < -1, -1..1, x > 1); each
// expert is the least-squares line fit to its regime. The router's argmax over
// three lines is exactly the upper envelope of those lines, so routing falls
// out as three contiguous x-intervals -- the classic "mixture of linear
// experts approximates a nonlinear function" picture.
//
// Everything is deployed int8 through cpp/qmoe.hpp::QMixtureOfExperts: the
// router argmaxes over raw int32 logits (no requant), each expert carries its
// own per-expert weight scale, and all experts share one output scale.
//
// The headline embedded property: of the three resident experts, exactly ONE
// runs per inference. Active compute is one expert + the router; resident
// memory is all three experts. `--bench` prints that accounting.
//
//   (default)  write output/moe_regimes_int8.csv: x, target, prediction, expert
//   --bench    print the active-vs-resident compute/memory accounting
//   --golden   print golden int8 bytes for a fixed probe set
//
// Standalone: float weights are fit in this binary, so `make check` runs it
// without numpy. The same router + per-expert weights could instead be loaded
// from a tinymind_import-emitted weights.hpp (see examples/import_moe_demo).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "qaffine.hpp"
#include "qcalibration.hpp"
#include "qdense.hpp"
#include "qmoe.hpp"

namespace {

constexpr std::size_t kIn       = 1;
constexpr std::size_t kOut      = 1;
constexpr std::size_t kExperts  = 3;
constexpr int32_t     kQmin     = -128;
constexpr int32_t     kQmax     =  127;

constexpr float kXLo = -3.0f;
constexpr float kXHi =  3.0f;
constexpr float kT0  = -1.0f;   // regime boundary e0|e1
constexpr float kT1  =  1.0f;   // regime boundary e1|e2

typedef tinymind::QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t,
                                    kIn, kOut, kExperts> MoE;

float target(float x)
{
    return 0.6f * std::sin(1.5f * x) + 0.25f * x;
}

// Least-squares line fit y = slope*x + intercept over [lo, hi].
void fitLine(float lo, float hi, float& slope, float& intercept)
{
    const int n = 200;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double x = lo + (hi - lo) * (static_cast<double>(i) / (n - 1));
        const double y = target(static_cast<float>(x));
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double dn = n;
    const double denom = dn * sxx - sx * sx;
    const double m = (dn * sxy - sx * sy) / denom;
    const double b = (sy - m * sx) / dn;
    slope = static_cast<float>(m);
    intercept = static_cast<float>(b);
}

float symmetricWeightScale(const float* w, std::size_t n)
{
    float ax = 0.0f;
    for (std::size_t i = 0; i < n; ++i) ax = std::max(ax, std::abs(w[i]));
    return (ax > 0.0f) ? ax / 127.0f : 1.0f;
}

int32_t quantizeBias(float b, float scale)
{
    const long q = std::lround(static_cast<double>(b) / static_cast<double>(scale));
    return static_cast<int32_t>(
        std::max<long>(INT32_MIN, std::min<long>(INT32_MAX, q)));
}

// Holds everything the int8 MoE needs, plus the float scales for reporting.
struct DeployedMoE
{
    MoE moe;

    // Router (raw int32 logits; no requant).
    int8_t  router_w[kExperts * kIn];
    int32_t router_b[kExperts];

    // Per-expert weights / biases.
    int8_t  expert_w[kExperts][kOut * kIn];
    int32_t expert_b[kExperts][kOut];

    float   expert_w_scale[kExperts];
    float   in_scale;
    int32_t in_zp;
    float   out_scale;
    int32_t out_zp;

    int8_t quantizeInput(float x) const
    {
        return tinymind::quantize<int8_t>(x, in_scale, in_zp, kQmin, kQmax);
    }

    float dequantizeOutput(int8_t q) const
    {
        return tinymind::dequantize<int8_t>(q, out_scale, out_zp);
    }
};

void buildDeployedMoE(DeployedMoE& d)
{
    // ---- Float model: hand-built router + least-squares expert lines. ----
    // Router lines (slope, bias) whose upper envelope crosses at kT0, kT1.
    const float router_slope[kExperts] = {-2.0f, 0.0f, 2.0f};
    const float router_bias[kExperts]  = {-2.0f, 0.0f, -2.0f};

    float expert_slope[kExperts];
    float expert_intercept[kExperts];
    fitLine(kXLo, kT0, expert_slope[0], expert_intercept[0]);
    fitLine(kT0,  kT1, expert_slope[1], expert_intercept[1]);
    fitLine(kT1,  kXHi, expert_slope[2], expert_intercept[2]);

    // ---- Calibration over a dense grid of the input domain. ----
    tinymind::RangeObserver in_obs;
    tinymind::RangeObserver out_obs;
    const int kCal = 241;
    for (int i = 0; i < kCal; ++i)
    {
        const float x = kXLo + (kXHi - kXLo) * (static_cast<float>(i) / (kCal - 1));
        in_obs.observe(&x, 1);

        // Float top-1 forward to observe the realized output range.
        std::size_t e = 0;
        float best = router_slope[0] * x + router_bias[0];
        for (std::size_t k = 1; k < kExperts; ++k)
        {
            const float logit = router_slope[k] * x + router_bias[k];
            if (logit > best) { best = logit; e = k; }
        }
        const float y = expert_slope[e] * x + expert_intercept[e];
        out_obs.observe(&y, 1);
    }

    const auto in_ap = tinymind::computeAffineParamsAsymmetric(
        in_obs.min_value, in_obs.max_value, kQmin, kQmax);
    const auto out_ap = tinymind::computeAffineParamsAsymmetric(
        out_obs.min_value, out_obs.max_value, kQmin, kQmax);

    d.in_scale  = in_ap.scale;
    d.in_zp     = in_ap.zero_point;
    d.out_scale = out_ap.scale;
    d.out_zp    = out_ap.zero_point;

    // ---- Router quantization (symmetric weights; bias at in*router scale). ----
    const float router_w_scale = symmetricWeightScale(router_slope, kExperts);
    for (std::size_t e = 0; e < kExperts; ++e)
    {
        d.router_w[e] = tinymind::quantize<int8_t>(
            router_slope[e], router_w_scale, 0, -127, 127);
        d.router_b[e] = quantizeBias(router_bias[e], d.in_scale * router_w_scale);
    }

    // ---- Per-expert quantization: own weight scale, shared output scale. ----
    for (std::size_t e = 0; e < kExperts; ++e)
    {
        d.expert_w_scale[e] = symmetricWeightScale(&expert_slope[e], 1);
        d.expert_w[e][0] = tinymind::quantize<int8_t>(
            expert_slope[e], d.expert_w_scale[e], 0, -127, 127);
        d.expert_b[e][0] = quantizeBias(
            expert_intercept[e], d.in_scale * d.expert_w_scale[e]);
    }

    // ---- Wire the int8 layer. ----
    d.moe.router_weights   = d.router_w;
    d.moe.router_biases    = d.router_b;
    d.moe.input_zero_point = static_cast<int8_t>(d.in_zp);
    for (std::size_t e = 0; e < kExperts; ++e)
    {
        d.moe.experts[e].weights          = d.expert_w[e];
        d.moe.experts[e].biases           = d.expert_b[e];
        d.moe.experts[e].input_zero_point = static_cast<int8_t>(d.in_zp);
        d.moe.experts[e].requantizer = tinymind::buildRequantizer<int8_t>(
            d.in_scale, d.expert_w_scale[e], d.out_scale, d.out_zp, kQmin, kQmax);
    }
}

int runCsv(const DeployedMoE& d)
{
    std::ofstream csv("moe_regimes_int8.csv");
    csv << "x,target,prediction,expert\n";
    csv << std::fixed << std::setprecision(6);

    const int n = 241;
    float max_abs_err = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        const float x = kXLo + (kXHi - kXLo) * (static_cast<float>(i) / (n - 1));
        const int8_t qx = d.quantizeInput(x);
        int8_t qy = 0;
        const std::size_t e = d.moe.forward(&qx, &qy);
        const float pred = d.dequantizeOutput(qy);
        const float tgt = target(x);
        max_abs_err = std::max(max_abs_err, std::abs(pred - tgt));
        csv << x << "," << tgt << "," << pred << "," << e << "\n";
    }
    csv.close();

    std::cout << "wrote moe_regimes_int8.csv (" << n << " points)\n";
    std::cout << std::fixed << std::setprecision(6)
              << "max |int8 pred - target| = " << max_abs_err << "\n";
    return 0;
}

int runBench(const DeployedMoE& d)
{
    // Parameter accounting. Each linear expert: kOut*kIn weights + kOut bias.
    const std::size_t expert_params = kOut * kIn + kOut;
    const std::size_t router_params = kExperts * kIn + kExperts;
    const std::size_t resident_params = router_params + kExperts * expert_params;
    const std::size_t active_params   = router_params + expert_params;

    // MAC accounting per inference (top-1).
    const std::size_t router_macs = kExperts * kIn;     // logits
    const std::size_t expert_macs = kOut * kIn;          // one expert
    const std::size_t active_macs = router_macs + expert_macs;
    const std::size_t dense_equiv_macs = router_macs + kExperts * expert_macs;

    std::cout << "MoE compute/memory accounting (top-1 routing)\n";
    std::cout << "  experts resident         : " << kExperts << "\n";
    std::cout << "  experts run per inference : 1\n";
    std::cout << "  resident params (bytes)   : " << resident_params
              << "  (router + all experts, all must be in flash)\n";
    std::cout << "  active params per call    : " << active_params
              << "  (router + selected expert)\n";
    std::cout << "  MACs per inference        : " << active_macs
              << "  (router " << router_macs << " + 1 expert " << expert_macs << ")\n";
    std::cout << "  MACs if all experts ran   : " << dense_equiv_macs
              << "  (dense-MoE upper bound)\n";

    // Probe the routing distribution over the domain.
    std::size_t counts[kExperts] = {0, 0, 0};
    const int n = 241;
    for (int i = 0; i < n; ++i)
    {
        const float x = kXLo + (kXHi - kXLo) * (static_cast<float>(i) / (n - 1));
        const int8_t qx = d.quantizeInput(x);
        int32_t logits[kExperts];
        const std::size_t e = d.moe.route(&qx, logits);
        counts[e]++;
    }
    std::cout << "  routing over domain       : ";
    for (std::size_t e = 0; e < kExperts; ++e)
    {
        std::cout << "e" << e << "=" << counts[e] << (e + 1 < kExperts ? " " : "\n");
    }
    return 0;
}

int runGolden(const DeployedMoE& d)
{
    std::cout << "moe_regimes_int8 golden probes\n";
    std::cout << "in_scale=" << d.in_scale << " in_zp=" << d.in_zp
              << " out_scale=" << d.out_scale << " out_zp=" << d.out_zp << "\n";
    const float probes[] = {-3.0f, -2.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.0f, 3.0f};
    std::cout << "x_q,expert,y_q\n";
    for (float x : probes)
    {
        const int8_t qx = d.quantizeInput(x);
        int8_t qy = 0;
        const std::size_t e = d.moe.forward(&qx, &qy);
        std::cout << static_cast<int>(qx) << "," << e << ","
                  << static_cast<int>(qy) << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    DeployedMoE d;
    buildDeployedMoE(d);

    std::string mode = (argc > 1) ? argv[1] : "";
    if (mode == "--bench")  return runBench(d);
    if (mode == "--golden") return runGolden(d);
    return runCsv(d);
}
