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

// Diagonal state-space (linear-recurrent / S4-lite) layer in int8, running as
// a streaming sequence filter. Two variants from cpp/qssm.hpp:
//
//   QStateSpace1D            (LTI)        s_t = a*s_{t-1} + b*x_t ; y = c*s + d*x
//   QSelectiveStateSpace1D   (gated)      s_t = a*s_{t-1} + g_t*b*x_t ; g_t = hardsigmoid(wg*x+bg)
//
// The recurrence runs NumChannels independent first-order IIRs -- a per-channel
// echo/smoothing filter whose pole is a[c]. The selective variant makes the
// input drive content-dependent through a cheap per-channel hard-sigmoid gate,
// the selectivity a Mamba-style model adds on top of a fixed linear recurrence.
//
// The decode state is a fixed NumChannels-wide int32 vector, CONSTANT in the
// sequence length, so the layer streams an arbitrarily long signal in O(C)
// memory -- ideal for always-on sensor / audio. The driver runs the int8
// layer two ways -- one full-sequence forward() and step() looped timestep by
// timestep -- and asserts they are byte-identical, then reports max-abs error
// vs a float reference for both the LTI and the selective variant.

#include "qaffine.hpp"
#include "qssm.hpp"
#include "include/qcalibration.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::size_t T = 32;  // sequence length
constexpr std::size_t C = 4;   // channels (independent recurrences)

float absmaxBuf(const float* b, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { const float a = (b[i] < 0.0f) ? -b[i] : b[i]; if (a > m) m = a; }
    return m;
}
float maxAbsDiff(const float* a, const float* b, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { const float d = std::fabs(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

void floatLTI(const float* x, const float* a, const float* b, const float* c, const float* d, float* y)
{
    float s[C]; for (std::size_t ch = 0; ch < C; ++ch) s[ch] = 0.0f;
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t ch = 0; ch < C; ++ch)
        {
            s[ch] = a[ch] * s[ch] + b[ch] * x[t * C + ch];
            y[t * C + ch] = c[ch] * s[ch] + d[ch] * x[t * C + ch];
        }
}
void floatSelective(const float* x, const float* a, const float* b, const float* c, const float* d,
                    const float* wg, const float* bg, float* y)
{
    float s[C]; for (std::size_t ch = 0; ch < C; ++ch) s[ch] = 0.0f;
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t ch = 0; ch < C; ++ch)
        {
            float g = wg[ch] * x[t * C + ch] + bg[ch];
            if (g < 0.0f) g = 0.0f;
            if (g > 1.0f) g = 1.0f;
            s[ch] = a[ch] * s[ch] + g * b[ch] * x[t * C + ch];
            y[t * C + ch] = c[ch] * s[ch] + d[ch] * x[t * C + ch];
        }
}

} // namespace

int main(int argc, char** argv)
{
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::quantizeBuffer;
    using tinymind::dequantizeBuffer;

    // ----- Synthetic multi-channel input signal (deterministic).
    float x[T * C];
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t ch = 0; ch < C; ++ch)
        {
            const float f = 0.25f + 0.20f * static_cast<float>(ch);
            const float tt = static_cast<float>(t);
            x[t * C + ch] = 0.8f * std::sin(f * tt) * std::exp(-0.02f * tt)
                          + 0.15f * std::sin(1.7f * tt + static_cast<float>(ch));
        }

    // ----- Coefficients. |a| < 1 for stability; a mix of poles.
    float a[C] = {0.88f, -0.55f, 0.70f, 0.30f};
    float b[C] = {0.50f, 0.70f, -0.45f, 0.60f};
    float c[C] = {0.90f, 0.65f, 0.80f, -0.70f};
    float d[C] = {0.10f, -0.08f, 0.12f, 0.05f};
    float wg[C] = {1.4f, 0.9f, -1.1f, 0.7f};
    float bg[C] = {0.3f, 0.1f, 0.5f, 0.2f};

    float y_lti[T * C], y_sel[T * C];
    floatLTI(x, a, b, c, d, y_lti);
    floatSelective(x, a, b, c, d, wg, bg, y_sel);

    // ----- State ranges for calibration (state stored int32).
    auto stateAbsmax = [&](bool selective) -> float
    {
        float s[C] = {0, 0, 0, 0}; float m = 0.0f;
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t ch = 0; ch < C; ++ch)
            {
                float drive = b[ch] * x[t * C + ch];
                if (selective)
                {
                    float g = wg[ch] * x[t * C + ch] + bg[ch];
                    if (g < 0.0f) g = 0.0f;
                    if (g > 1.0f) g = 1.0f;
                    drive *= g;
                }
                s[ch] = a[ch] * s[ch] + drive;
                const float av = std::fabs(s[ch]); if (av > m) m = av;
            }
        return m;
    };

    // ----- Grids.
    const float x_abs = absmaxBuf(x, T * C);
    const auto px = computeAffineParamsAsymmetric(-x_abs, x_abs, -128, 127);
    const auto py_lti = computeAffineParamsAsymmetric(-absmaxBuf(y_lti, T * C), absmaxBuf(y_lti, T * C), -128, 127);
    const auto py_sel = computeAffineParamsAsymmetric(-absmaxBuf(y_sel, T * C), absmaxBuf(y_sel, T * C), -128, 127);
    const float s_lti = stateAbsmax(false) / 32767.0f;
    const float s_sel = stateAbsmax(true) / 32767.0f;

    int8_t x_q[T * C];
    quantizeBuffer<int8_t>(x, x_q, T * C, px.scale, px.zero_point, -128, 127);

    // ----- LTI int8 layer.
    int32_t am[C], ash[C], bm[C], bsh[C], cm[C], csh[C], dm[C], dsh[C];
    tinymind::buildQSSMParams(a, b, c, d, px.scale, s_lti, py_lti.scale, C,
                              am, ash, bm, bsh, cm, csh, dm, dsh);
    tinymind::QStateSpace1D<int8_t, int32_t, int8_t, T, C> lti;
    lti.input_zero_point = static_cast<int8_t>(px.zero_point);
    lti.output_zero_point = static_cast<int8_t>(py_lti.zero_point);
    lti.qmin = -128; lti.qmax = 127;
    lti.a_multiplier = am; lti.a_shift = ash; lti.b_multiplier = bm; lti.b_shift = bsh;
    lti.c_multiplier = cm; lti.c_shift = csh; lti.d_multiplier = dm; lti.d_shift = dsh;

    decltype(lti)::State lti_full;
    int8_t y_lti_full[T * C];
    lti.forward(x_q, lti_full, y_lti_full);
    decltype(lti)::State lti_step; lti_step.reset();
    int8_t y_lti_step[T * C];
    for (std::size_t t = 0; t < T; ++t) lti.step(x_q + t * C, lti_step, y_lti_step + t * C);

    bool equiv = true; std::size_t mism = 0;
    for (std::size_t i = 0; i < T * C; ++i) if (y_lti_full[i] != y_lti_step[i]) { equiv = false; ++mism; }

    // ----- Selective int8 layer.
    int32_t s_am[C], s_ash[C], s_bm[C], s_bsh[C], s_cm[C], s_csh[C];
    int32_t s_dm[C], s_dsh[C], gm[C], gsh[C], gb[C];
    tinymind::buildQSSMParams(a, b, c, d, px.scale, s_sel, py_sel.scale, C,
                              s_am, s_ash, s_bm, s_bsh, s_cm, s_csh, s_dm, s_dsh);
    tinymind::buildQSelectiveGateParams(wg, bg, px.scale, C, gm, gsh, gb);
    tinymind::QSelectiveStateSpace1D<int8_t, int32_t, int8_t, T, C> sel;
    sel.input_zero_point = static_cast<int8_t>(px.zero_point);
    sel.output_zero_point = static_cast<int8_t>(py_sel.zero_point);
    sel.qmin = -128; sel.qmax = 127;
    sel.a_multiplier = s_am; sel.a_shift = s_ash; sel.b_multiplier = s_bm; sel.b_shift = s_bsh;
    sel.c_multiplier = s_cm; sel.c_shift = s_csh; sel.d_multiplier = s_dm; sel.d_shift = s_dsh;
    sel.gate_multiplier = gm; sel.gate_shift = gsh; sel.gate_bias = gb;

    decltype(sel)::State sel_full;
    int8_t y_sel_full[T * C];
    sel.forward(x_q, sel_full, y_sel_full);
    decltype(sel)::State sel_step; sel_step.reset();
    int8_t y_sel_step[T * C];
    for (std::size_t t = 0; t < T; ++t) sel.step(x_q + t * C, sel_step, y_sel_step + t * C);
    for (std::size_t i = 0; i < T * C; ++i) if (y_sel_full[i] != y_sel_step[i]) { equiv = false; ++mism; }

    // ----- Parity errors.
    float deq_lti[T * C], deq_sel[T * C];
    dequantizeBuffer<int8_t>(y_lti_full, deq_lti, T * C, py_lti.scale, py_lti.zero_point);
    dequantizeBuffer<int8_t>(y_sel_full, deq_sel, T * C, py_sel.scale, py_sel.zero_point);
    const float err_lti = maxAbsDiff(deq_lti, y_lti, T * C);
    const float err_sel = maxAbsDiff(deq_sel, y_sel, T * C);

    // ----- CSV (channel 0 LTI output over time: float vs int8).
    {
        std::FILE* csv = std::fopen("state_space_int8.csv", "w");
        std::fprintf(csv, "index,float,int8\n");
        for (std::size_t t = 0; t < T; ++t)
            std::fprintf(csv, "%zu,%.6f,%.6f\n", t, y_lti[t * C + 0], deq_lti[t * C + 0]);
        std::fclose(csv);
    }

    if (golden_mode)
    {
        std::printf("# state_space_int8 golden output\n");
        std::printf("# seq=%zu channels=%zu cache_equiv=%d\n",
                    static_cast<size_t>(T), static_cast<size_t>(C), equiv ? 1 : 0);
        std::printf("lti:");
        for (std::size_t i = 0; i < T * C; ++i) std::printf(" %d", static_cast<int>(y_lti_full[i]));
        std::printf("\nsel:");
        for (std::size_t i = 0; i < T * C; ++i) std::printf(" %d", static_cast<int>(y_sel_full[i]));
        std::printf("\n");
        return 0;
    }

    std::printf("Diagonal state-space (S4-lite) int8 vs float reference\n");
    std::printf("  seq=%zu  channels=%zu   state: %zu x int32 (fixed, any length)\n",
                static_cast<size_t>(T), static_cast<size_t>(C), static_cast<size_t>(C));
    std::printf("  streaming step() == full-sequence forward(): %s",
                equiv ? "YES (byte-identical, both variants)\n" : "NO\n");
    if (!equiv) std::printf("  FAIL: %zu cells diverged\n", mism);
    std::printf("  LTI       max-abs err: %.5f   (%.1f%% of range)\n",
                err_lti, 100.0f * err_lti / (absmaxBuf(y_lti, T * C) + 1e-6f));
    std::printf("  selective max-abs err: %.5f   (%.1f%% of range)\n",
                err_sel, 100.0f * err_sel / (absmaxBuf(y_sel, T * C) + 1e-6f));

    const bool ok = equiv && err_lti < 0.10f * absmaxBuf(y_lti, T * C)
                          && err_sel < 0.12f * absmaxBuf(y_sel, T * C);
    if (!ok) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
