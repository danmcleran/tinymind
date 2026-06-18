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

// Phase 3 Mixture-of-Experts importer round-trip.
//
// Consumes the Python-emitted weights.hpp (apps/import_pytorch importer) and
// rebuilds cpp/qmoe.hpp::QMixtureOfExperts directly from the generated
// router / per-expert constants. For each held-out test input it:
//
//   * quantizes the float input with the emitted input scale / zero point,
//   * runs the top-1 int8 MoE forward,
//   * checks the selected expert matches the float-model argmax, and
//   * dequantizes the int8 output and compares it to the float reference.
//
// The router argmaxes over raw int32 logits (no requant); each expert carries
// its own per-expert weight scale and shares the single MoE output scale. This
// binary builds against the checked-in weights.hpp / moe_reference.hpp without
// numpy.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "qaffine.hpp"
#include "qcalibration.hpp"
#include "qdense.hpp"
#include "qmoe.hpp"

#include "weights.hpp"
#include "moe_reference.hpp"

namespace {

constexpr int32_t kQmin = -128;
constexpr int32_t kQmax =  127;

using namespace import_moe_demo;

typedef tinymind::QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t,
                                    NumInputs, NumOutputs, NumExperts> MoE;

} // namespace

int main()
{
    MoE moe;

    // Router: raw int32 logits -> argmax (no requantizer).
    moe.router_weights    = kmoe_Router_Weights;
    moe.router_biases     = kmoe_Router_Biases;
    moe.input_zero_point  = static_cast<int8_t>(kinput_ZeroPoint);

    // Experts: each its own weight scale, all sharing the MoE output scale.
    const int8_t* expert_w[NumExperts] = {
        kmoe_Expert0_Weights, kmoe_Expert1_Weights, kmoe_Expert2_Weights,
    };
    const int32_t* expert_b[NumExperts] = {
        kmoe_Expert0_Biases, kmoe_Expert1_Biases, kmoe_Expert2_Biases,
    };
    const float expert_w_scale[NumExperts] = {
        kmoe_Expert0_WeightScale, kmoe_Expert1_WeightScale, kmoe_Expert2_WeightScale,
    };

    for (std::size_t e = 0; e < NumExperts; ++e)
    {
        moe.experts[e].weights          = expert_w[e];
        moe.experts[e].biases           = expert_b[e];
        moe.experts[e].input_zero_point = static_cast<int8_t>(kinput_ZeroPoint);
        moe.experts[e].requantizer = tinymind::buildRequantizer<int8_t>(
            kinput_Scale, expert_w_scale[e], kmoe_Scale, kmoe_ZeroPoint,
            kQmin, kQmax);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "import_moe_demo parity test (" << NumTest << " samples)\n";

    float max_abs_err = 0.0f;
    std::size_t expert_mismatches = 0;

    for (std::size_t t = 0; t < NumTest; ++t)
    {
        int8_t qx[NumInputs];
        for (std::size_t i = 0; i < NumInputs; ++i)
        {
            qx[i] = tinymind::quantize<int8_t>(
                kTestInputs[t][i], kinput_Scale, kinput_ZeroPoint, kQmin, kQmax);
        }

        int8_t qy[NumOutputs];
        const std::size_t selected = moe.forward(qx, qy);

        if (selected != kExpectedExpert[t])
        {
            ++expert_mismatches;
            std::cout << "  sample " << t << ": expert " << selected
                      << " != expected " << kExpectedExpert[t] << "\n";
        }

        for (std::size_t o = 0; o < NumOutputs; ++o)
        {
            const float yq = tinymind::dequantize<int8_t>(
                qy[o], kmoe_Scale, kmoe_ZeroPoint);
            const float err = std::abs(yq - kFloatRef[t][o]);
            if (err > max_abs_err) max_abs_err = err;
        }
    }

    std::cout << "  selected-expert mismatches = " << expert_mismatches
              << " / " << NumTest << "\n";
    std::cout << "  max |y_int8 - y_float|     = " << max_abs_err << "\n";

    // The MoE output spans roughly the calibrated range; 1 LSB of the output
    // grid is kmoe_Scale (~0.016). A few LSBs of weight + activation rounding
    // is expected; 0.05 (~3 LSBs) gates a single quantized linear expert.
    constexpr float kTolerance = 0.05f;
    const bool ok = (expert_mismatches == 0) && (max_abs_err <= kTolerance);
    std::cout << "  tolerance " << kTolerance << " : "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
