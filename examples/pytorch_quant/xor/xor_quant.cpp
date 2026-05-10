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

// PyTorch -> TinyMind int8 quantization end-to-end demo.
//
// Loads weights, biases, and calibration scales emitted by xor_quant.py
// (weights.hpp), reconstructs the per-layer Requantizers and the qsigmoid
// LUT host-side, then runs a pure-integer forward pass on the four XOR
// inputs. The pipeline mirrors the canonical TFLite int8 deployment:
//
//     int8 input
//       -> QDense (fc1)       -> int8 hidden    [requant: in*w1 / hidden]
//       -> qrelu (fold-free, applied pointwise on hidden)
//       -> QDense (fc2)       -> int8 logit     [requant: hidden*w2 / logit]
//       -> qsigmoid LUT       -> int8 output    [LUT: logit grid -> 1/256 grid]
//
// The C++ side uses qcalibration.hpp to turn the calibration scales into
// the integer (multiplier, shift, zero_point) tuples that Requantizer.apply
// consumes; that helper is gated on FLOAT && STD, which is fine here
// because this is a host demo. Drop the buildRequantizer / LUT-build calls
// for an MCU target and bake the integer constants in directly.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>

#include "qaffine.hpp"
#include "qcalibration.hpp"
#include "qactivations.hpp"
#include "qdense.hpp"

#include "weights.hpp"

namespace pq = pytorch_quant_xor;

typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t,
                         pq::NumInputs, pq::HiddenSize> QDenseFC1;
typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t,
                         pq::HiddenSize, pq::NumOutputs> QDenseFC2;

int main()
{
    static const float kXOR[4][2] = {
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    };
    static const int kXORLabels[4] = {0, 1, 1, 0};

    // Reconstruct Requantizers from the per-tensor scales emitted by the
    // calibration step. Inference-only MCU build would precompute these.
    QDenseFC1 fc1;
    fc1.weights = pq::kFc1Weights;
    fc1.biases  = pq::kFc1Biases;
    fc1.input_zero_point = static_cast<int8_t>(pq::kInputZeroPoint);
    fc1.requantizer = tinymind::buildRequantizer<int8_t>(
        pq::kInputScale, pq::kFc1WeightScale,
        pq::kHiddenScale, pq::kHiddenZeroPoint,
        -128, 127);

    QDenseFC2 fc2;
    fc2.weights = pq::kFc2Weights;
    fc2.biases  = pq::kFc2Biases;
    fc2.input_zero_point = static_cast<int8_t>(pq::kHiddenZeroPoint);
    fc2.requantizer = tinymind::buildRequantizer<int8_t>(
        pq::kHiddenScale, pq::kFc2WeightScale,
        pq::kLogitScale, pq::kLogitZeroPoint,
        -128, 127);

    // Build the int8 sigmoid LUT in the logit's input grid.
    int8_t sigmoidLUT[tinymind::kQActivationLUTSize];
    tinymind::buildQSigmoidLUT(pq::kLogitScale, pq::kLogitZeroPoint,
                               pq::kSigmoidOutScale, pq::kSigmoidOutZeroPoint,
                               sigmoidLUT);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "PyTorch -> TinyMind int8 XOR inference\n";
    std::cout << "  input_scale=" << pq::kInputScale
              << " input_zp="     << pq::kInputZeroPoint
              << " hidden_scale=" << pq::kHiddenScale
              << " hidden_zp="    << pq::kHiddenZeroPoint
              << "\n  logit_scale=" << pq::kLogitScale
              << " logit_zp="     << pq::kLogitZeroPoint
              << " out_scale="    << pq::kSigmoidOutScale
              << " out_zp="       << pq::kSigmoidOutZeroPoint
              << "\n\n";

    int correct = 0;
    for (int i = 0; i < 4; ++i)
    {
        int8_t input_q[pq::NumInputs];
        for (std::size_t k = 0; k < pq::NumInputs; ++k)
        {
            input_q[k] = tinymind::quantize<int8_t>(
                kXOR[i][k], pq::kInputScale, pq::kInputZeroPoint, -128, 127);
        }

        int8_t hidden_q[pq::HiddenSize];
        fc1.forward(input_q, hidden_q);
        // ReLU folds into the hidden tensor's zero_point clamp. For sanity
        // and clarity it is applied pointwise here.
        tinymind::qreluBuffer(hidden_q, pq::HiddenSize,
                              static_cast<int8_t>(pq::kHiddenZeroPoint));

        int8_t logit_q[pq::NumOutputs];
        fc2.forward(hidden_q, logit_q);

        int8_t output_q[pq::NumOutputs];
        for (std::size_t o = 0; o < pq::NumOutputs; ++o)
        {
            output_q[o] = tinymind::qApplyLUT(logit_q[o], sigmoidLUT);
        }

        const float prob = tinymind::dequantize<int8_t>(
            output_q[0], pq::kSigmoidOutScale, pq::kSigmoidOutZeroPoint);
        const int predicted = (prob > 0.5f) ? 1 : 0;
        if (predicted == kXORLabels[i])
        {
            ++correct;
        }

        std::cout << "  in=(" << kXOR[i][0] << "," << kXOR[i][1] << ")"
                  << "  q_input=("
                  << static_cast<int>(input_q[0]) << ","
                  << static_cast<int>(input_q[1]) << ")"
                  << "  q_logit=" << static_cast<int>(logit_q[0])
                  << "  prob="    << prob
                  << "  pred="    << predicted
                  << "  label="   << kXORLabels[i]
                  << "\n";
    }

    std::cout << "\nint8 XOR accuracy: " << correct << "/4\n";
    return (correct == 4) ? 0 : 1;
}
