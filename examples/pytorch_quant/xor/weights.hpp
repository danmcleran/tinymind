/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via xor_quant.py.
*
* Per-tensor int8 affine quantization metadata + weights for the XOR
* pytorch_quant example. Format mirrors TFLite / CMSIS-NN: symmetric
* per-tensor weights (zero_point = 0), asymmetric activations, int32
* biases at scale = input_scale * weight_scale.
*
* The shipped sample uses an exact textbook 2-4-1 ReLU+Sigmoid solution
* (fc1 = ones, fc1.bias = [0, -1, 0, -1], fc2 = [2, -4, 2, -4],
* fc2.bias = -2). Re-running xor_quant.py with PyTorch installed
* overwrites this file with whatever the SGD trainer converges to.
*/

#pragma once

#include <cstddef>
#include <cstdint>

namespace pytorch_quant_xor {

    constexpr std::size_t NumInputs   = 2;
    constexpr std::size_t HiddenSize  = 4;
    constexpr std::size_t NumOutputs  = 1;

    // Activation calibration (per-tensor, asymmetric int8).
    constexpr float   kInputScale         = 0.00392156862745098f;   // 1/255
    constexpr int32_t kInputZeroPoint     = -128;
    constexpr float   kHiddenScale        = 0.00784313725490196f;   // 2/255
    constexpr int32_t kHiddenZeroPoint    = -128;
    constexpr float   kLogitScale         = 0.01568627450980392f;   // 4/255
    constexpr int32_t kLogitZeroPoint     = -1;
    constexpr float   kSigmoidOutScale    = 0.00390625f;            // 1/256
    constexpr int32_t kSigmoidOutZeroPoint= -128;

    // Weight calibration (per-tensor, symmetric int8, zero_point = 0).
    constexpr float kFc1WeightScale = 0.007874015748031496f;        // 1/127
    constexpr float kFc2WeightScale = 0.031496062992125984f;        // 4/127

    constexpr int8_t kFc1Weights[HiddenSize * NumInputs] = {
          127,   127,   127,   127,   127,   127,   127,   127
    };
    constexpr int32_t kFc1Biases[HiddenSize] = {
            0, -32385,     0, -32385
    };
    constexpr int8_t kFc2Weights[NumOutputs * HiddenSize] = {
           64,  -127,    64,  -127
    };
    constexpr int32_t kFc2Biases[NumOutputs] = {
        -8096
    };
} // namespace pytorch_quant_xor
