/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import.
*/
#pragma once

#include <cstddef>
#include <cstdint>
namespace import_moe_demo {
    constexpr std::size_t NumInputs = 4;
    constexpr std::size_t NumOutputs = 3;
    constexpr std::size_t NumExperts = 3;

    // Activation calibration.
    constexpr float   kinput_Scale     = 0.007810055040845684f;
    constexpr int32_t kinput_ZeroPoint = -1;
    constexpr float   kmoe_Scale     = 0.01643507059882669f;
    constexpr int32_t kmoe_ZeroPoint = 16;

    // Weight calibration + int8 weights + int32 biases.

    // Mixture-of-Experts (cpp/qmoe.hpp): top-1 router + per-expert int8 experts.
    constexpr std::size_t kmoe_NumExperts = 3;
    constexpr float kmoe_Router_WeightScale = 0.015748031496062992f;
    constexpr int8_t kmoe_Router_Weights[12] = {
          127,     0,     0,    32,     0,   127,     0,   -32,     0,     0,   127,    32
    };
    constexpr int32_t kmoe_Router_Biases[3] = {
          813,     0,  -813
    };
    constexpr float kmoe_Expert0_WeightScale = 0.0062328778852627975f;
    constexpr int8_t kmoe_Expert0_Weights[12] = {
           32,   102,    71,   -71,   -51,    96,  -127,    82,    76,    -8,   -51,   -57
    };
    constexpr int32_t kmoe_Expert0_Biases[3] = {
        -1608, -6117,  4068
    };
    constexpr float kmoe_Expert1_WeightScale = 0.007803154273295965f;
    constexpr int8_t kmoe_Expert1_Weights[12] = {
          -63,   -14,     1,    14,   127,    75,    31,   125,   -73,   -87,    29,  -117
    };
    constexpr int32_t kmoe_Expert1_Biases[3] = {
        -3402, -2288,  3744
    };
    constexpr float kmoe_Expert2_WeightScale = 0.0038441414908161314f;
    constexpr int8_t kmoe_Expert2_Weights[12] = {
         -121,     4,    -9,   109,    34,     4,    -1,   -66,  -127,   -80,    50,   -78
    };
    constexpr int32_t kmoe_Expert2_Biases[3] = {
          196,  6938,  2792
    };
} // namespace import_moe_demo
