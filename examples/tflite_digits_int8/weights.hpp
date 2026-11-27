/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import_tflite.
*/
#pragma once

#include <cstddef>
#include <cstdint>

namespace digits {
    // Activation calibration per tensor.
    constexpr float   kserving_default_keras_tensor_0_Scale     = 0.003921568859368563f;
    constexpr int32_t kserving_default_keras_tensor_0_ZeroPoint = -128;
    constexpr float   kfunctional_1_conv2d_1_Relu_functional_1_conv2d_1_BiasAdd_functional_1_conv2d_1_convolution_functional_1_conv2d_1_Squeeze1_Scale     = 0.014159508049488068f;
    constexpr int32_t kfunctional_1_conv2d_1_Relu_functional_1_conv2d_1_BiasAdd_functional_1_conv2d_1_convolution_functional_1_conv2d_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_depthwise_conv2d_1_Relu6_functional_1_depthwise_conv2d_1_BiasAdd_functional_1_depthwise_conv2d_1_depthwise_functional_1_depthwise_conv2d_1_Squeeze1_Scale     = 0.023196693509817123f;
    constexpr int32_t kfunctional_1_depthwise_conv2d_1_Relu6_functional_1_depthwise_conv2d_1_BiasAdd_functional_1_depthwise_conv2d_1_depthwise_functional_1_depthwise_conv2d_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_conv2d_1_2_BiasAdd_functional_1_conv2d_1_2_convolution_functional_1_conv2d_1_2_Squeeze1_Scale     = 0.08557920157909393f;
    constexpr int32_t kfunctional_1_conv2d_1_2_BiasAdd_functional_1_conv2d_1_2_convolution_functional_1_conv2d_1_2_Squeeze1_ZeroPoint = -34;
    constexpr float   kfunctional_1_add_1_Add_Scale     = 0.08557920157909393f;
    constexpr int32_t kfunctional_1_add_1_Add_ZeroPoint = -34;
    constexpr float   kfunctional_1_max_pooling2d_1_MaxPool2d_Scale     = 0.08557920157909393f;
    constexpr int32_t kfunctional_1_max_pooling2d_1_MaxPool2d_ZeroPoint = -34;
    constexpr float   kfunctional_1_conv2d_2_1_Relu_functional_1_conv2d_2_1_BiasAdd_functional_1_conv2d_2_1_convolution_functional_1_conv2d_2_1_Squeeze1_Scale     = 0.21201619505882263f;
    constexpr int32_t kfunctional_1_conv2d_2_1_Relu_functional_1_conv2d_2_1_BiasAdd_functional_1_conv2d_2_1_convolution_functional_1_conv2d_2_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_global_average_pooling2d_1_Mean_Scale     = 0.09773630648851395f;
    constexpr int32_t kfunctional_1_global_average_pooling2d_1_Mean_ZeroPoint = -128;
    constexpr float   kfunctional_1_dense_1_MatMul_functional_1_dense_1_BiasAdd_Scale     = 0.2737049162387848f;
    constexpr int32_t kfunctional_1_dense_1_MatMul_functional_1_dense_1_BiasAdd_ZeroPoint = 13;
    constexpr float   kStatefulPartitionedCall_1_0_Scale     = 0.00390625f;
    constexpr int32_t kStatefulPartitionedCall_1_0_ZeroPoint = -128;

    // Conv conv0 (reads the model's input)
    constexpr float   kconv0_InputScale      = 0.003921568859368563f;
    constexpr int32_t kconv0_InputZeroPoint  = -128;
    constexpr float   kconv0_OutputScale     = 0.014159508049488068f;
    constexpr int32_t kconv0_OutputZeroPoint = -128;
    constexpr std::size_t kconv0_InputHeight = 8;
    constexpr std::size_t kconv0_InputWidth = 8;
    constexpr std::size_t kconv0_InputChannels = 1;
    constexpr std::size_t kconv0_KernelHeight = 3;
    constexpr std::size_t kconv0_KernelWidth = 3;
    constexpr std::size_t kconv0_StrideH = 1;
    constexpr std::size_t kconv0_StrideW = 1;
    constexpr std::size_t kconv0_NumFilters = 8;
    constexpr std::size_t kconv0_PadTop = 1;
    constexpr std::size_t kconv0_PadLeft = 1;
    constexpr std::size_t kconv0_PadBottom = 1;
    constexpr std::size_t kconv0_PadRight = 1;
    constexpr std::size_t kconv0_OutputHeight = 8;
    constexpr std::size_t kconv0_OutputWidth = 8;
    constexpr bool kconv0_PerChannel = true;
    constexpr float kconv0_WeightScales[8] = { 0.007977881468832493f, 0.008311018347740173f, 0.005789025686681271f, 0.005797780584543943f, 0.004464987199753523f, 0.011609192006289959f, 0.015311460942029953f, 0.007307530380785465f };
    constexpr int32_t kconv0_Multipliers[8] = {
        1214700166, 1265423084, 1762856591, 1765522606, 1359664396, 1767598017,
        1165649943, 1112633523
    };
    constexpr int32_t kconv0_Shifts[8] = {
            8,     8,     9,     9,     9,     8,     7,     8
    };
    constexpr int8_t kconv0_Weights[72] = {
          -10,    48,   -60,    70,    37,   -34,   127,    30,   -12,    74,    47,    17,
           43,    24,   -52,   127,    93,  -123,   127,    76,   125,    57,   -81,    39,
          -25,  -108,    34,   -84,  -127,   -89,    26,    44,   117,    99,    69,    71,
         -110,    37,  -127,   -18,   -32,   -23,   118,     8,    96,   -77,  -127,     0,
            4,    -5,    35,    45,    66,    55,    -9,    13,    28,    14,    32,    28,
          -76,  -127,   -58,  -118,  -127,    76,   -24,    79,    70,    76,   104,    -3
    };
    constexpr bool kconv0_HasBias = true;
    constexpr int32_t kconv0_Biases[8] = {
         4351,  5960,  5575,  1268, 11288, 13278,  8571,  4518
    };

    // DepthwiseConv dwconv1 (reads conv0)
    constexpr float   kdwconv1_InputScale      = 0.014159508049488068f;
    constexpr int32_t kdwconv1_InputZeroPoint  = -128;
    constexpr float   kdwconv1_OutputScale     = 0.023196693509817123f;
    constexpr int32_t kdwconv1_OutputZeroPoint = -128;
    constexpr std::size_t kdwconv1_InputHeight = 8;
    constexpr std::size_t kdwconv1_InputWidth = 8;
    constexpr std::size_t kdwconv1_InputChannels = 8;
    constexpr std::size_t kdwconv1_KernelHeight = 3;
    constexpr std::size_t kdwconv1_KernelWidth = 3;
    constexpr std::size_t kdwconv1_StrideH = 1;
    constexpr std::size_t kdwconv1_StrideW = 1;
    constexpr std::size_t kdwconv1_NumFilters = 8;
    constexpr std::size_t kdwconv1_PadTop = 1;
    constexpr std::size_t kdwconv1_PadLeft = 1;
    constexpr std::size_t kdwconv1_PadBottom = 1;
    constexpr std::size_t kdwconv1_PadRight = 1;
    constexpr std::size_t kdwconv1_OutputHeight = 8;
    constexpr std::size_t kdwconv1_OutputWidth = 8;
    constexpr bool kdwconv1_PerChannel = true;
    constexpr float kdwconv1_WeightScales[8] = { 0.0086336275562644f, 0.006497697904706001f, 0.006893975660204887f, 0.007952886633574963f, 0.008311612531542778f, 0.006464014295488596f, 0.008341532200574875f, 0.00652063125744462f };
    constexpr int32_t kdwconv1_Multipliers[8] = {
        1448622497, 1090238292, 1156729100, 1334402071, 1394592113, 1084586573,
        1399612286, 1094086243
    };
    constexpr int32_t kdwconv1_Shifts[8] = {
            7,     7,     7,     7,     7,     7,     7,     7
    };
    constexpr int8_t kdwconv1_Weights[72] = {
           42,   -20,    13,    -9,     1,    89,   -16,    49,   127,   -65,   -10,    82,
         -127,     5,   -36,    28,   -10,   -12,    87,   127,     6,    27,   114,   -14,
          -19,    67,   -30,   -34,    43,    25,   124,   127,   105,   -66,  -109,   -34,
           16,    33,    12,    53,   127,   108,    45,    92,    73,   -34,     3,    55,
          -46,   -93,   127,   -57,   -65,    81,  -127,   -21,    82,  -118,    14,    14,
          -73,    -6,    92,    87,   -15,  -127,   118,   -64,    50,    73,   -32,   105
    };
    constexpr bool kdwconv1_HasBias = true;
    constexpr int32_t kdwconv1_Biases[8] = {
          837,  3075,  1612,   -65, -1325,  1072,  5582,  -715
    };

    // Conv conv2 (reads dwconv1)
    constexpr float   kconv2_InputScale      = 0.023196693509817123f;
    constexpr int32_t kconv2_InputZeroPoint  = -128;
    constexpr float   kconv2_OutputScale     = 0.08557920157909393f;
    constexpr int32_t kconv2_OutputZeroPoint = -34;
    constexpr std::size_t kconv2_InputHeight = 8;
    constexpr std::size_t kconv2_InputWidth = 8;
    constexpr std::size_t kconv2_InputChannels = 8;
    constexpr std::size_t kconv2_KernelHeight = 1;
    constexpr std::size_t kconv2_KernelWidth = 1;
    constexpr std::size_t kconv2_StrideH = 1;
    constexpr std::size_t kconv2_StrideW = 1;
    constexpr std::size_t kconv2_NumFilters = 8;
    constexpr std::size_t kconv2_PadTop = 0;
    constexpr std::size_t kconv2_PadLeft = 0;
    constexpr std::size_t kconv2_PadBottom = 0;
    constexpr std::size_t kconv2_PadRight = 0;
    constexpr std::size_t kconv2_OutputHeight = 8;
    constexpr std::size_t kconv2_OutputWidth = 8;
    constexpr bool kconv2_PerChannel = true;
    constexpr float kconv2_WeightScales[8] = { 0.009113463573157787f, 0.007184389512985945f, 0.007940535433590412f, 0.007293037138879299f, 0.008100507780909538f, 0.008640866726636887f, 0.016856255009770393f, 0.011895745992660522f };
    constexpr int32_t kconv2_Multipliers[8] = {
        1358035575, 2141152256, 1183252615, 1086766168, 1207090768, 1287611929,
        1255910762, 1772635192
    };
    constexpr int32_t kconv2_Shifts[8] = {
            8,     9,     8,     8,     8,     8,     7,     8
    };
    constexpr int8_t kconv2_Weights[64] = {
           26,   -26,   -58,   -71,   -95,    93,   -73,   127,    66,    31,    37,     7,
          -80,   127,    30,    36,  -112,  -125,   127,    71,  -115,   -54,    23,   -55,
         -104,   -94,   -36,   100,    53,  -123,    52,   127,   -77,   -10,   -16,   -72,
          111,    52,   -71,   127,    31,   -66,   127,    29,   -75,   -85,   119,   -27,
          -33,    68,    -7,    29,   -59,   -74,  -127,     4,    47,    -8,   -93,   127,
           71,  -105,    88,   -70
    };
    constexpr bool kconv2_HasBias = true;
    constexpr int32_t kconv2_Biases[8] = {
        -1115,   731,  -149, -1407,  -501,   861,    96, -1174
    };

    // Add add3 (reads conv0, conv2)
    constexpr float   kadd3_InputScale      = 0.014159508049488068f;
    constexpr int32_t kadd3_InputZeroPoint  = -128;
    constexpr float   kadd3_InputBScale     = 0.08557920157909393f;
    constexpr int32_t kadd3_InputBZeroPoint = -34;
    constexpr float   kadd3_OutputScale     = 0.08557920157909393f;
    constexpr int32_t kadd3_OutputZeroPoint = -34;

    // MaxPool maxpool4 (reads add3)
    constexpr float   kmaxpool4_InputScale      = 0.08557920157909393f;
    constexpr int32_t kmaxpool4_InputZeroPoint  = -34;
    constexpr float   kmaxpool4_OutputScale     = 0.08557920157909393f;
    constexpr int32_t kmaxpool4_OutputZeroPoint = -34;
    constexpr std::size_t kmaxpool4_InputHeight = 8;
    constexpr std::size_t kmaxpool4_InputWidth = 8;
    constexpr std::size_t kmaxpool4_InputChannels = 8;
    constexpr std::size_t kmaxpool4_KernelHeight = 2;
    constexpr std::size_t kmaxpool4_KernelWidth = 2;
    constexpr std::size_t kmaxpool4_StrideH = 2;
    constexpr std::size_t kmaxpool4_StrideW = 2;
    constexpr std::size_t kmaxpool4_OutputHeight = 4;
    constexpr std::size_t kmaxpool4_OutputWidth = 4;

    // Conv conv5 (reads maxpool4)
    constexpr float   kconv5_InputScale      = 0.08557920157909393f;
    constexpr int32_t kconv5_InputZeroPoint  = -34;
    constexpr float   kconv5_OutputScale     = 0.21201619505882263f;
    constexpr int32_t kconv5_OutputZeroPoint = -128;
    constexpr std::size_t kconv5_InputHeight = 4;
    constexpr std::size_t kconv5_InputWidth = 4;
    constexpr std::size_t kconv5_InputChannels = 8;
    constexpr std::size_t kconv5_KernelHeight = 3;
    constexpr std::size_t kconv5_KernelWidth = 3;
    constexpr std::size_t kconv5_StrideH = 1;
    constexpr std::size_t kconv5_StrideW = 1;
    constexpr std::size_t kconv5_NumFilters = 16;
    constexpr std::size_t kconv5_PadTop = 1;
    constexpr std::size_t kconv5_PadLeft = 1;
    constexpr std::size_t kconv5_PadBottom = 1;
    constexpr std::size_t kconv5_PadRight = 1;
    constexpr std::size_t kconv5_OutputHeight = 4;
    constexpr std::size_t kconv5_OutputWidth = 4;
    constexpr bool kconv5_PerChannel = true;
    constexpr float kconv5_WeightScales[16] = { 0.015850679948925972f, 0.010366051457822323f, 0.018055472522974014f, 0.012455967254936695f, 0.015338493511080742f, 0.007995644584298134f, 0.011091121472418308f, 0.014843234792351723f, 0.015536836348474026f, 0.009643158875405788f, 0.011966808699071407f, 0.017679056152701378f, 0.009004716761410236f, 0.009362365119159222f, 0.008808720856904984f, 0.02085852436721325f };
    constexpr int32_t kconv5_Multipliers[16] = {
        1758680411, 1150144454, 2003308751, 1382026871, 1701851792, 1774281425,
        1230593144, 1646901354, 1723858524, 2139874715, 1327753264, 1961544227,
        1998200586, 2077564899, 1954707921, 1157157874
    };
    constexpr int32_t kconv5_Shifts[16] = {
            7,     7,     7,     7,     7,     8,     7,     7,     7,     8,     7,     7,
            8,     8,     8,     6
    };
    constexpr int8_t kconv5_Weights[1152] = {
           19,     3,    22,    34,    -4,     4,    78,    15,    -6,    -9,    14,    18,
           -4,     0,   127,    31,   -16,    -7,    11,    26,   -15,    -4,   114,    14,
           33,     8,    21,    38,    10,    28,    44,    32,     0,     1,    -6,    34,
           -3,     9,   116,    28,   -13,    -6,    -8,    27,   -13,    10,    96,     6,
           28,    12,   -36,     3,    26,    -4,   -44,   -11,     0,   -11,   -38,    23,
           39,     7,    20,    13,     7,   -21,   -56,     5,    10,    -1,    27,    29,
          -35,     8,    20,   -24,   -10,    11,   -15,    10,    15,    49,    -5,   -33,
            9,    39,   -88,   -22,    37,    15,     2,    12,    10,    25,   -52,     5,
          -28,     5,     0,     3,    -2,    15,    14,    28,    12,    40,    20,   -20,
           -4,    45,   -23,   -18,    25,    11,    27,    25,    31,     9,     3,     7,
          -36,   -10,    27,     9,   -13,    22,    97,    18,    17,    47,    31,    25,
          -25,    43,    90,     1,    14,    11,    38,    33,   -19,    12,   127,    -7,
          -37,   -39,   -10,    35,     9,   -12,   127,    13,   -19,   -26,   -19,    34,
           -5,    -2,    25,    27,    26,     6,    20,    35,    42,    41,    -2,    25,
          -20,   -27,   -61,    23,   -15,   -18,     5,    16,    22,     0,   -59,    -9,
            9,     2,   -77,    -5,    21,     9,    10,   -39,    22,    18,   -34,   -34,
           36,     0,   -25,    40,    47,     6,   -39,    33,    26,    14,     3,   -21,
          -19,    11,   -17,    -8,     5,     5,    34,   -48,   -35,    11,    69,   -38,
          -31,   -22,    12,    14,   -18,    -1,    -2,    17,   -23,    10,    32,    -4,
           40,     0,    -8,   -72,   -49,   -49,    -3,    -1,   -16,    21,    67,     5,
           -5,   -25,    79,    -1,   -35,    12,    44,    -5,   -37,     5,     7,    16,
          -28,    -5,   -18,   -16,   -10,   -12,     0,    15,    17,    16,   -76,    29,
          -11,   -13,   -22,     5,    39,    23,  -127,   -28,   -25,    13,     6,   -17,
          -15,    31,  -124,   -38,    50,    37,     7,     2,    30,    59,   -94,    27,
          -43,    -9,    16,    43,    -9,     3,   -11,    51,    -9,   -18,    31,    51,
           19,     6,   -20,    32,     0,     0,     8,    62,    41,   -13,   -63,    16,
           -6,    -8,    22,    -9,    34,   -10,   -60,    -1,    -5,   -23,    49,    44,
            9,     4,   -61,    -8,   -26,    -8,    46,    45,   -19,   -27,   -22,   -44,
          -17,    -6,    20,    21,     9,   -15,    -5,    -1,    -9,   -12,    48,    81,
          -22,   -25,   -41,   -27,   -49,   -24,    78,    22,   -71,    -6,   -31,  -127,
           58,    68,   107,    56,    28,    45,  -122,    49,   -56,    11,    45,    46,
            3,    44,   -29,    71,   -62,   -23,   -68,    93,   -51,   -11,   -12,    75,
           35,    79,    74,    36,   -18,    58,   -40,    51,   -20,     8,    14,    41,
           -8,    10,     9,    74,   -44,     0,   -76,    81,   -25,    -8,    26,    64,
           19,    29,    14,   -36,    23,     7,   -74,    28,     5,     0,   -29,   -16,
           73,    14,   -54,    46,   -28,   -24,  -127,    34,    60,   -60,  -116,   106,
           27,    45,    19,    -7,    20,    29,    -6,    18,    26,    27,    55,   -28,
           30,    11,   -31,   -26,    41,    44,    61,   -44,     9,    41,  -125,   -62,
          -12,   -11,    17,    44,    50,   -43,    33,   -20,   -25,   -19,    28,    26,
           39,    -9,    65,    15,     4,   -11,    48,    47,    45,    -9,     1,   -15,
           -1,    -7,   -76,    76,   127,   -11,    13,    24,    -9,   -10,    -5,    76,
           59,     1,    18,    35,   -51,   -25,    62,   116,    55,     9,   116,   -26,
           34,    31,   -59,   -12,     5,    -2,     0,    15,    20,     7,   -73,    -3,
           14,     1,    20,    12,    19,    -1,   -41,   -10,    -8,    -5,    59,    12,
          -20,    11,    -9,   -39,    -7,     4,    52,   -13,    23,    22,   -17,   -34,
          -15,     0,   -19,    -8,    -1,     6,   -23,   -27,   -22,    15,    47,     2,
          -32,     5,    30,    16,   -34,    23,   127,     0,    -2,    27,    25,    11,
          -54,     6,    85,    11,    14,    28,    21,    -1,   -47,    22,   104,    15,
           -2,   -64,   -42,    73,    39,   -63,   -24,   -66,   -39,   -48,   -10,    13,
           -5,   -11,    35,    10,    -5,   -15,     7,   -13,   -35,    29,    63,    18,
            7,    16,   -32,    65,   -71,    16,    93,    38,    -9,     0,    29,    76,
            1,    37,   127,    49,    12,     0,    23,    39,   -39,    38,    71,    32,
           38,    32,   -53,   -65,   -44,   -10,    18,   -17,   -15,    -4,   -22,    24,
           21,    -5,    45,   -20,   -17,   -33,    -5,   -28,   -12,    34,    50,    29,
           53,    93,    67,    -9,   -34,     3,   -51,   -10,   -74,   -35,    -1,    30,
           17,     6,    79,     0,   -95,   -53,    30,     7,   -49,    23,    96,    77,
           20,    51,    77,    19,     2,    -7,     4,   -38,   -88,   -45,     2,    25,
           23,    13,   -12,   -12,   -65,   -47,   -39,   -28,   -15,    27,   127,    67,
           -1,    40,    67,    55,    -7,    20,    41,   -54,   -73,   -36,   -33,    46,
           41,     2,    26,    45,    -3,   -10,   -22,   -53,   -75,    23,    42,    58,
           10,    -6,    -5,    -2,    92,   -11,   -71,   -37,    33,   -23,   -58,   -10,
           70,   -12,   -69,    -6,    23,     5,     4,    -1,    57,     0,   -39,    -6,
           28,    -8,    38,    -4,    45,     4,   -35,     8,    26,   -13,   -40,     0,
           67,    -7,   -74,    -9,    22,    -2,   -44,   -10,    52,   -30,   -85,    15,
            5,     6,    41,    -3,   -25,    11,   -56,     1,    50,    14,   -29,    37,
           10,     4,  -127,    27,    52,     3,   -68,    58,    19,   -43,   -86,    68,
           62,    25,   -25,   -99,   -91,   -18,   -33,   -26,     9,    -4,   -65,   -16,
           14,   -10,    21,   -32,   -33,   -40,    37,    31,    43,    20,    52,    34,
           47,    33,   -26,  -127,   -95,    -3,   -35,   -37,    15,    15,   -39,     9,
           27,     5,     2,   -21,   -45,   -48,    39,     4,    19,    29,    31,    18,
           49,    47,    -5,   -56,   -70,    -4,   -27,   -11,    17,    20,   -19,    52,
           32,    10,   -35,     7,   -40,   -36,    48,   -11,   -29,    20,    40,    47,
           79,    30,   -45,     0,    45,    10,  -111,     7,    26,     6,   -40,     0,
           33,    47,   -99,    -9,    46,     2,   -53,   -44,    12,    35,  -127,     7,
           51,     6,   -42,   -36,    -4,     3,   -51,   -27,    49,    -1,   -13,   -15,
           17,    28,   -37,    20,    25,     4,   -41,    17,    25,    18,    24,    30,
           54,    38,     0,   -95,     0,     5,   -77,     4,    16,    13,   -16,   -40,
           17,     2,   -28,    50,    56,    -3,   -60,    -5,    20,     3,   -56,    47,
           48,    27,   -62,    -9,    35,     0,   -72,    10,    79,    55,   -96,   -55,
           90,     8,  -127,    -3,    69,     9,   -60,   -84,    81,     9,   -71,   -19,
            7,   -17,     5,   -29,    21,   -12,  -123,    12,    57,    30,    -1,   -38,
           33,   -14,   -30,    -6,    54,   -21,    22,   -23,    50,   -20,    24,    36,
           25,   -20,   -46,   -32,   -45,   -20,    74,    66,    16,   -11,   -40,    19,
          -54,   -10,    24,    26,    46,    -2,   -20,   106,   -43,   -61,    30,    52,
          126,    42,  -117,  -109,    23,   -45,  -115,    18,    36,    13,   -47,    26,
           37,     1,   -19,    18,    24,   -72,   -15,    61,   113,     3,    96,    18,
           60,    39,  -121,   -38,    38,   -33,  -127,   -16,     8,   -31,   -57,     7,
           40,    -3,    41,    16,   -31,   -72,    43,    57,     4,    19,    76,   -18,
           41,    -1,    20,    21,    25,   -19,  -125,     8,   -34,   -55,    22,    72,
           34,   -11,    27,    12,    32,   -84,    68,    29,    14,    43,    84,    34,
          -28,   -33,    -4,    11,    10,    -6,    85,    -5,   -19,   -17,    13,    24,
           -4,     5,   127,    -6,    18,    13,    22,    14,    14,    22,    40,    -8,
          -18,   -33,    -8,    33,   -15,     1,    99,    16,     0,   -17,   -18,    27,
            5,     5,   107,    13,    12,     0,    24,    12,    29,     8,   -10,    -8,
           -1,   -32,   -37,    44,    39,   -15,    35,     3,    10,   -16,   -22,    43,
           24,    14,    43,     6,    12,     7,    21,    46,    32,    29,    26,   -23
    };
    constexpr bool kconv5_HasBias = true;
    constexpr int32_t kconv5_Biases[16] = {
          -17,   431,   -65,    48,  -102,   217,   -23,    80,   100,    65,   -94,   161,
          133,  -127,   -93,   -75
    };

    // Mean mean6 (reads conv5)
    constexpr float   kmean6_InputScale      = 0.21201619505882263f;
    constexpr int32_t kmean6_InputZeroPoint  = -128;
    constexpr float   kmean6_OutputScale     = 0.09773630648851395f;
    constexpr int32_t kmean6_OutputZeroPoint = -128;
    constexpr std::size_t kmean6_InputHeight = 4;
    constexpr std::size_t kmean6_InputWidth = 4;
    constexpr std::size_t kmean6_InputChannels = 16;

    // Dense fc7 (reads mean6)
    constexpr float   kfc7_InputScale      = 0.09773630648851395f;
    constexpr int32_t kfc7_InputZeroPoint  = -128;
    constexpr float   kfc7_OutputScale     = 0.2737049162387848f;
    constexpr int32_t kfc7_OutputZeroPoint = 13;
    constexpr std::size_t kfc7_NumInputs = 16;
    constexpr std::size_t kfc7_NumOutputs = 10;
    constexpr bool kfc7_PerChannel = true;
    constexpr float kfc7_WeightScales[10] = { 0.01773037575185299f, 0.014157614670693874f, 0.008988417685031891f, 0.009924975223839283f, 0.010016467422246933f, 0.013195245526731014f, 0.013247760944068432f, 0.011284108273684978f, 0.007084370125085115f, 0.00810128916054964f };
    constexpr int32_t kfc7_Multipliers[10] = {
        1740327967, 1389643012, 1764519251, 1948375172, 1966336036, 1295181509,
        1300336168, 1107593515, 1390735045, 1590366757
    };
    constexpr int32_t kfc7_Shifts[10] = {
            7,     7,     8,     8,     8,     7,     7,     7,     8,     8
    };
    constexpr int8_t kfc7_Weights[160] = {
          -47,    38,   -70,    80,    59,     7,    22,   -44,   -45,   -68,    11,   -65,
           -1,    74,  -127,   -69,   -51,    10,    36,   -42,   -87,   -64,    27,    44,
           31,    34,    26,   127,    21,    34,    40,    -6,    55,   -96,   -75,    48,
          -52,    40,    77,  -127,    52,    38,    11,   -57,    27,    55,    55,    22,
           95,     5,     1,   -19,  -127,   -25,   -52,    37,   109,   -14,   -70,    42,
          -33,    33,    75,    93,    16,    -3,    26,  -111,   111,   -28,    26,    21,
         -127,   -96,    63,   -35,    23,    76,   101,    73,    54,     3,  -127,   -57,
           -3,    50,    14,     8,    61,    81,   -47,   -75,   -18,   -96,   -77,   -53,
           27,     3,   -79,   -14,   -40,    54,   -87,    45,  -102,    13,    81,   -92,
           34,    -4,   -92,  -127,    10,   -47,   103,    96,    94,   -21,   -49,   -52,
           38,    67,    16,    14,   -26,  -127,    43,    62,   -57,    59,    77,    35,
         -127,    45,   -73,     2,    19,   -37,   -35,  -127,    92,   -75,   -54,  -114,
          -35,    74,   -43,    47,    -6,   -43,   127,    14,     1,   -16,   -21,   -46,
          -39,   -44,   -11,    82
    };
    constexpr bool kfc7_HasBias = true;
    constexpr int32_t kfc7_Biases[10] = {
           13,   -65,  -217,   -32,  -245,   106,    94,  -181,   251,   205
    };

    // Softmax softmax8 (reads fc7)
    constexpr float   ksoftmax8_InputScale      = 0.2737049162387848f;
    constexpr int32_t ksoftmax8_InputZeroPoint  = 13;
    constexpr float   ksoftmax8_OutputScale     = 0.00390625f;
    constexpr int32_t ksoftmax8_OutputZeroPoint = -128;
} // namespace digits
