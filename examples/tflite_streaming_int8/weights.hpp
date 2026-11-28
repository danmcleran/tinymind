/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import_tflite.
*/
#pragma once

#include <cstddef>
#include <cstdint>

namespace vibration {
    // Activation calibration per tensor.
    constexpr float   kserving_default_keras_tensor_0_Scale     = 0.02941484935581684f;
    constexpr int32_t kserving_default_keras_tensor_0_ZeroPoint = -8;
    constexpr float   kfunctional_1_conv2d_1_Relu_functional_1_conv2d_1_BiasAdd_functional_1_conv2d_1_convolution_functional_1_conv2d_1_Squeeze1_Scale     = 0.05681701377034187f;
    constexpr int32_t kfunctional_1_conv2d_1_Relu_functional_1_conv2d_1_BiasAdd_functional_1_conv2d_1_convolution_functional_1_conv2d_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_depthwise_conv2d_1_Relu_functional_1_depthwise_conv2d_1_BiasAdd_functional_1_depthwise_conv2d_1_depthwise_functional_1_depthwise_conv2d_1_Squeeze1_Scale     = 0.0440039224922657f;
    constexpr int32_t kfunctional_1_depthwise_conv2d_1_Relu_functional_1_depthwise_conv2d_1_BiasAdd_functional_1_depthwise_conv2d_1_depthwise_functional_1_depthwise_conv2d_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_conv2d_1_2_Relu_functional_1_conv2d_1_2_BiasAdd_functional_1_conv2d_1_2_convolution_functional_1_conv2d_1_2_Squeeze1_Scale     = 0.06782463937997818f;
    constexpr int32_t kfunctional_1_conv2d_1_2_Relu_functional_1_conv2d_1_2_BiasAdd_functional_1_conv2d_1_2_convolution_functional_1_conv2d_1_2_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_depthwise_conv2d_1_2_Relu_functional_1_depthwise_conv2d_1_2_BiasAdd_functional_1_depthwise_conv2d_1_2_depthwise_functional_1_depthwise_conv2d_1_2_Squeeze1_Scale     = 0.11491405218839645f;
    constexpr int32_t kfunctional_1_depthwise_conv2d_1_2_Relu_functional_1_depthwise_conv2d_1_2_BiasAdd_functional_1_depthwise_conv2d_1_2_depthwise_functional_1_depthwise_conv2d_1_2_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_conv2d_2_1_Relu_functional_1_conv2d_2_1_BiasAdd_functional_1_conv2d_2_1_convolution_functional_1_conv2d_2_1_Squeeze1_Scale     = 0.15498799085617065f;
    constexpr int32_t kfunctional_1_conv2d_2_1_Relu_functional_1_conv2d_2_1_BiasAdd_functional_1_conv2d_2_1_convolution_functional_1_conv2d_2_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_depthwise_conv2d_2_1_BiasAdd_functional_1_depthwise_conv2d_2_1_depthwise_functional_1_depthwise_conv2d_2_1_Squeeze1_Scale     = 0.5670965909957886f;
    constexpr int32_t kfunctional_1_depthwise_conv2d_2_1_BiasAdd_functional_1_depthwise_conv2d_2_1_depthwise_functional_1_depthwise_conv2d_2_1_Squeeze1_ZeroPoint = -43;
    constexpr float   kfunctional_1_conv2d_3_1_Relu_functional_1_conv2d_3_1_BiasAdd_functional_1_conv2d_3_1_convolution_functional_1_conv2d_3_1_Squeeze1_Scale     = 0.48135146498680115f;
    constexpr int32_t kfunctional_1_conv2d_3_1_Relu_functional_1_conv2d_3_1_BiasAdd_functional_1_conv2d_3_1_convolution_functional_1_conv2d_3_1_Squeeze1_ZeroPoint = -128;
    constexpr float   kfunctional_1_flatten_1_Reshape_Scale     = 0.48135146498680115f;
    constexpr int32_t kfunctional_1_flatten_1_Reshape_ZeroPoint = -128;
    constexpr float   kfunctional_1_dense_1_MatMul_functional_1_dense_1_BiasAdd_Scale     = 1.1422832012176514f;
    constexpr int32_t kfunctional_1_dense_1_MatMul_functional_1_dense_1_BiasAdd_ZeroPoint = -26;
    constexpr float   kStatefulPartitionedCall_1_0_Scale     = 0.00390625f;
    constexpr int32_t kStatefulPartitionedCall_1_0_ZeroPoint = -128;

    // Conv conv0 (reads the model's input)
    constexpr float   kconv0_InputScale      = 0.02941484935581684f;
    constexpr int32_t kconv0_InputZeroPoint  = -8;
    constexpr float   kconv0_OutputScale     = 0.05681701377034187f;
    constexpr int32_t kconv0_OutputZeroPoint = -128;
    constexpr std::size_t kconv0_InputHeight = 32;
    constexpr std::size_t kconv0_InputWidth = 1;
    constexpr std::size_t kconv0_InputChannels = 16;
    constexpr std::size_t kconv0_KernelHeight = 3;
    constexpr std::size_t kconv0_KernelWidth = 1;
    constexpr std::size_t kconv0_StrideH = 1;
    constexpr std::size_t kconv0_StrideW = 1;
    constexpr std::size_t kconv0_NumFilters = 16;
    constexpr std::size_t kconv0_PadTop = 0;
    constexpr std::size_t kconv0_PadLeft = 0;
    constexpr std::size_t kconv0_PadBottom = 0;
    constexpr std::size_t kconv0_PadRight = 0;
    constexpr std::size_t kconv0_OutputHeight = 30;
    constexpr std::size_t kconv0_OutputWidth = 1;
    constexpr bool kconv0_PerChannel = true;
    constexpr float kconv0_WeightScales[16] = { 0.00615027267485857f, 0.0020253071561455727f, 0.008735169656574726f, 0.004272740799933672f, 0.002027493203058839f, 0.005511089228093624f, 0.00443115783855319f, 0.007294246461242437f, 0.008772971108555794f, 0.0018848704639822245f, 0.0019395735580474138f, 0.0024332015309482813f, 0.005270627327263355f, 0.005364074371755123f, 0.0059562986716628075f, 0.004253461491316557f };
    constexpr int32_t kconv0_Multipliers[16] = {
        1750460948, 1152866311, 1243080930, 1216086880, 1154110675, 1568539638,
        1261174774, 2076053250, 1248460364, 2145850965, 1104064144, 1385052170,
        1500100531, 1526696978, 1695253002, 1210599696
    };
    constexpr int32_t kconv0_Shifts[16] = {
            8,     9,     7,     8,     9,     8,     8,     8,     7,    10,     9,     9,
            8,     8,     8,     8
    };
    constexpr int8_t kconv0_Weights[768] = {
          -28,   -46,   -25,    18,    -8,   -33,    18,     8,   127,   -13,    69,   -15,
            4,    -8,   -16,    13,     6,    67,   -50,    -3,    27,   -14,    46,    15,
          104,    25,    49,    -9,   -43,   -41,   -14,   -68,    12,    20,   -60,   -29,
          -38,    -3,    -2,   -45,    55,    19,     0,    36,     4,   -40,   -34,   -41,
           82,   -17,  -121,    62,  -117,  -114,   -58,   127,    68,   -49,    64,    63,
           54,   -22,    11,  -104,    93,    85,   -78,  -112,   -46,    59,     5,    80,
           16,    36,    -6,   -39,    -5,   -24,   -81,   -33,    31,     4,   -15,    36,
          -65,    53,   -39,   125,   -51,    88,   -58,    41,   -36,    32,  -104,    66,
           47,    54,   -70,   -36,    11,    97,    24,    10,    38,   -10,    79,    77,
          -47,    19,    42,    70,     7,    87,    -5,    30,    34,    -5,    21,   -14,
          -18,    47,    19,    44,  -109,   114,    52,    29,    33,    97,   -58,    32,
           47,    36,   127,    62,   -50,   -48,    22,   106,   -28,   -76,    51,   -16,
           -1,   -83,    19,    34,    28,   -38,     3,    60,   -81,   -49,   -75,    -7,
           48,   -34,   -38,    55,    -4,    -6,     6,    44,   -46,     6,    -4,     1,
         -111,   -77,   -39,    36,   -32,   -14,    83,    -3,    45,   -83,    36,   -61,
           12,   -22,   -40,   -25,  -127,   -73,    -3,    40,    18,    -3,   -11,    -5,
         -127,   -97,   -13,    54,    41,    99,   -10,   -77,   -13,    -7,   -18,    25,
         -111,    28,    19,    66,   -39,   -52,   103,   -67,   107,  -101,    48,   -78,
          -55,   102,   -12,     5,   -69,   -81,    88,   -91,   -63,    96,    57,   -77,
           71,    48,   -14,    13,   -81,    67,   -69,   120,     3,    70,   -43,   120,
          -24,    31,   -13,    44,   127,    24,    78,    16,   -87,   -70,   -38,    31,
          107,   -24,    44,    -5,   -16,    64,    31,   -28,    39,    15,    73,   -69,
          -93,   -81,    -5,   -22,    24,     6,   121,    68,    60,    -3,    38,   -28,
           24,   -19,    53,   -36,   -63,   -38,   -29,    -7,    38,    19,   -15,    89,
           -4,   -61,    25,    31,   -47,   -74,   -37,   -10,    22,   -33,    53,   -31,
           -3,   -82,   -70,    16,    -9,    31,   -68,   -20,   -19,   -53,   -11,    67,
          127,    96,    75,    27,     6,   -18,   -14,   -69,   -39,   120,   -38,     2,
          -75,   -45,    14,   -28,   117,   102,   104,   -58,   -30,   -41,   -43,     6,
          -27,    14,   -21,    11,    54,   -34,     9,    63,    93,    43,   -45,   -64,
            1,   -26,    15,   -21,    -7,    10,   -24,   -46,    15,    43,   -15,    75,
           70,    59,   -16,   -19,    -8,    40,   -47,    28,   -17,   -34,    57,   -11,
          -68,   -43,    67,    88,    59,    13,  -127,   -18,    33,   -47,   -18,   -23,
            9,   -37,    35,    27,    16,    -9,     6,   -38,    53,    -3,   113,   -47,
           19,     7,     2,     2,    20,    38,    -8,    16,    15,   -41,     6,   -50,
          -13,    -2,    36,   -18,    48,    -5,    15,    58,   -22,     7,    15,    -8,
          -54,   -92,     8,   -17,   -30,    43,    43,  -127,    23,    15,   -31,   -36,
          -56,   -70,   -34,    30,    50,  -114,  -109,   -82,    51,     2,   -31,   -12,
           12,     0,    80,     4,   127,   -98,  -125,   -49,    10,   116,   -95,    27,
           30,  -118,   109,   -30,   -25,   -60,    -3,  -119,    39,   106,  -102,   -74,
           46,   -70,   -34,   -91,   -94,   -71,   107,   -49,   -89,    83,     0,   -98,
           35,   100,   -35,  -113,    27,    80,     9,   -91,     4,  -127,    91,    47,
          -25,    38,   -45,   113,     7,    83,   -28,   -16,   -70,    91,     3,    31,
           13,   -12,  -126,   107,     2,   -14,    20,    96,    80,    51,    33,    31,
         -126,   -12,    24,   126,   -16,     4,   -18,   -64,  -117,   -23,  -117,   100,
          -98,   -31,    93,   -24,   -74,    66,    22,     7,    -6,   -50,    -8,    64,
           78,   -71,   -34,    40,   -62,   -24,   -71,    -6,   -56,   -62,    98,   -12,
          -44,    67,    24,   -58,    99,   -14,   -71,   -69,  -105,   -57,    32,   108,
          -58,   -42,    26,    83,   -75,    -7,    85,   -33,    61,   -78,    -5,   127,
           11,   -41,   -40,    31,   -76,   -83,    24,    55,    96,    33,   -28,    30,
           65,   -57,    -1,   -18,    -4,   -41,   -20,   -47,     2,    -2,   -55,    38,
           86,   127,     4,   -26,    -3,   -61,    -6,    13,     4,    21,   -49,   -25,
          -74,   -17,    38,    19,    74,    94,    11,    28,   -34,   -48,   -25,    40,
          -10,   -44,   -53,    -9,     8,    -7,    -6,     6,   127,    89,    49,    18,
          -69,   -36,   -58,   -31,   -17,    65,   -72,    -6,   -27,   -15,    27,    76,
           79,    67,    11,   -22,   -21,   -35,   -13,   -71,    25,     5,    16,    19,
          -65,   -45,    30,     7,    67,    45,    46,   -33,   -38,   -38,   -47,   -20,
           -1,   -41,    32,   -39,   -16,    20,   -39,   -20,   -62,   105,    41,   127,
            5,    30,   -12,    -1,   -37,   -44,   -69,   -45,   -52,   -61,    12,   -82,
           43,    90,    60,    42,     7,    -6,   -62,   -58,    -2,    45,   -59,    29,
            1,   -21,    -7,    90,   -47,    83,     5,    60,   -36,    30,    -4,    28,
          -34,    69,    15,    38,    28,   -29,   -34,   -52,   115,    73,    54,   -39,
          -28,   -13,    32,   -54,   -41,    17,    42,   -28,   -61,   -69,    39,  -112,
          127,    73,   -36,   -55,   -28,    37,     0,   -50,    18,    37,    21,    21,
           17,   -80,    -5,   -50,    96,    20,    -3,   -16,   -62,   -30,   -19,   -29
    };
    constexpr bool kconv0_HasBias = true;
    constexpr int32_t kconv0_Biases[16] = {
         -170,    69, -1865,    13,  -307,   -92, -1067,   212,  -558,    49,   -51,  -745,
          375,   185,   312,   240
    };

    // DepthwiseConv dwconv1 (reads conv0)
    constexpr float   kdwconv1_InputScale      = 0.05681701377034187f;
    constexpr int32_t kdwconv1_InputZeroPoint  = -128;
    constexpr float   kdwconv1_OutputScale     = 0.0440039224922657f;
    constexpr int32_t kdwconv1_OutputZeroPoint = -128;
    constexpr std::size_t kdwconv1_InputHeight = 30;
    constexpr std::size_t kdwconv1_InputWidth = 1;
    constexpr std::size_t kdwconv1_InputChannels = 16;
    constexpr std::size_t kdwconv1_KernelHeight = 5;
    constexpr std::size_t kdwconv1_KernelWidth = 1;
    constexpr std::size_t kdwconv1_StrideH = 1;
    constexpr std::size_t kdwconv1_StrideW = 1;
    constexpr std::size_t kdwconv1_NumFilters = 16;
    constexpr std::size_t kdwconv1_PadTop = 0;
    constexpr std::size_t kdwconv1_PadLeft = 0;
    constexpr std::size_t kdwconv1_PadBottom = 0;
    constexpr std::size_t kdwconv1_PadRight = 0;
    constexpr std::size_t kdwconv1_OutputHeight = 26;
    constexpr std::size_t kdwconv1_OutputWidth = 1;
    constexpr bool kdwconv1_PerChannel = true;
    constexpr float kdwconv1_WeightScales[16] = { 0.0038712911773473024f, 0.001533233793452382f, 0.007870790548622608f, 0.001973578706383705f, 0.0017568630864843726f, 0.0036624823696911335f, 0.005128578748553991f, 0.007646625395864248f, 0.005103172268718481f, 0.001808530418202281f, 0.001824822393245995f, 0.0014762593200430274f, 0.004517333582043648f, 0.003472971962764859f, 0.005914746318012476f, 0.0027336960192769766f };
    constexpr int32_t kdwconv1_Multipliers[16] = {
        1373987197, 1088341593, 1396738833, 1400913417, 1247081285, 1299877394,
        1820219978, 1356958817, 1811202785, 1283756518, 1295321117, 2095798340,
        1603278654, 1232616922, 2099244243, 1940470587
    };
    constexpr int32_t kdwconv1_Shifts[16] = {
            7,     8,     6,     8,     8,     7,     7,     6,     7,     8,     8,     9,
            7,     7,     7,     8
    };
    constexpr int8_t kdwconv1_Weights[80] = {
           73,     8,    77,    68,   127,   -63,    74,  -127,   -33,    14,    33,     6,
           57,  -127,   -46,    76,    38,   119,     8,   127,   -55,    74,   127,    57,
         -106,   100,    17,   -20,    89,   127,    34,    30,    14,    23,   127,   127,
           34,    58,    17,    94,   -45,    -2,   107,   127,   -47,   -47,  -127,     1,
           -9,   -75,   -25,  -127,   -42,    10,   -41,    87,   127,    99,  -107,   -47,
          127,   -45,    38,    41,   106,   127,    20,    74,    53,   -36,   115,    92,
          127,    60,   105,   127,   115,    74,   -69,    66
    };
    constexpr bool kdwconv1_HasBias = true;
    constexpr int32_t kdwconv1_Biases[16] = {
          -92,  -341,  -273,   -64,  -242,   -27,  -636,   -85,   -97,  -252,  -224,  -518,
         -137,  -340,  -244,  -178
    };

    // Conv conv2 (reads dwconv1)
    constexpr float   kconv2_InputScale      = 0.0440039224922657f;
    constexpr int32_t kconv2_InputZeroPoint  = -128;
    constexpr float   kconv2_OutputScale     = 0.06782463937997818f;
    constexpr int32_t kconv2_OutputZeroPoint = -128;
    constexpr std::size_t kconv2_InputHeight = 26;
    constexpr std::size_t kconv2_InputWidth = 1;
    constexpr std::size_t kconv2_InputChannels = 16;
    constexpr std::size_t kconv2_KernelHeight = 1;
    constexpr std::size_t kconv2_KernelWidth = 1;
    constexpr std::size_t kconv2_StrideH = 1;
    constexpr std::size_t kconv2_StrideW = 1;
    constexpr std::size_t kconv2_NumFilters = 16;
    constexpr std::size_t kconv2_PadTop = 0;
    constexpr std::size_t kconv2_PadLeft = 0;
    constexpr std::size_t kconv2_PadBottom = 0;
    constexpr std::size_t kconv2_PadRight = 0;
    constexpr std::size_t kconv2_OutputHeight = 26;
    constexpr std::size_t kconv2_OutputWidth = 1;
    constexpr bool kconv2_PerChannel = true;
    constexpr float kconv2_WeightScales[16] = { 0.005667288322001696f, 0.006882199551910162f, 0.011173254810273647f, 0.008798010647296906f, 0.005916442722082138f, 0.01104028057307005f, 0.010081801563501358f, 0.004095996264368296f, 0.016868503764271736f, 0.003463612636551261f, 0.0031231737229973078f, 0.013905838131904602f, 0.005588868632912636f, 0.003400614717975259f, 0.009922296740114689f, 0.009249521419405937f };
    constexpr int32_t kconv2_Multipliers[16] = {
        2021384990, 1227357255, 1992615187, 1569019049, 2110252352, 1968900836,
        1797967669, 1460943029, 1504147062, 1235387049, 1113960703, 1239969227,
        1993414579, 1212917211, 1769521909, 1649540547
    };
    constexpr int32_t kconv2_Shifts[16] = {
            8,     7,     7,     7,     8,     7,     7,     8,     6,     8,     8,     6,
            8,     8,     7,     7
    };
    constexpr int8_t kconv2_Weights[256] = {
          -34,    19,   127,    28,    67,   107,  -118,   -38,    62,   -39,    74,    64,
          -27,   -73,   -88,   -50,     0,    22,     1,   -55,   -25,    58,   -70,  -127,
           20,    37,   -59,    22,   -34,   -52,   -16,   -32,   -24,    24,    12,  -127,
          -37,    66,    58,   -79,   -35,   -37,   -11,   -35,    51,   -22,   -31,   -16,
         -119,   -19,   -64,     5,    22,    21,     2,   127,    50,    21,    31,   -13,
          -78,   -10,    98,    97,    -8,   -35,    -4,   -25,    56,   -26,    79,   -13,
         -103,    26,   -10,    19,    65,   127,   -52,    24,    -1,    38,   -13,    26,
           -8,   -26,  -127,    28,    -1,    10,     4,    31,    58,   -84,   -50,    12,
            5,     9,  -127,   -40,    43,   -28,    35,    84,    22,     4,   -32,     9,
           56,    23,    84,    -9,    42,   -97,    62,    36,   -46,   -74,    20,    22,
          -58,    62,   -27,    46,  -109,   -41,   -93,  -127,   -11,   -16,   127,     6,
           -3,    24,   -23,   -22,    21,    11,   -26,    -2,    22,   -12,   -28,   -29,
          -21,    53,    54,    65,    68,   -75,   -37,     9,   127,    65,    63,   -67,
           87,   -33,   -63,  -107,   -84,    94,   -94,   -80,   127,   -71,  -121,    88,
          -57,   116,   -80,    72,   -88,    -8,   -44,     0,     8,    13,   127,    21,
           24,    24,   -19,   -38,    26,   -25,   -20,    -1,    -5,    40,   -58,    32,
            9,   -22,    27,    28,    74,     0,    36,    28,  -127,    12,   -54,   -16,
           57,    83,   104,    52,  -117,    -6,   -34,   -93,    64,    51,    38,     6,
          -99,   -78,   106,   -34,  -122,   -59,  -127,    29,    38,     0,  -127,   -20,
           30,   -40,    44,    72,    33,   -33,   -10,     2,    53,    46,    43,    38,
           47,    28,  -127,   -39,    41,   -79,   -23,    56,    18,   -18,   -39,    -1,
           27,    10,    79,    29
    };
    constexpr bool kconv2_HasBias = true;
    constexpr int32_t kconv2_Biases[16] = {
          182,    59,     6,    79,   232,  -667,    30,  -356,    16,   429,    97,   -40,
        -1674,  -525,   -53,  -426
    };

    // DepthwiseConv dwconv3 (reads conv2)
    constexpr float   kdwconv3_InputScale      = 0.06782463937997818f;
    constexpr int32_t kdwconv3_InputZeroPoint  = -128;
    constexpr float   kdwconv3_OutputScale     = 0.11491405218839645f;
    constexpr int32_t kdwconv3_OutputZeroPoint = -128;
    constexpr std::size_t kdwconv3_InputHeight = 26;
    constexpr std::size_t kdwconv3_InputWidth = 1;
    constexpr std::size_t kdwconv3_InputChannels = 16;
    constexpr std::size_t kdwconv3_KernelHeight = 9;
    constexpr std::size_t kdwconv3_KernelWidth = 1;
    constexpr std::size_t kdwconv3_StrideH = 1;
    constexpr std::size_t kdwconv3_StrideW = 1;
    constexpr std::size_t kdwconv3_NumFilters = 16;
    constexpr std::size_t kdwconv3_PadTop = 0;
    constexpr std::size_t kdwconv3_PadLeft = 0;
    constexpr std::size_t kdwconv3_PadBottom = 0;
    constexpr std::size_t kdwconv3_PadRight = 0;
    constexpr std::size_t kdwconv3_OutputHeight = 18;
    constexpr std::size_t kdwconv3_OutputWidth = 1;
    constexpr bool kdwconv3_PerChannel = true;
    constexpr float kdwconv3_WeightScales[16] = { 0.005450981203466654f, 0.0045445580035448074f, 0.006567802745848894f, 0.008804644457995892f, 0.00424536969512701f, 0.010434302501380444f, 0.004305565729737282f, 0.0017006243579089642f, 0.003334473818540573f, 0.0013555207988247275f, 0.0014885412529110909f, 0.003746794303879142f, 0.004677938297390938f, 0.0010590374004095793f, 0.004276061896234751f, 0.004510908853262663f };
    constexpr int32_t kdwconv3_Multipliers[16] = {
        1768719115, 1474605453, 2131102241, 1428453188, 1377525669, 1692846627,
        1397057910, 1103627658, 1081960726, 1759342370, 1931990788, 1215749322,
        1517884318, 1374533959, 1387484588, 1463687071
    };
    constexpr int32_t kdwconv3_Shifts[16] = {
            8,     8,     8,     7,     8,     7,     8,     9,     8,    10,    10,     8,
            8,    10,     8,     8
    };
    constexpr int8_t kdwconv3_Weights[144] = {
         -127,   -44,   -14,    30,    16,   -13,    11,   -14,  -118,    47,    63,    51,
           62,   -21,    28,    50,   127,   127,    23,  -119,    84,   -36,   107,   -94,
           73,    17,   127,    89,    64,    22,     8,   -34,   -67,   -38,   -94,  -127,
          -84,   -15,    22,   105,    22,    65,    59,     6,  -127,    94,   127,    75,
           83,    63,    78,    28,    85,    65,   118,     7,    -1,    10,   -37,    35,
            1,    63,   127,    -8,    65,  -127,  -104,    66,   -33,    -8,    57,   -20,
           77,    46,    42,   100,   108,   127,    85,   111,    79,  -127,   -37,    16,
          -76,  -120,     4,   -12,  -126,   -67,   -76,    36,   -47,     2,  -119,   -79,
          -73,   -98,  -127,    51,   -14,    -2,    66,    15,    78,    13,    32,   127,
          127,    79,     8,    53,   -38,   -10,    48,    59,    32,  -127,   -45,    22,
          -30,    48,  -115,    86,   -23,    -2,   127,    45,     8,    20,    25,    30,
           13,    66,    99,  -127,   -55,    -7,   -50,    44,     6,    45,    34,   -37
    };
    constexpr bool kdwconv3_HasBias = true;
    constexpr int32_t kdwconv3_Biases[16] = {
         1017,   283,  -292,  -495,   201,    99,   -62,  -607,   314,  -202,  -553,   -44,
         -235, -1176,  -422,   708
    };

    // Conv conv4 (reads dwconv3)
    constexpr float   kconv4_InputScale      = 0.11491405218839645f;
    constexpr int32_t kconv4_InputZeroPoint  = -128;
    constexpr float   kconv4_OutputScale     = 0.15498799085617065f;
    constexpr int32_t kconv4_OutputZeroPoint = -128;
    constexpr std::size_t kconv4_InputHeight = 18;
    constexpr std::size_t kconv4_InputWidth = 1;
    constexpr std::size_t kconv4_InputChannels = 16;
    constexpr std::size_t kconv4_KernelHeight = 1;
    constexpr std::size_t kconv4_KernelWidth = 1;
    constexpr std::size_t kconv4_StrideH = 1;
    constexpr std::size_t kconv4_StrideW = 1;
    constexpr std::size_t kconv4_NumFilters = 16;
    constexpr std::size_t kconv4_PadTop = 0;
    constexpr std::size_t kconv4_PadLeft = 0;
    constexpr std::size_t kconv4_PadBottom = 0;
    constexpr std::size_t kconv4_PadRight = 0;
    constexpr std::size_t kconv4_OutputHeight = 18;
    constexpr std::size_t kconv4_OutputWidth = 1;
    constexpr bool kconv4_PerChannel = true;
    constexpr float kconv4_WeightScales[16] = { 0.00655340263620019f, 0.009351043961942196f, 0.008642113767564297f, 0.0040003033354878426f, 0.0046368851326406f, 0.01029727142304182f, 0.00707890372723341f, 0.003828091314062476f, 0.013793251477181911f, 0.003747275099158287f, 0.009018995799124241f, 0.009780142456293106f, 0.007306969724595547f, 0.0055151814594864845f, 0.003994956612586975f, 0.004233700223267078f };
    constexpr int32_t kconv4_Multipliers[16] = {
        1335616506, 1905789917, 1761306368, 1630563987, 1890041148, 2098635845,
        1442716279, 1560368630, 1405567105, 1527427126, 1838116827, 1993242354,
        1489197279, 1124021795, 1628384609, 1725698912
    };
    constexpr int32_t kconv4_Shifts[16] = {
            7,     7,     7,     8,     8,     7,     7,     8,     6,     8,     7,     7,
            7,     7,     8,     8
    };
    constexpr int8_t kconv4_Weights[256] = {
          -92,    58,    90,  -127,    23,    71,    12,   -58,    54,    42,    -5,    18,
          -28,   -44,   -34,   -59,   -74,     1,   -13,  -127,    31,    25,    21,    -8,
           23,    -9,     8,    48,     3,   -20,   -21,    -4,   -25,    43,   -61,  -127,
            8,    75,   -23,    -3,    78,   -47,   -36,    10,    15,   -43,   -31,    27,
           38,     9,   -72,   -95,   -64,   -22,    26,   -60,    19,    16,    59,  -127,
          -23,   -66,  -115,   -70,    85,   -50,   127,   -36,   -38,   -89,    50,   -75,
           57,    73,    33,   -32,   -68,    75,   -50,    55,   -11,    34,    44,  -127,
           61,    65,    -3,   -26,    63,   -18,   -22,    22,    50,   -38,   -22,    48,
           85,   -57,    58,   127,   -77,   -35,    47,   -13,    23,   -37,   -31,    -1,
           44,    34,    47,   -84,    86,   -33,    29,    43,    69,    28,    10,   -84,
          -98,   -48,    -7,  -108,    -9,   -60,  -127,    38,    37,    11,    46,   127,
          -33,    14,    26,     5,   -32,    15,     6,    17,     9,   -11,    30,   -48,
         -102,   -51,  -104,   -13,    90,    -3,   -96,    66,   -64,    71,   -93,   -51,
           46,    82,   -66,   127,   127,   -47,    47,   -66,   -35,   -23,    89,    24,
          -12,    42,     9,    -6,    71,    37,    50,   -25,    62,    35,  -127,     4,
           34,    42,    44,     6,   -39,   -31,    16,   -49,    11,    11,     1,   -64,
         -107,    41,    96,   -72,    88,   127,   -69,    17,   -15,   -56,   -35,    63,
          -23,   -25,   -64,    23,    19,    -3,    17,   -14,    25,    74,  -127,     2,
           69,    59,    23,   107,   -22,    78,   -59,     7,    47,   112,   -91,  -121,
          -51,   -62,   -81,    -3,   117,   -74,   -44,   127,  -120,    70,    70,     3,
          101,  -103,    90,    -9,   -79,   -38,   127,    20,   -58,    85,  -112,    52,
         -105,   -19,    68,   -53
    };
    constexpr bool kconv4_HasBias = true;
    constexpr int32_t kconv4_Biases[16] = {
           82,    79,    56,   -29,    99,    96,   -73,    93,   -36,   212,   124,   149,
          135,   403,  -393,  -399
    };

    // DepthwiseConv dwconv5 (reads conv4)
    constexpr float   kdwconv5_InputScale      = 0.15498799085617065f;
    constexpr int32_t kdwconv5_InputZeroPoint  = -128;
    constexpr float   kdwconv5_OutputScale     = 0.5670965909957886f;
    constexpr int32_t kdwconv5_OutputZeroPoint = -43;
    constexpr std::size_t kdwconv5_InputHeight = 18;
    constexpr std::size_t kdwconv5_InputWidth = 1;
    constexpr std::size_t kdwconv5_InputChannels = 16;
    constexpr std::size_t kdwconv5_KernelHeight = 18;
    constexpr std::size_t kdwconv5_KernelWidth = 1;
    constexpr std::size_t kdwconv5_StrideH = 1;
    constexpr std::size_t kdwconv5_StrideW = 1;
    constexpr std::size_t kdwconv5_NumFilters = 16;
    constexpr std::size_t kdwconv5_PadTop = 0;
    constexpr std::size_t kdwconv5_PadLeft = 0;
    constexpr std::size_t kdwconv5_PadBottom = 0;
    constexpr std::size_t kdwconv5_PadRight = 0;
    constexpr std::size_t kdwconv5_OutputHeight = 1;
    constexpr std::size_t kdwconv5_OutputWidth = 1;
    constexpr bool kdwconv5_PerChannel = true;
    constexpr float kdwconv5_WeightScales[16] = { 0.003899472998455167f, 0.0024902229197323322f, 0.005032298620790243f, 0.0012729187728837132f, 0.004240136593580246f, 0.0038494591135531664f, 0.005892242770642042f, 0.0012622805079445243f, 0.006765905302017927f, 0.0023508109152317047f, 0.006301378831267357f, 0.009305530227720737f, 0.007558474317193031f, 0.008550012484192848f, 0.004628672730177641f, 0.0034595010802149773f };
    constexpr int32_t kdwconv5_Multipliers[16] = {
        1171781820, 1496611438, 1512193068, 1530035548, 1274150373, 1156752774,
        1770604121, 1517248461, 2033137512, 1412825525, 1893548477, 1398144205,
        1135651253, 1284628615, 1390904504, 2079142733
    };
    constexpr int32_t kdwconv5_Shifts[16] = {
            9,    10,     9,    11,     9,     9,     9,    11,     9,    10,     9,     8,
            8,     8,     9,    10
    };
    constexpr int8_t kdwconv5_Weights[288] = {
          -87,    -8,   -24,   -44,    13,   -70,   -34,   -98,    19,   -41,    -6,   -64,
           35,    14,     5,   -35,   -15,  -127,    97,    23,    73,    -2,   -37,   -41,
           71,   127,    19,    66,    71,    73,   110,    33,    44,    39,    94,    99,
          127,    -9,    45,    -9,    23,    16,    69,    21,    52,    -3,    77,    63,
           13,     4,    23,    14,    34,   111,    13,    45,   107,    54,    13,    54,
          -95,   -41,    65,    -1,    49,    19,   -68,    39,   127,  -102,   -28,   -18,
           55,   -19,    70,   -22,    65,  -127,    54,   -28,    25,  -110,    39,   -75,
           36,   -51,    79,   -29,    51,    -8,    80,    55,    59,    66,    18,    47,
           38,   111,    38,    59,    28,    92,    73,    65,    15,    82,    64,   127,
         -127,   -17,   -29,   -27,   -29,   -21,   -44,    28,    17,    21,    -2,   -33,
          -43,   -15,   -24,   -20,    18,   -69,   -40,   -69,   112,   -65,    -4,    39,
           22,    66,   125,   108,   100,    -3,   124,    74,   -20,    45,   -23,   127,
          127,    22,    37,     7,    52,    43,    65,     5,    25,    31,    43,    42,
           26,    38,    40,    37,    19,    45,   -57,   -18,   -91,    -3,  -127,   -55,
          -41,     5,   -48,    14,   -59,    97,  -117,   -40,   -30,    -8,   -22,   -15,
         -127,   -30,   -17,    20,   -28,    12,    -3,    25,    54,    61,    45,    30,
           16,    -3,    -8,    -4,   -13,  -120,   -49,     6,     7,    99,    32,    35,
           49,    49,     2,    13,    51,   -21,    48,   -36,    15,     3,    75,   127,
           68,   -30,    45,     8,    28,   -20,    60,    16,   -21,    -1,   101,    30,
          -36,    44,   -11,    81,    16,   127,    49,    21,    11,     9,    20,    19,
            9,     3,   -15,   -21,    -2,    42,    33,    28,    42,    52,    81,   127,
          -74,   -69,   -50,   -85,   -31,   -70,   -88,   -46,   -40,   -30,   -27,   -72,
          -62,   -72,   -33,   -64,   -71,  -127,  -127,   -43,  -105,   -41,   -29,   -17,
          -31,    -7,    44,    33,   -50,   -15,    18,     7,     8,   -29,   -71,   -91
    };
    constexpr bool kdwconv5_HasBias = true;
    constexpr int32_t kdwconv5_Biases[16] = {
          -32,    16,   -29,  -132,     5,    85,    72,     6,   -20,     3,   -61,   -76,
          -60,   -41,   432,   287
    };

    // Conv conv6 (reads dwconv5)
    constexpr float   kconv6_InputScale      = 0.5670965909957886f;
    constexpr int32_t kconv6_InputZeroPoint  = -43;
    constexpr float   kconv6_OutputScale     = 0.48135146498680115f;
    constexpr int32_t kconv6_OutputZeroPoint = -128;
    constexpr std::size_t kconv6_InputHeight = 1;
    constexpr std::size_t kconv6_InputWidth = 1;
    constexpr std::size_t kconv6_InputChannels = 16;
    constexpr std::size_t kconv6_KernelHeight = 1;
    constexpr std::size_t kconv6_KernelWidth = 1;
    constexpr std::size_t kconv6_StrideH = 1;
    constexpr std::size_t kconv6_StrideW = 1;
    constexpr std::size_t kconv6_NumFilters = 8;
    constexpr std::size_t kconv6_PadTop = 0;
    constexpr std::size_t kconv6_PadLeft = 0;
    constexpr std::size_t kconv6_PadBottom = 0;
    constexpr std::size_t kconv6_PadRight = 0;
    constexpr std::size_t kconv6_OutputHeight = 1;
    constexpr std::size_t kconv6_OutputWidth = 1;
    constexpr bool kconv6_PerChannel = true;
    constexpr float kconv6_WeightScales[8] = { 0.0038578181993216276f, 0.004713717382401228f, 0.005176362115889788f, 0.006532689556479454f, 0.005608872044831514f, 0.005363513249903917f, 0.008455125615000725f, 0.008238543756306171f };
    constexpr int32_t kconv6_Multipliers[8] = {
        1249327591, 1526504589, 1676328868, 2115566076, 1816394200, 1736936461,
        1369066811, 1333997547
    };
    constexpr int32_t kconv6_Shifts[8] = {
            7,     7,     7,     7,     7,     7,     6,     6
    };
    constexpr int8_t kconv6_Weights[128] = {
           29,    -2,  -107,    23,    49,   -84,    46,   -26,   -53,  -100,   127,   -18,
          -53,    12,    30,   -67,    23,     7,   -39,   -90,   -61,    27,    75,    11,
         -127,    48,    64,   -63,    37,   -25,    57,   104,    28,    90,   109,    37,
           36,    63,    32,    62,   -80,    95,    92,   -20,    75,   105,  -127,    71,
          -23,   -22,   -67,    64,    41,   -82,   -94,    31,    43,    26,  -127,    27,
           40,    36,   -64,    -1,  -127,   -13,    43,    95,   -49,   117,    81,     1,
          -11,   -33,    26,  -107,    18,    46,   -62,   -26,   -37,  -104,    17,   -30,
           86,    12,    68,    68,  -127,   -10,   -10,    91,    11,   -48,   -49,    -3,
           -3,   -68,    -1,   -21,   -19,   -44,   -59,    -2,   -23,   -35,    52,   127,
          -37,   -36,    34,   -23,    67,    26,   -16,   -18,    16,   -69,   -46,   -10,
           73,     0,   -17,    63,  -127,   -72,   -38,   -83
    };
    constexpr bool kconv6_HasBias = true;
    constexpr int32_t kconv6_Biases[8] = {
           -3,   -39,    34,    -5,    38,   -53,    -6,   -25
    };

    // Dense fc11 (reads conv6)
    constexpr float   kfc11_InputScale      = 0.48135146498680115f;
    constexpr int32_t kfc11_InputZeroPoint  = -128;
    constexpr float   kfc11_OutputScale     = 1.1422832012176514f;
    constexpr int32_t kfc11_OutputZeroPoint = -26;
    constexpr std::size_t kfc11_NumInputs = 8;
    constexpr std::size_t kfc11_NumOutputs = 2;
    constexpr bool kfc11_PerChannel = true;
    constexpr float kfc11_WeightScales[2] = { 0.006273617502301931f, 0.007254547905176878f };
    constexpr int32_t kfc11_Multipliers[2] = {
        1453370616, 1680616766
    };
    constexpr int32_t kfc11_Shifts[2] = {
            8,     8
    };
    constexpr int8_t kfc11_Weights[16] = {
         -111,     3,    23,  -127,    -5,    38,   -46,   -46,     0,    53,   -77,    65,
          -95,    43,   127,    92
    };
    constexpr bool kfc11_HasBias = true;
    constexpr int32_t kfc11_Biases[2] = {
           37,   -32
    };

    // Softmax softmax12 (reads fc11)
    constexpr float   ksoftmax12_InputScale      = 1.1422832012176514f;
    constexpr int32_t ksoftmax12_InputZeroPoint  = -26;
    constexpr float   ksoftmax12_OutputScale     = 0.00390625f;
    constexpr int32_t ksoftmax12_OutputZeroPoint = -128;
} // namespace vibration
