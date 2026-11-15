/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import.
*/
#pragma once

#include <cstddef>
#include <cstdint>
namespace import_lstm_demo_cell16 {
    constexpr std::size_t NumInputs = 3;
    constexpr std::size_t NumHidden = 8;
    constexpr std::size_t NumOutputs = 2;
    constexpr std::size_t SequenceLength = 24;

    // Activation calibration.
    constexpr float   kinput_Scale     = 0.008613460671667959f;
    constexpr int32_t kinput_ZeroPoint = 2;
    constexpr float   klstm_Scale     = 0.005900940087836559f;
    constexpr int32_t klstm_ZeroPoint = 0;
    constexpr float   khead_Scale     = 0.005084513916688807f;
    constexpr int32_t khead_ZeroPoint = -8;

    // Weight calibration + int8 weights + int32 biases.
    constexpr float khead_WeightScale = 0.006263596335733969f;
    constexpr int8_t khead_Weights[16] = {
          -64,   114,   -79,   -82,   -38,   -69,    44,   -98,   101,    91,  -127,    11,
         -100,   -62,   -21,   -12
    };
    constexpr int32_t khead_Biases[2] = {
         -172,  2313
    };

    // LSTM lstm (cpp/qlstm.hpp QLSTMCell): gates i, f, g, o; weights
    // gate-major [gate][hidden][input]; runs from zero state. klstm_Scale /
    // klstm_ZeroPoint above are the hidden state's grid.
    constexpr std::size_t klstm_NumInputs = 3;
    constexpr std::size_t klstm_NumHidden = 8;
    constexpr std::size_t klstm_SequenceLength = 24;
    typedef int16_t lstm_CellType;
    constexpr float   klstm_CellScale     = 6.325539516079112e-05f;
    constexpr int32_t klstm_CellZeroPoint = 0;
    constexpr int32_t klstm_CellQMin      = -32768;
    constexpr int32_t klstm_CellQMax      = 32767;
    constexpr float   klstm_LutInputScale = 0.06299212598425197f;
    constexpr float klstm_InputWeightScales[4] = { 0.00856837325208769f, 0.00860773296806756f, 0.008844930355943094f, 0.008588405106011339f };
    constexpr float klstm_RecurrentWeightScales[4] = { 0.004443547857089306f, 0.004373411963305135f, 0.004194053139273576f, 0.004434993417244258f };
    constexpr int8_t klstm_InputWeights[96] = {
          -98,     0,    27,  -124,   -93,   113,  -113,   -98,   118,    32,   -35,     3,
           43,   -59,   -96,    76,    45,     3,    84,    13,   127,   -78,    14,    -4,
          -39,    24,   -70,    79,    97,   -98,    -9,   -59,  -110,   104,   -18,   -93,
           46,   -78,   106,   -74,  -123,   -79,   -41,    -8,   107,    52,   -42,  -127,
          -87,   127,   -10,    49,  -114,  -119,    88,    22,   -49,   -47,  -105,   -84,
         -122,    87,    -9,   -95,    61,   -78,  -112,    25,   101,  -121,    78,   -79,
         -107,  -127,   -55,    60,    -2,    93,   -75,   -49,   -64,   126,   116,   -42,
          -17,   -49,    65,  -121,  -114,   -25,   -67,    91,    64,    12,    43,    51
    };
    constexpr int8_t klstm_RecurrentWeights[256] = {
           72,   109,   -89,    32,   -91,   -14,    73,   100,    66,  -118,   -36,   -86,
          127,   -91,   -65,   -36,  -112,    94,    35,   -87,     0,  -107,    28,   -68,
         -117,   -98,    14,    35,   -45,    37,   -38,   -94,   -47,   -27,   105,   -98,
         -105,    16,   118,   104,    51,  -110,    78,    47,   -91,    -9,  -115,    77,
           56,    78,    66,   -59,    74,   -64,   -92,   -28,    -1,   -55,    27,    26,
          -66,    31,   -36,    60,   -54,    77,   -22,    14,    45,     5,   -63,   124,
         -104,   -45,    13,  -122,   -89,    77,    73,    35,   104,    66,   -52,    37,
          -41,    65,   -30,   -90,    97,    49,    63,    15,    73,   -13,    17,  -113,
           14,    81,    53,    78,    -1,    99,  -101,    97,   -33,  -106,    31,   -12,
          -19,    91,   -94,    30,   -22,     7,     0,   -95,     3,    94,   -85,  -126,
         -112,   -10,   123,  -118,   127,     9,   -98,   -21,   -79,    58,    11,   -57,
          -66,    99,    72,   -17,   -25,    64,   127,  -113,   -92,   -37,     2,  -116,
         -120,   -76,   -31,    65,    30,  -127,  -123,   -13,   101,   112,   -36,   104,
          116,   -25,    66,   -36,   -93,    20,  -112,    44,    83,   112,   -14,  -103,
          109,   100,   126,    25,    47,   -34,   -85,   -56,    60,   -47,    52,     1,
          -14,   126,   -90,    -3,   -92,   118,    -3,   -45,   -89,    25,   -27,  -101,
         -119,   -28,    21,     5,   102,   105,   111,    76,    -6,     6,   -26,   -26,
         -117,    76,   -65,  -120,    -9,    56,   -16,   -35,    40,   -86,  -125,    24,
            8,    93,   -21,    73,  -123,  -121,    25,   -70,  -110,   -95,   -32,   -42,
           16,    99,   -34,   -81,    20,  -127,   -29,   -47,   -29,  -115,    14,    32,
           64,   117,   -37,    40,   -15,   -73,    99,    63,   113,    82,   106,   -95,
         -124,   -77,   -22,    48
    };
    constexpr int32_t klstm_Biases[32] = {
            2,    -1,     1,    -1,     3,    -2,    -4,    -1,    19,    17,    17,    20,
           14,    11,    17,    14,     2,    -1,     2,    -3,    -1,     0,     0,     2,
            0,    -1,    -4,     1,     2,     1,    -4,     1
    };
} // namespace import_lstm_demo_cell16
