/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import_onnx (model code generation).
*/
#pragma once

#include <cstddef>
#include <cstdint>

#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdense.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpool2d.hpp"
#include "qadd.hpp"
#include "qint4.hpp"
#include "weights.hpp"

#if !TINYMIND_ENABLE_QUANTIZATION
#error "a generated tinymind model needs TINYMIND_ENABLE_QUANTIZATION=1"
#endif

namespace digits {
namespace int4 {
    // Model interface. input and output are caller-owned int8_t buffers in
    // NHWC order; arena is caller-owned scratch of kModelArenaSize bytes,
    // aligned to kModelArenaAlignment, overlapping neither of them.
    typedef int8_t ModelActivationType;
    constexpr std::size_t kModelInputSize      = 64;
    constexpr std::size_t kModelOutputSize     = 10;
    constexpr std::size_t kModelArenaSize      = 1024;
    constexpr std::size_t kModelArenaAlignment = 4;
    constexpr float   kModelInputScale      = 0.003921568859368563f;
    constexpr int32_t kModelInputZeroPoint  = -128;
    constexpr float   kModelOutputScale     = 0.30850204825401306f;
    constexpr int32_t kModelOutputZeroPoint = 50;

    // Memory plan: 5 arena buffers in 1024 bytes (unshared: 2048; busiest layer: 1024).
    //   offset    bytes  live      outputs
    //        0      512    0-3    c1_Conv, Add
    //        0      256    5-6    pw_Conv
    //      256      128    4-5    dw_Conv
    //      512      512    1-2    c2_Conv
    //      512      128    3-4    MaxPool
    constexpr std::size_t kModelUnsharedSize   = 2048;
    constexpr std::size_t kModelPeakLiveSize   = 1024;
    constexpr std::size_t kModelLayerCount     = 7;
    constexpr std::size_t kModelBufferCount    = 5;
    constexpr const char* kModelLayerNames[7] = {"c1_Conv", "c2_Conv", "Add", "MaxPool", "dw_Conv", "pw_Conv", "fc_Gemm"};
    struct ModelBuffer
    {
        std::size_t offset;
        std::size_t size;
        std::size_t first_layer;
        std::size_t last_layer;
        const char* outputs; // the layers whose output lives here, in order
    };
    constexpr ModelBuffer kModelBuffers[5] = {
        {0, 512, 0, 3, "c1_Conv + Add"},
        {0, 256, 5, 6, "pw_Conv"},
        {256, 128, 4, 5, "dw_Conv"},
        {512, 512, 1, 2, "c2_Conv"},
        {512, 128, 3, 4, "MaxPool"}
    };

    namespace model_detail {

        // [0] Conv c1_Conv <- input
        // int4 per channel: 36 bytes, against 72 as int8.
        constexpr uint8_t c1_Conv_weights_int4[36] = {
            160,  57,  59,  98, 165, 204, 164,  76, 119, 102, 194,  19, 235, 119,  65,  84,
            198, 217,  51,  22,  39,  86,  13,  43, 242,  25,  22, 116, 194,  52,  43, 101,
            191,  35, 233,   5
        };
        constexpr int32_t c1_Conv_biases_int4[8] = {
            634, 428, 682, 348, 559, 962, -1275, 235
        };
        constexpr tinymind::Requantizer<int32_t, int8_t> c1_Conv_requantizers[8] = {
            {1383139935, 5, -128, -128, 127}, {1680710352, 6, -128, -128, 127}, {1234845072, 5, -128, -128, 127},
            {1902139141, 5, -128, -128, 127}, {2024569964, 5, -128, -128, 127}, {1395011169, 5, -128, -128, 127},
            {1575782529, 5, -128, -128, 127}, {1850618972, 5, -128, -128, 127}
        };
        typedef tinymind::QConv2DInt4<int8_t, int32_t, int8_t, 8, 8, 1, 3, 3, 1, 1, 8, 1, 1, 1, 1> c1_Conv_type;
        constexpr c1_Conv_type c1_Conv_layer = {c1_Conv_weights_int4, c1_Conv_biases_int4, -128, c1_Conv_requantizers};
        static_assert(c1_Conv_type::OutputSize == 512, "c1_Conv: output size");

        // [1] Conv c2_Conv <- c1_Conv
        // int4 per channel: 288 bytes, against 576 as int8.
        constexpr uint8_t c2_Conv_weights_int4[288] = {
             46,  37,   2,  36, 224,  52, 227,  33, 192,  69, 224, 209, 241, 240,  79,  47,
             51, 224,  79, 239,  32,  13,  47, 220,  49, 210,  49,  51, 118, 223,  82,  38,
            101, 221,  98,  30, 254,  47,  30,  47, 254, 225,  45, 240,   9, 239, 162, 177,
            205, 192, 252, 192,  13, 243, 189,  16, 253, 212,  48, 177,   3,  49, 241,  25,
            206, 191,  44, 179, 236, 223, 225, 164, 237,   2,  17,  49, 255, 192,   0,  33,
            226, 225, 221, 223, 193,  34,  17,  66, 226, 241,  31,  35,  71,  46, 254, 178,
             65,   2, 113,  15,  35,   3, 226,  19, 254,  64, 245, 227, 220, 211, 213,  19,
            159,  27, 187, 192,  77, 237, 111, 189,  11, 179, 253,  77,  78,   1, 236, 242,
            236, 177, 219,  19, 208, 236, 192, 245, 157,  62, 175, 144, 238,  30, 187, 193,
             16,   3, 191,  17,  48,  49,  58,  34,  44,  14,  43,  31,   1,  62, 156, 241,
            144,  35,  82, 222,  63, 227, 253,  90, 226,  48, 253,  13, 223, 254, 176,  46,
             28, 165,  81,  77, 205,   5, 236, 242,  55,  46, 211,   3,  76, 228, 116, 228,
            114, 251,  26, 236, 239, 246,  17, 156,  65, 115,  27,  34,  94,  49, 255,  10,
            245,  49,  80, 186, 222,  58, 163,  11, 252,  80, 178, 242, 238,  35, 227,  84,
              1, 224,   1,   4, 219,  50, 225, 243, 153,   3, 177,  69, 238, 207,  48, 100,
            255,  53, 255, 241, 206,  18, 225,  50, 236,  79,  35,  67,  80,  16,  66,  94,
             66,  50,  49,  32,   0,  67,   2, 245, 220,  49, 226,  30,  21,  85,   1,  67,
              0,  51,  18, 229, 114, 239,  32, 255,  35,  68,  34, 243,  31,  33, 241, 242
        };
        constexpr int32_t c2_Conv_biases_int4[8] = {
            36, 161, -98, -1, -333, 204, -121, 2
        };
        constexpr tinymind::Requantizer<int32_t, int8_t> c2_Conv_requantizers[8] = {
            {1272459327, 4, -128, -128, 127}, {1252629164, 6, -128, -128, 127}, {1521955440, 4, -128, -128, 127},
            {1719876593, 7, -128, -128, 127}, {2125304703, 7, -128, -128, 127}, {1807032905, 7, -128, -128, 127},
            {1119978155, 4, -128, -128, 127}, {1074444510, 4, -128, -128, 127}
        };
        typedef tinymind::QConv2DInt4<int8_t, int32_t, int8_t, 8, 8, 8, 3, 3, 1, 1, 8, 1, 1, 1, 1> c2_Conv_type;
        constexpr c2_Conv_type c2_Conv_layer = {c2_Conv_weights_int4, c2_Conv_biases_int4, -128, c2_Conv_requantizers};
        static_assert(c2_Conv_type::OutputSize == 512, "c2_Conv: output size");

        // [2] Add Add <- c1_Conv, c2_Conv
        typedef tinymind::QAdd<int8_t, int8_t, int8_t, 512> Add_type;
        constexpr Add_type Add_layer = {-128, -128, 20, 1388594262, 2, 1073741824, 0, {1879263757, 19, -128, -128, 127}};

        // [3] MaxPool MaxPool <- Add
        typedef tinymind::QMaxPool2D<int8_t, 8, 8, 8, 2, 2, 2, 2> MaxPool_type;
        constexpr MaxPool_type MaxPool_layer = {};
        static_assert(MaxPool_type::OutputSize == 128, "MaxPool: output size");

        // [4] DepthwiseConv dw_Conv <- MaxPool
        // int4 per channel: 36 bytes, against 72 as int8.
        constexpr uint8_t dw_Conv_weights_int4[36] = {
             22, 245, 233,  53,  15,  79, 155, 180,  84,  14, 243, 215, 160, 246, 238, 174,
            193, 116, 223,  62,  15, 118, 151, 238, 167, 251,  61, 192, 102, 148,  99, 241,
            243, 117, 208,   3
        };
        constexpr int32_t dw_Conv_biases_int4[8] = {
            38, -119, 49, 6, -24, -113, 14, 37
        };
        constexpr tinymind::Requantizer<int32_t, int8_t> dw_Conv_requantizers[8] = {
            {1556315626, 3, -128, -128, 127}, {1138090552, 4, -128, -128, 127}, {1222575795, 3, -128, -128, 127},
            {1593816573, 3, -128, -128, 127}, {1431844007, 3, -128, -128, 127}, {2059053174, 5, -128, -128, 127},
            {1182502475, 3, -128, -128, 127}, {1405262181, 3, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2DInt4<int8_t, int32_t, int8_t, 4, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1> dw_Conv_type;
        constexpr dw_Conv_type dw_Conv_layer = {dw_Conv_weights_int4, dw_Conv_biases_int4, -128, dw_Conv_requantizers};
        static_assert(dw_Conv_type::OutputSize == 128, "dw_Conv: output size");

        // [5] Conv pw_Conv <- dw_Conv
        // int4 per channel: 64 bytes, against 128 as int8.
        constexpr uint8_t pw_Conv_weights_int4[64] = {
             25, 118, 242,  52, 229,  75,  39,  28, 254, 149,   1,  45,  15,  96,  45,  23,
            161, 169,  43,  14, 239, 201,  22,  44,  39,  23, 244, 178,  83, 103,  11,  51,
            250, 148,   1,  47,  55, 147,  51,  30, 239,  19, 249,  35, 224, 146, 227,  30,
            228,  25, 214,  28,  31, 153,  13,  95, 219, 201, 247,  76,  52, 156,  37,  42
        };
        constexpr int32_t pw_Conv_biases_int4[16] = {
            -9, 30, 19, -5, -120, 46, -51, -126,
            10, -43, -25, 1, 52, 17, 65, 75
        };
        constexpr tinymind::Requantizer<int32_t, int8_t> pw_Conv_requantizers[16] = {
            {1224246091, 2, -128, -128, 127}, {1969528532, 3, -128, -128, 127}, {1747793712, 2, -128, -128, 127},
            {1970009170, 3, -128, -128, 127}, {1212858947, 4, -128, -128, 127}, {1104975114, 2, -128, -128, 127},
            {1621643541, 3, -128, -128, 127}, {1532276275, 3, -128, -128, 127}, {1327224269, 2, -128, -128, 127},
            {1083447933, 2, -128, -128, 127}, {1186188567, 2, -128, -128, 127}, {1466615873, 2, -128, -128, 127},
            {2007287620, 3, -128, -128, 127}, {2132635087, 3, -128, -128, 127}, {1880535342, 3, -128, -128, 127},
            {1624949024, 3, -128, -128, 127}
        };
        typedef tinymind::QConv2DInt4<int8_t, int32_t, int8_t, 4, 4, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0> pw_Conv_type;
        constexpr pw_Conv_type pw_Conv_layer = {pw_Conv_weights_int4, pw_Conv_biases_int4, -128, pw_Conv_requantizers};
        static_assert(pw_Conv_type::OutputSize == 256, "pw_Conv: output size");

        // [6] Dense fc_Gemm <- pw_Conv
        // int4 per channel: 1280 bytes, against 2560 as int8.
        constexpr uint8_t fc_Gemm_weights_int4[1280] = {
            240,  45, 240,  11, 255, 240, 224, 239, 208, 241, 225,   1,   0,   0, 222, 255,
            241, 253,   0,  10, 225,   0,  47, 225, 209,  28, 224, 249, 223, 240,  47, 241,
            178,  31, 193,   1,   0,   2, 158, 221, 240, 251,   0, 250, 255, 255,  17,  17,
             47, 253,  32,   2,  28,  31, 193,  16, 241,  31, 192,  17,  11,   1, 239, 239,
            255, 242, 240,   1,   0,   0,   0,  31, 194,  46, 208,  13, 240, 241, 253, 207,
             31, 237, 255,   5,  31,  14, 241,  15, 224,   0,  15,  29,   1, 241,  16,   1,
             61, 221,  17, 252,  12, 255, 241,  15, 210,   2, 224,  30,  18,  18, 221, 240,
             15,  29, 224,   0,  12, 240,  17,  15,   0, 241, 240,  25,   0,  17,  16,   0,
             47, 239,  48, 252,   0, 255,   3,  18,   0,  15, 224,   1,   0,   0,  15, 255,
            224,  31, 240,   0, 255, 240, 254, 223,   0, 242,  32, 254, 241,  15, 208,  17,
              0, 224, 240,   0,  47,  15, 207,  16,  47, 240, 240,   1,  31,  16,   2,  16,
            224,  28, 176,  15,  14, 241,  15, 251,   1,  14,   0, 241,  12,   0,   0,  11,
            224,  31, 225,  29, 240, 240, 221, 239, 208,   0, 192,   9, 241,   0, 253, 240,
            225,  47, 208,  25,   0,   1,  15,  16,   0,  28, 224, 246, 227, 255, 192,  30,
            240, 241, 240, 251, 240,  15,  31,  15,  30, 223,   0, 253,   0,  15,  33,  18,
            225,  18, 208,  21,   3,   1,  14, 255, 210,  47, 208,  15, 227,   0,  14, 192,
            240, 242, 224,   0,   1,   0, 238, 255, 240,   0,   0,   0,  32,  16,  16,  32,
            192,  18, 208,  30, 240,   1,  45, 223, 239,  30, 208,   0, 254, 241,  31, 254,
             46, 225,  32, 251,   0,  15,  18,  17,  30, 225,  32, 249,   0,  15, 225,  18,
            240, 243,  16, 240, 225,   0,  17,  35,  44, 237, 224, 241,  27, 254, 225,  30,
            208,  47,  16,  16, 240, 242,  65, 241, 193,  29, 160,   2, 255, 240, 236, 204,
            240, 253,  48, 239, 227,  30, 240, 243,   0, 241,  32, 240, 225,  14, 209,  18,
             31, 224,  16,   2,   0,  16,  33,  17,  45, 223,   0,  14,  29,  31,   1,  30,
              1,  18, 240,  27,  18,  33,   0, 254,  16,  45, 240,  43,  28,   1,  17,  15,
            240, 210,   0, 255,  17,  15, 239,   1,  31, 227,  48, 242,  33,  47,  34,  66,
            176,  20, 208,  30, 193, 241,  44, 223,   0,  13, 224,  19,  29,   1, 240,  30,
            254, 209,   0,  15,   0,   0,   0,  15, 224, 240, 224,   0,   0,   0, 239, 255,
            192,   2, 240, 255, 225,  15,  30,   1, 240,  14, 192, 239, 254, 255, 208, 253,
             32,  28,  16, 240, 237, 240,  50,   1,  62, 208,  32, 252,  15,  14, 242,  33,
              0, 240, 224,  31,   0,  16, 223, 255,   0,   0, 240,  17,  45,   1,   0, 254,
            242,   1, 192,  16,  19,  17, 192,  12, 241,  31, 208,   2,  29,  16, 224,  13,
             30, 242,  32,   9,  16,   0,  33,  17,  15,  13,  16,  13,  13, 240,   1,  16,
            240,  15,   0,   1,   0,   0,  32,   0, 240,   0, 224,   0,  15, 240, 239, 255,
             31, 224,  32, 238, 240,  14,  33,  34,  31, 225,  16, 225,   1,  15, 241,  16,
              1,  47, 240,   2,   0, 241,  33,  15,  16,  15,  16,   0,  31,  16, 225,   0,
             16,  12,   0,   0,  16,   0, 240, 240,   1,  15,  16,  30,  15,  16, 240,   0,
              0,  16,  16,  15, 241,   0,   0,   1, 240,  17, 240,   1,   0,   1,  15, 240,
            208,  15, 192,  31, 241, 241,  30, 223, 224,  14, 224,  16,  14, 241,  31, 239,
            225,  32,  16,  19,   1,   1,  15, 241,  15, 240,  32,  14,   0,   0,  17,  17,
            224,  31, 224,   9, 240, 240,  15, 255, 224,   1,  32, 253,   0,   0,  15,  17,
             16,  30, 240,  12, 240,   0, 192,   0, 192,  47, 192,  16,   0, 241, 220, 190,
             63, 252,  16,   2,  79,  15,   3,  32,  16,  29, 240,  45,  46,  17,  49,  31,
            224, 242,   0,   1, 241,   0,  15,   0, 240,  16,   0,   9, 241, 241,  32,   0,
            240,  28, 224,   0,   1, 240,  16, 240, 208, 242,  16,  10, 227, 241,  79,  18,
             31, 240,  32,   0,  15,   0,  49,  16,  47, 225,  32, 253,   0,  30, 241,  33,
             15,   2,  48, 243,   0,   0,   1,  18,  31, 240,  16, 241,  31,  15, 225,  17,
              0, 240, 208,   0,  31,  16, 223,   0,  31,  14, 224,   2,  13,   0, 193, 254,
             47, 240,   0,   1,  14,  15, 241,  14,  78, 239,  16, 244,  45,  15, 226,  47,
            239,  15, 224, 237, 255, 240, 208, 255, 239,  13, 208, 255, 239, 255, 238, 238,
             30, 211,  16, 241,  16,  15, 210,  48,  28, 196,  17, 239,  29,  15,  17,  32,
             15,  13, 224, 253, 239, 240, 223, 240, 193,   1, 192, 241, 240,   0, 252, 239,
              0, 255,   0,   0, 241,  15,  16,  17,  30, 222,  32, 255, 255,  29, 209,  33,
            240, 179, 208,  23,  33,  16, 144,  12, 241,  28, 240,  15,   0, 241, 241, 254,
             18,  26, 255,   0,   0,  15,  17, 241,  18,  29,  16,  29,  15,  16,  15, 226,
             43, 220,  64, 219, 235, 238,  51,  35, 242,  18, 208,  12,   2,  17, 239, 255,
            225,  16, 240,  47,  49,   1,  15, 240, 225,  18, 240,  25,   2,   1,  48,   2,
             47, 209,  16, 255,  17,  31,   0,  17,  31, 242,  16,   3,  17,  16,  33,  33,
            241,  30, 224,  15,   0, 240,  31,  31,   1,  28, 240,  16,  31,  16, 255, 240,
              0, 240,   0,  12,   0,   0,   0,   0,   0, 240,   0,  11,   0,  16,   0,  17,
            240,  15, 240,   2, 240,   0, 255, 240, 225,  17, 240,   2, 240, 241, 238, 192,
             46, 239,  16, 254,  14,  15, 225,  17,  15, 241,   0,   2,  16,   0,  16,  31,
             31, 225,  16,   0,  16,   0,  33,  17,  15, 240, 224,   0,   0, 241,  16,  15,
            210,  64, 224,  17, 240,   1, 207, 223,  15, 255,  48, 233, 239, 255,   1,   1,
            224,   1, 224, 240, 240,   0, 239, 240,  31, 210,  48, 255, 241,  31,   1,  34,
            241,  16, 240,  19,   0,   1,  32, 255,  17,  12,   0, 253, 239, 255, 241, 240,
              0,  14, 240, 241, 240,  15, 191, 225,   1, 255,  32,  13,   3,  31,  16,  18,
             47, 225,  48, 252,  17,  15, 241,  18, 209,  16, 192,   3,   0,   0, 239, 253,
            241,  14,   0,  13,   3,   0,  17, 243,  31, 253,   0,  14,  30,   0,  17,   1,
             17,  33, 208,  30,   0,   1, 192, 254, 239,  14,   0,   2,  15,   0,  16,   0,
            240,  11, 160,   1, 255, 224, 208, 236, 225, 240,   0,  11,   0,   1,  63,   1,
             30, 177,  32, 252, 240,  15,  33,  16,   1,  16, 224,  25,   0,   1, 240, 253,
            255,  30, 240, 255,  13, 240,  17,   1,  15,  13, 192, 244, 255, 240, 224, 255,
            225,  16, 208,  17,   0,   1, 236, 238,  32,  13,   0,  12, 255,   0,   1,   1,
              0,  17, 240,  17,  16,   0, 240,   0, 225,  41, 240,  15, 225, 241, 254, 208,
              0,  17,  16,  15, 240, 240,  16,   1,   0,  15, 240,  31,  15,   1,  15,   0,
              0,  15, 224,  14,  29,   1,   0,  29,   0,  31,   0,  18,  18,   0,  16, 209,
            239, 224,   0, 255,   0,  15,  16,  16,  15, 241,  16,  13,   0,   0,  17,  17,
             31, 240,  16,  12,  15,  15,   0,  16,   0,  14,   0,   1,   0,   0,   1,  16,
            240,  31, 208, 240,   1,   0, 223,   1, 225,   0,  16,  15, 241,   0,  14, 242,
             31, 239,  32, 226, 241,  14, 241,  18, 240,   0, 224,  12,   0, 240, 255, 254
        };
        constexpr int32_t fc_Gemm_biases_int4[10] = {
            -9, -5, 6, -5, -1, 7, -1, 1,
            2, -3
        };
        constexpr tinymind::Requantizer<int32_t, int8_t> fc_Gemm_requantizers[10] = {
            {1815878324, 5, 50, -128, 127}, {1785370676, 5, 50, -128, 127}, {2019033453, 5, 50, -128, 127},
            {1938686588, 5, 50, -128, 127}, {1592924104, 4, 50, -128, 127}, {2091784610, 5, 50, -128, 127},
            {1771099907, 5, 50, -128, 127}, {1355606715, 4, 50, -128, 127}, {1947408861, 5, 50, -128, 127},
            {1316744246, 4, 50, -128, 127}
        };
        typedef tinymind::QDenseInt4<int8_t, int32_t, int8_t, 256, 10> fc_Gemm_type;
        constexpr fc_Gemm_type fc_Gemm_layer = {fc_Gemm_weights_int4, fc_Gemm_biases_int4, -128, fc_Gemm_requantizers};
        static_assert(fc_Gemm_type::OutputLength == 10, "fc_Gemm: output size");
    } // namespace model_detail

    /**
     * Run the model: input (kModelInputSize int8 values on the input grid)
     * to output (kModelOutputSize values on the output grid). arena
     * holds every intermediate activation; its contents need not be
     * kept between calls.
     */
    inline void forward(const int8_t* input, int8_t* output, int8_t* arena)
    {
        model_detail::c1_Conv_layer.forward(input, (arena + 0));
        model_detail::c2_Conv_layer.forward((arena + 0), (arena + 512));
        model_detail::Add_layer.forward((arena + 0), (arena + 512), (arena + 0));
        model_detail::MaxPool_layer.forward((arena + 0), (arena + 512));
        model_detail::dw_Conv_layer.forward((arena + 512), (arena + 256));
        model_detail::pw_Conv_layer.forward((arena + 256), (arena + 0));
        model_detail::fc_Gemm_layer.forward((arena + 0), output);
    }
    constexpr std::size_t kModelWeightBytes = 1704;
} // namespace int4
} // namespace digits
