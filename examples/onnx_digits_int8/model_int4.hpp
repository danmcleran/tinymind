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
    constexpr float   kModelOutputScale     = 0.32083380222320557f;
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
        constexpr tinymind::Requantizer<int32_t, int8_t> c1_Conv_requantizers[8] = {
            {1223013959, 9, -128, -128, 127}, {1523593256, 10, -128, -128, 127}, {1077312074, 9, -128, -128, 127},
            {1689086850, 9, -128, -128, 127}, {1797262620, 9, -128, -128, 127}, {1221314614, 9, -128, -128, 127},
            {1397797218, 9, -128, -128, 127}, {1635708787, 9, -128, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 8, 8, 1, 3, 3, 1, 1, 8, 1, 1, 1, 1> c1_Conv_type;
        constexpr c1_Conv_type c1_Conv_layer = {kc1_Conv_Weights, kc1_Conv_Biases, -128, c1_Conv_requantizers};
        static_assert(c1_Conv_type::OutputSize == 512, "c1_Conv: output size");

        // [1] Conv c2_Conv <- c1_Conv
        constexpr tinymind::Requantizer<int32_t, int8_t> c2_Conv_requantizers[8] = {
            {1115667282, 8, -128, -128, 127}, {1084604887, 10, -128, -128, 127}, {1312256952, 8, -128, -128, 127},
            {1499176406, 11, -128, -128, 127}, {1852578656, 11, -128, -128, 127}, {1575148535, 11, -128, -128, 127},
            {1953813635, 9, -128, -128, 127}, {1877412341, 9, -128, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 8, 8, 8, 3, 3, 1, 1, 8, 1, 1, 1, 1> c2_Conv_type;
        constexpr c2_Conv_type c2_Conv_layer = {kc2_Conv_Weights, kc2_Conv_Biases, -128, c2_Conv_requantizers};
        static_assert(c2_Conv_type::OutputSize == 512, "c2_Conv: output size");

        // [2] Add Add <- c1_Conv, c2_Conv
        typedef tinymind::QAdd<int8_t, int8_t, int8_t, 512> Add_type;
        constexpr Add_type Add_layer = {-128, -128, 20, 1372513161, 2, 1073741824, 0, {1990989998, 19, -128, -128, 127}};

        // [3] MaxPool MaxPool <- Add
        typedef tinymind::QMaxPool2D<int8_t, 8, 8, 8, 2, 2, 2, 2> MaxPool_type;
        constexpr MaxPool_type MaxPool_layer = {};
        static_assert(MaxPool_type::OutputSize == 128, "MaxPool: output size");

        // [4] DepthwiseConv dw_Conv <- MaxPool
        constexpr tinymind::Requantizer<int32_t, int8_t> dw_Conv_requantizers[8] = {
            {1313265133, 7, -128, -128, 127}, {1966709800, 9, -128, -128, 127}, {2132588349, 8, -128, -128, 127},
            {1373839068, 7, -128, -128, 127}, {1226852937, 7, -128, -128, 127}, {1769861063, 9, -128, -128, 127},
            {2032657621, 8, -128, -128, 127}, {1201689121, 7, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 4, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1> dw_Conv_type;
        constexpr dw_Conv_type dw_Conv_layer = {kdw_Conv_Weights, kdw_Conv_Biases, -128, dw_Conv_requantizers};
        static_assert(dw_Conv_type::OutputSize == 128, "dw_Conv: output size");

        // [5] Conv pw_Conv <- dw_Conv
        constexpr tinymind::Requantizer<int32_t, int8_t> pw_Conv_requantizers[16] = {
            {1110908696, 6, -128, -128, 127}, {1797043519, 7, -128, -128, 127}, {1573853700, 6, -128, -128, 127},
            {1780047226, 7, -128, -128, 127}, {1097553974, 8, -128, -128, 127}, {2009775858, 7, -128, -128, 127},
            {1463569653, 7, -128, -128, 127}, {1387065814, 7, -128, -128, 127}, {1205505734, 6, -128, -128, 127},
            {1940666946, 7, -128, -128, 127}, {1077167233, 6, -128, -128, 127}, {1317541903, 6, -128, -128, 127},
            {1805791928, 7, -128, -128, 127}, {1921967464, 7, -128, -128, 127}, {1713489867, 7, -128, -128, 127},
            {1461158728, 7, -128, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 4, 4, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0> pw_Conv_type;
        constexpr pw_Conv_type pw_Conv_layer = {kpw_Conv_Weights, kpw_Conv_Biases, -128, pw_Conv_requantizers};
        static_assert(pw_Conv_type::OutputSize == 256, "pw_Conv: output size");

        // [6] Dense fc_Gemm <- pw_Conv
        // int4 per channel: 1280 bytes, against 2560 as int8.
        constexpr uint8_t fc_Gemm_weights_int4[1280] = {
            240,  45, 240,  10, 255, 240, 224, 239, 208, 241, 225,   1,   0,   0, 221, 255,
            241, 253,   0,  10, 225,   0,  47, 225, 209,  29, 224, 249, 208, 240,  31, 224,
            178,  31, 193,   1,   0,   1, 158, 221, 240, 251,   0, 250, 255, 255,  17,  17,
             47, 253,  32,   2,  28,  31, 193,  16, 241,  31, 192,  17,  11,   1, 239, 239,
            255, 242, 240,   1,   0,   0,   0,  31, 210,  46, 208,  13, 240, 241, 254, 207,
             31, 237, 240,   5,  31,  14, 241,  15, 224,   0,   0,  29,   1, 241,  16,   1,
             61, 221,  17, 252,  12, 255, 241,  15, 210,   2, 224,  30,  18,  18, 221, 240,
             15,  29, 224,   0,  12, 240,  17,  15,   0, 240, 240,  25,   0,   1,   0,   0,
             47, 239,  48, 252,   0, 255,   3,  18,   0,  15, 224,   1,   0,   0,  15, 255,
            240,  31, 224,   0, 255, 240, 254, 223,   0, 242,  32, 254, 241,  31, 208,  17,
              0, 224, 240,   0,  31,  15, 207,  16,  47, 240, 240,   1,  31,  16,   2,  16,
            224,  28, 176,  15,  14, 240,  15, 251,   1,  14,   0, 241,  12,   0,   0,  11,
            224,  31, 224,  29, 240, 240, 221, 239, 208,   0, 192,   9, 241,   0, 253, 240,
            225,  47, 208,  25,   0,   0,  15,  16,   0,  28, 224, 245, 227, 255, 192,  30,
            240, 241, 240, 251, 240,  15,  31,  15,  30, 223,   0, 253,   0,  15,  33,  18,
            225,  17, 208,  21,   3,   1,  14, 255, 210,  47, 208,  15, 227,   0,  14, 192,
            240, 242, 224,   0,   1,   0, 238, 255, 240,   0,   0,   0,  32,  16,  16,  32,
            192,  18, 208,  30,   0,   1,  45, 223, 239,  30, 208,   0, 254, 241,  31, 254,
             46, 225,  32, 251,   0,  15,  18,  17,  30, 225,  32, 249,   0,  15, 241,  18,
            240, 243,  16, 240, 226,   0,  17,  35,  44, 237, 224, 241,  27, 254, 225,  30,
            208,  47,  16,  16, 240, 242,  65, 240, 193,  29, 160,   2, 255, 240, 236, 204,
            240, 253,  48, 239, 227,  31, 240, 243,   0, 241,  32, 240, 225,  14, 209,  18,
             31, 224,  16,   1,   0,  16,  33,  17,  45, 223,   0,  14,  29,  31,   1,  30,
              1,  18, 240,  27,  18,  33,   0,  14,  16,  45, 240,  43,  28,   1,  17,  15,
            240, 210,   0, 255,  17,  15, 255,   1,  31, 227,  48, 242,  33,  47,  35,  66,
            176,  20, 208,  30, 193, 241,  28, 223,   0,  13, 224,  19,  29,   1, 240,  30,
            254, 209,   0,  15,   0,   0,   0,  15, 240, 240, 224,   0,   0,   0, 239, 255,
            192,   2, 240, 255, 225,  15,  30,   1, 240,  14, 192, 239, 254, 255, 208, 253,
             32,  28,  16,   0, 237, 240,  50,   1,  62, 224,  32, 252,  15,  14, 242,  33,
              0, 240, 224,  31,   0,  16, 223,  15,   0,   0, 240,  17,  45,   1,   0, 254,
            242,   1, 192,  16,  19,  17, 192,  12,   1,  31, 208,   2,  29,  16, 224,  13,
             30, 242,  32,   9,  16,   0,  33,  17,  15,  13,  16,  13,  13, 240,   1,  16,
            240,  15,   0,   1,   0,   0,  32,   0, 240,   0, 224,   0,  15, 240, 239, 255,
             31, 224,  32, 238, 240,  14,  33,  34,  31, 225,  16, 225,   1,  15, 240,  16,
              1,  47,   0,   2,   0, 241,  33,  15,  16,  15,  16,   0,  31,   0, 225,   0,
             16,  12,   0,   0,  16,   0, 240, 240,   1,  15,   0,  30,  15,  16, 240,   0,
              0,  16,  16,  15, 241,   0,   0,   1, 240,  17, 240,   1,   0,   0,  15, 240,
            224,  15, 192,  31, 241, 241,  30, 223, 224,  14, 224,  16,  14, 241,  31, 239,
            225,  32,  16,  19,   1,   1,  15, 241,  15, 240,  32,  14,   0,   0,  17,  17,
            224,  31, 224,   9, 240, 240,  15, 255, 224,   1,  32, 253,   0,   0,  15,  17,
             16,  30, 240,  12, 240,   0, 192,   0, 192,  47, 192,  16,   0, 241, 220, 190,
             63, 252,  16,   1,  79,  15,   3,  32,  16,  29, 240,  45,  46,  17,  49,  31,
            224, 242,   0,   1, 241,   0,  15,   0, 240,  16,   0,   9, 241, 241,  32,   0,
            240,  13, 224,   0,   1, 240,  16, 240, 208, 242,  16,  10, 227, 241,  79,  18,
             31, 240,  32,   0,  15,   0,  33,  16,  47, 225,  32, 253,   0,  30, 241,  17,
             15,   1,  48, 243,   0,   0,   1,  18,  31, 240,  16, 241,  16,  15, 225,  17,
              0, 240, 208,   0,  31,  16, 223,   0,  31,  14, 224,   2,  13,   0, 193, 254,
             31, 240,   0,   1,  14,  15, 241,  14,  78, 239,  16, 244,  45,  15, 226,  47,
            239,  15, 224, 253, 255, 240, 224, 255, 239,  13, 208, 255, 239, 255, 238, 238,
             30, 210,  16, 241,  16,  15, 209,  48,  28, 196,  16, 255,  29,  15,  17,  32,
              0,  13, 224, 253, 239, 240, 223, 240, 193,   1, 192, 241, 240,   0, 253, 239,
              0, 255,   0,   0,   1,  15,  16,  17,  30, 222,  32, 255,  15,  29, 209,  33,
            240, 179, 208,  23,  33,  16, 144,  12, 241,  28, 240,  15,   0, 241, 241, 254,
             18,  26, 240,   0,   0,  15,  17, 241,  18,  29,   0,  29,   0,  16,  15, 226,
             43, 236,  64, 219, 235, 238,  51,  35, 242,  18, 208,  12,   2,  17, 239, 255,
            225,  16, 240,  47,  49,   1,   0, 240, 225,  17, 240,  25,   2,   1,  48,   2,
             31, 209,  16, 255,  17,  31,   0,  17,  16, 242,  16,   3,  17,  16,  33,  17,
            225,  30, 224,  15,   0, 240,  31,  15,   1,  28, 240,   0,  31,  16, 255, 240,
              0, 240,   0,  12,   0,   0,   0,   0,   0, 240,   0,  11,   0,  16,   0,  17,
            240,  15, 240,   2, 240, 240, 255, 240, 225,  17, 240,   2, 240, 241, 238, 192,
             46, 239,  16, 254,  14,  15, 225,  17,  15, 241,   0,   2,  16,   0,  16,  31,
             31, 225,  16,   0,  16,   0,  33,  17,  15, 240, 224,   0,   0, 241,  16,  15,
            210,  64, 224,  17, 240,   1, 207, 223,  15, 255,  48, 233, 239, 255,   1,   1,
            224,   1, 224, 240, 240,   0, 239, 240,  31, 210,  48, 255, 241,  31,   1,  34,
            241,  16, 240,  19,   0,   1,  16, 255,  17,  12,   0,  13, 239, 255, 241, 240,
              0,  14, 240, 241, 240,  15, 191, 225,   1, 255,  32,  14,   2,  31,  16,  18,
             47, 225,  48, 252,  17,  15, 241,  18, 209,  16, 192,   3,   0,   0, 239, 253,
            241,  14,   0,  13,   2,   0,  17, 243,  16, 253,   0,  14,  30,   0,  17,   1,
             17,  33, 208,  30,   0,   1, 192, 254, 239,  14,   0,   2,  15,   0,  16,   0,
            240,  11, 160,   1, 255, 224, 208, 236, 225, 240,   0,  11,   0,   1,  63,   1,
             30, 177,  32, 252, 240,  15,  33,  16,   1,  16, 224,  25,   0,   1, 240, 253,
            255,  30, 240,  15,  13, 240,  17,   1,  15,  13, 192, 244, 255, 240, 240, 255,
            225,  16, 208,  17,   0,   1, 236, 238,  32,  13,   0,  12, 255,   0,   1,   0,
              0,  17, 240,   1,  16,   0, 240,   0, 225,  41, 240,  15, 225, 241, 254, 208,
              0,  17,  16,  15, 240, 240,   0,   1,   0,  15, 240,  31,  15,   1,  31,   0,
              0,  15, 224,  14,  29,   1,   0,  29,   0,  31,   0,  18,  18,   0,  16, 209,
            239, 224,   0, 255,   0,  15,  16,  16,  15, 241,  16,  13,   0,   0,  17,  17,
             31, 240,  16,  12,  15,  15,   0,   0,   0,  14,   0,   1,   0,   0,   1,  16,
            240,  31, 208, 240,   1,   0, 223,   1, 225,   0,  16,  15, 241,   0,  14, 242,
             31, 239,  32, 226, 240,  14, 241,  18, 240,   0, 224,  12,   0, 240, 255, 254
        };
        constexpr int32_t fc_Gemm_biases_int4[10] = {
            -10, -6, 7, -5, -1, 8, -1, 1,
            3, -3
        };
        constexpr tinymind::Requantizer<int32_t, int8_t> fc_Gemm_requantizers[10] = {
            {1631440526, 5, 50, -128, 127}, {1630711523, 5, 50, -128, 127}, {1828791720, 5, 50, -128, 127},
            {1759044856, 5, 50, -128, 127}, {1451124821, 4, 50, -128, 127}, {1910900524, 5, 50, -128, 127},
            {1607691375, 5, 50, -128, 127}, {1225481842, 4, 50, -128, 127}, {1763604571, 5, 50, -128, 127},
            {1182922704, 4, 50, -128, 127}
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
    constexpr std::size_t kModelWeightBytes = 2128;
} // namespace int4
} // namespace digits
