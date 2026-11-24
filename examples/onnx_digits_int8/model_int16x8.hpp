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
#include "weights.hpp"

#if !TINYMIND_ENABLE_QUANTIZATION
#error "a generated tinymind model needs TINYMIND_ENABLE_QUANTIZATION=1"
#endif

namespace digits {
namespace int16x8 {
    // Model interface. input and output are caller-owned int16_t buffers in
    // NHWC order; arena is caller-owned scratch of kModelArenaSize bytes,
    // aligned to kModelArenaAlignment, overlapping neither of them.
    typedef int16_t ModelActivationType;
    constexpr std::size_t kModelInputSize      = 64;
    constexpr std::size_t kModelOutputSize     = 10;
    constexpr std::size_t kModelArenaSize      = 2048;
    constexpr std::size_t kModelArenaAlignment = 4;
    constexpr float   kModelInputScale      = 3.051851308555342e-05f;
    constexpr int32_t kModelInputZeroPoint  = 0;
    constexpr float   kModelOutputScale     = 0.0017428637947887182f;
    constexpr int32_t kModelOutputZeroPoint = 0;

    // Memory plan: 5 arena buffers in 2048 bytes (unshared: 4096; busiest layer: 2048).
    //   offset    bytes  live      outputs
    //        0     1024    0-3    c1_Conv, Add
    //        0      512    5-6    pw_Conv
    //      512      256    4-5    dw_Conv
    //     1024     1024    1-2    c2_Conv
    //     1024      256    3-4    MaxPool
    constexpr std::size_t kModelUnsharedSize   = 4096;
    constexpr std::size_t kModelPeakLiveSize   = 2048;
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
        {0, 1024, 0, 3, "c1_Conv + Add"},
        {0, 512, 5, 6, "pw_Conv"},
        {512, 256, 4, 5, "dw_Conv"},
        {1024, 1024, 1, 2, "c2_Conv"},
        {1024, 256, 3, 4, "MaxPool"}
    };

    namespace model_detail {

        // [0] Conv c1_Conv <- input
        constexpr int64_t c1_Conv_biases16[8] = {
            1571274, 1090820, 1612650, 836265, 1322759, 2369118,
            -3010966, 590191
        };
        constexpr tinymind::Requantizer<int64_t, int16_t> c1_Conv_requantizers[8] = {
            {1223013959, 9, 0, 0, 32767}, {1523593256, 10, 0, 0, 32767}, {1077312074, 9, 0, 0, 32767},
            {1689086850, 9, 0, 0, 32767}, {1797262620, 9, 0, 0, 32767}, {1221314614, 9, 0, 0, 32767},
            {1397797218, 9, 0, 0, 32767}, {1635708787, 9, 0, 0, 32767}
        };
        typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int64_t, int16_t, 8, 8, 1, 3, 3, 1, 1, 8, 1, 1, 1, 1> c1_Conv_type;
        constexpr c1_Conv_type c1_Conv_layer = {kc1_Conv_Weights, c1_Conv_biases16, 0, c1_Conv_requantizers};
        static_assert(c1_Conv_type::OutputSize == 512, "c1_Conv: output size");

        // [1] Conv c2_Conv <- c1_Conv
        constexpr int64_t c2_Conv_biases16[8] = {
            85194, 400914, -230911, -3212, -786665, 481996,
            -285651, 5911
        };
        constexpr tinymind::Requantizer<int64_t, int16_t> c2_Conv_requantizers[8] = {
            {1115667282, 8, 0, 0, 32767}, {1084604887, 10, 0, 0, 32767}, {1312256952, 8, 0, 0, 32767},
            {1499176406, 11, 0, 0, 32767}, {1852578656, 11, 0, 0, 32767}, {1575148535, 11, 0, 0, 32767},
            {1953813635, 9, 0, 0, 32767}, {1877412341, 9, 0, 0, 32767}
        };
        typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int64_t, int16_t, 8, 8, 8, 3, 3, 1, 1, 8, 1, 1, 1, 1> c2_Conv_type;
        constexpr c2_Conv_type c2_Conv_layer = {kc2_Conv_Weights, c2_Conv_biases16, 0, c2_Conv_requantizers};
        static_assert(c2_Conv_type::OutputSize == 512, "c2_Conv: output size");

        // [2] Add Add <- c1_Conv, c2_Conv
        typedef tinymind::QAdd<int16_t, int16_t, int16_t, 512> Add_type;
        constexpr Add_type Add_layer = {0, 0, 15, 1372513147, 2, 1073741824, 0, {1990989975, 14, 0, 0, 32767}};

        // [3] MaxPool MaxPool <- Add
        typedef tinymind::QMaxPool2D<int16_t, 8, 8, 8, 2, 2, 2, 2> MaxPool_type;
        constexpr MaxPool_type MaxPool_layer = {};
        static_assert(MaxPool_type::OutputSize == 128, "MaxPool: output size");

        // [4] DepthwiseConv dw_Conv <- MaxPool
        constexpr int64_t dw_Conv_biases16[8] = {
            96888, -294389, 120917, 15548, -56539, -278327,
            33795, 93418
        };
        constexpr tinymind::Requantizer<int64_t, int16_t> dw_Conv_requantizers[8] = {
            {1313265133, 7, 0, 0, 32767}, {1966709800, 9, 0, 0, 32767}, {2132588349, 8, 0, 0, 32767},
            {1373839068, 7, 0, 0, 32767}, {1226852937, 7, 0, 0, 32767}, {1769861063, 9, 0, 0, 32767},
            {2032657621, 8, 0, 0, 32767}, {1201689121, 7, 0, 0, 32767}
        };
        typedef tinymind::QDepthwiseConv2D<int16_t, int8_t, int64_t, int16_t, 4, 4, 8, 3, 3, 1, 1, 1, 1, 1, 1> dw_Conv_type;
        constexpr dw_Conv_type dw_Conv_layer = {kdw_Conv_Weights, dw_Conv_biases16, 0, dw_Conv_requantizers};
        static_assert(dw_Conv_type::OutputSize == 128, "dw_Conv: output size");

        // [5] Conv pw_Conv <- dw_Conv
        constexpr int64_t pw_Conv_biases16[16] = {
            -20688, 73501, 46773, -13621, -289378, 108067,
            -120403, -300685, 22487, -103312, -62065, 3726,
            125800, 43047, 153298, 183624
        };
        constexpr tinymind::Requantizer<int64_t, int16_t> pw_Conv_requantizers[16] = {
            {1110908696, 6, 0, 0, 32767}, {1797043519, 7, 0, 0, 32767}, {1573853700, 6, 0, 0, 32767},
            {1780047226, 7, 0, 0, 32767}, {1097553974, 8, 0, 0, 32767}, {2009775858, 7, 0, 0, 32767},
            {1463569653, 7, 0, 0, 32767}, {1387065814, 7, 0, 0, 32767}, {1205505734, 6, 0, 0, 32767},
            {1940666946, 7, 0, 0, 32767}, {1077167233, 6, 0, 0, 32767}, {1317541903, 6, 0, 0, 32767},
            {1805791928, 7, 0, 0, 32767}, {1921967464, 7, 0, 0, 32767}, {1713489867, 7, 0, 0, 32767},
            {1461158728, 7, 0, 0, 32767}
        };
        typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int64_t, int16_t, 4, 4, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0> pw_Conv_type;
        constexpr pw_Conv_type pw_Conv_layer = {kpw_Conv_Weights, pw_Conv_biases16, 0, pw_Conv_requantizers};
        static_assert(pw_Conv_type::OutputSize == 256, "pw_Conv: output size");

        // [6] Dense fc_Gemm <- pw_Conv
        constexpr int64_t fc_Gemm_biases16[10] = {
            -22230, -12978, 15420, -11051, -2184, 17861,
            -3212, 2313, 6682, -7581
        };
        constexpr tinymind::Requantizer<int64_t, int16_t> fc_Gemm_requantizers[10] = {
            {2061131621, 9, 0, -32767, 14174}, {2060210612, 9, 0, -32767, 14174}, {1155230725, 8, 0, -32767, 14174},
            {1111172279, 8, 0, -32767, 14174}, {1833324113, 8, 0, -32767, 14174}, {1207098093, 8, 0, -32767, 14174},
            {2031127385, 9, 0, -32767, 14174}, {1548250969, 8, 0, -32767, 14174}, {1114052609, 8, 0, -32767, 14174},
            {1494482546, 8, 0, -32767, 14174}
        };
        typedef tinymind::QConv2DPerChannel<int16_t, int8_t, int64_t, int16_t, 1, 1, 256, 1, 1, 1, 1, 10> fc_Gemm_type;
        constexpr fc_Gemm_type fc_Gemm_layer = {kfc_Gemm_Weights, fc_Gemm_biases16, 0, fc_Gemm_requantizers};
        static_assert(fc_Gemm_type::OutputSize == 10, "fc_Gemm: output size");
    } // namespace model_detail

    /**
     * Run the model: input (kModelInputSize int16 values on the input grid)
     * to output (kModelOutputSize values on the output grid). arena
     * holds every intermediate activation; its contents need not be
     * kept between calls.
     */
    inline void forward(const int16_t* input, int16_t* output, int16_t* arena)
    {
        model_detail::c1_Conv_layer.forward(input, (arena + 0));
        model_detail::c2_Conv_layer.forward((arena + 0), (arena + 512));
        model_detail::Add_layer.forward((arena + 0), (arena + 512), (arena + 0));
        model_detail::MaxPool_layer.forward((arena + 0), (arena + 512));
        model_detail::dw_Conv_layer.forward((arena + 512), (arena + 256));
        model_detail::pw_Conv_layer.forward((arena + 256), (arena + 0));
        model_detail::fc_Gemm_layer.forward((arena + 0), output);
    }
    constexpr std::size_t kModelWeightBytes = 3408;
} // namespace int16x8
} // namespace digits
