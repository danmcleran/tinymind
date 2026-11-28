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
#include "qpointwiseconv2d.hpp"
#include "weights.hpp"

#if !TINYMIND_ENABLE_QUANTIZATION
#error "a generated tinymind model needs TINYMIND_ENABLE_QUANTIZATION=1"
#endif

namespace digits {
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
    constexpr std::size_t kModelLayerOutputSize[7] = {512, 512, 512, 128, 128, 256, 10};
    constexpr float kModelLayerOutputScale[7] = {0.017960580065846443f, 0.05620354413986206f, 0.060621194541454315f, 0.060621194541454315f, 0.07558827102184296f, 0.0674242153763771f, 0.32083380222320557f};
    constexpr int32_t kModelLayerOutputZeroPoint[7] = {-128, -128, -128, -128, -128, -128, 50};
    constexpr std::size_t kModelInputLastReader = 0; // the last layer that reads the input
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
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 4, 4, 8, 16> pw_Conv_type;
        constexpr pw_Conv_type pw_Conv_layer = {kpw_Conv_Weights, kpw_Conv_Biases, -128, pw_Conv_requantizers};
        static_assert(pw_Conv_type::OutputSize == 256, "pw_Conv: output size");

        // [6] Dense fc_Gemm <- pw_Conv
        constexpr tinymind::Requantizer<int32_t, int8_t> fc_Gemm_requantizers[10] = {
            {1438750700, 9, 50, -128, 127}, {1438107800, 9, 50, -128, 127}, {1612792698, 9, 50, -128, 127},
            {1551283652, 9, 50, -128, 127}, {1279732126, 8, 50, -128, 127}, {1685203612, 9, 50, -128, 127},
            {1417806567, 9, 50, -128, 127}, {1080739892, 8, 50, -128, 127}, {1555304819, 9, 50, -128, 127},
            {2086414848, 9, 50, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 256, 1, 1, 1, 1, 10> fc_Gemm_type;
        constexpr fc_Gemm_type fc_Gemm_layer = {kfc_Gemm_Weights, kfc_Gemm_Biases, -128, fc_Gemm_requantizers};
        static_assert(fc_Gemm_type::OutputSize == 10, "fc_Gemm: output size");
    } // namespace model_detail

    // A forward() / step() hook: before(layer) runs before layer `layer` (an
    // index into kModelLayerNames), after(layer, out, size) after it, with its
    // output (size values on the grid kModelLayerOutputScale[layer] /
    // kModelLayerOutputZeroPoint[layer]), valid until a later layer reuses its
    // bytes. Use it to time layers or to trace activations. ModelNoHook does
    // nothing and compiles away.
    struct ModelNoHook
    {
        void before(std::size_t) {}
        void after(std::size_t, const int8_t*, std::size_t) {}
    };

    /**
     * Run the model: input (kModelInputSize int8 values on the input grid)
     * to output (kModelOutputSize values on the output grid). arena
     * holds every intermediate activation; its contents need not be
     * kept between calls. hook (see ModelNoHook) is called around
     * every layer.
     */
    template<typename Hook>
    inline void forward(const int8_t* input, int8_t* output, int8_t* arena, Hook& hook)
    {
        hook.before(0);
        model_detail::c1_Conv_layer.forward(input, (arena + 0));
        hook.after(0, (arena + 0), 512);
        hook.before(1);
        model_detail::c2_Conv_layer.forward((arena + 0), (arena + 512));
        hook.after(1, (arena + 512), 512);
        hook.before(2);
        model_detail::Add_layer.forward((arena + 0), (arena + 512), (arena + 0));
        hook.after(2, (arena + 0), 512);
        hook.before(3);
        model_detail::MaxPool_layer.forward((arena + 0), (arena + 512));
        hook.after(3, (arena + 512), 128);
        hook.before(4);
        model_detail::dw_Conv_layer.forward((arena + 512), (arena + 256));
        hook.after(4, (arena + 256), 128);
        hook.before(5);
        model_detail::pw_Conv_layer.forward((arena + 256), (arena + 0));
        hook.after(5, (arena + 0), 256);
        hook.before(6);
        model_detail::fc_Gemm_layer.forward((arena + 0), output);
        hook.after(6, output, 10);
    }

    inline void forward(const int8_t* input, int8_t* output, int8_t* arena)
    {
        ModelNoHook hook;
        forward(input, output, arena, hook);
    }

    /**
     * Run one layer of forward(): calling it for layer 0, 1, ...,
     * kModelLayerCount - 1 in turn computes what forward() does, so the
     * work can be spread across ticks or between other tasks. arena must
     * be left alone between the calls, and input must stay valid until
     * layer kModelInputLastReader has run. A layer index out of range does
     * nothing.
     */
    inline void forwardLayer(std::size_t layer, const int8_t* input, int8_t* output, int8_t* arena)
    {
        (void)input;
        (void)output;
        (void)arena;
        switch (layer)
        {
        case 0:
        {
            model_detail::c1_Conv_layer.forward(input, (arena + 0));
            break;
        }
        case 1:
        {
            model_detail::c2_Conv_layer.forward((arena + 0), (arena + 512));
            break;
        }
        case 2:
        {
            model_detail::Add_layer.forward((arena + 0), (arena + 512), (arena + 0));
            break;
        }
        case 3:
        {
            model_detail::MaxPool_layer.forward((arena + 0), (arena + 512));
            break;
        }
        case 4:
        {
            model_detail::dw_Conv_layer.forward((arena + 512), (arena + 256));
            break;
        }
        case 5:
        {
            model_detail::pw_Conv_layer.forward((arena + 256), (arena + 0));
            break;
        }
        case 6:
        {
            model_detail::fc_Gemm_layer.forward((arena + 0), output);
            break;
        }
        default:
            break;
        }
    }
    constexpr std::size_t kModelWeightBytes = 3408;
} // namespace digits
