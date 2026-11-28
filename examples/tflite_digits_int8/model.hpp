/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import_tflite (model code generation).
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
#include "qsoftmax_fixedpoint.hpp"
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
    constexpr std::size_t kModelArenaSize      = 1536;
    constexpr std::size_t kModelArenaAlignment = 4;
    constexpr float   kModelInputScale      = 0.003921568859368563f;
    constexpr int32_t kModelInputZeroPoint  = -128;
    constexpr float   kModelOutputScale     = 0.00390625f;
    constexpr int32_t kModelOutputZeroPoint = -128;

    // Memory plan: 7 arena buffers in 1536 bytes (unshared: 2460; busiest layer: 1536).
    //   offset    bytes  live      outputs
    //        0      512    0-4    conv0, add3
    //        0      256    5-6    conv5
    //        0       12    7-8    fc7
    //      256       16    6-7    mean6
    //      512      512    1-2    dwconv1
    //      512      128    4-5    maxpool4
    //     1024      512    2-3    conv2
    constexpr std::size_t kModelUnsharedSize   = 2460;
    constexpr std::size_t kModelPeakLiveSize   = 1536;
    constexpr std::size_t kModelLayerCount     = 9;
    constexpr std::size_t kModelBufferCount    = 7;
    constexpr const char* kModelLayerNames[9] = {"conv0", "dwconv1", "conv2", "add3", "maxpool4", "conv5", "mean6", "fc7", "softmax8"};
    constexpr std::size_t kModelLayerOutputSize[9] = {512, 512, 512, 512, 128, 256, 16, 10, 10};
    constexpr float kModelLayerOutputScale[9] = {0.014159508049488068f, 0.023196693509817123f, 0.08557920157909393f, 0.08557920157909393f, 0.08557920157909393f, 0.21201619505882263f, 0.09773630648851395f, 0.2737049162387848f, 0.00390625f};
    constexpr int32_t kModelLayerOutputZeroPoint[9] = {-128, -128, -34, -34, -34, -128, -128, 13, -128};
    constexpr std::size_t kModelInputLastReader = 0; // the last layer that reads the input
    struct ModelBuffer
    {
        std::size_t offset;
        std::size_t size;
        std::size_t first_layer;
        std::size_t last_layer;
        const char* outputs; // the layers whose output lives here, in order
    };
    constexpr ModelBuffer kModelBuffers[7] = {
        {0, 512, 0, 4, "conv0 + add3"},
        {0, 256, 5, 6, "conv5"},
        {0, 12, 7, 8, "fc7"},
        {256, 16, 6, 7, "mean6"},
        {512, 512, 1, 2, "dwconv1"},
        {512, 128, 4, 5, "maxpool4"},
        {1024, 512, 2, 3, "conv2"}
    };

    namespace model_detail {

        // [0] Conv conv0 <- input
        constexpr tinymind::Requantizer<int32_t, int8_t> conv0_requantizers[8] = {
            {1214700166, 8, -128, -128, 127}, {1265423084, 8, -128, -128, 127}, {1762856591, 9, -128, -128, 127},
            {1765522606, 9, -128, -128, 127}, {1359664396, 9, -128, -128, 127}, {1767598017, 8, -128, -128, 127},
            {1165649943, 7, -128, -128, 127}, {1112633523, 8, -128, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 8, 8, 1, 3, 3, 1, 1, 8, 1, 1, 1, 1> conv0_type;
        constexpr conv0_type conv0_layer = {kconv0_Weights, kconv0_Biases, -128, conv0_requantizers};
        static_assert(conv0_type::OutputSize == 512, "conv0: output size");

        // [1] DepthwiseConv dwconv1 <- conv0
        constexpr tinymind::Requantizer<int32_t, int8_t> dwconv1_requantizers[8] = {
            {1448622497, 7, -128, -128, 127}, {1090238292, 7, -128, -128, 127}, {1156729100, 7, -128, -128, 127},
            {1334402071, 7, -128, -128, 127}, {1394592113, 7, -128, -128, 127}, {1084586573, 7, -128, -128, 127},
            {1399612286, 7, -128, -128, 127}, {1094086243, 7, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1> dwconv1_type;
        constexpr dwconv1_type dwconv1_layer = {kdwconv1_Weights, kdwconv1_Biases, -128, dwconv1_requantizers};
        static_assert(dwconv1_type::OutputSize == 512, "dwconv1: output size");

        // [2] Conv conv2 <- dwconv1
        constexpr tinymind::Requantizer<int32_t, int8_t> conv2_requantizers[8] = {
            {1358035575, 8, -34, -128, 127}, {2141152256, 9, -34, -128, 127}, {1183252615, 8, -34, -128, 127},
            {1086766168, 8, -34, -128, 127}, {1207090768, 8, -34, -128, 127}, {1287611929, 8, -34, -128, 127},
            {1255910762, 7, -34, -128, 127}, {1772635192, 8, -34, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 8, 8, 8, 8> conv2_type;
        constexpr conv2_type conv2_layer = {kconv2_Weights, kconv2_Biases, -128, conv2_requantizers};
        static_assert(conv2_type::OutputSize == 512, "conv2: output size");

        // [3] Add add3 <- conv0, conv2
        typedef tinymind::QAdd<int8_t, int8_t, int8_t, 512> add3_type;
        constexpr add3_type add3_layer = {-128, -34, 20, 1421247754, 3, 1073741824, 0, {1073741824, 18, -34, -128, 127}};

        // [4] MaxPool maxpool4 <- add3
        typedef tinymind::QMaxPool2D<int8_t, 8, 8, 8, 2, 2, 2, 2> maxpool4_type;
        constexpr maxpool4_type maxpool4_layer = {};
        static_assert(maxpool4_type::OutputSize == 128, "maxpool4: output size");

        // [5] Conv conv5 <- maxpool4
        constexpr tinymind::Requantizer<int32_t, int8_t> conv5_requantizers[16] = {
            {1758680411, 7, -128, -128, 127}, {1150144454, 7, -128, -128, 127}, {2003308751, 7, -128, -128, 127},
            {1382026871, 7, -128, -128, 127}, {1701851792, 7, -128, -128, 127}, {1774281425, 8, -128, -128, 127},
            {1230593144, 7, -128, -128, 127}, {1646901354, 7, -128, -128, 127}, {1723858524, 7, -128, -128, 127},
            {2139874715, 8, -128, -128, 127}, {1327753264, 7, -128, -128, 127}, {1961544227, 7, -128, -128, 127},
            {1998200586, 8, -128, -128, 127}, {2077564899, 8, -128, -128, 127}, {1954707921, 8, -128, -128, 127},
            {1157157874, 6, -128, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 4, 4, 8, 3, 3, 1, 1, 16, 1, 1, 1, 1> conv5_type;
        constexpr conv5_type conv5_layer = {kconv5_Weights, kconv5_Biases, -34, conv5_requantizers};
        static_assert(conv5_type::OutputSize == 256, "conv5: output size");

        // [6] Mean mean6 <- conv5
        constexpr tinymind::Requantizer<int32_t, int8_t> mean6_requantizer = {1164616631, 2, -128, -128, 127};

        // [7] Dense fc7 <- mean6
        constexpr tinymind::Requantizer<int32_t, int8_t> fc7_requantizers[10] = {
            {1740327967, 7, 13, -128, 127}, {1389643012, 7, 13, -128, 127}, {1764519251, 8, 13, -128, 127},
            {1948375172, 8, 13, -128, 127}, {1966336036, 8, 13, -128, 127}, {1295181509, 7, 13, -128, 127},
            {1300336168, 7, 13, -128, 127}, {1107593515, 7, 13, -128, 127}, {1390735045, 8, 13, -128, 127},
            {1590366757, 8, 13, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 16, 10> fc7_type;
        constexpr fc7_type fc7_layer = {kfc7_Weights, kfc7_Biases, -128, fc7_requantizers};
        static_assert(fc7_type::OutputSize == 10, "fc7: output size");

        // [8] Softmax softmax8 <- fc7
        typedef tinymind::QSoftmaxFixedPoint1D<1, 10> softmax8_type;
        constexpr softmax8_type softmax8_layer = {1175553664, 25, -62};
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
        model_detail::conv0_layer.forward(input, (arena + 0));
        hook.after(0, (arena + 0), 512);
        hook.before(1);
        model_detail::dwconv1_layer.forward((arena + 0), (arena + 512));
        hook.after(1, (arena + 512), 512);
        hook.before(2);
        model_detail::conv2_layer.forward((arena + 512), (arena + 1024));
        hook.after(2, (arena + 1024), 512);
        hook.before(3);
        model_detail::add3_layer.forward((arena + 0), (arena + 1024), (arena + 0));
        hook.after(3, (arena + 0), 512);
        hook.before(4);
        model_detail::maxpool4_layer.forward((arena + 0), (arena + 512));
        hook.after(4, (arena + 512), 128);
        hook.before(5);
        model_detail::conv5_layer.forward((arena + 512), (arena + 0));
        hook.after(5, (arena + 0), 256);
        hook.before(6);
        for (std::size_t c = 0; c < 16; ++c)
        {
            int32_t sum = 0;
            for (std::size_t j = 0; j < 16; ++j)
            {
                sum += static_cast<int32_t>((arena + 0)[j * 16 + c]) - (-128);
            }
            (arena + 256)[c] = model_detail::mean6_requantizer.apply(sum);
        }
        hook.after(6, (arena + 256), 16);
        hook.before(7);
        model_detail::fc7_layer.forward((arena + 256), (arena + 0));
        hook.after(7, (arena + 0), 10);
        hook.before(8);
        model_detail::softmax8_layer.forward((arena + 0), output);
        hook.after(8, output, 10);
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
            model_detail::conv0_layer.forward(input, (arena + 0));
            break;
        }
        case 1:
        {
            model_detail::dwconv1_layer.forward((arena + 0), (arena + 512));
            break;
        }
        case 2:
        {
            model_detail::conv2_layer.forward((arena + 512), (arena + 1024));
            break;
        }
        case 3:
        {
            model_detail::add3_layer.forward((arena + 0), (arena + 1024), (arena + 0));
            break;
        }
        case 4:
        {
            model_detail::maxpool4_layer.forward((arena + 0), (arena + 512));
            break;
        }
        case 5:
        {
            model_detail::conv5_layer.forward((arena + 512), (arena + 0));
            break;
        }
        case 6:
        {
            for (std::size_t c = 0; c < 16; ++c)
            {
                int32_t sum = 0;
                for (std::size_t j = 0; j < 16; ++j)
                {
                    sum += static_cast<int32_t>((arena + 0)[j * 16 + c]) - (-128);
                }
                (arena + 256)[c] = model_detail::mean6_requantizer.apply(sum);
            }
            break;
        }
        case 7:
        {
            model_detail::fc7_layer.forward((arena + 256), (arena + 0));
            break;
        }
        case 8:
        {
            model_detail::softmax8_layer.forward((arena + 0), output);
            break;
        }
        default:
            break;
        }
    }
    constexpr std::size_t kModelWeightBytes = 1520;
} // namespace digits
