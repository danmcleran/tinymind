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
#include "qsoftmax_fixedpoint.hpp"
#include "qpointwiseconv2d.hpp"
#include "weights.hpp"

#if !TINYMIND_ENABLE_QUANTIZATION
#error "a generated tinymind model needs TINYMIND_ENABLE_QUANTIZATION=1"
#endif

namespace vibration {
namespace stream {
    // Model interface. input and output are caller-owned int8_t buffers in
    // NHWC order; arena is caller-owned scratch of kModelArenaSize bytes,
    // aligned to kModelArenaAlignment, overlapping neither of them.
    typedef int8_t ModelActivationType;
    constexpr std::size_t kModelInputSize      = 16;
    constexpr std::size_t kModelOutputSize     = 2;
    constexpr std::size_t kModelArenaSize      = 24;
    constexpr std::size_t kModelArenaAlignment = 4;
    constexpr float   kModelInputScale      = 0.02941484935581684f;
    constexpr int32_t kModelInputZeroPoint  = -8;
    constexpr float   kModelOutputScale     = 0.00390625f;
    constexpr int32_t kModelOutputZeroPoint = -128;

    // Memory plan: 5 arena buffers in 24 bytes (unshared: 60; busiest layer: 24).
    //   offset    bytes  live      outputs
    //        0       16    1-2    dwconv1
    //        0       16    3-4    dwconv3
    //        0       16    5-6    dwconv5
    //        0        4    7-8    fc11
    //       16        8    6-7    conv6
    constexpr std::size_t kModelUnsharedSize   = 60;
    constexpr std::size_t kModelPeakLiveSize   = 24;
    constexpr std::size_t kModelLayerCount     = 9;
    constexpr std::size_t kModelBufferCount    = 5;
    constexpr const char* kModelLayerNames[9] = {"conv0", "dwconv1", "conv2", "dwconv3", "conv4", "dwconv5", "conv6", "fc11", "softmax12"};
    constexpr std::size_t kModelLayerOutputSize[9] = {16, 16, 16, 16, 16, 16, 8, 2, 2};
    constexpr float kModelLayerOutputScale[9] = {0.05681701377034187f, 0.0440039224922657f, 0.06782463937997818f, 0.11491405218839645f, 0.15498799085617065f, 0.5670965909957886f, 0.48135146498680115f, 1.1422832012176514f, 0.00390625f};
    constexpr int32_t kModelLayerOutputZeroPoint[9] = {-128, -128, -128, -128, -128, -43, -128, -26, -128};
    constexpr std::size_t kModelInputLastReader = 0; // the last layer that reads the input
    // Streaming: each step() takes one input frame; the time axis is H of the
    // whole-window model, whose input spans 32 frames.
    constexpr std::size_t kModelWindowFrames     = 32;
    constexpr std::size_t kModelReceptiveFrames  = 32;
    constexpr std::size_t kModelStateSize        = 560;
    constexpr std::size_t kModelStateAlignment   = 4;
    // State layout: one history per tensor a layer reads several frames of.
    //        0    3 frames x    16  input
    //       48    5 frames x    16  conv0
    //      128    9 frames x    16  conv2
    //      272   18 frames x    16  conv4
    struct ModelBuffer
    {
        std::size_t offset;
        std::size_t size;
        std::size_t first_layer;
        std::size_t last_layer;
        const char* outputs; // the layers whose output lives here, in order
    };
    constexpr ModelBuffer kModelBuffers[5] = {
        {0, 16, 1, 2, "dwconv1"},
        {0, 16, 3, 4, "dwconv3"},
        {0, 16, 5, 6, "dwconv5"},
        {0, 4, 7, 8, "fc11"},
        {16, 8, 6, 7, "conv6"}
    };

    namespace model_detail {

        // [0] Conv conv0 <- input
        constexpr tinymind::Requantizer<int32_t, int8_t> conv0_requantizers[16] = {
            {1750460948, 8, -128, -128, 127}, {1152866311, 9, -128, -128, 127}, {1243080930, 7, -128, -128, 127},
            {1216086880, 8, -128, -128, 127}, {1154110675, 9, -128, -128, 127}, {1568539638, 8, -128, -128, 127},
            {1261174774, 8, -128, -128, 127}, {2076053250, 8, -128, -128, 127}, {1248460364, 7, -128, -128, 127},
            {2145850965, 10, -128, -128, 127}, {1104064144, 9, -128, -128, 127}, {1385052170, 9, -128, -128, 127},
            {1500100531, 8, -128, -128, 127}, {1526696978, 8, -128, -128, 127}, {1695253002, 8, -128, -128, 127},
            {1210599696, 8, -128, -128, 127}
        };
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 3, 1, 16, 3, 1, 1, 1, 16, 0, 0, 0, 0> conv0_type;
        constexpr conv0_type conv0_layer = {kconv0_Weights, kconv0_Biases, -8, conv0_requantizers};
        static_assert(conv0_type::OutputSize == 16, "conv0: output size");

        // [1] DepthwiseConv dwconv1 <- conv0
        constexpr tinymind::Requantizer<int32_t, int8_t> dwconv1_requantizers[16] = {
            {1373987197, 7, -128, -128, 127}, {1088341593, 8, -128, -128, 127}, {1396738833, 6, -128, -128, 127},
            {1400913417, 8, -128, -128, 127}, {1247081285, 8, -128, -128, 127}, {1299877394, 7, -128, -128, 127},
            {1820219978, 7, -128, -128, 127}, {1356958817, 6, -128, -128, 127}, {1811202785, 7, -128, -128, 127},
            {1283756518, 8, -128, -128, 127}, {1295321117, 8, -128, -128, 127}, {2095798340, 9, -128, -128, 127},
            {1603278654, 7, -128, -128, 127}, {1232616922, 7, -128, -128, 127}, {2099244243, 7, -128, -128, 127},
            {1940470587, 8, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 5, 1, 16, 5, 1, 1, 1, 0, 0, 0, 0> dwconv1_type;
        constexpr dwconv1_type dwconv1_layer = {kdwconv1_Weights, kdwconv1_Biases, -128, dwconv1_requantizers};
        static_assert(dwconv1_type::OutputSize == 16, "dwconv1: output size");

        // [2] Conv conv2 <- dwconv1
        constexpr tinymind::Requantizer<int32_t, int8_t> conv2_requantizers[16] = {
            {2021384990, 8, -128, -128, 127}, {1227357255, 7, -128, -128, 127}, {1992615187, 7, -128, -128, 127},
            {1569019049, 7, -128, -128, 127}, {2110252352, 8, -128, -128, 127}, {1968900836, 7, -128, -128, 127},
            {1797967669, 7, -128, -128, 127}, {1460943029, 8, -128, -128, 127}, {1504147062, 6, -128, -128, 127},
            {1235387049, 8, -128, -128, 127}, {1113960703, 8, -128, -128, 127}, {1239969227, 6, -128, -128, 127},
            {1993414579, 8, -128, -128, 127}, {1212917211, 8, -128, -128, 127}, {1769521909, 7, -128, -128, 127},
            {1649540547, 7, -128, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 16, 16> conv2_type;
        constexpr conv2_type conv2_layer = {kconv2_Weights, kconv2_Biases, -128, conv2_requantizers};
        static_assert(conv2_type::OutputSize == 16, "conv2: output size");

        // [3] DepthwiseConv dwconv3 <- conv2
        constexpr tinymind::Requantizer<int32_t, int8_t> dwconv3_requantizers[16] = {
            {1768719115, 8, -128, -128, 127}, {1474605453, 8, -128, -128, 127}, {2131102241, 8, -128, -128, 127},
            {1428453188, 7, -128, -128, 127}, {1377525669, 8, -128, -128, 127}, {1692846627, 7, -128, -128, 127},
            {1397057910, 8, -128, -128, 127}, {1103627658, 9, -128, -128, 127}, {1081960726, 8, -128, -128, 127},
            {1759342370, 10, -128, -128, 127}, {1931990788, 10, -128, -128, 127}, {1215749322, 8, -128, -128, 127},
            {1517884318, 8, -128, -128, 127}, {1374533959, 10, -128, -128, 127}, {1387484588, 8, -128, -128, 127},
            {1463687071, 8, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 9, 1, 16, 9, 1, 1, 1, 0, 0, 0, 0> dwconv3_type;
        constexpr dwconv3_type dwconv3_layer = {kdwconv3_Weights, kdwconv3_Biases, -128, dwconv3_requantizers};
        static_assert(dwconv3_type::OutputSize == 16, "dwconv3: output size");

        // [4] Conv conv4 <- dwconv3
        constexpr tinymind::Requantizer<int32_t, int8_t> conv4_requantizers[16] = {
            {1335616506, 7, -128, -128, 127}, {1905789917, 7, -128, -128, 127}, {1761306368, 7, -128, -128, 127},
            {1630563987, 8, -128, -128, 127}, {1890041148, 8, -128, -128, 127}, {2098635845, 7, -128, -128, 127},
            {1442716279, 7, -128, -128, 127}, {1560368630, 8, -128, -128, 127}, {1405567105, 6, -128, -128, 127},
            {1527427126, 8, -128, -128, 127}, {1838116827, 7, -128, -128, 127}, {1993242354, 7, -128, -128, 127},
            {1489197279, 7, -128, -128, 127}, {1124021795, 7, -128, -128, 127}, {1628384609, 8, -128, -128, 127},
            {1725698912, 8, -128, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 16, 16> conv4_type;
        constexpr conv4_type conv4_layer = {kconv4_Weights, kconv4_Biases, -128, conv4_requantizers};
        static_assert(conv4_type::OutputSize == 16, "conv4: output size");

        // [5] DepthwiseConv dwconv5 <- conv4
        constexpr tinymind::Requantizer<int32_t, int8_t> dwconv5_requantizers[16] = {
            {1171781820, 9, -43, -128, 127}, {1496611438, 10, -43, -128, 127}, {1512193068, 9, -43, -128, 127},
            {1530035548, 11, -43, -128, 127}, {1274150373, 9, -43, -128, 127}, {1156752774, 9, -43, -128, 127},
            {1770604121, 9, -43, -128, 127}, {1517248461, 11, -43, -128, 127}, {2033137512, 9, -43, -128, 127},
            {1412825525, 10, -43, -128, 127}, {1893548477, 9, -43, -128, 127}, {1398144205, 8, -43, -128, 127},
            {1135651253, 8, -43, -128, 127}, {1284628615, 8, -43, -128, 127}, {1390904504, 9, -43, -128, 127},
            {2079142733, 10, -43, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 18, 1, 16, 18, 1, 1, 1, 0, 0, 0, 0> dwconv5_type;
        constexpr dwconv5_type dwconv5_layer = {kdwconv5_Weights, kdwconv5_Biases, -128, dwconv5_requantizers};
        static_assert(dwconv5_type::OutputSize == 16, "dwconv5: output size");

        // [6] Conv conv6 <- dwconv5
        constexpr tinymind::Requantizer<int32_t, int8_t> conv6_requantizers[8] = {
            {1249327591, 7, -128, -128, 127}, {1526504589, 7, -128, -128, 127}, {1676328868, 7, -128, -128, 127},
            {2115566076, 7, -128, -128, 127}, {1816394200, 7, -128, -128, 127}, {1736936461, 7, -128, -128, 127},
            {1369066811, 6, -128, -128, 127}, {1333997547, 6, -128, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 16, 8> conv6_type;
        constexpr conv6_type conv6_layer = {kconv6_Weights, kconv6_Biases, -43, conv6_requantizers};
        static_assert(conv6_type::OutputSize == 8, "conv6: output size");

        // [7] Dense fc11 <- conv6
        constexpr tinymind::Requantizer<int32_t, int8_t> fc11_requantizers[2] = {
            {1453370616, 8, -26, -128, 127}, {1680616766, 8, -26, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 8, 2> fc11_type;
        constexpr fc11_type fc11_layer = {kfc11_Weights, kfc11_Biases, -128, fc11_requantizers};
        static_assert(fc11_type::OutputSize == 2, "fc11: output size");

        // [8] Softmax softmax12 <- fc11
        typedef tinymind::QSoftmaxFixedPoint1D<1, 2> softmax12_type;
        constexpr softmax12_type softmax12_layer = {1226517248, 27, -15};
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
     * Clear the state: every history frame becomes the real value 0 (its
     * tensor's zero point). Call once before the first step().
     */
    inline void reset(int8_t* state)
    {
        for (std::size_t b = 0; b < 48; ++b)
        {
            state[0 + b] = -8;
        }
        for (std::size_t b = 0; b < 80; ++b)
        {
            state[48 + b] = -128;
        }
        for (std::size_t b = 0; b < 144; ++b)
        {
            state[128 + b] = -128;
        }
        for (std::size_t b = 0; b < 288; ++b)
        {
            state[272 + b] = -128;
        }
    }

    /**
     * Feed one input frame (kModelInputSize int8 values on the input grid)
     * and compute the output for the window of kModelReceptiveFrames frames
     * that ends with it (kModelOutputSize values on the output grid): the
     * whole-window model's output for that window, bit for bit, from the
     * kModelReceptiveFrames-th frame after reset() on. state
     * (kModelStateSize bytes, aligned to kModelStateAlignment) carries the
     * history between calls; arena is scratch, as for forward(). hook (see
     * ModelNoHook) is called around every layer.
     */
    template<typename Hook>
    inline void step(const int8_t* frame, int8_t* output, int8_t* state, int8_t* arena, Hook& hook)
    {
        // Slide the input history one frame and append the new frame.
        for (std::size_t b = 0; b < 32; ++b)
        {
            state[0 + b] = state[16 + b];
        }
        for (std::size_t b = 0; b < 16; ++b)
        {
            state[32 + b] = frame[b];
        }
        hook.before(0);
        // Slide conv0's history one frame.
        for (std::size_t b = 0; b < 64; ++b)
        {
            state[48 + b] = state[64 + b];
        }
        model_detail::conv0_layer.forward((state + 0), (state + 112));
        hook.after(0, (state + 112), 16);
        hook.before(1);
        model_detail::dwconv1_layer.forward((state + 48), (arena + 0));
        hook.after(1, (arena + 0), 16);
        hook.before(2);
        // Slide conv2's history one frame.
        for (std::size_t b = 0; b < 128; ++b)
        {
            state[128 + b] = state[144 + b];
        }
        model_detail::conv2_layer.forward((arena + 0), (state + 256));
        hook.after(2, (state + 256), 16);
        hook.before(3);
        model_detail::dwconv3_layer.forward((state + 128), (arena + 0));
        hook.after(3, (arena + 0), 16);
        hook.before(4);
        // Slide conv4's history one frame.
        for (std::size_t b = 0; b < 272; ++b)
        {
            state[272 + b] = state[288 + b];
        }
        model_detail::conv4_layer.forward((arena + 0), (state + 544));
        hook.after(4, (state + 544), 16);
        hook.before(5);
        model_detail::dwconv5_layer.forward((state + 272), (arena + 0));
        hook.after(5, (arena + 0), 16);
        hook.before(6);
        model_detail::conv6_layer.forward((arena + 0), (arena + 16));
        hook.after(6, (arena + 16), 8);
        hook.before(7);
        model_detail::fc11_layer.forward((arena + 16), (arena + 0));
        hook.after(7, (arena + 0), 2);
        hook.before(8);
        model_detail::softmax12_layer.forward((arena + 0), output);
        hook.after(8, output, 2);
    }

    inline void step(const int8_t* frame, int8_t* output, int8_t* state, int8_t* arena)
    {
        ModelNoHook hook;
        step(frame, output, state, arena, hook);
    }
    constexpr std::size_t kModelWeightBytes = 1936;
} // namespace stream
} // namespace vibration
