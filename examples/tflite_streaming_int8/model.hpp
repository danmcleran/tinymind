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
    // Model interface. input and output are caller-owned int8_t buffers in
    // NHWC order; arena is caller-owned scratch of kModelArenaSize bytes,
    // aligned to kModelArenaAlignment, overlapping neither of them.
    typedef int8_t ModelActivationType;
    constexpr std::size_t kModelInputSize      = 512;
    constexpr std::size_t kModelOutputSize     = 2;
    constexpr std::size_t kModelArenaSize      = 896;
    constexpr std::size_t kModelArenaAlignment = 4;
    constexpr float   kModelInputScale      = 0.02941484935581684f;
    constexpr int32_t kModelInputZeroPoint  = -8;
    constexpr float   kModelOutputScale     = 0.00390625f;
    constexpr int32_t kModelOutputZeroPoint = -128;

    // Memory plan: 8 arena buffers in 896 bytes (unshared: 1916; busiest layer: 896).
    //   offset    bytes  live      outputs
    //        0      480    0-1    conv0
    //        0      416    2-3    conv2
    //        0      288    4-5    conv4
    //        0        8    6-7    conv6
    //        8        4    7-8    fc11
    //      288       16    5-6    dwconv5
    //      416      288    3-4    dwconv3
    //      480      416    1-2    dwconv1
    constexpr std::size_t kModelUnsharedSize   = 1916;
    constexpr std::size_t kModelPeakLiveSize   = 896;
    constexpr std::size_t kModelLayerCount     = 9;
    constexpr std::size_t kModelBufferCount    = 8;
    constexpr const char* kModelLayerNames[9] = {"conv0", "dwconv1", "conv2", "dwconv3", "conv4", "dwconv5", "conv6", "fc11", "softmax12"};
    constexpr std::size_t kModelLayerOutputSize[9] = {480, 416, 416, 288, 288, 16, 8, 2, 2};
    constexpr float kModelLayerOutputScale[9] = {0.05681701377034187f, 0.0440039224922657f, 0.06782463937997818f, 0.11491405218839645f, 0.15498799085617065f, 0.5670965909957886f, 0.48135146498680115f, 1.1422832012176514f, 0.00390625f};
    constexpr int32_t kModelLayerOutputZeroPoint[9] = {-128, -128, -128, -128, -128, -43, -128, -26, -128};
    constexpr std::size_t kModelInputLastReader = 0; // the last layer that reads the input
    struct ModelBuffer
    {
        std::size_t offset;
        std::size_t size;
        std::size_t first_layer;
        std::size_t last_layer;
        const char* outputs; // the layers whose output lives here, in order
    };
    constexpr ModelBuffer kModelBuffers[8] = {
        {0, 480, 0, 1, "conv0"},
        {0, 416, 2, 3, "conv2"},
        {0, 288, 4, 5, "conv4"},
        {0, 8, 6, 7, "conv6"},
        {8, 4, 7, 8, "fc11"},
        {288, 16, 5, 6, "dwconv5"},
        {416, 288, 3, 4, "dwconv3"},
        {480, 416, 1, 2, "dwconv1"}
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
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 32, 1, 16, 3, 1, 1, 1, 16, 0, 0, 0, 0> conv0_type;
        constexpr conv0_type conv0_layer = {kconv0_Weights, kconv0_Biases, -8, conv0_requantizers};
        static_assert(conv0_type::OutputSize == 480, "conv0: output size");

        // [1] DepthwiseConv dwconv1 <- conv0
        constexpr tinymind::Requantizer<int32_t, int8_t> dwconv1_requantizers[16] = {
            {1373987197, 7, -128, -128, 127}, {1088341593, 8, -128, -128, 127}, {1396738833, 6, -128, -128, 127},
            {1400913417, 8, -128, -128, 127}, {1247081285, 8, -128, -128, 127}, {1299877394, 7, -128, -128, 127},
            {1820219978, 7, -128, -128, 127}, {1356958817, 6, -128, -128, 127}, {1811202785, 7, -128, -128, 127},
            {1283756518, 8, -128, -128, 127}, {1295321117, 8, -128, -128, 127}, {2095798340, 9, -128, -128, 127},
            {1603278654, 7, -128, -128, 127}, {1232616922, 7, -128, -128, 127}, {2099244243, 7, -128, -128, 127},
            {1940470587, 8, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 30, 1, 16, 5, 1, 1, 1, 0, 0, 0, 0> dwconv1_type;
        constexpr dwconv1_type dwconv1_layer = {kdwconv1_Weights, kdwconv1_Biases, -128, dwconv1_requantizers};
        static_assert(dwconv1_type::OutputSize == 416, "dwconv1: output size");

        // [2] Conv conv2 <- dwconv1
        constexpr tinymind::Requantizer<int32_t, int8_t> conv2_requantizers[16] = {
            {2021384990, 8, -128, -128, 127}, {1227357255, 7, -128, -128, 127}, {1992615187, 7, -128, -128, 127},
            {1569019049, 7, -128, -128, 127}, {2110252352, 8, -128, -128, 127}, {1968900836, 7, -128, -128, 127},
            {1797967669, 7, -128, -128, 127}, {1460943029, 8, -128, -128, 127}, {1504147062, 6, -128, -128, 127},
            {1235387049, 8, -128, -128, 127}, {1113960703, 8, -128, -128, 127}, {1239969227, 6, -128, -128, 127},
            {1993414579, 8, -128, -128, 127}, {1212917211, 8, -128, -128, 127}, {1769521909, 7, -128, -128, 127},
            {1649540547, 7, -128, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 26, 1, 16, 16> conv2_type;
        constexpr conv2_type conv2_layer = {kconv2_Weights, kconv2_Biases, -128, conv2_requantizers};
        static_assert(conv2_type::OutputSize == 416, "conv2: output size");

        // [3] DepthwiseConv dwconv3 <- conv2
        constexpr tinymind::Requantizer<int32_t, int8_t> dwconv3_requantizers[16] = {
            {1768719115, 8, -128, -128, 127}, {1474605453, 8, -128, -128, 127}, {2131102241, 8, -128, -128, 127},
            {1428453188, 7, -128, -128, 127}, {1377525669, 8, -128, -128, 127}, {1692846627, 7, -128, -128, 127},
            {1397057910, 8, -128, -128, 127}, {1103627658, 9, -128, -128, 127}, {1081960726, 8, -128, -128, 127},
            {1759342370, 10, -128, -128, 127}, {1931990788, 10, -128, -128, 127}, {1215749322, 8, -128, -128, 127},
            {1517884318, 8, -128, -128, 127}, {1374533959, 10, -128, -128, 127}, {1387484588, 8, -128, -128, 127},
            {1463687071, 8, -128, -128, 127}
        };
        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 26, 1, 16, 9, 1, 1, 1, 0, 0, 0, 0> dwconv3_type;
        constexpr dwconv3_type dwconv3_layer = {kdwconv3_Weights, kdwconv3_Biases, -128, dwconv3_requantizers};
        static_assert(dwconv3_type::OutputSize == 288, "dwconv3: output size");

        // [4] Conv conv4 <- dwconv3
        constexpr tinymind::Requantizer<int32_t, int8_t> conv4_requantizers[16] = {
            {1335616506, 7, -128, -128, 127}, {1905789917, 7, -128, -128, 127}, {1761306368, 7, -128, -128, 127},
            {1630563987, 8, -128, -128, 127}, {1890041148, 8, -128, -128, 127}, {2098635845, 7, -128, -128, 127},
            {1442716279, 7, -128, -128, 127}, {1560368630, 8, -128, -128, 127}, {1405567105, 6, -128, -128, 127},
            {1527427126, 8, -128, -128, 127}, {1838116827, 7, -128, -128, 127}, {1993242354, 7, -128, -128, 127},
            {1489197279, 7, -128, -128, 127}, {1124021795, 7, -128, -128, 127}, {1628384609, 8, -128, -128, 127},
            {1725698912, 8, -128, -128, 127}
        };
        typedef tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 18, 1, 16, 16> conv4_type;
        constexpr conv4_type conv4_layer = {kconv4_Weights, kconv4_Biases, -128, conv4_requantizers};
        static_assert(conv4_type::OutputSize == 288, "conv4: output size");

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
        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, 8, 1, 1, 1, 1, 2> fc11_type;
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
        hook.after(0, (arena + 0), 480);
        hook.before(1);
        model_detail::dwconv1_layer.forward((arena + 0), (arena + 480));
        hook.after(1, (arena + 480), 416);
        hook.before(2);
        model_detail::conv2_layer.forward((arena + 480), (arena + 0));
        hook.after(2, (arena + 0), 416);
        hook.before(3);
        model_detail::dwconv3_layer.forward((arena + 0), (arena + 416));
        hook.after(3, (arena + 416), 288);
        hook.before(4);
        model_detail::conv4_layer.forward((arena + 416), (arena + 0));
        hook.after(4, (arena + 0), 288);
        hook.before(5);
        model_detail::dwconv5_layer.forward((arena + 0), (arena + 288));
        hook.after(5, (arena + 288), 16);
        hook.before(6);
        model_detail::conv6_layer.forward((arena + 288), (arena + 0));
        hook.after(6, (arena + 0), 8);
        hook.before(7);
        model_detail::fc11_layer.forward((arena + 0), (arena + 8));
        hook.after(7, (arena + 8), 2);
        hook.before(8);
        model_detail::softmax12_layer.forward((arena + 8), output);
        hook.after(8, output, 2);
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
            model_detail::dwconv1_layer.forward((arena + 0), (arena + 480));
            break;
        }
        case 2:
        {
            model_detail::conv2_layer.forward((arena + 480), (arena + 0));
            break;
        }
        case 3:
        {
            model_detail::dwconv3_layer.forward((arena + 0), (arena + 416));
            break;
        }
        case 4:
        {
            model_detail::conv4_layer.forward((arena + 416), (arena + 0));
            break;
        }
        case 5:
        {
            model_detail::dwconv5_layer.forward((arena + 0), (arena + 288));
            break;
        }
        case 6:
        {
            model_detail::conv6_layer.forward((arena + 288), (arena + 0));
            break;
        }
        case 7:
        {
            model_detail::fc11_layer.forward((arena + 0), (arena + 8));
            break;
        }
        case 8:
        {
            model_detail::softmax12_layer.forward((arena + 8), output);
            break;
        }
        default:
            break;
        }
    }
    constexpr std::size_t kModelWeightBytes = 1936;
} // namespace vibration
