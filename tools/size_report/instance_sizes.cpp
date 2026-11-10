/**
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// Prints the instance sizes quoted in docs/size-comparison.md, one
// "key<TAB>bytes" line per configuration. Driven by
// tools/size_report/size_report.py, which formats the tables; run that rather
// than this file directly. Only sizeof() is taken, so nothing is executed
// beyond the prints. The activation policy does not affect object layout, so
// the linear policy stands in for every value type (the LUT policies need a
// QValue).

#include <cstdio>

#include "qformat.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "kan.hpp"
#include "kanTransferFunctions.hpp"
#include "conv1d.hpp"
#include "pool1d.hpp"
#include "dropout.hpp"
#include "conv2d.hpp"
#include "depthwiseconv2d.hpp"
#include "pointwiseconv2d.hpp"
#include "pool2d.hpp"
#include "binarylayer.hpp"
#include "ternarylayer.hpp"
#include "ltc.hpp"
#include "cfc.hpp"

namespace
{
    template<typename ValueType>
    struct ZeroRandom
    {
        static ValueType generateRandomWeight()
        {
            return ValueType(0);
        }
    };

    template<typename ValueType>
    struct Types
    {
        typedef tinymind::FixedPointTransferFunctions<
            ValueType,
            ZeroRandom<ValueType>,
            tinymind::LinearActivationPolicy<ValueType>,
            tinymind::LinearActivationPolicy<ValueType>> TF;

        typedef tinymind::KanTransferFunctions<ValueType, ZeroRandom<ValueType>, 1> KanTF;

        template<size_t Hidden, bool Trainable>
        struct Mlp
        {
            typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<Hidden>, 1, TF, Trainable> type;
        };

        template<size_t Hidden, bool Trainable>
        struct Elman
        {
            typedef tinymind::ElmanNeuralNetwork<ValueType, 2, Hidden, 1, TF, Trainable> type;
        };

        template<size_t Hidden, bool Trainable>
        struct Lstm
        {
            typedef tinymind::LstmNeuralNetwork<ValueType, 2, tinymind::HiddenLayers<Hidden>, 1, TF, Trainable> type;
        };

        template<size_t Hidden, bool Trainable>
        struct Gru
        {
            typedef tinymind::GruNeuralNetwork<ValueType, 2, tinymind::HiddenLayers<Hidden>, 1, TF, Trainable> type;
        };

        // KAN 2->5->1, GridSize 5, SplineDegree 1 (the kan_xor shape).
        template<bool Trainable>
        struct Kan
        {
            typedef tinymind::KolmogorovArnoldNetwork<ValueType, 2, 1, 5, 1, KanTF, Trainable, 1, 5, 1> type;
        };
    };

    void row(const char* key, size_t bytes)
    {
        std::printf("%s\t%zu\n", key, bytes);
    }

    template<typename T>
    void networks(const char* prefix)
    {
        char key[96];
#define TM_ROW(name, ...) std::snprintf(key, sizeof(key), "%s.%s", prefix, name); row(key, sizeof(__VA_ARGS__))
        TM_ROW("mlp3.train", typename Types<T>::template Mlp<3, true>::type);
        TM_ROW("mlp3.infer", typename Types<T>::template Mlp<3, false>::type);
        TM_ROW("mlp5.train", typename Types<T>::template Mlp<5, true>::type);
        TM_ROW("mlp5.infer", typename Types<T>::template Mlp<5, false>::type);
        TM_ROW("elman3.train", typename Types<T>::template Elman<3, true>::type);
        TM_ROW("elman3.infer", typename Types<T>::template Elman<3, false>::type);
        TM_ROW("lstm3.train", typename Types<T>::template Lstm<3, true>::type);
        TM_ROW("lstm3.infer", typename Types<T>::template Lstm<3, false>::type);
        TM_ROW("gru3.train", typename Types<T>::template Gru<3, true>::type);
        TM_ROW("gru3.infer", typename Types<T>::template Gru<3, false>::type);
        TM_ROW("kan5.train", typename Types<T>::template Kan<true>::type);
        TM_ROW("kan5.infer", typename Types<T>::template Kan<false>::type);
#undef TM_ROW
    }
}

int main()
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> Q88;

    networks<double>("double");
    networks<Q88>("q88");

    // Liquid cells: parameter counts (bytes = count x sizeof(ValueType)).
    row("ltc_2_3.params", tinymind::ltc::LtcCell<2, 3>::NumParams);
    row("cfc_2_3_4.params", tinymind::cfc::CfCCell<2, 3, 4>::NumParams);
    row("ltc_4_8.params", tinymind::ltc::LtcCell<4, 8>::NumParams);
    row("cfc_4_8_16.params", tinymind::cfc::CfCCell<4, 8, 16>::NumParams);

    // 1D signal-processing layers.
    row("conv1d_100_5_2_8.double", sizeof(tinymind::Conv1D<double, 100, 5, 2, 8>));
    row("conv1d_100_5_2_8.q88", sizeof(tinymind::Conv1D<Q88, 100, 5, 2, 8>));
    row("conv1d_100_5_1_4.double", sizeof(tinymind::Conv1D<double, 100, 5, 1, 4>));
    row("conv1d_100_5_1_4.q88", sizeof(tinymind::Conv1D<Q88, 100, 5, 1, 4>));
    row("maxpool1d_96_2_2_4", sizeof(tinymind::MaxPool1D<double, 96, 2, 2, 4>));
    row("maxpool1d_48_2_2_1", sizeof(tinymind::MaxPool1D<double, 48, 2, 2, 1>));
    row("maxpool1d_6_2_2_1", sizeof(tinymind::MaxPool1D<double, 6, 2, 2, 1>));
    row("maxpool1d_96_2_2_4.q88", sizeof(tinymind::MaxPool1D<Q88, 96, 2, 2, 4>));
    row("avgpool1d_96_2_2_4", sizeof(tinymind::AvgPool1D<double, 96, 2, 2, 4>));
    row("dropout_192_50", sizeof(tinymind::Dropout<double, 192, 50>));
    row("dropout_32_50", sizeof(tinymind::Dropout<double, 32, 50>));
    row("dropout_5_50", sizeof(tinymind::Dropout<double, 5, 50>));
    row("dropout_192_50.q88", sizeof(tinymind::Dropout<Q88, 192, 50>));

    // 2D layers in float (the kws_cortex_m value type).
    row("conv2d_20_20_1_3_3_1_1_8", sizeof(tinymind::Conv2D<float, 20, 20, 1, 3, 3, 1, 1, 8>));
    row("dwconv2d_9_9_8_3_3", sizeof(tinymind::DepthwiseConv2D<float, 9, 9, 8, 3, 3>));
    row("pwconv2d_7_7_8_16", sizeof(tinymind::PointwiseConv2D<float, 7, 7, 8, 16>));
    row("pwconv2d_1_1_16_10", sizeof(tinymind::PointwiseConv2D<float, 1, 1, 16, 10>));
    row("maxpool2d_18_18_8_2_2", sizeof(tinymind::MaxPool2D<float, 18, 18, 8, 2, 2>));
    row("gap2d_7_7_16", sizeof(tinymind::GlobalAvgPool2D<float, 7, 7, 16>));

    // Binary / ternary dense layers.
    row("binary_4_2.double", sizeof(tinymind::BinaryDense<double, 4, 2>));
    row("binary_4_2.q88", sizeof(tinymind::BinaryDense<Q88, 4, 2>));
    row("binary_16_8.double", sizeof(tinymind::BinaryDense<double, 16, 8>));
    row("binary_16_8.q88", sizeof(tinymind::BinaryDense<Q88, 16, 8>));
    row("binary_64_16.double", sizeof(tinymind::BinaryDense<double, 64, 16>));
    row("binary_64_16.q88", sizeof(tinymind::BinaryDense<Q88, 64, 16>));
    row("ternary_4_2.double", sizeof(tinymind::TernaryDense<double, 4, 2, 50>));
    row("ternary_4_2.q88", sizeof(tinymind::TernaryDense<Q88, 4, 2, 50>));
    row("ternary_16_8.double", sizeof(tinymind::TernaryDense<double, 16, 8, 50>));
    row("ternary_16_8.q88", sizeof(tinymind::TernaryDense<Q88, 16, 8, 50>));
    row("ternary_64_16.double", sizeof(tinymind::TernaryDense<double, 64, 16, 50>));
    row("ternary_64_16.q88", sizeof(tinymind::TernaryDense<Q88, 64, 16, 50>));

    return 0;
}
