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

// A fixed-point SGD step must round learning rate x gradient to nearest.
//
// Rounded with the default TruncatePolicy (a floor), a step smaller than one
// LSB is 0 when the gradient is positive and a full -1 LSB when it is
// negative, so every weight with a small gradient drifts down by about half
// an LSB per training step. In Q8.8 that swamps the gradient: the xor
// example built in Q8.8 trained on 25 of 100 seeds with it and on 100 of 100
// without. Rounded toward zero instead, every step below one LSB is lost, and
// recurrent Q16.16 training with a small learning rate stalls. To nearest,
// ties away from zero, the step is symmetric and only steps below half an LSB
// are lost.
//
// The layer cases drive exact gradients of each sign through the layer's own
// backward pass and check the move against the rounded step; the cases are
// chosen so that floor and toward-zero rounding each fail some of them. The
// attention, LSTM, GRU (and KAN, in unit_test/kan) cases train with no
// momentum and require every parameter to move by exactly its gradient step
// rounded to nearest. The network case trains the xor example's network in
// Q8.8.
//
// Included from nn_unit_test.cpp.

namespace sgd_step_rounding_tests
{
    // Q8.8 with the library's default policies (TruncatePolicy).
    typedef tinymind::QValue<8, 8, true> Q8;
    typedef Q8::FullWidthFieldType RawType;

    static Q8 raw(const int value)
    {
        return Q8(static_cast<RawType>(value));
    }

    // 0.25: one quarter of a one-LSB gradient is a quarter-LSB step.
    static Q8 learningRate()
    {
        return Q8(0, 64);
    }

    static Q8 one()
    {
        return Q8(1, 0);
    }

    // Steps a fresh layer once with output delta `deltaLsbs` (input 1.0, so
    // the gradient is the delta) and returns how far parameter `index` moved,
    // in LSBs. Each layer exposes a slightly different API, hence the policy.
    template<typename Policy>
    static int stepMoves(const int deltaLsbs, const size_t index)
    {
        typename Policy::LayerType layer;
        const Q8 before = Policy::parameter(layer, index);
        Policy::backward(layer, raw(deltaLsbs));
        BOOST_TEST(Policy::gradient(layer, index).getValue() == deltaLsbs);
        Policy::update(layer);

        return static_cast<int>(Policy::parameter(layer, index).getValue()) - static_cast<int>(before.getValue());
    }

    template<typename Policy>
    static void checkLayer(const size_t numberOfParameters)
    {
        for (size_t index = 0; index < numberOfParameters; ++index)
        {
            // Steps of 0.25 x (gradient in LSBs). A floor fails -1 (gives
            // -1) and +3 (gives 0); toward zero fails +-3 and +-6.
            BOOST_TEST(stepMoves<Policy>(-1, index) == 0);   // -0.25
            BOOST_TEST(stepMoves<Policy>(1, index) == 0);    // +0.25
            BOOST_TEST(stepMoves<Policy>(-2, index) == -1);  // -0.5, tie away from zero
            BOOST_TEST(stepMoves<Policy>(2, index) == 1);    // +0.5
            BOOST_TEST(stepMoves<Policy>(-3, index) == -1);  // -0.75
            BOOST_TEST(stepMoves<Policy>(3, index) == 1);    // +0.75
            BOOST_TEST(stepMoves<Policy>(-4, index) == -1);  // exact
            BOOST_TEST(stepMoves<Policy>(4, index) == 1);
            BOOST_TEST(stepMoves<Policy>(-6, index) == -2);  // -1.5
            BOOST_TEST(stepMoves<Policy>(6, index) == 2);    // +1.5
        }
    }

    // Conv1D / Conv2D / DepthwiseConv2D / PointwiseConv2D share the
    // computeGradients(deltas, input) + updateWeights(lr) + getWeight API.
    template<typename Layer>
    struct ConvPolicy
    {
        typedef Layer LayerType;

        static void backward(LayerType& layer, const Q8& delta)
        {
            const Q8 input[1] = {one()};
            const Q8 deltas[1] = {delta};
            layer.computeGradients(&deltas[0], &input[0]);
        }

        static void update(LayerType& layer)
        {
            layer.updateWeights(learningRate());
        }

        static Q8 parameter(const LayerType& layer, const size_t index)
        {
            return layer.getWeight(index);
        }

        static Q8 gradient(const LayerType& layer, const size_t index)
        {
            return layer.getGradient(index);
        }
    };

    // BinaryDense / TernaryDense: index 0 is the latent weight, 1 the bias.
    // Each keeps its gradients private, so the gradient is recomputed here
    // from the documented rule (latent gradient = delta x sign(input), bias
    // gradient = delta), which with input 1.0 is the delta itself.
    template<typename Layer>
    struct DensePolicy
    {
        typedef Layer LayerType;

        static Q8& lastDelta()
        {
            static Q8 delta;
            return delta;
        }

        static void backward(LayerType& layer, const Q8& delta)
        {
            const Q8 input[1] = {one()};
            const Q8 deltas[1] = {delta};
            Q8 output[1];
            lastDelta() = delta;
            layer.forward(&input[0], &output[0]);
            layer.backward(&deltas[0], &input[0], nullptr);
        }

        static void update(LayerType& layer)
        {
            layer.updateWeights(learningRate());
        }

        static Q8 parameter(const LayerType& layer, const size_t index)
        {
            return (index == 0) ? layer.getLatentWeight(0, 0) : layer.getBias(0);
        }

        static Q8 gradient(const LayerType&, const size_t)
        {
            return lastDelta();
        }
    };

    // DenseHead: index 0 is the weight, 1 the bias. The update is fused into
    // the backward pass, so the delta is kept here and applied by update();
    // with input 1.0 both gradients are the delta itself.
    struct DenseHeadPolicy
    {
        typedef tinymind::DenseHead<Q8, 1, 1, tinymind::SquaredErrorHeadLoss<Q8>> LayerType;

        static Q8& lastDelta()
        {
            static Q8 delta;
            return delta;
        }

        static void backward(LayerType&, const Q8& delta)
        {
            lastDelta() = delta;
        }

        static void update(LayerType& layer)
        {
            const Q8 input[1] = {one()};
            const Q8 deltas[1] = {lastDelta()};
            layer.update(&deltas[0], &input[0], learningRate());
        }

        static Q8 parameter(const LayerType& layer, const size_t index)
        {
            return (index == 0) ? layer.getWeight(0, 0) : layer.getBias(0);
        }

        static Q8 gradient(const LayerType&, const size_t)
        {
            return lastDelta();
        }
    };

    // BatchNorm1D: beta's gradient is the delta itself.
    struct BatchNormPolicy
    {
        typedef tinymind::BatchNorm1D<Q8, 1> LayerType;

        static void backward(LayerType& layer, const Q8& delta)
        {
            const Q8 input[1] = {Q8(0)};
            const Q8 deltas[1] = {delta};
            Q8 output[1];
            Q8 inputDeltas[1];
            layer.forward(&input[0], &output[0]);
            layer.backward(&deltas[0], &inputDeltas[0]);
        }

        static void update(LayerType& layer)
        {
            layer.updateParameters(learningRate());
        }

        static Q8 parameter(const LayerType& layer, const size_t)
        {
            return layer.getBeta(0);
        }

        static Q8 gradient(const LayerType& layer, const size_t)
        {
            return layer.getBetaGradient(0);
        }
    };

    // multiplyRoundNearest against the exact product of the raw values,
    // rounded to nearest with ties away from zero.
    template<typename ValueType>
    static void checkNearest(const bool withNegatives)
    {
        typedef typename ValueType::FullWidthFieldType FieldType;
        const int64_t one = static_cast<int64_t>(1) << ValueType::NumberOfFractionalBits;
        const int64_t magnitudes[] = {0, 1, 2, 3, one / 4, one / 2 - 1, one / 2, one / 2 + 1, one - 1, one, one + 1, 3 * one / 2, 3 * one / 2 + 1};
        const size_t count = sizeof(magnitudes) / sizeof(magnitudes[0]);
        for (size_t a = 0; a < 2 * count; ++a)
        {
            for (size_t b = 0; b < 2 * count; ++b)
            {
                if (!withNegatives && ((a >= count) || (b >= count)))
                {
                    continue;
                }
                const int64_t left = (a < count) ? magnitudes[a] : -magnitudes[a - count];
                const int64_t right = (b < count) ? magnitudes[b] : -magnitudes[b - count];
                const int64_t exact = left * right;
                const int64_t magnitude = ((exact < 0) ? -exact : exact);
                const int64_t rounded = (magnitude + one / 2) / one;
                const int64_t expected = (exact < 0) ? -rounded : rounded;
                const ValueType result = tinymind::multiplyRoundNearest(ValueType(static_cast<FieldType>(left)), ValueType(static_cast<FieldType>(right)));
                BOOST_TEST_INFO(left << " x " << right);
                BOOST_TEST(static_cast<int64_t>(result.getValue()) == expected);
            }
        }
    }

    // The xor example's network (2-4-1, tanh hidden, sigmoid output, the
    // library's default learning rate 0.25, momentum 0.5 and acceleration
    // 0.0625) in Q8.8, initial weights uniform in [-0.5, 0.5).
    struct XorWeights
    {
        static PortableUniformReal& generator()
        {
            static PortableUniformReal g(-0.5, 0.5, 1u);
            return g;
        }

        static Q8 generateRandomWeight()
        {
            return tinymind::ValueConverter<double, Q8>::convertToDestinationType(generator()());
        }
    };

    typedef tinymind::FixedPointTransferFunctions<
        Q8,
        XorWeights,
        tinymind::TanhActivationPolicy<Q8>,
        tinymind::SigmoidActivationPolicy<Q8>> XorTransferFunctions;
    typedef tinymind::MultilayerPerceptron<Q8, 2, 1, 4, 1, XorTransferFunctions> XorNetwork;

    // value / 256 rounded to nearest, ties away from zero: what one Q8.8
    // learning rate x gradient step must come to, from their raw product.
    static int64_t nearestStep(const int64_t product)
    {
        return (product < 0) ? -((-product + 128) / 256) : ((product + 128) / 256);
    }

    // Raw weight and the gradient stored for it by the last training step.
    struct Slot
    {
        int64_t weight;
        int64_t gradient;
    };

    template<typename LayerType>
    static void addWeightSlots(const LayerType& layer, const size_t neurons, const size_t connections, std::vector<Slot>& slots)
    {
        for (size_t n = 0; n < neurons; ++n)
        {
            for (size_t c = 0; c < connections; ++c)
            {
                slots.push_back({layer.getWeightForNeuronAndConnection(n, c).getValue(), layer.getGradientForNeuronAndConnection(n, c).getValue()});
            }
        }
    }

    template<typename LayerType>
    static void addBiasSlots(const LayerType& layer, const size_t connections, std::vector<Slot>& slots)
    {
        for (size_t c = 0; c < connections; ++c)
        {
            slots.push_back({layer.getBiasNeuronWeightForConnection(c).getValue(), layer.getBiasNeuronGradientForConnection(c).getValue()});
        }
    }

    // Input (gated) weights and biases, recurrent weights, then the hidden
    // layer's output weights and biases.
    template<typename NetworkType>
    static std::vector<Slot> recurrentSlots(NetworkType& nn, const size_t inputs, const size_t hidden, const size_t outputs, const size_t gates)
    {
        std::vector<Slot> slots;
        addWeightSlots(nn.getInputLayer(), inputs, gates * hidden, slots);
        addBiasSlots(nn.getInputLayer(), gates * hidden, slots);
        addWeightSlots(nn.getRecurrentLayer(), hidden, gates * hidden, slots);
        addWeightSlots(nn.getLastHiddenLayer(), hidden, outputs, slots);
        addBiasSlots(nn.getLastHiddenLayer(), outputs, slots);

        return slots;
    }

    // A recurrent network with one hidden layer (Gates gate weights per
    // hidden neuron) trains with learning rate 0.25 and no momentum, so every
    // weight, gated weight and bias must move by exactly the gradient step
    // rounded to nearest. The steps include sub-LSB negative ones (a floor
    // makes them -1 LSB) and fractions of at least one half (toward zero
    // drops them), so both wrong roundings fail.
    template<typename NetworkType, size_t Gates>
    static void checkRecurrentSteps()
    {
        static const size_t Inputs = 3;
        static const size_t Hidden = 4;
        static const size_t Outputs = 2;
        XorWeights::generator().seed(7u);
        static NetworkType nn;
        nn.setLearningRate(learningRate());
        nn.setMomentumRate(Q8(0));
        nn.setAccelerationRate(Q8(0));

        PortableUniformReal values(-0.5, 0.5, 11u);
        Q8 inputs[Inputs];
        Q8 target[Outputs];
        size_t mismatches = 0;
        size_t negativeSubLsb = 0;
        size_t roundedUpHalves = 0;
        for (size_t step = 0; step < 20; ++step)
        {
            for (size_t i = 0; i < Inputs; ++i)
            {
                inputs[i] = tinymind::ValueConverter<double, Q8>::convertToDestinationType(values());
            }
            for (size_t o = 0; o < Outputs; ++o)
            {
                target[o] = tinymind::ValueConverter<double, Q8>::convertToDestinationType(values());
            }
            const std::vector<Slot> before = recurrentSlots(nn, Inputs, Hidden, Outputs, Gates);
            nn.feedForward(&inputs[0]);
            nn.trainNetwork(&target[0]);
            const std::vector<Slot> after = recurrentSlots(nn, Inputs, Hidden, Outputs, Gates);

            for (size_t i = 0; i < after.size(); ++i)
            {
                const int64_t product = static_cast<int64_t>(learningRate().getValue()) * after[i].gradient;
                const int64_t expected = nearestStep(product);
                mismatches += ((after[i].weight - before[i].weight) != expected) ? 1u : 0u;
                negativeSubLsb += ((product < 0) && (product > -256)) ? 1u : 0u;
                const int64_t magnitude = (product < 0) ? -product : product;
                roundedUpHalves += ((magnitude % 256) >= 128) ? 1u : 0u;
            }
        }

        BOOST_TEST(mismatches == 0u);
        BOOST_TEST(negativeSubLsb > 0u);
        BOOST_TEST(roundedUpHalves > 0u);
    }

    // Trains one network from `seed` and returns how many of the four XOR
    // patterns it classifies correctly.
    static unsigned trainXor(const uint32_t seed)
    {
        static const int cases[4][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
        XorWeights::generator().seed(seed);
        XorNetwork nn;
        PortableUniformReal patterns(0.0, 4.0, seed + 1000u);
        Q8 inputs[2];
        Q8 target[1];
        Q8 output[1];

        for (size_t step = 0; step < 20000; ++step)
        {
            const int c = static_cast<int>(patterns()) & 3;
            inputs[0] = Q8(cases[c][0], 0);
            inputs[1] = Q8(cases[c][1], 0);
            target[0] = Q8(cases[c][2], 0);
            nn.feedForward(&inputs[0]);
            nn.trainNetwork(&target[0]);
        }

        unsigned correct = 0;
        for (int c = 0; c < 4; ++c)
        {
            inputs[0] = Q8(cases[c][0], 0);
            inputs[1] = Q8(cases[c][1], 0);
            nn.feedForward(&inputs[0]);
            nn.getLearnedValues(&output[0]);
            const bool high = !(output[0] < Q8(0, 128));
            correct += (high == (cases[c][2] == 1)) ? 1u : 0u;
        }

        return correct;
    }
}

BOOST_AUTO_TEST_SUITE(test_suite_sgd_step_rounding)

BOOST_AUTO_TEST_CASE(test_case_multiply_round_nearest_is_exact)
{
    using namespace sgd_step_rounding_tests;
    checkNearest<tinymind::QValue<8, 8, true>>(true);
    checkNearest<tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy>>(true);
    checkNearest<tinymind::QValue<16, 16, true>>(true);
    checkNearest<tinymind::QValue<24, 8, true>>(true);
    checkNearest<tinymind::QValue<8, 8, false>>(false);
    checkNearest<tinymind::QValue<16, 16, false>>(false);

    // Floating point multiplies as usual.
    BOOST_TEST(tinymind::multiplyRoundNearest(0.5, -0.375) == -0.1875);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_conv1d)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::Conv1D<Q8, 1, 1> LayerType;
    checkLayer<ConvPolicy<LayerType>>(LayerType::TotalWeights);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_conv2d)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::Conv2D<Q8, 1, 1, 1, 1, 1> LayerType;
    checkLayer<ConvPolicy<LayerType>>(LayerType::TotalWeights);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_depthwise_conv2d)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::DepthwiseConv2D<Q8, 1, 1, 1, 1, 1> LayerType;
    checkLayer<ConvPolicy<LayerType>>(LayerType::TotalWeights);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_pointwise_conv2d)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::PointwiseConv2D<Q8, 1, 1, 1, 1> LayerType;
    checkLayer<ConvPolicy<LayerType>>(LayerType::TotalWeights);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_binary_dense)
{
    using namespace sgd_step_rounding_tests;
    checkLayer<DensePolicy<tinymind::BinaryDense<Q8, 1, 1>>>(2);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_ternary_dense)
{
    using namespace sgd_step_rounding_tests;
    checkLayer<DensePolicy<tinymind::TernaryDense<Q8, 1, 1>>>(2);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_batchnorm)
{
    using namespace sgd_step_rounding_tests;
    checkLayer<BatchNormPolicy>(1);
}

BOOST_AUTO_TEST_CASE(test_case_sub_lsb_step_dense_head)
{
    using namespace sgd_step_rounding_tests;
    checkLayer<DenseHeadPolicy>(2);
}

BOOST_AUTO_TEST_CASE(test_case_step_rounding_self_attention)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::SelfAttention1D<Q8, 2, 2, 2> LayerType;
    LayerType layer;
    // Fixed raw patterns in about [-0.5, 0.5]; with them several weight and
    // bias gradients are negative fractions of one.
    int k = 1;
    for (size_t i = 0; i < LayerType::TotalWeights; ++i, ++k)
    {
        layer.setWeight(i, raw((k * 37) % 200 - 100));
    }
    for (size_t i = 0; i < LayerType::TotalBiases; ++i, ++k)
    {
        layer.setBias(i, raw((k * 37) % 200 - 100));
    }
    Q8 input[LayerType::InputSize];
    Q8 output[LayerType::OutputSize];
    Q8 deltas[LayerType::OutputSize];
    for (size_t i = 0; i < LayerType::InputSize; ++i, ++k)
    {
        input[i] = raw((k * 53) % 256 - 128);
    }
    for (size_t i = 0; i < LayerType::OutputSize; ++i, ++k)
    {
        deltas[i] = raw((k * 29) % 256 - 128);
    }
    layer.forward(&input[0], &output[0]);
    layer.computeGradients(&deltas[0]);

    // Learning rate 0.25: every weight and bias must move by exactly the
    // step rounded to nearest. The gradients include steps a floor gets
    // wrong (sub-LSB negative) and steps toward zero gets wrong (fraction of
    // at least one half).
    std::vector<Slot> before;
    for (size_t i = 0; i < LayerType::TotalWeights; ++i)
    {
        before.push_back({layer.getWeight(i).getValue(), layer.getGradient(i).getValue()});
    }
    for (size_t i = 0; i < LayerType::TotalBiases; ++i)
    {
        before.push_back({layer.getBias(i).getValue(), layer.getBiasGradient(i).getValue()});
    }
    layer.updateWeights(learningRate());

    size_t negativeSubLsb = 0;
    size_t roundedUpHalves = 0;
    for (size_t i = 0; i < before.size(); ++i)
    {
        const int64_t after = (i < LayerType::TotalWeights) ? layer.getWeight(i).getValue() : layer.getBias(i - LayerType::TotalWeights).getValue();
        const int64_t product = static_cast<int64_t>(learningRate().getValue()) * before[i].gradient;
        BOOST_TEST_INFO("parameter " << i);
        BOOST_TEST(after - before[i].weight == nearestStep(product));
        negativeSubLsb += ((product < 0) && (product > -256)) ? 1u : 0u;
        const int64_t magnitude = (product < 0) ? -product : product;
        roundedUpHalves += ((magnitude % 256) >= 128) ? 1u : 0u;
    }
    BOOST_TEST(negativeSubLsb > 0u);
    BOOST_TEST(roundedUpHalves > 0u);
}

BOOST_AUTO_TEST_CASE(test_case_step_rounding_lstm)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::FixedPointTransferFunctions<
        Q8,
        XorWeights,
        tinymind::TanhActivationPolicy<Q8>,
        tinymind::SigmoidActivationPolicy<Q8>,
        2> TransferFunctionsType;
    typedef tinymind::LstmNeuralNetwork<Q8, 3, tinymind::HiddenLayers<4>, 2, TransferFunctionsType> NetworkType;
    checkRecurrentSteps<NetworkType, 4>();
}

BOOST_AUTO_TEST_CASE(test_case_step_rounding_gru)
{
    using namespace sgd_step_rounding_tests;
    typedef tinymind::FixedPointTransferFunctions<
        Q8,
        XorWeights,
        tinymind::TanhActivationPolicy<Q8>,
        tinymind::SigmoidActivationPolicy<Q8>,
        2> TransferFunctionsType;
    typedef tinymind::GruNeuralNetwork<Q8, 3, tinymind::HiddenLayers<4>, 2, TransferFunctionsType> NetworkType;
    checkRecurrentSteps<NetworkType, 3>();
}

BOOST_AUTO_TEST_CASE(test_case_q8_8_xor_trains_with_default_sgd)
{
    // Every one of 20 initializations must learn XOR. With the step floored
    // instead, 1 of these 20 did.
    unsigned solved = 0;
    for (uint32_t seed = 1; seed <= 20; ++seed)
    {
        const unsigned correct = sgd_step_rounding_tests::trainXor(seed);
        BOOST_TEST_INFO("seed " << seed);
        BOOST_TEST(correct == 4u);
        solved += (correct == 4u) ? 1u : 0u;
    }
    BOOST_TEST(solved == 20u);
}

BOOST_AUTO_TEST_SUITE_END()
