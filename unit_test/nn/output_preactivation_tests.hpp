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

// NeuralNetwork::getOutputPreActivations() and getPredictedClass(): the
// output layer's logits, recomputed from the last hidden layer, and the
// argmax over them.
//
// The pre-activations must reproduce the output layer exactly (its
// activation applied to them gives getLearnedValues() bit for bit), and the
// predicted class must separate two outputs that the saturating sigmoid
// lookup table maps to the same value. That tie is what cost
// gas_sensor_drift about 4 points of batch-10 accuracy when it ranked the
// sigmoid outputs.
//
// Included from nn_unit_test.cpp.

namespace output_preactivation_tests
{
    typedef tinymind::QValue<16, 16, true> Q16;
    static const size_t Outputs = 3;

    typedef tinymind::FixedPointTransferFunctions<
        Q16,
        UniformRealRandomNumberGenerator<Q16>,
        tinymind::ReluActivationPolicy<Q16>,
        tinymind::SigmoidActivationPolicy<Q16>,
        Outputs> QTransferFunctions;
    typedef tinymind::MultilayerPerceptron<Q16, 2, 1, 4, Outputs, QTransferFunctions> QNetwork;

    typedef FloatingPointTransferFunctions<
        double,
        UniformRealRandomNumberGenerator,
        tinymind::ReluActivationPolicy,
        tinymind::SigmoidActivationPolicy,
        tinymind::NullActivationPolicy,
        tinymind::ZeroToleranceCalculator,
        Outputs> DoubleTransferFunctions;
    typedef tinymind::MultilayerPerceptron<double, 2, 1, 4, Outputs, DoubleTransferFunctions> DoubleNetwork;

    static Q16 q16(const double value)
    {
        return tinymind::ValueConverter<double, Q16>::convertToDestinationType(value);
    }

    template<typename ValueType>
    static size_t argmax(const ValueType* values)
    {
        size_t best = 0;
        for (size_t i = 1; i < Outputs; ++i)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }
        return best;
    }
}

BOOST_AUTO_TEST_SUITE(test_suite_output_preactivations)

BOOST_AUTO_TEST_CASE(q16_preactivations_reproduce_the_output_layer)
{
    using namespace output_preactivation_tests;
    srand(RANDOM_SEED);
    static QNetwork nn;

    Q16 input[2];
    Q16 learned[Outputs];
    Q16 pre[Outputs];
    for (int sample = 0; sample < 200; ++sample)
    {
        input[0] = q16(((rand() % 2001) - 1000) / 250.0);
        input[1] = q16(((rand() % 2001) - 1000) / 250.0);
        nn.feedForward(&input[0]);
        nn.getLearnedValues(&learned[0]);
        nn.getOutputPreActivations(&pre[0]);
        for (size_t o = 0; o < Outputs; ++o)
        {
            BOOST_REQUIRE(QTransferFunctions::outputNeuronActivationFunction(pre[o]) == learned[o]);
        }
        BOOST_REQUIRE_EQUAL(nn.getPredictedClass(), argmax(&pre[0]));
    }
}

BOOST_AUTO_TEST_CASE(double_preactivations_reproduce_the_output_layer)
{
    using namespace output_preactivation_tests;
    srand(RANDOM_SEED);
    static DoubleNetwork nn;

    double input[2];
    double learned[Outputs];
    double pre[Outputs];
    for (int sample = 0; sample < 200; ++sample)
    {
        input[0] = ((rand() % 2001) - 1000) / 250.0;
        input[1] = ((rand() % 2001) - 1000) / 250.0;
        nn.feedForward(&input[0]);
        nn.getLearnedValues(&learned[0]);
        nn.getOutputPreActivations(&pre[0]);
        for (size_t o = 0; o < Outputs; ++o)
        {
            BOOST_REQUIRE_EQUAL(DoubleTransferFunctions::outputNeuronActivationFunction(pre[o]), learned[o]);
        }
        // The double sigmoid is strictly increasing, so ranking either agrees.
        BOOST_REQUIRE_EQUAL(nn.getPredictedClass(), argmax(&learned[0]));
    }
}

BOOST_AUTO_TEST_CASE(predicted_class_separates_outputs_the_sigmoid_table_ties)
{
    // One hidden unit at ReLU(1) = 1 drives logits of 7, 9 and 0. The
    // sigmoid table is flat beyond 5.875, so outputs 0 and 1 read the same
    // value and an argmax over them takes output 0; the logits say 1.
    using namespace output_preactivation_tests;
    srand(RANDOM_SEED);
    static QNetwork nn;

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t k = 0; k < 4; ++k)
        {
            nn.setInputLayerWeightForNeuronAndConnection(i, k, q16(0.0));
        }
    }
    for (size_t k = 0; k < 4; ++k)
    {
        nn.setInputLayerBiasWeightForConnection(k, q16(k == 0 ? 1.0 : 0.0));
        for (size_t o = 0; o < Outputs; ++o)
        {
            nn.setHiddenLayerWeightForNeuronAndConnection(0, k, o, q16(0.0));
        }
    }
    for (size_t o = 0; o < Outputs; ++o)
    {
        nn.setHiddenLayerBiasNeuronWeightForConnection(0, o, q16(0.0));
    }
    nn.setHiddenLayerWeightForNeuronAndConnection(0, 0, 0, q16(7.0));
    nn.setHiddenLayerWeightForNeuronAndConnection(0, 0, 1, q16(9.0));

    const Q16 input[2] = {q16(0.5), q16(-0.5)};
    nn.feedForward(&input[0]);

    Q16 learned[Outputs];
    Q16 pre[Outputs];
    nn.getLearnedValues(&learned[0]);
    nn.getOutputPreActivations(&pre[0]);

    BOOST_TEST(pre[0] == q16(7.0));
    BOOST_TEST(pre[1] == q16(9.0));
    BOOST_TEST(pre[2] == q16(0.0));
    BOOST_TEST(learned[0] == learned[1]);
    BOOST_TEST(argmax(&learned[0]) == 0u);
    BOOST_TEST(nn.getPredictedClass() == 1u);
}

BOOST_AUTO_TEST_SUITE_END()
