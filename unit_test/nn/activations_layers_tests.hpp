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

// Regression tests for the activation policies and standalone layers fixed
// after the 2026-11 training audit: softmax classifier output, GELU, capped
// ReLU, Xavier normal initialisation, BatchNorm1D and BinaryDense.
// Included from nn_unit_test.cpp, which supplies Boost.Test, the double
// specialisations of Constants / Tanh / Sigmoid, and the RNG helpers.

#pragma once

BOOST_AUTO_TEST_SUITE(test_suite_activations_layers)

namespace activations_layers_tests
{
    struct Rng
    {
        static double generateRandomWeight()
        {
            return (static_cast<double>(rand()) / RAND_MAX) - 0.5;
        }
    };

    struct Init
    {
        static double initialAccelerationRate() { return 0.0; }
        static double initialBiasOutputValue() { return 1.0; }
        static double initialDeltaWeight() { return 0.0; }
        static double initialGradientValue() { return 0.0; }
        static double initialLearningRate() { return 1e-3; }
        static double initialMomentumRate() { return 0.0; }
        static double initialOutputValue() { return 0.0; }
        static double noOpDeltaWeight() { return 1.0; }
        static double noOpWeight() { return 1.0; }
    };

    template<typename Hidden, typename Output, unsigned NumberOfOutputs>
    struct TransferFunctions
    {
        typedef tinymind::FixedPointTransferFunctions<
            double, Rng, Hidden, Output, NumberOfOutputs, Init,
            tinymind::MeanSquaredErrorCalculator<double, NumberOfOutputs>,
            tinymind::ZeroToleranceCalculator<double> > type;
    };

    enum LossKind
    {
        HalfSquaredError,
        CrossEntropy
    };

    // Every weight of a feed-forward network, addressed by stage: stage s
    // connects layer s to layer s + 1 (layer 0 = inputs); neuron -1 is the
    // bias neuron of layer s.
    struct ParameterRef
    {
        size_t stage;
        int neuron;
        size_t connection;
    };

    template<typename NetworkType>
    struct FeedForwardParameters
    {
        NetworkType& nn;
        std::vector<size_t> sizes;
        std::vector<ParameterRef> refs;

        FeedForwardParameters(NetworkType& network, const std::vector<size_t>& layerSizes) : nn(network), sizes(layerSizes)
        {
            for (size_t stage = 0; stage + 1 < sizes.size(); ++stage)
            {
                for (int neuron = -1; neuron < static_cast<int>(sizes[stage]); ++neuron)
                {
                    for (size_t connection = 0; connection < sizes[stage + 1]; ++connection)
                    {
                        ParameterRef ref = { stage, neuron, connection };
                        refs.push_back(ref);
                    }
                }
            }
        }

        double get(const ParameterRef& r)
        {
            if (r.stage == 0)
            {
                return (r.neuron < 0) ? nn.getInputLayerBiasNeuronWeightForConnection(r.connection)
                                      : nn.getInputLayerWeightForNeuronAndConnection(static_cast<size_t>(r.neuron), r.connection);
            }
            return (r.neuron < 0) ? nn.getHiddenLayerBiasNeuronWeightForConnection(r.stage - 1, r.connection)
                                  : nn.getHiddenLayerWeightForNeuronAndConnection(r.stage - 1, static_cast<size_t>(r.neuron), r.connection);
        }

        void set(const ParameterRef& r, const double value)
        {
            if (r.stage == 0)
            {
                if (r.neuron < 0)
                {
                    nn.setInputLayerBiasWeightForConnection(r.connection, value);
                }
                else
                {
                    nn.setInputLayerWeightForNeuronAndConnection(static_cast<size_t>(r.neuron), r.connection, value);
                }
                return;
            }
            if (r.neuron < 0)
            {
                nn.setHiddenLayerBiasNeuronWeightForConnection(r.stage - 1, r.connection, value);
            }
            else
            {
                nn.setHiddenLayerWeightForNeuronAndConnection(r.stage - 1, static_cast<size_t>(r.neuron), r.connection, value);
            }
        }
    };

    template<typename NetworkType, size_t Inputs, size_t Outputs>
    double lossOf(NetworkType& nn, const double* x, const double* t, const LossKind kind)
    {
        double out[Outputs];
        nn.feedForward(x);
        nn.getLearnedValues(out);
        double loss = 0.0;
        for (size_t o = 0; o < Outputs; ++o)
        {
            if (kind == HalfSquaredError)
            {
                loss += 0.5 * (t[o] - out[o]) * (t[o] - out[o]);
            }
            else
            {
                loss -= t[o] * std::log(out[o]);
            }
        }
        return loss;
    }

    // One SGD step (no momentum) must move every weight by
    // learningRate * (-dL/dw), with dL/dw from central differences of the
    // network's own forward pass. Returns the worst error relative to the
    // largest expected step; counts weights whose expected step is non-zero.
    template<typename NetworkType, size_t Inputs, size_t Outputs>
    double gradientCheck(NetworkType& nn, const std::vector<size_t>& sizes, const LossKind kind, size_t& nonZero, size_t& total)
    {
        FeedForwardParameters<NetworkType> params(nn, sizes);
        double x[Inputs];
        double t[Outputs];
        for (size_t i = 0; i < Inputs; ++i)
        {
            x[i] = (rand() % 2001 - 1000) / 1000.0;
        }
        double targetSum = 0.0;
        for (size_t o = 0; o < Outputs; ++o)
        {
            t[o] = (kind == CrossEntropy) ? (0.1 + (rand() % 1000) / 1000.0) : ((rand() % 2001 - 1000) / 1000.0);
            targetSum += t[o];
        }
        if (kind == CrossEntropy)
        {
            for (size_t o = 0; o < Outputs; ++o)
            {
                t[o] /= targetSum;
            }
        }

        static const double eps = 1e-6;
        std::vector<double> before;
        std::vector<double> expected;
        for (size_t p = 0; p < params.refs.size(); ++p)
        {
            const double w = params.get(params.refs[p]);
            params.set(params.refs[p], w + eps);
            const double lp = lossOf<NetworkType, Inputs, Outputs>(nn, x, t, kind);
            params.set(params.refs[p], w - eps);
            const double lm = lossOf<NetworkType, Inputs, Outputs>(nn, x, t, kind);
            params.set(params.refs[p], w);
            before.push_back(w);
            expected.push_back(-(lp - lm) / (2.0 * eps));
        }

        const double learningRate = 1e-3;
        nn.setLearningRate(learningRate);
        nn.setMomentumRate(0.0);
        nn.setAccelerationRate(0.0);
        nn.feedForward(x);
        nn.trainNetwork(t);

        double largest = 0.0;
        for (size_t p = 0; p < expected.size(); ++p)
        {
            largest = std::max(largest, std::fabs(learningRate * expected[p]));
        }
        double worst = 0.0;
        nonZero = 0;
        total = expected.size();
        for (size_t p = 0; p < expected.size(); ++p)
        {
            const double step = params.get(params.refs[p]) - before[p];
            worst = std::max(worst, std::fabs(step - learningRate * expected[p]) / largest);
            if (std::fabs(expected[p]) > 1e-9)
            {
                ++nonZero;
            }
        }
        return worst;
    }

    inline double qToDouble(const tinymind::QValue<16, 16, true>& q)
    {
        return static_cast<double>(q.getValue()) / 65536.0;
    }

    inline tinymind::QValue<16, 16, true> doubleToQ(const double d)
    {
        return tinymind::QValue<16, 16, true>(static_cast<tinymind::QValue<16, 16, true>::FullWidthValueType>(std::llround(d * 65536.0)));
    }
}

// ---------------------------------------------------------------------------
// Softmax classifier output
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_case_softmax_derivative_is_target_minus_output)
{
    // The node delta is -dL/dz = t - y for softmax + cross-entropy, matching
    // the (t - y) * f'(y) convention of the feed-forward output layer.
    const double y[3] = { 0.2, 0.5, 0.3 };
    const double t[3] = { 0.0, 1.0, 0.0 };
    double delta[3];
    tinymind::SoftmaxActivationPolicy<double>::activationFunctionDerivative(y, t, delta, 3);
    BOOST_TEST(std::fabs(delta[0] - (-0.2)) < 1e-12);
    BOOST_TEST(std::fabs(delta[1] - 0.5) < 1e-12);
    BOOST_TEST(std::fabs(delta[2] - (-0.3)) < 1e-12);
}

BOOST_AUTO_TEST_CASE(test_case_softmax_classifier_gradients_match_finite_differences)
{
    using namespace activations_layers_tests;
    typedef TransferFunctions<tinymind::TanhActivationPolicy<double>, tinymind::SoftmaxActivationPolicy<double>, 3>::type TF;
    typedef tinymind::NeuralNetwork<double, 2, tinymind::HiddenLayers<4>, 3, TF, true, 1, false,
                                    tinymind::NonRecurrentHiddenLayerConfig, 0,
                                    tinymind::ClassifierOutputLayerConfiguration> NetworkType;
    srand(RANDOM_SEED);
    static NetworkType nn;
    size_t nonZero = 0;
    size_t total = 0;
    const double worst = gradientCheck<NetworkType, 2, 3>(nn, { 2, 4, 3 }, CrossEntropy, nonZero, total);
    BOOST_TEST(worst < 1e-6);
    BOOST_TEST(nonZero == total);
}

BOOST_AUTO_TEST_CASE(test_case_softmax_classifier_q16_trains_three_blobs)
{
    // Three Gaussian blobs, Q16.16, exp lookup-table softmax, cross-entropy.
    typedef tinymind::QValue<16, 16, true> Q;
    struct QRng
    {
        static Q generateRandomWeight()
        {
            return activations_layers_tests::doubleToQ((static_cast<double>(rand()) / RAND_MAX) - 0.5);
        }
    };
    typedef tinymind::FixedPointTransferFunctions<Q, QRng, tinymind::TanhActivationPolicy<Q>,
                                                  tinymind::SoftmaxActivationPolicy<Q>, 3> TF;
    typedef tinymind::NeuralNetwork<Q, 2, tinymind::HiddenLayers<6>, 3, TF, true, 1, false,
                                    tinymind::NonRecurrentHiddenLayerConfig, 0,
                                    tinymind::ClassifierOutputLayerConfiguration> NetworkType;
    srand(RANDOM_SEED);
    static NetworkType nn;
    nn.setLearningRate(activations_layers_tests::doubleToQ(0.05));
    nn.setMomentumRate(Q(0));
    nn.setAccelerationRate(Q(0));

    const double cx[3] = { -1.0, 1.0, 0.0 };
    const double cy[3] = { -1.0, -1.0, 1.0 };
    double xs[300][2];
    int classes[300];
    for (int i = 0; i < 300; ++i)
    {
        classes[i] = i % 3;
        xs[i][0] = cx[classes[i]] + 0.3 * ((static_cast<double>(rand()) / RAND_MAX) - 0.5);
        xs[i][1] = cy[classes[i]] + 0.3 * ((static_cast<double>(rand()) / RAND_MAX) - 0.5);
    }

    auto evaluate = [&](double& crossEntropy, double& accuracy)
    {
        crossEntropy = 0.0;
        int correct = 0;
        for (int i = 0; i < 300; ++i)
        {
            Q in[2] = { activations_layers_tests::doubleToQ(xs[i][0]), activations_layers_tests::doubleToQ(xs[i][1]) };
            Q out[3];
            nn.feedForward(in);
            nn.getLearnedValues(out);
            int best = 0;
            for (int k = 1; k < 3; ++k)
            {
                if (out[k] > out[best])
                {
                    best = k;
                }
            }
            correct += (best == classes[i]) ? 1 : 0;
            crossEntropy -= std::log(std::max(activations_layers_tests::qToDouble(out[classes[i]]), 1e-6));
        }
        crossEntropy /= 300.0;
        accuracy = correct / 300.0;
    };

    double ceBefore = 0.0;
    double accBefore = 0.0;
    evaluate(ceBefore, accBefore);
    for (int it = 0; it < 3000; ++it)
    {
        const int i = rand() % 300;
        Q in[2] = { activations_layers_tests::doubleToQ(xs[i][0]), activations_layers_tests::doubleToQ(xs[i][1]) };
        Q target[3] = { Q(0), Q(0), Q(0) };
        target[classes[i]] = Q(1, 0);
        nn.feedForward(in);
        nn.trainNetwork(target);
    }
    double ceAfter = 0.0;
    double accAfter = 0.0;
    evaluate(ceAfter, accAfter);

    BOOST_TEST(ceAfter < 0.5 * ceBefore);
    BOOST_TEST(ceAfter < 0.1);
    BOOST_TEST(accAfter > 0.95);
}

// ---------------------------------------------------------------------------
// GELU: the derivative is taken at the pre-activation
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_case_gelu_hidden_gradients_match_finite_differences)
{
    using namespace activations_layers_tests;
    typedef TransferFunctions<tinymind::GeluActivationPolicy<double>, tinymind::LinearActivationPolicy<double>, 2>::type TF;
    typedef tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<5, 4>, 2, TF> NetworkType;
    srand(RANDOM_SEED);
    static NetworkType nn;
    // Larger weights put many pre-activations below zero, where GELU's
    // derivative differs most from a function of its output.
    FeedForwardParameters<NetworkType> params(nn, { 3, 5, 4, 2 });
    for (size_t p = 0; p < params.refs.size(); ++p)
    {
        params.set(params.refs[p], 4.0 * params.get(params.refs[p]));
    }
    size_t nonZero = 0;
    size_t total = 0;
    const double worst = gradientCheck<NetworkType, 3, 2>(nn, { 3, 5, 4, 2 }, HalfSquaredError, nonZero, total);
    BOOST_TEST(worst < 1e-6);
    BOOST_TEST(nonZero == total);
}

BOOST_AUTO_TEST_CASE(test_case_gelu_output_gradients_match_finite_differences)
{
    using namespace activations_layers_tests;
    typedef TransferFunctions<tinymind::TanhActivationPolicy<double>, tinymind::GeluActivationPolicy<double>, 2>::type TF;
    typedef tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, TF> NetworkType;
    srand(RANDOM_SEED + 1);
    static NetworkType nn;
    FeedForwardParameters<NetworkType> params(nn, { 3, 4, 2 });
    for (size_t p = 0; p < params.refs.size(); ++p)
    {
        params.set(params.refs[p], 4.0 * params.get(params.refs[p]));
    }
    size_t nonZero = 0;
    size_t total = 0;
    const double worst = gradientCheck<NetworkType, 3, 2>(nn, { 3, 4, 2 }, HalfSquaredError, nonZero, total);
    BOOST_TEST(worst < 1e-6);
    BOOST_TEST(nonZero == total);
}

typedef FloatingPointTransferFunctions<
                                        double,
                                        UniformRealRandomNumberGenerator,
                                        tinymind::GeluActivationPolicy,
                                        tinymind::TanhActivationPolicy,
                                        tinymind::SigmoidActivationPolicy,
                                        tinymind::ZeroToleranceCalculator,
                                        2> GeluElmanGradientCheckTransferFunctions;

BOOST_AUTO_TEST_CASE(test_case_gelu_elman_gradients_match_finite_differences)
{
    typedef tinymind::ElmanNeuralNetwork<double, 3, 4, 2, GeluElmanGradientCheckTransferFunctions> NetworkType;
    srand(RANDOM_SEED);
    static NetworkType nn;
    RecurrentGradientCheck<NetworkType, 3, 4, 1, 2> check(nn);
    check.run();
}

BOOST_AUTO_TEST_CASE(test_case_gelu_q16_derivative_matches_reference)
{
    typedef tinymind::QValue<16, 16, true> Q;
    for (int i = -30; i <= 30; ++i)
    {
        const double x = i / 10.0;
        const double s = 1.0 / (1.0 + std::exp(-1.702 * x));
        const double expected = s + 1.702 * x * s * (1.0 - s);
        const double actual = activations_layers_tests::qToDouble(
            tinymind::GeluActivationPolicy<Q>::activationFunctionDerivative(activations_layers_tests::doubleToQ(x)));
        BOOST_TEST(std::fabs(actual - expected) < 0.02);
    }
    // A float GELU agrees with the same reference exactly.
    BOOST_TEST(std::fabs(tinymind::GeluActivationPolicy<double>::activationFunctionDerivative(-1.0)
                         - (1.0 / (1.0 + std::exp(1.702)) - 1.702 / (1.0 + std::exp(1.702)) * (1.0 - 1.0 / (1.0 + std::exp(1.702))))) < 1e-12);
}

// ---------------------------------------------------------------------------
// Capped ReLU
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_case_capped_relu_caps_at_whole_number)
{
    typedef tinymind::QValue<16, 16, true> Q;
    typedef tinymind::CappedReluActivationPolicy<Q, 6> Relu6Q;
    typedef tinymind::CappedReluActivationPolicy<double, 6> Relu6D;
    const double inputs[] = { -2.0, 0.0, 0.5, 3.0, 5.99, 6.0, 7.5, 10.0 };
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i)
    {
        const double x = inputs[i];
        const double expected = std::min(std::max(x, 0.0), 6.0);
        BOOST_TEST(std::fabs(Relu6D::activationFunction(x) - expected) < 1e-12);
        BOOST_TEST(std::fabs(activations_layers_tests::qToDouble(Relu6Q::activationFunction(activations_layers_tests::doubleToQ(x))) - expected) < 1e-4);

        // Derivative at the output: 1 strictly inside (0, 6), else 0.
        const double expectedDerivative = (expected > 0.0 && expected < 6.0) ? 1.0 : 0.0;
        BOOST_TEST(Relu6D::activationFunctionDerivative(expected) == expectedDerivative);
        BOOST_TEST(activations_layers_tests::qToDouble(Relu6Q::activationFunctionDerivative(activations_layers_tests::doubleToQ(expected))) == expectedDerivative);
    }
}

// ---------------------------------------------------------------------------
// Xavier normal initialisation
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_case_xavier_normal_is_normal_with_glorot_std)
{
    // One stage of 40 inputs into 10 outputs: std = sqrt(2 / (40 + 10)) = 0.2.
    // Draw many networks' worth of the first stage and check mean, std and
    // that the distribution is not bounded at sqrt(2 / fanSum) like a uniform.
    typedef tinymind::XavierWeightInitializer<40, 1, 10, 1> Init;
    srand(RANDOM_SEED);
    double sum = 0.0;
    double sumSq = 0.0;
    double sumFourth = 0.0;
    size_t count = 0;
    size_t beyondOneSigma = 0;
    for (int network = 0; network < 200; ++network)
    {
        Init init;
        for (size_t w = 0; w < 40 * 10; ++w)
        {
            const double v = init.generateNormalWeight();
            sum += v;
            sumSq += v * v;
            sumFourth += v * v * v * v;
            beyondOneSigma += (std::fabs(v) > 0.2) ? 1 : 0;
            ++count;
        }
    }
    const double mean = sum / count;
    const double variance = sumSq / count - mean * mean;
    const double kurtosis = (sumFourth / count) / (variance * variance);
    BOOST_TEST(std::fabs(mean) < 0.005);
    BOOST_TEST(std::fabs(std::sqrt(variance) - 0.2) < 0.004);
    // Normal: kurtosis 3, 31.7% beyond one sigma. Uniform: 1.8, 0%.
    BOOST_TEST(std::fabs(kurtosis - 3.0) < 0.15);
    BOOST_TEST(std::fabs(static_cast<double>(beyondOneSigma) / count - 0.317) < 0.01);
}

// ---------------------------------------------------------------------------
// BatchNorm1D
// ---------------------------------------------------------------------------

namespace activations_layers_tests
{
    inline double uniformRand()
    {
        return (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
    }

    // Loss L = sum(c * y). The deltas passed to backward are -dL/dy = -c, so
    // the input deltas it returns must equal -dL/dx.
    template<typename LayerType, size_t Count>
    double weightedOutput(LayerType layer, const double* x, const double* c, const size_t batchSize)
    {
        double y[Count];
        layer.forwardBatch(x, y, batchSize);
        double loss = 0.0;
        for (size_t k = 0; k < batchSize * LayerType::NumberOfFeatures; ++k)
        {
            loss += c[k] * y[k];
        }
        return loss;
    }

    // Checks backwardBatch's input deltas and gamma/beta gradients against
    // central differences of forwardBatch, on copies of the layer so each
    // evaluation starts from the same running statistics. Returns the worst
    // absolute error.
    template<typename LayerType, size_t Features, size_t Batch>
    double batchNormGradientCheck(LayerType& layer, const size_t batchSize)
    {
        static const size_t Count = Features * Batch;
        static const double eps = 1e-6;
        double x[Count];
        double c[Count];
        for (size_t k = 0; k < Count; ++k)
        {
            x[k] = 2.0 * uniformRand() + static_cast<double>(k % Features);
            c[k] = uniformRand();
        }

        double deltas[Count];
        for (size_t m = 0; m < Count; ++m)
        {
            deltas[m] = -c[m];
        }
        LayerType analytic = layer;
        double y[Count];
        double inputDeltas[Count];
        analytic.forwardBatch(x, y, batchSize);
        analytic.backwardBatch(deltas, inputDeltas, batchSize);

        double worst = 0.0;
        for (size_t k = 0; k < batchSize * Features; ++k)
        {
            double xp[Count];
            double xm[Count];
            for (size_t m = 0; m < Count; ++m)
            {
                xp[m] = x[m];
                xm[m] = x[m];
            }
            xp[k] += eps;
            xm[k] -= eps;
            const double negGrad = -(weightedOutput<LayerType, Count>(layer, xp, c, batchSize)
                                   - weightedOutput<LayerType, Count>(layer, xm, c, batchSize)) / (2.0 * eps);
            worst = std::max(worst, std::fabs(inputDeltas[k] - negGrad));
        }

        for (size_t i = 0; i < Features; ++i)
        {
            LayerType gp = layer;
            LayerType gm = layer;
            gp.setGamma(i, layer.getGamma(i) + eps);
            gm.setGamma(i, layer.getGamma(i) - eps);
            const double negGammaGrad = -(weightedOutput<LayerType, Count>(gp, x, c, batchSize)
                                        - weightedOutput<LayerType, Count>(gm, x, c, batchSize)) / (2.0 * eps);
            LayerType bp = layer;
            LayerType bm = layer;
            bp.setBeta(i, layer.getBeta(i) + eps);
            bm.setBeta(i, layer.getBeta(i) - eps);
            const double negBetaGrad = -(weightedOutput<LayerType, Count>(bp, x, c, batchSize)
                                       - weightedOutput<LayerType, Count>(bm, x, c, batchSize)) / (2.0 * eps);
            worst = std::max(worst, std::fabs(analytic.getGammaGradient(i) - negGammaGrad));
            worst = std::max(worst, std::fabs(analytic.getBetaGradient(i) - negBetaGrad));
        }
        return worst;
    }

    template<typename LayerType>
    void setNonTrivialParameters(LayerType& bn, const size_t features)
    {
        for (size_t i = 0; i < features; ++i)
        {
            bn.setGamma(i, 0.5 + 0.25 * static_cast<double>(i));
            bn.setBeta(i, 0.1 * static_cast<double>(i) - 0.2);
            bn.setRunningMean(i, 0.3 * static_cast<double>(i));
            bn.setRunningVariance(i, 0.5 + 0.7 * static_cast<double>(i));
        }
    }
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_batch_gradients_match_finite_differences)
{
    // Mini-batch training mode: the input gradient flows through the batch
    // mean and variance, so it is not delta * gamma (nor delta * gamma / std).
    typedef tinymind::BatchNorm1D<double, 3, 10, 1, 5> LayerType;
    srand(RANDOM_SEED);
    LayerType bn;
    activations_layers_tests::setNonTrivialParameters(bn, 3);
    bn.setTraining(true);
    BOOST_TEST((activations_layers_tests::batchNormGradientCheck<LayerType, 3, 5>(bn, 5)) < 1e-7);
    BOOST_TEST((activations_layers_tests::batchNormGradientCheck<LayerType, 3, 5>(bn, 2)) < 1e-7);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_running_statistics_gradients_match_finite_differences)
{
    typedef tinymind::BatchNorm1D<double, 4, 10, 1, 3> LayerType;
    srand(RANDOM_SEED);
    LayerType bn;
    activations_layers_tests::setNonTrivialParameters(bn, 4);

    // Single-sample training mode: normalized with the running statistics.
    bn.setTraining(true);
    BOOST_TEST((activations_layers_tests::batchNormGradientCheck<LayerType, 4, 3>(bn, 1)) < 1e-7);

    // Inference mode, single sample and batch.
    bn.setTraining(false);
    BOOST_TEST((activations_layers_tests::batchNormGradientCheck<LayerType, 4, 3>(bn, 1)) < 1e-7);
    BOOST_TEST((activations_layers_tests::batchNormGradientCheck<LayerType, 4, 3>(bn, 3)) < 1e-7);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_batch_normalizes_each_feature)
{
    // Each feature is normalized over the batch: per-feature output mean 0
    // and variance 1 (gamma 1, beta 0), and gamma/beta scale and shift it.
    tinymind::BatchNorm1D<double, 2, 10, 1, 4> bn;
    const double x[8] = { 1.0, 100.0,   3.0, 300.0,   5.0, 500.0,   7.0, 700.0 };
    double y[8];
    bn.forwardBatch(x, y, 4);
    for (size_t i = 0; i < 2; ++i)
    {
        double mean = 0.0;
        double variance = 0.0;
        for (size_t n = 0; n < 4; ++n)
        {
            mean += y[n * 2 + i] / 4.0;
        }
        for (size_t n = 0; n < 4; ++n)
        {
            variance += (y[n * 2 + i] - mean) * (y[n * 2 + i] - mean) / 4.0;
        }
        BOOST_TEST(std::fabs(mean) < 1e-9);
        BOOST_TEST(std::fabs(variance - 1.0) < 0.01);
    }
    // Feature 1 is feature 0 scaled by 100: the same normalized values, up to
    // epsilon's effect on feature 0's small variance.
    for (size_t n = 0; n < 4; ++n)
    {
        BOOST_TEST(std::fabs(y[n * 2] - y[n * 2 + 1]) < 0.01);
    }

    bn.setGamma(0, 2.0);
    bn.setBeta(0, 3.0);
    double y2[8];
    bn.forwardBatch(x, y2, 4);
    for (size_t n = 0; n < 4; ++n)
    {
        BOOST_TEST(std::fabs(y2[n * 2] - (2.0 * y[n * 2] + 3.0)) < 1e-9);
    }

    // A constant feature has variance 0: its output is beta.
    const double constant[4] = { 5.0, -1.0,   5.0, 1.0 };
    double yc[4];
    bn.forwardBatch(constant, yc, 2);
    BOOST_TEST(std::fabs(yc[0] - 3.0) < 1e-9);
    BOOST_TEST(std::fabs(yc[2] - 3.0) < 1e-9);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_running_statistics_are_per_feature)
{
    // Features with means 0..5 and standard deviations 1..6. After many
    // training batches the running statistics hold each feature's own mean
    // and variance (not one value shared by every feature), and the inference
    // output for a batch matches the training output for the same batch.
    static const size_t F = 6;
    static const size_t N = 64;
    static tinymind::BatchNorm1D<double, F, 10, 1, N> bn;
    srand(RANDOM_SEED);
    static double x[F * N];
    static double yTrain[F * N];
    static double yInfer[F * N];
    for (int batch = 0; batch < 300; ++batch)
    {
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t i = 0; i < F; ++i)
            {
                // Uniform with mean i and standard deviation i + 1.
                x[n * F + i] = static_cast<double>(i) + std::sqrt(3.0) * static_cast<double>(i + 1) * activations_layers_tests::uniformRand();
            }
        }
        bn.setTraining(true);
        bn.forwardBatch(x, yTrain, N);
    }
    for (size_t i = 0; i < F; ++i)
    {
        const double sd = static_cast<double>(i + 1);
        BOOST_TEST(std::fabs(bn.getRunningMean(i) - static_cast<double>(i)) < 0.15 * sd);
        BOOST_TEST(std::fabs(bn.getRunningVariance(i) / (sd * sd) - 1.0) < 0.15);
    }

    // With momentum 100% the running statistics are the last batch's (the
    // variance with the unbiased N/(N-1) factor), so inference on that batch
    // reproduces the training output up to sqrt((N-1)/N).
    static tinymind::BatchNorm1D<double, F, 100, 1, N> last;
    last.setTraining(true);
    last.forwardBatch(x, yTrain, N);
    last.setTraining(false);
    last.forwardBatch(x, yInfer, N);
    double worst = 0.0;
    for (size_t k = 0; k < F * N; ++k)
    {
        worst = std::max(worst, std::fabs(yInfer[k] - yTrain[k]));
    }
    BOOST_TEST(worst < 0.02);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_single_sample_train_matches_inference)
{
    // One sample has no batch statistics, so single-sample training mode
    // normalizes with the running statistics, like inference, then updates
    // them. A sample's training output equals the inference output computed
    // with the statistics as they were before that sample.
    tinymind::BatchNorm1D<double, 3> bn;
    srand(RANDOM_SEED);
    for (int step = 0; step < 50; ++step)
    {
        double x[3];
        for (size_t i = 0; i < 3; ++i)
        {
            x[i] = 10.0 * static_cast<double>(i) + activations_layers_tests::uniformRand();
        }
        tinymind::BatchNorm1D<double, 3> before = bn;
        before.setTraining(false);
        double yInfer[3];
        double yTrain[3];
        before.forward(x, yInfer);
        bn.setTraining(true);
        bn.forward(x, yTrain);
        for (size_t i = 0; i < 3; ++i)
        {
            BOOST_TEST(yTrain[i] == yInfer[i]);
        }
    }
    // Running means track each feature (means 0, 10, 20).
    for (size_t i = 0; i < 3; ++i)
    {
        BOOST_TEST(std::fabs(bn.getRunningMean(i) - 10.0 * static_cast<double>(i)) < 1.0);
    }
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_update_parameters)
{
    tinymind::BatchNorm1D<double, 2, 10, 1, 3> bn;
    const double x[6] = { 1.0, 2.0,   2.0, 4.0,   4.0, 1.0 };
    double y[6];
    bn.forwardBatch(x, y, 3);
    const double deltas[6] = { 1.0, 0.5,   1.0, 0.5,   1.0, 0.5 };
    double inputDeltas[6];
    bn.backwardBatch(deltas, inputDeltas, 3);
    BOOST_TEST(std::fabs(bn.getBetaGradient(0) - 3.0) < 1e-12);
    BOOST_TEST(std::fabs(bn.getBetaGradient(1) - 1.5) < 1e-12);
    // A uniform delta shifts every output equally: no input gradient, and the
    // gamma gradient (sum of delta * xhat) is zero because xhat has mean 0.
    for (size_t k = 0; k < 6; ++k)
    {
        BOOST_TEST(std::fabs(inputDeltas[k]) < 1e-12);
    }
    BOOST_TEST(std::fabs(bn.getGammaGradient(0)) < 1e-12);
    bn.updateParameters(0.1);
    BOOST_TEST(std::fabs(bn.getBeta(0) - 0.3) < 1e-12);
    BOOST_TEST(std::fabs(bn.getBeta(1) - 0.15) < 1e-12);
    BOOST_TEST(bn.getGammaGradient(0) == 0.0);
    BOOST_TEST(bn.getBetaGradient(1) == 0.0);

    BOOST_TEST(bn.isTraining() == true);
    bn.setTraining(false);
    BOOST_TEST(bn.isTraining() == false);
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_q16_matches_double)
{
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> Q;
    tinymind::BatchNorm1D<Q, 3, 10, 1, 4> bq;
    tinymind::BatchNorm1D<double, 3, 10, 1, 4> bd;
    srand(RANDOM_SEED);
    double xd[12];
    double cd[12];
    Q xq[12];
    Q cq[12];
    for (size_t k = 0; k < 12; ++k)
    {
        xq[k] = Q(static_cast<Q::FullWidthValueType>(std::llround((3.0 * activations_layers_tests::uniformRand() + static_cast<double>(k % 3)) * 65536.0)));
        xd[k] = static_cast<double>(xq[k].getValue()) / 65536.0;
        cq[k] = Q(static_cast<Q::FullWidthValueType>(std::llround(activations_layers_tests::uniformRand() * 65536.0)));
        cd[k] = static_cast<double>(cq[k].getValue()) / 65536.0;
    }

    // Batch training, batch inference, single-sample training.
    for (int mode = 0; mode < 3; ++mode)
    {
        const size_t batchSize = (mode == 2) ? 1 : 4;
        bq.setTraining(mode != 1);
        bd.setTraining(mode != 1);
        double yd[12];
        double dxd[12];
        Q yq[12];
        Q dxq[12];
        bd.forwardBatch(xd, yd, batchSize);
        bq.forwardBatch(xq, yq, batchSize);
        bd.backwardBatch(cd, dxd, batchSize);
        bq.backwardBatch(cq, dxq, batchSize);
        for (size_t k = 0; k < batchSize * 3; ++k)
        {
            BOOST_TEST(std::fabs(static_cast<double>(yq[k].getValue()) / 65536.0 - yd[k]) < 2e-3);
            BOOST_TEST(std::fabs(static_cast<double>(dxq[k].getValue()) / 65536.0 - dxd[k]) < 2e-3);
        }
        for (size_t i = 0; i < 3; ++i)
        {
            BOOST_TEST(std::fabs(static_cast<double>(bq.getRunningMean(i).getValue()) / 65536.0 - bd.getRunningMean(i)) < 2e-3);
            BOOST_TEST(std::fabs(static_cast<double>(bq.getRunningVariance(i).getValue()) / 65536.0 - bd.getRunningVariance(i)) < 2e-3);
        }
    }

    // All-zero input with zero variance needs a non-zero epsilon (no divide
    // by zero in invStd): the output is beta = 0.
    tinymind::BatchNorm1D<Q, 2, 10, 1, 2> zeros;
    const Q zx[4] = { Q(0, 0), Q(0, 0), Q(0, 0), Q(0, 0) };
    Q zy[4];
    zeros.forwardBatch(zx, zy, 2);
    for (size_t k = 0; k < 4; ++k)
    {
        BOOST_TEST(zy[k].getValue() == 0);
    }
}

BOOST_AUTO_TEST_CASE(test_case_batchnorm_conv1d_pipeline)
{
    // Conv1D -> BatchNorm1D over a batch of four sequences: each of the six
    // conv output positions is normalized over the batch.
    tinymind::Conv1D<double, 8, 3, 1, 1> conv;
    tinymind::BatchNorm1D<double, 6, 10, 1, 4> bn;
    conv.setFilterWeight(0, 0, 1.0);
    conv.setFilterWeight(0, 1, 1.0);
    conv.setFilterWeight(0, 2, 1.0);
    conv.setFilterWeight(0, 3, 0.0);

    double convOutputs[4 * 6];
    for (size_t n = 0; n < 4; ++n)
    {
        double input[8];
        for (size_t t = 0; t < 8; ++t)
        {
            input[t] = static_cast<double>((n + 1) * (t + 1));
        }
        conv.forward(input, &convOutputs[n * 6]);
    }
    double bnOutputs[4 * 6];
    bn.forwardBatch(convOutputs, bnOutputs, 4);
    for (size_t i = 0; i < 6; ++i)
    {
        double mean = 0.0;
        for (size_t n = 0; n < 4; ++n)
        {
            mean += bnOutputs[n * 6 + i] / 4.0;
        }
        BOOST_TEST(std::fabs(mean) < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// BinaryDense: latent weights are clipped to [-1, 1]
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_case_binary_dense_keeps_learning_after_teacher_changes)
{
    // Learn a random binary teacher from noisy targets, then flip half the
    // teacher's weights and keep training. Without clipping, latent weights
    // pushed past +/-1 have their STE gradient zeroed forever and the student
    // cannot follow the change (64 -> 69 of 128 correct in the audit).
    static const size_t I = 32;
    static const size_t O = 4;
    static tinymind::BinaryDense<double, I, O> student;
    static double teacher[O][I];
    srand(RANDOM_SEED);
    for (size_t j = 0; j < O; ++j)
    {
        for (size_t i = 0; i < I; ++i)
        {
            teacher[j][i] = (rand() & 1) ? 1.0 : -1.0;
            student.setLatentWeight(j, i, 0.1 * activations_layers_tests::uniformRand());
        }
    }
    student.binarizeWeights();

    auto train = [&](const int steps)
    {
        double x[I];
        double y[O];
        double delta[O];
        for (int step = 0; step < steps; ++step)
        {
            for (size_t i = 0; i < I; ++i)
            {
                x[i] = activations_layers_tests::uniformRand();
            }
            student.forward(x, y);
            for (size_t j = 0; j < O; ++j)
            {
                double target = 0.0;
                for (size_t i = 0; i < I; ++i)
                {
                    target += teacher[j][i] * ((x[i] < 0.0) ? -1.0 : 1.0);
                }
                // Gaussian label noise, standard deviation 2 (Box-Muller).
                const double u1 = (static_cast<double>(rand()) + 1.0) / (static_cast<double>(RAND_MAX) + 1.0);
                const double u2 = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
                target += 2.0 * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
                delta[j] = target - y[j];
            }
            student.backward(delta, x, nullptr);
            student.updateWeights(0.005);
        }
    };
    auto correct = [&]()
    {
        size_t count = 0;
        for (size_t j = 0; j < O; ++j)
        {
            for (size_t i = 0; i < I; ++i)
            {
                count += (student.getBinaryWeight(j, i) == teacher[j][i]) ? 1 : 0;
            }
        }
        return count;
    };

    train(20000);
    BOOST_TEST(correct() >= 124u);
    size_t outside = 0;
    for (size_t j = 0; j < O; ++j)
    {
        for (size_t i = 0; i < I; ++i)
        {
            outside += (std::fabs(student.getLatentWeight(j, i)) > 1.0) ? 1 : 0;
        }
    }
    BOOST_TEST(outside == 0u);

    for (size_t j = 0; j < O; ++j)
    {
        for (size_t i = 0; i < I; i += 2)
        {
            teacher[j][i] = -teacher[j][i];
        }
    }
    train(20000);
    BOOST_TEST(correct() >= 120u);
}

BOOST_AUTO_TEST_SUITE_END()
