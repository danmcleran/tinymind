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

// Optimizer, weight-decay, clipping and fixed-point-default tests.
//
// Each network test runs a few real training steps and compares every weight
// and every bias, parameter by parameter, with the textbook update computed
// from central finite differences of the network's own forward pass and loss
// L = 0.5 * sum((target - output)^2). The library adds learningRate * (-dL/dw)
// style updates, so the reference uses g = -dL/dw throughout.
//
// Included from nn_unit_test.cpp after its helpers (CellStateAccess,
// RecurrentGradientCheckTransferFunctions, FloatingPointTransferFunctions).

#include "rmsprop.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace optimizer_tests
{
    struct Param
    {
        std::string group;
        std::function<double()> get;
        std::function<void(double)> set;
    };

    static bool isBiasGroup(const std::string& group)
    {
        return (group.size() >= 2) && (group.compare(group.size() - 2, 2, "_b") == 0);
    }

    template<typename ValueType>
    static double toDouble(const ValueType& value)
    {
        return static_cast<double>(value.getValue()) / static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    }

    static double toDouble(const double value)
    {
        return value;
    }

    template<typename ValueType>
    static ValueType fromDouble(const double value)
    {
        const double scaled = value * static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
        return ValueType(static_cast<typename ValueType::FullWidthValueType>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
    }

    template<>
    double fromDouble<double>(const double value)
    {
        return value;
    }

    // Every weight and bias of a feed-forward network, grouped per matrix.
    template<typename ValueType, typename NetworkType>
    static void addMlpParams(NetworkType& nn, const size_t inputs, const size_t hidden, const size_t hiddenLayers, const size_t outputs, std::vector<Param>& params)
    {
        for (size_t i = 0; i < inputs; ++i)
        {
            for (size_t k = 0; k < hidden; ++k)
            {
                params.push_back({"input_w",
                    [&nn, i, k]() { return toDouble(nn.getInputLayerWeightForNeuronAndConnection(i, k)); },
                    [&nn, i, k](double v) { nn.setInputLayerWeightForNeuronAndConnection(i, k, fromDouble<ValueType>(v)); }});
            }
        }
        for (size_t k = 0; k < hidden; ++k)
        {
            params.push_back({"input_b",
                [&nn, k]() { return toDouble(nn.getInputLayerBiasNeuronWeightForConnection(k)); },
                [&nn, k](double v) { nn.setInputLayerBiasWeightForConnection(k, fromDouble<ValueType>(v)); }});
        }
        for (size_t l = 0; l < hiddenLayers; ++l)
        {
            const size_t next = (l == hiddenLayers - 1) ? outputs : hidden;
            const std::string name = "hidden" + std::to_string(l);
            for (size_t j = 0; j < hidden; ++j)
            {
                for (size_t k = 0; k < next; ++k)
                {
                    params.push_back({name + "_w",
                        [&nn, l, j, k]() { return toDouble(nn.getHiddenLayerWeightForNeuronAndConnection(l, j, k)); },
                        [&nn, l, j, k](double v) { nn.setHiddenLayerWeightForNeuronAndConnection(l, j, k, fromDouble<ValueType>(v)); }});
                }
            }
            for (size_t k = 0; k < next; ++k)
            {
                params.push_back({name + "_b",
                    [&nn, l, k]() { return toDouble(nn.getHiddenLayerBiasNeuronWeightForConnection(l, k)); },
                    [&nn, l, k](double v) { nn.setHiddenLayerBiasNeuronWeightForConnection(l, k, fromDouble<ValueType>(v)); }});
            }
        }
    }

    enum class Rule
    {
        Sgd,
        Adam,
        RmsProp
    };

    struct Hyper
    {
        Rule rule;
        double beta1;      // Adam
        double beta2;      // Adam; RMSprop decay
        double epsilon;
        double lambda;     // decoupled weight decay on weights (0 = none)
        double clip;       // |gradient| bound (0 = none)
    };

    struct Report
    {
        std::map<std::string, double> worst;    // max |library - reference| / lr
        std::map<std::string, double> moved;    // max |change|
    };

    // Runs `steps` training steps on fresh random inputs and targets and
    // compares every parameter with the reference rule. `restore` puts any
    // recurrent state back before each forward pass.
    template<typename NetworkType, typename ValueType, size_t Inputs, size_t Outputs>
    static Report compareWithReference(NetworkType& nn, std::vector<Param>& params, const Hyper& hp, const double lr, const size_t steps, std::function<void()> restore, const double fdStep = 1e-6)
    {
        Report report;
        std::vector<double> m(params.size(), 0.0);
        std::vector<double> v(params.size(), 0.0);
        double beta1Power = 1.0;
        double beta2Power = 1.0;
        ValueType x[Inputs];
        ValueType target[Outputs];

        nn.setLearningRate(fromDouble<ValueType>(lr));

        auto loss = [&]()
        {
            ValueType out[Outputs];
            restore();
            nn.feedForward(&x[0]);
            nn.getLearnedValues(&out[0]);
            double l = 0.0;
            for (size_t o = 0; o < Outputs; ++o)
            {
                const double d = toDouble(target[o]) - toDouble(out[o]);
                l += 0.5 * d * d;
            }
            return l;
        };

        for (size_t step = 1; step <= steps; ++step)
        {
            for (size_t i = 0; i < Inputs; ++i)
            {
                x[i] = fromDouble<ValueType>((rand() % 2001 - 1000) / 1000.0);
            }
            for (size_t o = 0; o < Outputs; ++o)
            {
                target[o] = fromDouble<ValueType>((rand() % 1001) / 1000.0);
            }
            beta1Power *= hp.beta1;
            beta2Power *= hp.beta2;

            std::vector<double> before(params.size());
            std::vector<double> expected(params.size());
            std::vector<bool> resolved(params.size(), true);
            for (size_t p = 0; p < params.size(); ++p)
            {
                const double w = params[p].get();
                before[p] = w;
                params[p].set(w + fdStep);
                const double lp = loss();
                params[p].set(w - fdStep);
                const double lm = loss();
                params[p].set(w);
                double g = -(lp - lm) / (2.0 * fdStep);
                if (hp.clip > 0.0)
                {
                    g = std::max(-hp.clip, std::min(hp.clip, g));
                }

                double update = 0.0;
                switch (hp.rule)
                {
                case Rule::Sgd:
                    update = lr * g;
                    break;
                case Rule::Adam:
                    m[p] = hp.beta1 * m[p] + (1.0 - hp.beta1) * g;
                    v[p] = hp.beta2 * v[p] + (1.0 - hp.beta2) * g * g;
                    update = lr * (m[p] / (1.0 - beta1Power)) / (std::sqrt(v[p] / (1.0 - beta2Power)) + hp.epsilon);
                    break;
                case Rule::RmsProp:
                    v[p] = hp.beta2 * v[p] + (1.0 - hp.beta2) * g * g;
                    update = lr * g / (std::sqrt(v[p]) + hp.epsilon);
                    break;
                }

                const double decayed = isBiasGroup(params[p].group) ? w : (w - lr * hp.lambda * w);
                expected[p] = decayed + update;
                // An adaptive step is about lr * sign(g) whatever |g| is, so
                // for a gradient near the finite-difference noise floor the
                // reference's sign, and so its step, is not resolved.
                resolved[p] = (hp.rule == Rule::Sgd) || (std::fabs(g) > 1e-6);
            }

            restore();
            nn.feedForward(&x[0]);
            nn.trainNetwork(&target[0]);

            for (size_t p = 0; p < params.size(); ++p)
            {
                const double after = params[p].get();
                const std::string& group = params[p].group;
                if (resolved[p])
                {
                    report.worst[group] = std::max(report.worst[group], std::fabs(after - expected[p]) / lr);
                }
                report.moved[group] = std::max(report.moved[group], std::fabs(after - before[p]));
            }
        }

        return report;
    }

    static void expectMatch(const Report& report, const double tolerance, const char* label)
    {
        for (const auto& entry : report.worst)
        {
            BOOST_TEST_INFO(label << " group " << entry.first << ": worst |library - reference| / lr = " << entry.second);
            BOOST_TEST(entry.second < tolerance);
        }
        for (const auto& entry : report.moved)
        {
            BOOST_TEST_INFO(label << " group " << entry.first << " never moved");
            BOOST_TEST(entry.second > 0.0);
        }
    }

    // A feed-forward network's restore is a no-op: it has no state.
    static void noState()
    {
    }

    template<typename Optimizer>
    struct DoubleTF : public test_suite_tier1_features::RecurrentGradientCheckTransferFunctions
    {
        typedef Optimizer OptimizerPolicyType;
    };

    // lambda = 0.05, applied as w - lr * lambda * w (L2WeightDecay's rule).
    struct DoubleDecay
    {
        static double applyDecay(const double& weight, const double& learningRate)
        {
            return weight - learningRate * 0.05 * weight;
        }
    };

    struct DoubleClip
    {
        static double clip(const double& gradient)
        {
            return std::max(-1e-3, std::min(1e-3, gradient));
        }
    };

    template<typename Optimizer>
    struct DoubleDecayTF : public test_suite_tier1_features::RecurrentGradientCheckTransferFunctions
    {
        typedef Optimizer OptimizerPolicyType;
        typedef DoubleDecay WeightDecayPolicyType;
    };

    struct DoubleClipSgdTF : public test_suite_tier1_features::RecurrentGradientCheckTransferFunctions
    {
        typedef DoubleClip GradientClippingPolicyType;
    };

    template<typename TF, size_t HiddenLayers>
    static Report runMlp(const Hyper& hp, const double lr, const size_t steps)
    {
        typedef tinymind::MultilayerPerceptron<double, 3, HiddenLayers, 4, 2, TF> NetworkType;
        srand(RANDOM_SEED);
        static NetworkType nn;
        nn.setMomentumRate(0.0);
        nn.setAccelerationRate(0.0);
        std::vector<Param> params;
        addMlpParams<double>(nn, 3, 4, HiddenLayers, 2, params);
        return compareWithReference<NetworkType, double, 3, 2>(nn, params, hp, lr, steps, noState);
    }

    // Single-layer LSTM/GRU: input weights and biases for every gate, the
    // recurrent weights, and the hidden-to-output weights and bias. The state
    // h_{t-1} (and c_{t-1}) is warmed up once, then restored before every
    // forward pass, as in the #257 finite-difference tests.
    template<typename NetworkType, size_t Gates>
    static Report runRecurrent(const Hyper& hp, const double lr, const size_t steps)
    {
        static const size_t Inputs = 3;
        static const size_t Hidden = 4;
        srand(RANDOM_SEED);
        static NetworkType nn;
        double warm[Inputs];
        for (int s = 0; s < 3; ++s)
        {
            for (size_t i = 0; i < Inputs; ++i) warm[i] = (rand() % 2001 - 1000) / 1000.0;
            nn.feedForward(&warm[0]);
        }
        double h[Hidden];
        double c[Hidden];
        for (size_t n = 0; n < Hidden; ++n)
        {
            h[n] = nn.getLastHiddenLayer().getNeuron(n)->getOutputValue();
            c[n] = test_suite_tier1_features::CellStateAccess<Gates>::get(nn.getLastHiddenLayer().getNeuron(n));
        }
        auto restore = [&]()
        {
            nn.resetState();
            for (size_t n = 0; n < Hidden; ++n)
            {
                nn.getLastHiddenLayer().getNeuron(n)->setOutputValue(h[n]);
                test_suite_tier1_features::CellStateAccess<Gates>::set(nn.getLastHiddenLayer().getNeuron(n), c[n]);
            }
        };

        std::vector<Param> params;
        const size_t G = Gates * Hidden;
        for (size_t i = 0; i < Inputs; ++i)
        {
            for (size_t k = 0; k < G; ++k)
            {
                params.push_back({"input_w",
                    [i, k]() { return nn.getInputLayerWeightForNeuronAndConnection(i, k); },
                    [i, k](double v) { nn.setInputLayerWeightForNeuronAndConnection(i, k, v); }});
            }
        }
        for (size_t k = 0; k < G; ++k)
        {
            params.push_back({"input_b",
                [k]() { return nn.getInputLayerBiasNeuronWeightForConnection(k); },
                [k](double v) { nn.setInputLayerBiasWeightForConnection(k, v); }});
        }
        for (size_t j = 0; j < Hidden; ++j)
        {
            for (size_t k = 0; k < G; ++k)
            {
                params.push_back({"recurrent_w",
                    [j, k]() { return nn.getRecurrentLayer().getWeightForNeuronAndConnection(j, k); },
                    [j, k](double v) { nn.getRecurrentLayer().setWeightForNeuronAndConnection(j, k, v); }});
            }
            for (size_t o = 0; o < 2; ++o)
            {
                params.push_back({"hidden0_w",
                    [j, o]() { return nn.getHiddenLayerWeightForNeuronAndConnection(0, j, o); },
                    [j, o](double v) { nn.setHiddenLayerWeightForNeuronAndConnection(0, j, o, v); }});
            }
        }
        for (size_t o = 0; o < 2; ++o)
        {
            params.push_back({"hidden0_b",
                [o]() { return nn.getHiddenLayerBiasNeuronWeightForConnection(0, o); },
                [o](double v) { nn.setHiddenLayerBiasNeuronWeightForConnection(0, o, v); }});
        }

        return compareWithReference<NetworkType, double, Inputs, 2>(nn, params, hp, lr, steps, restore);
    }

    static const Hyper AdamFloat = {Rule::Adam, 0.9, 0.999, 1e-8, 0.0, 0.0};
    static const Hyper RmsPropFloat = {Rule::RmsProp, 0.0, 0.9, 1e-8, 0.0, 0.0};

    // An optimizer driven directly, one parameter, no network: enough to
    // follow the bias correction out to t = 150.
    template<typename ValueType>
    struct OneParameterLayer
    {
        ValueType weight;
        ValueType m;
        ValueType v;

        OneParameterLayer() : weight(0), m(0), v(0) {}

        ValueType getFirstMomentForNeuronAndConnection(size_t, size_t) const { return m; }
        void setFirstMomentForNeuronAndConnection(size_t, size_t, const ValueType& value) { m = value; }
        ValueType getSecondMomentForNeuronAndConnection(size_t, size_t) const { return v; }
        void setSecondMomentForNeuronAndConnection(size_t, size_t, const ValueType& value) { v = value; }
        ValueType getWeightForNeuronAndConnection(size_t, size_t) const { return weight; }
        void setWeightForNeuronAndConnection(size_t, size_t, const ValueType& value) { weight = value; }
    };
}

BOOST_AUTO_TEST_SUITE(test_suite_optimizers)

// Adam and RMSprop must update every weight AND every bias with the textbook
// rule, and Adam's bias-correction timestep must count training steps. Before
// the fix no bias moved under either optimizer, and Adam's t advanced once per
// layer, so every layer but the output one got the wrong correction at step 1.
BOOST_AUTO_TEST_CASE(test_case_adam_float_mlp_two_hidden_layers_matches_textbook)
{
    using namespace optimizer_tests;
    const Report report = runMlp<DoubleTF<tinymind::AdamOptimizerFloat<double>>, 2>(AdamFloat, 0.01, 3);
    expectMatch(report, 1e-3, "Adam 2 hidden");
    BOOST_TEST(report.worst.size() == 6u);
}

BOOST_AUTO_TEST_CASE(test_case_adam_float_mlp_three_hidden_layers_matches_textbook)
{
    using namespace optimizer_tests;
    const Report report = runMlp<DoubleTF<tinymind::AdamOptimizerFloat<double>>, 3>(AdamFloat, 0.01, 3);
    expectMatch(report, 1e-3, "Adam 3 hidden");
    BOOST_TEST(report.worst.size() == 8u);
}

BOOST_AUTO_TEST_CASE(test_case_rmsprop_float_mlp_two_hidden_layers_matches_textbook)
{
    using namespace optimizer_tests;
    const Report report = runMlp<DoubleTF<tinymind::RmsPropOptimizerFloat<double>>, 2>(RmsPropFloat, 0.01, 3);
    expectMatch(report, 1e-3, "RMSprop 2 hidden");
}

BOOST_AUTO_TEST_CASE(test_case_rmsprop_float_mlp_three_hidden_layers_matches_textbook)
{
    using namespace optimizer_tests;
    const Report report = runMlp<DoubleTF<tinymind::RmsPropOptimizerFloat<double>>, 3>(RmsPropFloat, 0.01, 3);
    expectMatch(report, 1e-3, "RMSprop 3 hidden");
}

BOOST_AUTO_TEST_CASE(test_case_adam_float_lstm_matches_textbook)
{
    using namespace optimizer_tests;
    typedef tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, DoubleTF<tinymind::AdamOptimizerFloat<double>>> NetworkType;
    const Report report = runRecurrent<NetworkType, 4>(AdamFloat, 0.01, 3);
    expectMatch(report, 1e-3, "Adam LSTM");
    BOOST_TEST(report.worst.size() == 5u);
}

BOOST_AUTO_TEST_CASE(test_case_adam_float_gru_matches_textbook)
{
    using namespace optimizer_tests;
    typedef tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, DoubleTF<tinymind::AdamOptimizerFloat<double>>> NetworkType;
    const Report report = runRecurrent<NetworkType, 3>(AdamFloat, 0.01, 3);
    expectMatch(report, 1e-3, "Adam GRU");
}

BOOST_AUTO_TEST_CASE(test_case_rmsprop_float_lstm_matches_textbook)
{
    using namespace optimizer_tests;
    typedef tinymind::LstmNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, DoubleTF<tinymind::RmsPropOptimizerFloat<double>>> NetworkType;
    const Report report = runRecurrent<NetworkType, 4>(RmsPropFloat, 0.01, 3);
    expectMatch(report, 1e-3, "RMSprop LSTM");
}

BOOST_AUTO_TEST_CASE(test_case_rmsprop_float_gru_matches_textbook)
{
    using namespace optimizer_tests;
    typedef tinymind::GruNeuralNetwork<double, 3, tinymind::HiddenLayers<4>, 2, DoubleTF<tinymind::RmsPropOptimizerFloat<double>>> NetworkType;
    const Report report = runRecurrent<NetworkType, 3>(RmsPropFloat, 0.01, 3);
    expectMatch(report, 1e-3, "RMSprop GRU");
}

// Weight decay used to be skipped entirely under Adam and RMSprop. It is now
// the same shrink SGD applies, w - lr * lambda * w, before the adaptive step
// (decoupled, as in AdamW), on weights only.
BOOST_AUTO_TEST_CASE(test_case_weight_decay_applies_under_adam_and_rmsprop)
{
    using namespace optimizer_tests;
    Hyper adam = AdamFloat;
    adam.lambda = 0.05;
    expectMatch(runMlp<DoubleDecayTF<tinymind::AdamOptimizerFloat<double>>, 2>(adam, 0.1, 3), 1e-3, "Adam + decay");

    Hyper rms = RmsPropFloat;
    rms.lambda = 0.05;
    expectMatch(runMlp<DoubleDecayTF<tinymind::RmsPropOptimizerFloat<double>>, 2>(rms, 0.1, 3), 1e-3, "RMSprop + decay");
}

// SGD clipped weight gradients but used bias gradients unclipped.
BOOST_AUTO_TEST_CASE(test_case_sgd_clips_bias_gradients)
{
    using namespace optimizer_tests;
    const Hyper sgd = {Rule::Sgd, 0.0, 0.0, 0.0, 0.0, 1e-3};
    expectMatch(runMlp<DoubleClipSgdTF, 2>(sgd, 0.1, 3), 1e-6, "SGD + clip");
}

BOOST_AUTO_TEST_CASE(test_case_gradient_clip_is_symmetric_for_fractional_bounds)
{
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> QType;
    typedef tinymind::GradientClipByValue<QType, 1, 32768> Clip; // 1.5
    const QType big(10, 0);
    const QType small = QType(static_cast<QType::FullWidthValueType>(0)) - big;
    BOOST_TEST(Clip::clip(big).getValue() == 98304);
    BOOST_TEST(Clip::clip(small).getValue() == -98304);

    typedef tinymind::GradientClipByValue<QType> UnitClip; // 1.0
    BOOST_TEST(UnitClip::clip(big).getValue() == 65536);
    BOOST_TEST(UnitClip::clip(small).getValue() == -65536);
}

// The fixed-point defaults were Q8.8 raw fractions, so in Q16.16 Adam's beta1
// was 230/65536 = 0.0035. They are now computed for each format.
BOOST_AUTO_TEST_CASE(test_case_fixed_point_defaults_are_format_correct)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> Q8;
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> Q16;
    typedef tinymind::QValue<32, 32, true, tinymind::RoundUpPolicy> Q32;

    // Q8.8 keeps its old values where they were right: 0.9 -> 230, 0.999 and
    // 1e-8 clamp to 255 and 1.
    BOOST_TEST((tinymind::QFractionLsbs<Q8, 9, 10>::value) == 230u);
    BOOST_TEST((tinymind::QFractionLsbs<Q8, 999, 1000>::value) == 255u);
    BOOST_TEST((tinymind::QFractionLsbs<Q8, 1, 100000000>::value) == 1u);
    BOOST_TEST((tinymind::QFractionLsbs<Q16, 9, 10>::value) == 58982u);
    BOOST_TEST((tinymind::QFractionLsbs<Q16, 99, 100>::value) == 64881u);
    BOOST_TEST((tinymind::QFractionLsbs<Q16, 1, 1000>::value) == 66u);
    BOOST_TEST((tinymind::QFractionLsbs<Q32, 9, 10>::value) == 3865470566ULL);
    // 64 fractional bits: long division, no 2^64 overflow.
    BOOST_TEST(tinymind::detail::fractionLsbs(9, 10, 64) == 16602069666338596454ULL);
    BOOST_TEST(tinymind::detail::fractionLsbs(1, 3, 64) == 6148914691236517205ULL);

    const double b1 = optimizer_tests::toDouble(tinymind::AdamOptimizer<Q16>::beta1());
    const double b2 = optimizer_tests::toDouble(tinymind::AdamOptimizer<Q16>::beta2());
    const double eps = optimizer_tests::toDouble(tinymind::AdamOptimizer<Q16>::epsilon());
    BOOST_TEST(std::fabs(b1 - 0.9) < 1e-4);
    BOOST_TEST(std::fabs(b2 - 0.99) < 1e-4);
    BOOST_TEST(std::fabs(eps - 2560.0 / 65536.0) < 1e-9);
    // epsilon = sqrt(LSB / (1 - beta2)) with 1 - beta2 = 1/100.
    BOOST_TEST((tinymind::QAdamEpsilonLsbs<Q16, 100>::value) == 2560u);
    BOOST_TEST((tinymind::QAdamEpsilonLsbs<Q8, 100>::value) == 160u);
    BOOST_TEST((tinymind::QAdamEpsilonLsbs<Q32, 100>::value) == 655360u);
    BOOST_TEST(tinymind::detail::adamEpsilonLsbs(100, 64) == 42949672960ULL);
    BOOST_TEST(tinymind::detail::adamEpsilonLsbs(100, 63) == tinymind::detail::isqrtRounded(200) << 31);
    BOOST_TEST(std::fabs(optimizer_tests::toDouble(tinymind::RmsPropOptimizer<Q16>::decay()) - 0.9) < 1e-4);

    // L2WeightDecay default lambda = 0.001: w * (1 - lr * lambda) with lr = 1.
    const Q16 w(1, 0);
    const Q16 lr(1, 0);
    BOOST_TEST((w - tinymind::L2WeightDecay<Q16>::applyDecay(w, lr)).getValue() == 66);

    // StepDecaySchedule default factor = 0.9 after one interval.
    tinymind::StepDecaySchedule<Q16, 1> schedule;
    BOOST_TEST(schedule.step(Q16(1, 0)).getValue() == 58982);
}

// Fixed-point Adam's bias correction stopped at t = 100, so from step 100 on
// the update jumped (1 / sqrt(1 - 0.99^100) = 1.26x with the new beta2).
// Driven with a constant gradient for 150 steps, every update must match the
// textbook update with the optimizer's own constants.
BOOST_AUTO_TEST_CASE(test_case_fixed_point_adam_bias_correction_past_t_100)
{
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> Q16;
    typedef tinymind::AdamOptimizer<Q16> Adam;
    using optimizer_tests::toDouble;

    Adam adam;
    optimizer_tests::OneParameterLayer<Q16> layer;
    const Q16 gradient(0, 16384); // 0.25
    const Q16 lr(0, 655);         // ~0.01
    const double b1 = toDouble(Adam::beta1());
    const double b2 = toDouble(Adam::beta2());
    const double eps = toDouble(Adam::epsilon());
    const double g = toDouble(gradient);
    double m = 0.0;
    double v = 0.0;
    double worstRatio = 0.0;
    for (size_t t = 1; t <= 150; ++t)
    {
        adam.step();
        const double before = toDouble(layer.weight);
        adam.updateWeights<void>(layer, 0, 0, gradient, lr);
        const double update = toDouble(layer.weight) - before;

        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        const double expected = toDouble(lr) * (m / (1.0 - std::pow(b1, static_cast<double>(t)))) / (std::sqrt(v / (1.0 - std::pow(b2, static_cast<double>(t)))) + eps);
        worstRatio = std::max(worstRatio, std::fabs(update / expected - 1.0));
    }
    BOOST_TEST(adam.getTimestep() == 150u);
    BOOST_TEST_INFO("worst relative deviation from the textbook update over t = 1..150: " << worstRatio);
    BOOST_TEST(worstRatio < 0.02);
}

// Q16.16 Adam with its defaults, on a two-hidden-layer network, against the
// textbook rule in double with the same constants and the same weights: every
// weight and bias group moves and stays within quantization error.
BOOST_AUTO_TEST_CASE(test_case_fixed_point_adam_defaults_track_float_reference)
{
    using namespace optimizer_tests;
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> Q16;
    typedef tinymind::FixedPointTransferFunctions<
        Q16,
        UniformRealRandomNumberGenerator<Q16>,
        tinymind::TanhActivationPolicy<Q16>,
        tinymind::SigmoidActivationPolicy<Q16>,
        2,
        tinymind::DefaultNetworkInitializer<Q16>,
        tinymind::MeanSquaredErrorCalculator<Q16, 2>,
        tinymind::ZeroToleranceCalculator<Q16>,
        tinymind::GradientClipByValue<Q16>,
        tinymind::NullWeightDecayPolicy<Q16>,
        tinymind::FixedLearningRatePolicy<Q16>,
        tinymind::AdamOptimizer<Q16>> TF;
    typedef tinymind::MultilayerPerceptron<Q16, 3, 2, 4, 2, TF> NetworkType;

    srand(RANDOM_SEED);
    static NetworkType nn;
    std::vector<Param> params;
    addMlpParams<Q16>(nn, 3, 4, 2, 2, params);

    const Hyper hp = {Rule::Adam,
                      toDouble(tinymind::AdamOptimizer<Q16>::beta1()),
                      toDouble(tinymind::AdamOptimizer<Q16>::beta2()),
                      toDouble(tinymind::AdamOptimizer<Q16>::epsilon()),
                      0.0,
                      1.0};
    const double lr = 655.0 / 65536.0;
    nn.setLearningRate(Q16(0, 655));

    // Gradients come from the fixed-point network's own forward pass (LUT
    // activations, Q16.16 rounding), so the finite differences use a step
    // of 2^-8 (256 LSBs) per side and the comparison allows quantization error.
    const Report report = compareWithReference<NetworkType, Q16, 3, 2>(nn, params, hp, lr, 3, noState, 1.0 / 256.0);
    for (const auto& entry : report.moved)
    {
        BOOST_TEST_INFO("Q16.16 Adam group " << entry.first << " never moved");
        BOOST_TEST(entry.second > 0.0);
    }
    for (const auto& entry : report.worst)
    {
        BOOST_TEST_INFO("Q16.16 Adam group " << entry.first << ": worst |library - reference| / lr = " << entry.second);
        BOOST_TEST(entry.second < 0.4);
    }
}

// With BatchSize > 1 the weights move once per batch, so Adam's timestep
// must advance once per batch, not once per sample (or per layer).
namespace optimizer_tests
{
    struct CountingAdam : public tinymind::AdamOptimizerFloat<double>
    {
        static size_t steps;

        void step()
        {
            ++steps;
            tinymind::AdamOptimizerFloat<double>::step();
        }
    };

    size_t CountingAdam::steps = 0;
}

BOOST_AUTO_TEST_CASE(test_case_adam_timestep_advances_once_per_batch)
{
    using namespace optimizer_tests;
    typedef tinymind::NeuralNetwork<double, 3, tinymind::HiddenLayers<4, 4>, 2, ::optimizer_tests::DoubleTF<CountingAdam>, true, 3> NetworkType;
    srand(RANDOM_SEED);
    static NetworkType nn;
    nn.setLearningRate(0.01);
    const double biasBefore = nn.getInputLayerBiasNeuronWeightForConnection(0);
    double x[3];
    double target[2];
    CountingAdam::steps = 0;
    for (size_t sample = 1; sample <= 9; ++sample)
    {
        for (size_t i = 0; i < 3; ++i) x[i] = (rand() % 2001 - 1000) / 1000.0;
        for (size_t o = 0; o < 2; ++o) target[o] = (rand() % 1001) / 1000.0;
        nn.feedForward(&x[0]);
        nn.trainNetwork(&target[0]);
        BOOST_TEST(CountingAdam::steps == sample / 3);
    }
    BOOST_TEST(nn.getInputLayerBiasNeuronWeightForConnection(0) != biasBefore);
}

BOOST_AUTO_TEST_SUITE_END()
