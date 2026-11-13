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

// Head-only fine-tuning: DenseHead (cpp/densehead.hpp) and PrototypeHead
// (cpp/prototypehead.hpp), the trainable and gradient-free classifiers that
// sit on top of a frozen model's embedding.
//
// The DenseHead cases check the forward pass and backward() against hand
// computation, the softmax loss against overflow in fixed point, that a head
// learns a separable task in double and Q16.16 under both loss policies, and
// that reloading the shipped weights undoes an adaptation exactly. The SGD
// step's rounding is pinned in sgd_step_rounding_tests.hpp with the other
// trainable layers. The PrototypeHead cases check the running mean, the
// count cap that turns it into a moving average, and that a class never
// learned is never predicted.
//
// Included from nn_unit_test.cpp.

namespace head_finetune_tests
{
    typedef tinymind::QValue<16, 16, true> Q16;

    static Q16 q16(const double value)
    {
        return tinymind::ValueConverter<double, Q16>::convertToDestinationType(value);
    }

    static double toDouble(const Q16& value)
    {
        return static_cast<double>(value.getValue()) / 65536.0;
    }

    template<typename ValueType>
    static ValueType make(const double value);

    template<>
    Q16 make<Q16>(const double value)
    {
        return q16(value);
    }

    template<>
    double make<double>(const double value)
    {
        return value;
    }

    // Four Gaussian-free clusters in 3-D: class c sits at a fixed centre,
    // samples are the centre plus a small deterministic jitter.
    static const double centres[4][3] = {
        {0.8, 0.1, -0.3},
        {-0.6, 0.7, 0.2},
        {0.1, -0.8, 0.6},
        {-0.2, -0.1, -0.9}};

    static void sample(PortableUniformReal& jitter, const size_t c, double* x)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            x[i] = centres[c][i] + jitter();
        }
    }

    // Trains a 3 -> 4 head on the clusters and returns its accuracy on fresh
    // samples.
    template<typename ValueType, typename LossPolicy>
    static double trainClusters(const double learningRate, const size_t steps)
    {
        typedef tinymind::DenseHead<ValueType, 3, 4, LossPolicy> HeadType;
        HeadType head;
        PortableUniformReal jitter(-0.15, 0.15, 3u);
        double x[3];
        ValueType input[3];
        ValueType inputDeltas[3];
        const ValueType rate = make<ValueType>(learningRate);
        for (size_t step = 0; step < steps; ++step)
        {
            const size_t c = step % 4;
            sample(jitter, c, x);
            for (size_t i = 0; i < 3; ++i)
            {
                input[i] = make<ValueType>(x[i]);
            }
            // Every other step also asks for the embedding gradient, which
            // must not change what the head learns.
            head.trainClass(&input[0], c, rate, ((step & 1u) != 0u) ? &inputDeltas[0] : nullptr);
        }

        size_t correct = 0;
        const size_t trials = 400;
        for (size_t t = 0; t < trials; ++t)
        {
            const size_t c = t % 4;
            sample(jitter, c, x);
            for (size_t i = 0; i < 3; ++i)
            {
                input[i] = make<ValueType>(x[i]);
            }
            correct += (head.predict(&input[0]) == c) ? 1u : 0u;
        }

        return static_cast<double>(correct) / static_cast<double>(trials);
    }
}

BOOST_AUTO_TEST_SUITE(test_suite_head_finetune)

BOOST_AUTO_TEST_CASE(test_case_dense_head_forward_and_backward)
{
    typedef tinymind::DenseHead<double, 3, 2, tinymind::SquaredErrorHeadLoss<double>> HeadType;
    HeadType head;
    const double weights[6] = {0.5, -1.0, 2.0, 0.25, 0.75, -0.5};
    const double biases[2] = {0.125, -0.25};
    head.loadWeights(&weights[0], &biases[0]);

    // Output-major layout: weights[o * 3 + i].
    BOOST_TEST(head.getWeight(0, 2) == 2.0);
    BOOST_TEST(head.getWeight(1, 0) == 0.25);
    BOOST_TEST(head.getBias(1) == -0.25);
    const size_t parameters = HeadType::NumberOfParameters;
    BOOST_TEST(parameters == 8u);

    const double input[3] = {1.0, 2.0, -1.0};
    double logits[2];
    head.forward(&input[0], &logits[0]);
    BOOST_TEST(logits[0] == 0.125 + 0.5 - 2.0 - 2.0);
    BOOST_TEST(logits[1] == -0.25 + 0.25 + 1.5 + 0.5);
    BOOST_TEST(head.predict(&input[0]) == 1u);

    const double deltas[2] = {0.5, -2.0};
    double inputDeltas[3];
    head.backward(&deltas[0], &inputDeltas[0]);
    BOOST_TEST(inputDeltas[0] == 0.5 * 0.5 + 0.25 * -2.0);
    BOOST_TEST(inputDeltas[1] == -1.0 * 0.5 + 0.75 * -2.0);
    BOOST_TEST(inputDeltas[2] == 2.0 * 0.5 + -0.5 * -2.0);

    // update(): parameter += rate x delta x input.
    head.update(&deltas[0], &input[0], 0.5);
    BOOST_TEST(head.getWeight(0, 1) == -1.0 + 0.5 * 0.5 * 2.0);
    BOOST_TEST(head.getWeight(1, 2) == -0.5 + 0.5 * -2.0 * -1.0);
    BOOST_TEST(head.getBias(0) == 0.125 + 0.5 * 0.5);
    BOOST_TEST(head.getBias(1) == -0.25 + 0.5 * -2.0);
}

BOOST_AUTO_TEST_CASE(test_case_dense_head_train_returns_prediction_before_update)
{
    typedef tinymind::DenseHead<double, 1, 2, tinymind::SquaredErrorHeadLoss<double>> HeadType;
    HeadType head;
    const double weights[2] = {1.0, 0.0};
    const double biases[2] = {0.0, 0.0};
    head.loadWeights(&weights[0], &biases[0]);
    const double input[1] = {1.0};
    double inputDeltas[1];

    // Predicts 0, learns 1 with a large step: the return is the old guess,
    // the head now predicts 1, and inputDeltas used the old weights:
    // deltas (0 - 1, 1 - 0) -> 1 x -1 + 0 x 1.
    BOOST_TEST(head.trainClass(&input[0], 1, 1.0, &inputDeltas[0]) == 0u);
    BOOST_TEST(inputDeltas[0] == -1.0);
    BOOST_TEST(head.predict(&input[0]) == 1u);
}

BOOST_AUTO_TEST_CASE(test_case_softmax_head_loss)
{
    using namespace head_finetune_tests;

    // double: probabilities sum to 1, deltas to 0, and a shift of every
    // logit changes nothing.
    const double logits[3] = {1.0, 2.0, 0.5};
    const double shifted[3] = {101.0, 102.0, 100.5};
    double p[3];
    double ps[3];
    tinymind::SoftmaxCrossEntropyHeadLoss<double>::probabilities(&logits[0], &p[0], 3);
    tinymind::SoftmaxCrossEntropyHeadLoss<double>::probabilities(&shifted[0], &ps[0], 3);
    const double denominator = std::exp(1.0) + std::exp(2.0) + std::exp(0.5);
    for (size_t k = 0; k < 3; ++k)
    {
        BOOST_TEST(std::fabs(p[k] - std::exp(logits[k]) / denominator) < 1e-12);
        BOOST_TEST(std::fabs(ps[k] - p[k]) < 1e-12);
    }
    const double target[3] = {0.0, 1.0, 0.0};
    double deltas[3];
    tinymind::SoftmaxCrossEntropyHeadLoss<double>::outputDeltas(&logits[0], &target[0], &deltas[0], 3);
    BOOST_TEST(std::fabs(deltas[0] + deltas[1] + deltas[2]) < 1e-12);
    BOOST_TEST(deltas[1] > 0.0);

    // Q16.16: the same logits match double to the exp table's accuracy, and
    // logits far beyond exp's range (e^200 overflows Q16.16 many times over)
    // still give a clean one-hot instead of a wrapped sum.
    Q16 ql[3] = {q16(1.0), q16(2.0), q16(0.5)};
    Q16 qp[3];
    tinymind::SoftmaxCrossEntropyHeadLoss<Q16>::probabilities(&ql[0], &qp[0], 3);
    for (size_t k = 0; k < 3; ++k)
    {
        BOOST_TEST(std::fabs(toDouble(qp[k]) - p[k]) < 0.01);
    }
    Q16 big[3] = {q16(200.0), q16(150.0), q16(-200.0)};
    tinymind::SoftmaxCrossEntropyHeadLoss<Q16>::probabilities(&big[0], &qp[0], 3);
    BOOST_TEST(std::fabs(toDouble(qp[0]) - 1.0) < 0.001);
    BOOST_TEST(std::fabs(toDouble(qp[1])) < 0.001);
    BOOST_TEST(std::fabs(toDouble(qp[2])) < 0.001);
}

BOOST_AUTO_TEST_CASE(test_case_dense_head_learns_clusters)
{
    using namespace head_finetune_tests;
    BOOST_TEST((trainClusters<double, tinymind::SoftmaxCrossEntropyHeadLoss<double>>(0.1, 4000)) == 1.0);
    BOOST_TEST((trainClusters<double, tinymind::SquaredErrorHeadLoss<double>>(0.05, 4000)) == 1.0);
    BOOST_TEST((trainClusters<Q16, tinymind::SoftmaxCrossEntropyHeadLoss<Q16>>(0.1, 4000)) == 1.0);
    BOOST_TEST((trainClusters<Q16, tinymind::SquaredErrorHeadLoss<Q16>>(0.05, 4000)) == 1.0);
}

BOOST_AUTO_TEST_CASE(test_case_dense_head_adapts_and_reloads)
{
    using namespace head_finetune_tests;
    typedef tinymind::DenseHead<Q16, 3, 4> HeadType;

    // Ship a head trained on the clusters.
    HeadType head;
    PortableUniformReal jitter(-0.15, 0.15, 5u);
    double x[3];
    Q16 input[3];
    for (size_t step = 0; step < 4000; ++step)
    {
        const size_t c = step % 4;
        sample(jitter, c, x);
        for (size_t i = 0; i < 3; ++i)
        {
            input[i] = q16(x[i]);
        }
        head.trainClass(&input[0], c, q16(0.1));
    }
    Q16 shippedWeights[HeadType::NumberOfWeights];
    Q16 shippedBiases[HeadType::NumberOfBiases];
    for (size_t o = 0; o < 4; ++o)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            shippedWeights[o * 3 + i] = head.getWeight(o, i);
        }
        shippedBiases[o] = head.getBias(o);
    }

    // On the device the labels of classes 0 and 1 swap (a sensor remapped):
    // the shipped head gets those wrong, a few labelled samples fix it.
    size_t wrongBefore = 0;
    for (size_t c = 0; c < 2; ++c)
    {
        sample(jitter, c, x);
        for (size_t i = 0; i < 3; ++i)
        {
            input[i] = q16(x[i]);
        }
        wrongBefore += (head.predict(&input[0]) != (1u - c)) ? 1u : 0u;
    }
    BOOST_TEST(wrongBefore == 2u);

    static const size_t relabel[4] = {1, 0, 2, 3};
    for (size_t step = 0; step < 800; ++step)
    {
        const size_t c = step % 4;
        sample(jitter, c, x);
        for (size_t i = 0; i < 3; ++i)
        {
            input[i] = q16(x[i]);
        }
        head.trainClass(&input[0], relabel[c], q16(0.1));
    }
    size_t correct = 0;
    for (size_t t = 0; t < 400; ++t)
    {
        const size_t c = t % 4;
        sample(jitter, c, x);
        for (size_t i = 0; i < 3; ++i)
        {
            input[i] = q16(x[i]);
        }
        correct += (head.predict(&input[0]) == relabel[c]) ? 1u : 0u;
    }
    BOOST_TEST(correct == 400u);

    // Reloading the shipped parameters undoes the adaptation bit for bit.
    head.loadWeights(&shippedWeights[0], &shippedBiases[0]);
    for (size_t o = 0; o < 4; ++o)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            BOOST_TEST(head.getWeight(o, i) == shippedWeights[o * 3 + i]);
        }
        BOOST_TEST(head.getBias(o) == shippedBiases[o]);
    }
}

BOOST_AUTO_TEST_CASE(test_case_prototype_head_running_mean)
{
    typedef tinymind::PrototypeHead<double, 2, 3> HeadType;
    HeadType head;
    const double probe[2] = {0.0, 0.0};

    // Nothing learned: no class to predict.
    BOOST_TEST(head.predict(&probe[0]) == 3u);

    const double a[2] = {1.0, 2.0};
    const double b[2] = {3.0, -2.0};
    const double c[2] = {2.0, 0.0};
    head.learn(&a[0], 1);
    head.learn(&b[0], 1);
    head.learn(&c[0], 1);
    BOOST_TEST(head.getCount(1) == 3u);
    BOOST_TEST(std::fabs(head.getPrototype(1, 0) - 2.0) < 1e-12);
    BOOST_TEST(std::fabs(head.getPrototype(1, 1) - 0.0) < 1e-12);

    // Only class 1 is known, so even a far input maps to it.
    const double far[2] = {-50.0, 50.0};
    BOOST_TEST(head.predict(&far[0]) == 1u);

    const double d[2] = {-4.0, 4.0};
    head.learn(&d[0], 2);
    BOOST_TEST(head.predict(&far[0]) == 2u);
    BOOST_TEST(head.predict(&c[0]) == 1u);
    BOOST_TEST(head.distance(&c[0], 2) == 36.0 + 16.0);

    head.forget(2);
    BOOST_TEST(head.getCount(2) == 0u);
    BOOST_TEST(head.predict(&far[0]) == 1u);
}

BOOST_AUTO_TEST_CASE(test_case_prototype_head_count_cap)
{
    typedef tinymind::PrototypeHead<double, 1, 1> HeadType;
    HeadType head;
    head.setMaximumCount(4);
    const double zero[1] = {0.0};
    const double one[1] = {1.0};

    // Four zeros fill the cap; every later sample moves the prototype by a
    // quarter of the way, an exponential moving average.
    for (size_t i = 0; i < 4; ++i)
    {
        head.learn(&zero[0], 0);
    }
    BOOST_TEST(head.getCount(0) == 4u);
    head.learn(&one[0], 0);
    BOOST_TEST(head.getCount(0) == 4u);
    BOOST_TEST(head.getPrototype(0, 0) == 0.25);
    head.learn(&one[0], 0);
    BOOST_TEST(head.getPrototype(0, 0) == 0.25 + 0.75 * 0.25);

    // Lowering the cap clamps the counts; 0 means 1 (track the last sample).
    head.setMaximumCount(0);
    BOOST_TEST(head.getMaximumCount() == 1u);
    BOOST_TEST(head.getCount(0) == 1u);
    head.learn(&zero[0], 0);
    BOOST_TEST(head.getPrototype(0, 0) == 0.0);

    // setPrototype loads a shipped prototype with its weight.
    head.setMaximumCount(100);
    const double shipped[1] = {2.0};
    head.setPrototype(0, &shipped[0], 3);
    head.learn(&zero[0], 0);
    BOOST_TEST(head.getCount(0) == 4u);
    BOOST_TEST(head.getPrototype(0, 0) == 1.5);
}

BOOST_AUTO_TEST_CASE(test_case_prototype_head_fixed_point)
{
    using namespace head_finetune_tests;
    typedef tinymind::PrototypeHead<Q16, 3, 4> EuclideanHead;
    typedef tinymind::PrototypeHead<Q16, 3, 4, tinymind::ManhattanDistance<Q16>> ManhattanHead;
    EuclideanHead euclidean;
    ManhattanHead manhattan;
    PortableUniformReal jitter(-0.15, 0.15, 9u);
    double x[3];
    Q16 input[3];
    double sums[4][3] = {{0.0}};
    const size_t perClass = 50;
    for (size_t step = 0; step < 4 * perClass; ++step)
    {
        const size_t c = step % 4;
        sample(jitter, c, x);
        for (size_t i = 0; i < 3; ++i)
        {
            input[i] = q16(x[i]);
            sums[c][i] += toDouble(input[i]);
        }
        euclidean.learn(&input[0], c);
        manhattan.learn(&input[0], c);
    }

    // The running mean in Q16.16 stays within a few LSBs of the exact mean
    // of the same quantized samples.
    for (size_t c = 0; c < 4; ++c)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            const double exact = sums[c][i] / static_cast<double>(perClass);
            BOOST_TEST(std::fabs(toDouble(euclidean.getPrototype(c, i)) - exact) < 8.0 / 65536.0);
        }
    }

    size_t correctEuclidean = 0;
    size_t correctManhattan = 0;
    for (size_t t = 0; t < 400; ++t)
    {
        const size_t c = t % 4;
        sample(jitter, c, x);
        for (size_t i = 0; i < 3; ++i)
        {
            input[i] = q16(x[i]);
        }
        correctEuclidean += (euclidean.predict(&input[0]) == c) ? 1u : 0u;
        correctManhattan += (manhattan.predict(&input[0]) == c) ? 1u : 0u;
    }
    BOOST_TEST(correctEuclidean == 400u);
    BOOST_TEST(correctManhattan == 400u);
}

BOOST_AUTO_TEST_SUITE_END()
