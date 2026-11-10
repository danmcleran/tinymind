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

// Fixed-point SGD momentum must let an undriven delta weight decay to zero.
//
// The momentum terms are momentum x (last delta) + acceleration x (the delta
// before). Rounded with the format's own policy, 0.5 x (+1 LSB) rounds back up
// to +1 LSB under RoundUpPolicy (and 0.5 x (-1 LSB) back down to -1 LSB under
// TruncatePolicy), so a one-LSB delta never decays and every such weight moves
// by an LSB on every training step, whatever the gradient. Over the hundreds
// of thousands of steps a recurrent example trains for, that drift saturated
// the Q16.16 networks. The tests seed nonzero deltas with one real training
// step, turn the learning rate off so only the momentum terms act, and require
// every weight and bias to stop moving, as they do in floating point.
//
// Included from nn_unit_test.cpp after optimizers_tests.hpp, whose
// parameter-collection helpers it reuses.

namespace momentum_rounding_tests
{
    template<typename ValueType>
    struct Network
    {
        typedef tinymind::FixedPointTransferFunctions<
            ValueType,
            UniformRealRandomNumberGenerator<ValueType>,
            tinymind::TanhActivationPolicy<ValueType>,
            tinymind::SigmoidActivationPolicy<ValueType>> TransferFunctionsType;
        typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<3, 3>, 1, TransferFunctionsType> type;
    };

    template<typename ValueType>
    static double lsb()
    {
        return 1.0 / static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    }

    // Returns the largest change, in LSBs, of any weight or bias over the
    // last `quietSteps` of `steps` momentum-only training steps.
    template<typename ValueType>
    static double driftAfterMomentumOnly(const size_t steps, const size_t quietSteps, double& firstStepMove)
    {
        typedef typename Network<ValueType>::type NetworkType;
        static NetworkType nn;

        std::vector<optimizer_tests::Param> params;
        optimizer_tests::addMlpParams<ValueType>(nn, 2, 3, 2, 1, params);

        ValueType inputs[2] = {optimizer_tests::fromDouble<ValueType>(0.5), optimizer_tests::fromDouble<ValueType>(-0.25)};
        ValueType target[1] = {optimizer_tests::fromDouble<ValueType>(0.75)};

        // The library's default rates: learning 0.25, momentum 0.5,
        // acceleration 0.0625. One real step leaves deltas of many LSBs.
        // Write each parameter back to itself: the round trip through the
        // setters must be exact, and it leaves the trajectory unchanged.
        std::vector<double> before;
        for (size_t i = 0; i < params.size(); ++i)
        {
            const double value = params[i].get();
            params[i].set(value);
            BOOST_TEST(params[i].get() == value);
            before.push_back(value);
        }
        nn.feedForward(&inputs[0]);
        nn.trainNetwork(&target[0]);
        firstStepMove = 0.0;
        for (size_t i = 0; i < params.size(); ++i)
        {
            firstStepMove = std::max(firstStepMove, std::fabs(params[i].get() - before[i]) / lsb<ValueType>());
        }

        nn.setLearningRate(ValueType(0));
        std::vector<double> snapshot;
        for (size_t step = 0; step < steps; ++step)
        {
            if (step == steps - quietSteps)
            {
                snapshot.clear();
                for (size_t i = 0; i < params.size(); ++i)
                {
                    snapshot.push_back(params[i].get());
                }
            }
            nn.feedForward(&inputs[0]);
            nn.trainNetwork(&target[0]);
        }

        double drift = 0.0;
        for (size_t i = 0; i < params.size(); ++i)
        {
            drift = std::max(drift, std::fabs(params[i].get() - snapshot[i]) / lsb<ValueType>());
        }
        return drift;
    }

    // multiplyRoundTowardZero against the exact product of the raw values.
    template<typename ValueType>
    static void checkTowardZero()
    {
        typedef typename ValueType::FullWidthFieldType RawType;
        const int64_t one = static_cast<int64_t>(1) << ValueType::NumberOfFractionalBits;
        const int64_t raws[] = {1, -1, 2, -2, 3, -3, one / 2, -(one / 2), one / 16, -(one / 16), one - 1, -(one - 1), one, -one, 3 * one / 2 + 1, -(3 * one / 2 + 1)};
        for (size_t a = 0; a < sizeof(raws) / sizeof(raws[0]); ++a)
        {
            for (size_t b = 0; b < sizeof(raws) / sizeof(raws[0]); ++b)
            {
                const ValueType left(static_cast<RawType>(raws[a]));
                const ValueType right(static_cast<RawType>(raws[b]));
                const int64_t exact = raws[a] * raws[b];
                const int64_t expected = (exact < 0) ? -((-exact) / one) : (exact / one);
                const ValueType result = tinymind::multiplyRoundTowardZero(left, right);
                BOOST_TEST(static_cast<int64_t>(result.getValue()) == expected);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE(test_suite_momentum_rounding)

BOOST_AUTO_TEST_CASE(test_case_momentum_decays_q16_16_round_up)
{
    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
    double firstStepMove = 0.0;
    const double drift = momentum_rounding_tests::driftAfterMomentumOnly<ValueType>(300, 200, firstStepMove);
    BOOST_TEST(firstStepMove > 16.0);
    BOOST_TEST(drift == 0.0);
}

BOOST_AUTO_TEST_CASE(test_case_momentum_decays_q16_16_truncate)
{
    typedef tinymind::QValue<16, 16, true, tinymind::TruncatePolicy> ValueType;
    double firstStepMove = 0.0;
    const double drift = momentum_rounding_tests::driftAfterMomentumOnly<ValueType>(300, 200, firstStepMove);
    BOOST_TEST(firstStepMove > 16.0);
    BOOST_TEST(drift == 0.0);
}

BOOST_AUTO_TEST_CASE(test_case_momentum_decays_q8_8_round_up)
{
    typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
    double firstStepMove = 0.0;
    const double drift = momentum_rounding_tests::driftAfterMomentumOnly<ValueType>(300, 200, firstStepMove);
    BOOST_TEST(firstStepMove > 1.0);
    BOOST_TEST(drift == 0.0);
}

BOOST_AUTO_TEST_CASE(test_case_multiply_round_toward_zero_is_exact)
{
    momentum_rounding_tests::checkTowardZero<tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy>>();
    momentum_rounding_tests::checkTowardZero<tinymind::QValue<16, 16, true, tinymind::TruncatePolicy>>();
    momentum_rounding_tests::checkTowardZero<tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy>>();
    momentum_rounding_tests::checkTowardZero<tinymind::QValue<24, 8, true, tinymind::TruncatePolicy>>();

    // Floating point multiplies as usual.
    BOOST_TEST(tinymind::multiplyRoundTowardZero(0.5, -0.375) == -0.1875);
}

BOOST_AUTO_TEST_SUITE_END()
