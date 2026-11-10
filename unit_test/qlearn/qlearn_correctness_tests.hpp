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

// Correctness tests for the Q-learning update itself, not only for the policy
// it ends up with. The 6-room maze is easy enough that a wrong Q table can
// still rank the actions correctly, so these compare the learned values with
// Q* from value iteration and check single updates against the textbook
// formula. Included from qlearn_unit_test.cpp inside test_suite_qlearn.

#pragma once

#include <algorithm>
#include <cmath>

namespace qlearn_correctness
{
    static const size_t GoalState = 5;

    static double toDouble(const QValueType& value)
    {
        return static_cast<double>(value.getValue()) / 65536.0;
    }

    // Q*(s, a) = r(s, a) + gamma * max over the valid actions a' of s' of
    // Q*(s', a'), with s' = a in this maze. Value iteration in double.
    template<typename LearnerType>
    static void computeOptimalQ(LearnerType& learner, const double gamma, double qStar[NUMBER_OF_STATES][NUMBER_OF_ACTIONS], bool valid[NUMBER_OF_STATES][NUMBER_OF_ACTIONS])
    {
        double reward[NUMBER_OF_STATES][NUMBER_OF_ACTIONS];

        for (size_t s = 0; s < NUMBER_OF_STATES; ++s)
        {
            for (size_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
            {
                const QValueType r = learner.getEnvironment().getRewardForStateAndAction(static_cast<state_t>(s), static_cast<action_t>(a));
                valid[s][a] = (r != MazeEnvironmentType::EnvironmentInvalidActionValue);
                reward[s][a] = toDouble(r);
                qStar[s][a] = 0.0;
            }
        }

        for (unsigned iteration = 0; iteration < 5000; ++iteration)
        {
            double next[NUMBER_OF_STATES][NUMBER_OF_ACTIONS];
            for (size_t s = 0; s < NUMBER_OF_STATES; ++s)
            {
                for (size_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
                {
                    next[s][a] = 0.0;
                    if (!valid[s][a])
                    {
                        continue;
                    }
                    const size_t nextState = a;
                    bool any = false;
                    double best = 0.0;
                    for (size_t nextAction = 0; nextAction < NUMBER_OF_ACTIONS; ++nextAction)
                    {
                        if (valid[nextState][nextAction] && (!any || (qStar[nextState][nextAction] > best)))
                        {
                            best = qStar[nextState][nextAction];
                            any = true;
                        }
                    }
                    next[s][a] = reward[s][a] + gamma * best;
                }
            }
            std::copy(&next[0][0], &next[0][0] + (NUMBER_OF_STATES * NUMBER_OF_ACTIONS), &qStar[0][0]);
        }
    }

    // Episodes from random start rooms (the goal included, so its own row is
    // learned too), each ended by reaching the goal or by the step cap.
    template<typename LearnerType>
    static void runEpisodes(LearnerType& learner, const unsigned episodes, std::default_random_engine& engine)
    {
        std::uniform_int_distribution<size_t> stateDistribution(0, NUMBER_OF_STATES - 1);
        typename LearnerType::experience_t experience;

        learner.getEnvironment().setGoalState(GoalState);
        for (unsigned episode = 0; episode < episodes; ++episode)
        {
            learner.startNewEpisode();
            learner.setState(static_cast<state_t>(stateDistribution(engine)));
            size_t steps = 0;
            do
            {
                const state_t state = learner.getState();
                const action_t action = learner.takeAction(state);
                experience.state = state;
                experience.action = action;
                experience.reward = learner.getEnvironment().getRewardForStateAndAction(state, action);
                experience.newState = static_cast<state_t>(action);
                learner.updateFromExperience(experience);
                ++steps;
            } while ((learner.getState() != GoalState) && (steps < MAX_STEPS_PER_EPISODE));
        }
    }

    // Largest and RMS distance of the learned Q values from Q* over every
    // valid (state, action) pair.
    template<typename LearnerType>
    static void distanceFromOptimal(LearnerType& learner, const double gamma, double& maxError, double& rmsError)
    {
        double qStar[NUMBER_OF_STATES][NUMBER_OF_ACTIONS];
        bool valid[NUMBER_OF_STATES][NUMBER_OF_ACTIONS];
        double sumSquares = 0.0;
        size_t count = 0;

        computeOptimalQ(learner, gamma, qStar, valid);
        maxError = 0.0;
        for (size_t s = 0; s < NUMBER_OF_STATES; ++s)
        {
            for (size_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
            {
                if (!valid[s][a])
                {
                    continue;
                }
                const double error = toDouble(learner.getQValue(static_cast<state_t>(s), static_cast<action_t>(a))) - qStar[s][a];
                maxError = std::max(maxError, std::fabs(error));
                sumSquares += error * error;
                ++count;
            }
        }
        rmsError = std::sqrt(sumSquares / static_cast<double>(count));
    }

    // One-hot DQN: one input per room and a linear output layer, so the
    // network can represent Q* (which reaches 1 / (1 - 0.8) = 5 with a reward
    // of 1). The scalar-input DQN above cannot, which is why it is only used
    // for machinery tests.
    template<typename ValueType, typename RewardPolicyType>
    struct OneHotDQNMazeEnvironment : public DQNMazeEnvironment<state_t, action_t, ValueType, NUMBER_OF_STATES, NUMBER_OF_ACTIONS, RewardPolicyType>
    {
        typedef DQNMazeEnvironment<state_t, action_t, ValueType, NUMBER_OF_STATES, NUMBER_OF_ACTIONS, RewardPolicyType> BaseType;
        typedef typename BaseType::ParentType ParentType;

        OneHotDQNMazeEnvironment(const ValueType& learningRate, const ValueType& discountFactor, const size_t randomActionDecisionPoint) :
            BaseType(learningRate, discountFactor, randomActionDecisionPoint)
        {
        }

        static void getInputValues(const state_t state, ValueType* pInputs)
        {
            for (size_t i = 0; i < NUMBER_OF_STATES; ++i)
            {
                pInputs[i] = ValueType((i == state) ? 1 : 0, 0);
            }
        }
    };

    typedef OneHotDQNMazeEnvironment<QValueType, QTableRewardPolicyType> OneHotEnvironmentType;
    typedef tinymind::FixedPointTransferFunctions<  QValueType,
                                                    UniformRealRandomNumberGenerator<QValueType>,
                                                    tinymind::TanhActivationPolicy<QValueType>,
                                                    tinymind::LinearActivationPolicy<QValueType>,
                                                    NUMBER_OF_ACTIONS> OneHotTransferFunctionsType;
    typedef tinymind::MultilayerPerceptron< QValueType,
                                            NUMBER_OF_STATES,
                                            1,
                                            NUMBER_OF_ACTIONS + 2,
                                            NUMBER_OF_ACTIONS,
                                            OneHotTransferFunctionsType> OneHotNetworkType;
    typedef tinymind::QValueNeuralNetworkPolicy<OneHotEnvironmentType, OneHotNetworkType, 10> OneHotQValuePolicyType;
    typedef tinymind::QLearner<OneHotEnvironmentType, OneHotQValuePolicyType> OneHotDQNLearnerType;
}

BOOST_AUTO_TEST_CASE(test_qlearn_tabular_converges_to_optimal_q)
{
    // Q-learning is off-policy: with every action chosen at random, the
    // update's max over next actions still drives the table to Q*. A
    // bootstrap from the exploring policy's action instead learns the value
    // of the random policy and stays far from Q* however long it trains.
    using namespace qlearn_correctness;
    static QLearnerType learner(learningRate, discountFactor, 100);
    std::default_random_engine engine(RANDOM_SEED);
    double maxError;
    double rmsError;

    initializeRewardMatrix(learner);
    // The goal's own row is only updated in episodes that start there, and
    // its self-loop shrinks the error by just 4% per update (1 - alpha +
    // alpha * gamma = 0.96), so it needs thousands of episodes.
    runEpisodes(learner, 10000, engine);
    distanceFromOptimal(learner, toDouble(discountFactor), maxError, rmsError);

    // Q* reaches 100 / (1 - 0.8) = 500. Each Q16.16 update rounds by about
    // one LSB (1.5e-5); 0.05 is 1e-4 of the largest value and far above that
    // rounding, far below the error of any wrong bootstrap.
    BOOST_TEST_MESSAGE("tabular Q vs Q*: max " << maxError << ", RMS " << rmsError);
    BOOST_TEST(maxError < 0.05);
}

BOOST_AUTO_TEST_CASE(test_qlearn_single_update_uses_the_max_while_exploring)
{
    // With exploration on, every single update must still equal
    // (1 - alpha) * Q(s, a) + alpha * (r + gamma * max_a' Q(s', a')).
    using namespace qlearn_correctness;
    static QLearnerType learner(learningRate, discountFactor, 100);
    std::default_random_engine engine(RANDOM_SEED);
    std::uniform_int_distribution<int> valueDistribution(-200, 200);
    std::uniform_int_distribution<size_t> stateDistribution(0, NUMBER_OF_STATES - 1);
    const QValueType one(1, 0);
    action_t validActions[NUMBER_OF_ACTIONS];
    size_t mismatches = 0;

    initializeRewardMatrix(learner);
    for (unsigned trial = 0; trial < 500; ++trial)
    {
        for (state_t s = 0; s < NUMBER_OF_STATES; ++s)
        {
            for (action_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
            {
                learner.setQValue(s, a, QValueType(valueDistribution(engine), 0) / QValueType(4, 0));
            }
        }

        const state_t state = static_cast<state_t>(stateDistribution(engine));
        const size_t numberOfValidActions = learner.getEnvironment().getValidActionsForState(state, &validActions[0]);
        std::uniform_int_distribution<size_t> actionDistribution(0, numberOfValidActions - 1);
        const action_t action = validActions[actionDistribution(engine)];
        const state_t nextState = static_cast<state_t>(action);
        action_t nextActions[NUMBER_OF_ACTIONS];
        const size_t numberOfNextActions = learner.getEnvironment().getValidActionsForState(nextState, &nextActions[0]);
        QValueType best = learner.getQValue(nextState, nextActions[0]);
        for (size_t i = 1; i < numberOfNextActions; ++i)
        {
            best = std::max(best, learner.getQValue(nextState, nextActions[i]));
        }

        typename QLearnerType::experience_t experience;
        experience.state = state;
        experience.action = action;
        experience.reward = learner.getEnvironment().getRewardForStateAndAction(state, action);
        experience.newState = nextState;
        const QValueType expected = ((one - learningRate) * learner.getQValue(state, action)) + (learningRate * (experience.reward + (discountFactor * best)));

        learner.setState(state);
        learner.updateFromExperience(experience);
        if (learner.getQValue(state, action) != expected)
        {
            ++mismatches;
        }
    }

    BOOST_TEST(mismatches == 0u);
}

BOOST_AUTO_TEST_CASE(test_qlearn_update_uses_the_experience_state)
{
    // updateFromExperience updates Q(experience.state, experience.action),
    // whatever state the learner was last in, and then moves to newState.
    using namespace qlearn_correctness;
    static QLearnerType learner(learningRate, discountFactor, 0);
    const QValueType zero(0, 0);

    initializeRewardMatrix(learner);
    learner.setQValue(4, 5, QValueType(10, 0));
    learner.setState(1);

    typename QLearnerType::experience_t experience;
    experience.state = 3;
    experience.action = 4;
    experience.reward = zero;
    experience.newState = 4;
    learner.updateFromExperience(experience);

    const QValueType expected = learningRate * (discountFactor * QValueType(10, 0));
    BOOST_TEST(learner.getQValue(3, 4) == expected);
    BOOST_TEST(learner.getQValue(1, 4) == zero);
    BOOST_TEST(learner.getQValue(1, 3) == zero);
    BOOST_TEST(learner.getState() == static_cast<state_t>(4));
}

BOOST_AUTO_TEST_CASE(test_qlearn_random_action_probability_is_p_percent)
{
    static QLearnerType learner(learningRate, discountFactor, 0);
    const size_t draws = 200000;
    const size_t points[] = { 1, 10, 50, 99 };

    for (size_t i = 0; i < (sizeof(points) / sizeof(points[0])); ++i)
    {
        learner.getEnvironment().setRandomActionDecisionPoint(points[i]);
        size_t randomCount = 0;
        for (size_t draw = 0; draw < draws; ++draw)
        {
            if (learner.getEnvironment().shouldChooseRandomAction())
            {
                ++randomCount;
            }
        }
        const double expected = static_cast<double>(points[i]) / 100.0;
        const double measured = static_cast<double>(randomCount) / static_cast<double>(draws);
        // Five binomial standard deviations; an off-by-one of 1/101 in the
        // probability is 4 to 40 standard deviations away.
        const double tolerance = 5.0 * std::sqrt(expected * (1.0 - expected) / static_cast<double>(draws));
        BOOST_TEST_MESSAGE("decision point " << points[i] << ": " << measured);
        BOOST_TEST(std::fabs(measured - expected) < tolerance);
    }
}

BOOST_AUTO_TEST_CASE(test_qlearn_rates_of_one_are_accepted)
{
    const QValueType one(1, 0);
    const QValueType two(2, 0);
    const QValueType zero(0, 0);
    const QValueType half = one / two;
    static QLearnerType learner(one, one, 0);

    BOOST_TEST(learner.getEnvironment().getLearningRate() == one);
    BOOST_TEST(learner.getEnvironment().getDiscountFactor() == one);

    // Out of range is ignored; the previous value stays.
    learner.getEnvironment().setLearningRate(two);
    learner.getEnvironment().setDiscountFactor(zero - half);
    BOOST_TEST(learner.getEnvironment().getLearningRate() == one);
    BOOST_TEST(learner.getEnvironment().getDiscountFactor() == one);

    learner.getEnvironment().setLearningRate(zero);
    learner.getEnvironment().setDiscountFactor(half);
    BOOST_TEST(learner.getEnvironment().getLearningRate() == zero);
    BOOST_TEST(learner.getEnvironment().getDiscountFactor() == half);
}

BOOST_AUTO_TEST_CASE(test_qlearn_two_tied_actions_are_chosen_at_random)
{
    static QLearnerType learner(learningRate, discountFactor, 0);
    size_t chosen[NUMBER_OF_ACTIONS] = { 0 };
    const state_t state = 0;

    for (action_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
    {
        learner.getEnvironment().setRewardForStateAndAction(state, a, MazeEnvironmentType::EnvironmentInvalidActionValue);
    }
    learner.getEnvironment().setRewardForStateAndAction(state, 1, MazeEnvironmentType::EnvironmentNoRewardValue);
    learner.getEnvironment().setRewardForStateAndAction(state, 2, MazeEnvironmentType::EnvironmentNoRewardValue);

    for (unsigned i = 0; i < 1000; ++i)
    {
        ++chosen[learner.takeAction(state)];
    }

    BOOST_TEST(chosen[1] + chosen[2] == 1000u);
    BOOST_TEST(chosen[1] > 400u);
    BOOST_TEST(chosen[2] > 400u);
}

BOOST_AUTO_TEST_CASE(test_qlearn_dqn_target_network_starts_as_a_copy)
{
    static QValuePolicyType policy;

    for (state_t state = 0; state < NUMBER_OF_STATES; ++state)
    {
        for (action_t action = 0; action < NUMBER_OF_ACTIONS; ++action)
        {
            BOOST_TEST(policy.getQValue(state, action) == policy.getFutureQValue(state, action));
        }
    }
}

BOOST_AUTO_TEST_CASE(test_qlearn_dqn_trains_on_a_small_error)
{
    // A target 0.05 above the current output must still train the network
    // toward it; a tolerance on the mean error over all six outputs skipped
    // every update closer than about 0.22.
    static QValuePolicyType policy;
    const state_t state = 2;
    const action_t action = 3;
    const QValueType before = policy.getQValue(state, action);
    const QValueType target = before + (QValueType(1, 0) / QValueType(20, 0));

    policy.setQValue(state, action, target);
    const QValueType after = policy.getQValue(state, action);

    BOOST_TEST(after > before);
    BOOST_TEST(after <= target);
}

BOOST_AUTO_TEST_CASE(test_qlearn_dqn_converges_to_optimal_q)
{
    // A one-hot DQN trained with every action random must approach Q*, not
    // freeze early or learn the random policy's values.
    using namespace qlearn_correctness;
    static OneHotDQNLearnerType learner(learningRate, discountFactor, 100);
    std::default_random_engine engine(RANDOM_SEED);
    const QValueType rewardOfOne(1, 0);
    double maxError;
    double rmsError;

    initializeRewardMatrix(learner);
    learner.getEnvironment().setRewardForStateAndAction(1, 5, rewardOfOne);
    learner.getEnvironment().setRewardForStateAndAction(4, 5, rewardOfOne);
    learner.getEnvironment().setRewardForStateAndAction(5, 5, rewardOfOne);

    runEpisodes(learner, 3000, engine);
    distanceFromOptimal(learner, toDouble(discountFactor), maxError, rmsError);

    // Q* runs from 2.05 to 5. A function approximator with a fixed learning
    // rate and target-network lag keeps some error (about 0.02 RMS here);
    // 0.1 RMS is 2% of the largest value, against about 3 RMS when training
    // stopped early or the bootstrap followed the random policy.
    BOOST_TEST_MESSAGE("DQN Q vs Q*: max " << maxError << ", RMS " << rmsError);
    BOOST_TEST(rmsError < 0.1);
}
