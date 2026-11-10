/**
* Copyright (c) 2020 Intel Corporation
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

#pragma once
#include <cstdint>

#include <cstddef>

#include "constants.hpp"
#include "limits.hpp"
#include "signed.hpp"
#include "neuralnet.hpp"

namespace tinymind {
    template<typename StateType, typename ActionType, typename ValueType, size_t NumberOfStates, size_t NumberOfActions>
    class NullRewardPolicy
    {
    public:
        ValueType getRewardForStateAndAction(const StateType state, const ActionType action) const
        {
            (void)state; // Unused parameter
            (void)action; // Unused parameter
            return 0;
        }

        void setRewardForStateAndAction(const StateType state, const ActionType action, const ValueType& reward)
        {
            (void)state; // Unused parameter
            (void)action; // Unused parameter
            (void)reward; // Unused parameter
        }
    };

    template<typename StateType, typename ActionType, typename ValueType, size_t NumberOfStates, size_t NumberOfActions>
    class QTableRewardPolicy
    {
    public:
        QTableRewardPolicy()
        {
            size_t bufferIndex = 0;

            for(StateType state = 0;state < NumberOfStates;++state)
            {
                for(ActionType action = 0;action < NumberOfActions;++action)
                {
                    new(&this->mRewardTableBuffer[bufferIndex]) ValueType(0);
                    bufferIndex += sizeof(ValueType);
                }
            }
        }

        ValueType getRewardForStateAndAction(const StateType state, const ActionType action) const
        {
            size_t stateIndex;
            size_t actionIndex;
            size_t bufferIndex;

            if((state < NumberOfStates) && (action < NumberOfActions))
            {
                stateIndex = (state * NumberOfActions * sizeof(ValueType));
                actionIndex = (action * sizeof(ValueType));
                bufferIndex = (stateIndex + actionIndex);
                const ValueType* pValue = reinterpret_cast<const ValueType*>(&this->mRewardTableBuffer[bufferIndex]);
                return *pValue;
            }
            else
            {
                return 0;
            }
        }

        void setRewardForStateAndAction(const StateType state, const ActionType action, const ValueType& reward)
        {
            ValueType* pValue;
            size_t stateIndex;
            size_t actionIndex;
            size_t bufferIndex;

            if((state < NumberOfStates) && (action < NumberOfActions))
            {
                stateIndex = (state * NumberOfActions * sizeof(ValueType));
                actionIndex = (action * sizeof(ValueType));
                bufferIndex = (stateIndex + actionIndex);
                pValue = reinterpret_cast<ValueType*>(&this->mRewardTableBuffer[bufferIndex]);
                *pValue = reward;
            }
        }
    private:
        alignas(ValueType) unsigned char mRewardTableBuffer[sizeof(ValueType) * NumberOfStates * NumberOfActions];
    };

    template<typename ValueType>
    class NullLearningPolicy
    {
    public:
        ValueType getDiscountFactor() const
        {
            return 0;
        }

        ValueType getLearningRate() const
        {
            return 0;
        }

        void setDiscountFactor(const ValueType& discountFactor)
        {
            (void)discountFactor; // Unused parameter
        }

        void setLearningRate(const ValueType& learningRate)
        {
            (void)learningRate; // Unused parameter
        }
    };

    template<typename ValueType>
    class DefaultLearningPolicy
    {
    public:
        DefaultLearningPolicy() : mLearningRate(0), mDiscountFactor(0) {}

        ValueType getDiscountFactor() const
        {
            return this->mDiscountFactor;
        }

        ValueType getLearningRate() const
        {
            return this->mLearningRate;
        }

        // Both rates must lie in [0, 1]; 1 is valid (alpha = 1 replaces the
        // old estimate outright, gamma = 1 is an undiscounted return). A value
        // outside the range is ignored and the previous one is kept.
        void setDiscountFactor(const ValueType& discountFactor)
        {
            if(isInUnitInterval(discountFactor))
            {
                this->mDiscountFactor = discountFactor;
            }
        }

        void setLearningRate(const ValueType& learningRate)
        {
            if(isInUnitInterval(learningRate))
            {
                this->mLearningRate = learningRate;
            }
        }
    private:
        static bool isInUnitInterval(const ValueType& value)
        {
            return (!(value < tinymind::Constants<ValueType>::zero()) && !(tinymind::Constants<ValueType>::one() < value));
        }

        ValueType mLearningRate;
        ValueType mDiscountFactor;
    };

    template<   typename StateType,
                typename ActionType,
                typename ValueType,
                size_t NumberOfStates,
                size_t NumberOfActions,
                template<typename> class QLearningRandomNumberPolicy,
                template<typename> class QLearningPolicy = DefaultLearningPolicy
            >
    class QLearningEnvironment
    {
    public:
        typedef StateType EnvironmentStateType;
        typedef ActionType EnvironmentActionType;
        typedef ValueType EnvironmentValueType;
        typedef QLearningRandomNumberPolicy<ActionType> RandomNumberPolicy;
        typedef QLearningPolicy<ValueType> LearningPolicy;

        static const size_t EnvironmentNumberOfStates = NumberOfStates;
        static const size_t EnvironmentNumberOfActions = NumberOfActions;
        static const StateType EnvironmentInvalidState = tinymind::limits<StateType>::max;
        static const ActionType EnvironmentInvalidAction = tinymind::limits<ActionType>::max;

        struct experience_t
        {
            experience_t() : state(EnvironmentInvalidState), action(EnvironmentInvalidAction), reward(0), newState(EnvironmentInvalidState) {};

            experience_t(const experience_t& other)
            {
                this->state = other.state;
                this->action = other.action;
                this->reward = other.reward;
                this->newState = other.newState;
            }

            experience_t& operator=(const experience_t& other)
            {
                this->state = other.state;
                this->action = other.action;
                this->reward = other.reward;
                this->newState = other.newState;

                return *this;
            }

            StateType state;
            ActionType action;
            ValueType reward;
            StateType newState;
        };

        QLearningEnvironment(const ValueType& learningRate, const ValueType& discountFactor, const size_t randomActionDecisionPoint)
        {
            this->setLearningRate(learningRate);
            this->setDiscountFactor(discountFactor);
            this->setRandomActionDecisionPoint(randomActionDecisionPoint);
        }

        ActionType chooseRandomActionFromValidActions(ActionType const* const validActions, const size_t numberOfValidActions)
        {
            ActionType actionIndex;
            ActionType action;

            switch(numberOfValidActions)
            {
                case 0:
                    action = EnvironmentInvalidAction;
                    break;
                case 1:
                    action = validActions[0];
                    break;
                default:
                    actionIndex = this->mRandomNumberPolicy.randInt(0, static_cast<ActionType>(numberOfValidActions - 1));
                    action = validActions[actionIndex];
                    break;
            }

            return action;
        }

        ValueType getDiscountFactor() const
        {
            return this->mLearningPolicy.getDiscountFactor();
        }

        ValueType getInitialQTableValue() const
        {
            return static_cast<ValueType>(0);
        }

        ValueType getLearningRate() const
        {
            return this->mLearningPolicy.getLearningRate();
        }

        size_t getRandomActionDecisionPoint() const
        {
            return this->mRandomNumberPolicy.getRandomActionDecisionPoint();
        }

        void setDiscountFactor(const ValueType& discountFactor)
        {
            this->mLearningPolicy.setDiscountFactor(discountFactor);
        }

        void setLearningRate(const ValueType& learningRate)
        {
            this->mLearningPolicy.setLearningRate(learningRate);
        }

        void setRandomActionDecisionPoint(const size_t randomActionDecisionPoint)
        {
            this->mRandomNumberPolicy.setRandomActionDecisionPoint(randomActionDecisionPoint);
        }

        bool shouldChooseRandomAction()
        {
            const size_t randomActionDecisionPoint = this->mRandomNumberPolicy.getRandomActionDecisionPoint();
            size_t value;
            bool chooseRandomAction;
            
            if (0 == randomActionDecisionPoint)
            {
                return false;
            }

            if (100 == randomActionDecisionPoint)
            {
                return true;
            }

            // randInt is inclusive at both ends: 100 equally likely values
            // 0..99, of which randomActionDecisionPoint are below the point,
            // so a random action is chosen with probability exactly p/100.
            value = this->mRandomNumberPolicy.randInt(0, 99);
            chooseRandomAction = (value < randomActionDecisionPoint);
            
            return chooseRandomAction;
        }
    protected:
        RandomNumberPolicy mRandomNumberPolicy;
        LearningPolicy mLearningPolicy;
    };

    template<typename EnvironmentType, typename QLearnerQValuePolicy>
    class ArgMaxPolicy
    {
    public:
        typedef typename EnvironmentType::EnvironmentStateType StateType;
        typedef typename EnvironmentType::EnvironmentActionType ActionType;
        typedef typename EnvironmentType::EnvironmentValueType ValueType;

        static ActionType selectBestActionForState(const StateType state, ActionType const* const actions, const size_t numberOfValidActions, const QLearnerQValuePolicy& qValuePolicy)
        {
            ActionType bestAction;
            ValueType bestQValue;
            ValueType qValue;
            ValueType qValue0;
            ValueType qValue1;
            ActionType action;
            size_t actionIndex; // index/counter over actions[]; matches numberOfValidActions' type
            bool allTied;

            switch(numberOfValidActions)
            {
                case 0:
                    bestAction = EnvironmentType::EnvironmentInvalidAction;
                    break;
                case 1:
                    bestAction = actions[0];
                    break;
                case 2:
                    qValue0 = qValuePolicy.getQValue(state, actions[0]);
                    qValue1 = qValuePolicy.getQValue(state, actions[1]);
                    if(qValue0 > qValue1)
                    {
                        bestAction = actions[0];
                    }
                    else if(qValue1 > qValue0)
                    {
                        bestAction = actions[1];
                    }
                    else
                    {
                        // A tie is no preference, as below.
                        bestAction = EnvironmentType::EnvironmentInvalidAction;
                    }
                    break;
                default:
                    // Largest Q value, whatever its sign. Only when every valid
                    // action ties (e.g. an unvisited state of a zero-initialised
                    // Q table) is there no preference, and InvalidAction tells
                    // the learner to pick a random action instead (and its
                    // bootstrap to use any of them, since all are the max).
                    bestAction = actions[0];
                    bestQValue = qValuePolicy.getQValue(state, bestAction);
                    allTied = true;
                    for(actionIndex = 1;actionIndex < numberOfValidActions;++actionIndex)
                    {
                        action = actions[actionIndex];
                        qValue = qValuePolicy.getQValue(state, action);
                        if(qValue != bestQValue)
                        {
                            allTied = false;
                        }
                        if(qValue > bestQValue)
                        {
                            bestQValue = qValue;
                            bestAction = action;
                        }
                    }
                    if(allTied)
                    {
                        bestAction = EnvironmentType::EnvironmentInvalidAction;
                    }
                    break;
            }
            
            return bestAction;
        }
    };

    template<typename EnvironmentType>
    class QValueTablePolicy
    {
    public:
        typedef typename EnvironmentType::EnvironmentStateType StateType;
        typedef typename EnvironmentType::EnvironmentActionType ActionType;
        typedef typename EnvironmentType::EnvironmentValueType ValueType;

        static const size_t NumberOfStates = EnvironmentType::EnvironmentNumberOfStates;
        static const size_t NumberOfActions = EnvironmentType::EnvironmentNumberOfActions;

        QValueTablePolicy()
        {
        }

        ValueType getQValue(const StateType state, const ActionType action) const
        {
            size_t stateIndex;
            size_t actionIndex;
            size_t bufferIndex;

            if((state < NumberOfStates) && (action < NumberOfActions))
            {
                stateIndex = (state * NumberOfActions * sizeof(ValueType));
                actionIndex = (action * sizeof(ValueType));
                bufferIndex = (stateIndex + actionIndex);
                const ValueType* pValue = reinterpret_cast<const ValueType*>(&this->mQTableBuffer[bufferIndex]);
                return *pValue;
            }
            else
            {
                return 0;
            }
        }

        ValueType getFutureQValue(const StateType state, const ActionType action) const
        {
            return this->getQValue(state, action);
        }

        void init(const EnvironmentType& environment)
        {
            for(StateType state = 0;state < NumberOfStates;++state)
            {
                for(ActionType action = 0;action < NumberOfActions;++action)
                {
                    this->setQValue(state, action, environment.getInitialQTableValue());
                }
            }
        }

        void setQValue(const StateType state, const ActionType action, const ValueType& value)
        {
            ValueType* pValue;
            size_t stateIndex;
            size_t actionIndex;
            size_t bufferIndex;

            if((state < NumberOfStates) && (action < NumberOfActions))
            {
                stateIndex = (state * NumberOfActions * sizeof(ValueType));
                actionIndex = (action * sizeof(ValueType));
                bufferIndex = (stateIndex + actionIndex);
                pValue = reinterpret_cast<ValueType*>(&this->mQTableBuffer[bufferIndex]);
                *pValue  = value;
            }
        }
    private:
        alignas(ValueType) unsigned char mQTableBuffer[sizeof(ValueType) * NumberOfStates * NumberOfActions];
    };

    template<typename EnvironmentType, typename NeuralNetworkType, size_t NumberOfIterationsBeforeTargetNetworkUpdate>
    class QValueNeuralNetworkPolicy
    {
    public:
        typedef typename EnvironmentType::EnvironmentStateType StateType;
        typedef typename EnvironmentType::EnvironmentActionType ActionType;
        typedef typename EnvironmentType::EnvironmentValueType ValueType;

        static const size_t NumberOfActions = EnvironmentType::EnvironmentNumberOfActions;
        static const size_t NumberOfInputLayerNeurons = NeuralNetworkType::NumberOfInputLayerNeurons;

        QValueNeuralNetworkPolicy() : mIterations(0), mFedState(0), mFedStateValid(false)
        {
            // The target network starts as a copy of the online one, as in
            // standard DQN, not as a second, independent random network.
            this->copyNetworkWeights();
        }

        /**
         * Q(state, action) from the online network. The network's outputs are
         * for whichever input it last saw, so the state is fed forward first,
         * unless it was the last one fed (ArgMaxPolicy asks once per action).
         * getQValue is const for ArgMaxPolicy, so the network and that record
         * are mutable: a forward pass changes no weights.
         */
        ValueType getQValue(const StateType state, const ActionType action) const
        {
            ValueType learnedValues[NumberOfActions];

            if (!this->mFedStateValid || (this->mFedState != state))
            {
                ValueType inputs[NumberOfInputLayerNeurons];

                EnvironmentType::getInputValues(state, &inputs[0]);
                this->mNeuralNet.feedForward(&inputs[0]);
                this->mFedState = state;
                this->mFedStateValid = true;
            }

            this->mNeuralNet.getLearnedValues(&learnedValues[0]);

            return learnedValues[action];
        }

        ValueType getFutureQValue(const StateType state, const ActionType action)
        {
            ValueType learnedValues[NumberOfActions];
            ValueType inputs[NumberOfInputLayerNeurons];

            EnvironmentType::getInputValues(state, &inputs[0]);

            this->mTargetNeuralNet.feedForward(&inputs[0]);
            this->mTargetNeuralNet.getLearnedValues(&learnedValues[0]);

            return learnedValues[action];
        }

        void init(const EnvironmentType& environment)
        {
            (void)environment; // Unused parameter
        }

        void setQValue(const StateType state, const ActionType action, const ValueType& value)
        {
            ValueType values[NumberOfActions];
            ValueType inputs[NumberOfInputLayerNeurons];

            EnvironmentType::getInputValues(state, &inputs[0]);

            // Training needs this state's forward pass immediately before it,
            // and changes the weights, so no earlier forward pass stays valid.
            this->mNeuralNet.feedForward(&inputs[0]);
            this->mNeuralNet.getLearnedValues(&values[0]);
            this->mFedStateValid = false;

            // Train whenever the target differs from the current output. Only
            // the taken action's output gets an error, and the learner's target
            // already moves Q by alpha times the TD error, so a tolerance on the
            // network's mean error over all outputs would skip every update
            // whose TD error is small, and learning would stop early.
            if (values[action] != value)
            {
                values[action] = value;
                this->mNeuralNet.trainNetwork(&values[0]);
                ++mIterations;
                if(mIterations >= NumberOfIterationsBeforeTargetNetworkUpdate)
                {
                    mIterations = 0;
                    this->copyNetworkWeights();
                }
            }
        }
    private:
        void copyNetworkWeights()
        {
            this->mTargetNeuralNet.setWeights(this->mNeuralNet);
        }
    private:
        size_t mIterations;
        mutable NeuralNetworkType mNeuralNet;
        NeuralNetworkType mTargetNeuralNet;
        mutable StateType mFedState;
        mutable bool mFedStateValid;
    };

    template<   typename EnvironmentType,
                typename QValuePolicy = QValueTablePolicy<EnvironmentType>,
                template<typename, typename> class QLearnerStateToActionPolicy = ArgMaxPolicy
            >
    class QLearner
    {
    public:
        typedef typename EnvironmentType::EnvironmentStateType StateType;
        typedef typename EnvironmentType::EnvironmentActionType ActionType;
        typedef typename EnvironmentType::EnvironmentValueType ValueType;
        typedef typename EnvironmentType::experience_t experience_t;
        typedef QLearnerStateToActionPolicy<EnvironmentType, QValuePolicy> StateToActionPolicy;

        static const size_t NumberOfStates = EnvironmentType::EnvironmentNumberOfStates;
        static const size_t NumberOfActions = EnvironmentType::EnvironmentNumberOfActions;
        static const StateType InvalidState = EnvironmentType::EnvironmentInvalidState;
        static const ActionType InvalidAction = EnvironmentType::EnvironmentInvalidAction;

        QLearner(const ValueType& learningRate, const ValueType& discountFactor, const size_t randomActionDecisionPoint) : 
            mEnvironment(learningRate, discountFactor, randomActionDecisionPoint), mState(InvalidState)
        {
            size_t bufferIndex = 0;

            for(ActionType action = 0;action < NumberOfActions;++action)
            {
                new(&this->mActionsBuffer[bufferIndex]) ActionType(0);
                bufferIndex += sizeof(ActionType);
            }

            this->mQValuePolicy.init(this->mEnvironment);
        }

        EnvironmentType& getEnvironment()
        {
            return this->mEnvironment;
        }

        ValueType getQValue(const StateType state, const ActionType action) const
        {
            return this->mQValuePolicy.getQValue(state, action);
        }

        StateType getState() const
        {
            return this->mState;
        }

        void setQValue(const StateType state, const ActionType action, const ValueType& value)
        {
            this->mQValuePolicy.setQValue(state, action, value);
        }

        void setState(const StateType state)
        {
            if(state < NumberOfStates)
            {
                this->mState = state;
            }
        }

        void startNewEpisode()
        {
            this->mState = InvalidState;
        }

        ActionType takeAction(const StateType state)
        {
            const ActionType action = this->chooseAction(state);
            experience_t experience;

            this->setState(state);

            experience.state = state;
            experience.action = action;

            this->mEnvironment.takeAction(experience);

            return action;
        }

        void updateFromExperience(const experience_t& experience)
        {
            const ValueType learningRate = this->mEnvironment.getLearningRate();
            ValueType oneMinusLearningRate;
            ValueType discountFactor;
            ValueType oldQValue;
            ValueType futureQValue;
            ValueType qValue;

            // Q(s, a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s, a)), for
            // the transition's own state s = experience.state, whatever state
            // the learner was last in (so a replayed or externally generated
            // experience updates the right entry).
            if((learningRate > 0) && (experience.state < NumberOfStates))
            {
                oneMinusLearningRate = (tinymind::Constants<ValueType>::one() - learningRate);
                discountFactor = this->mEnvironment.getDiscountFactor();

                oldQValue = this->mQValuePolicy.getQValue(experience.state, experience.action);

                futureQValue = calculateFutureQValue(experience.newState);

                qValue = ((oneMinusLearningRate * oldQValue) + (learningRate * (experience.reward + (discountFactor * futureQValue))));

                this->mQValuePolicy.setQValue(experience.state, experience.action, qValue);
            }

            this->mState = experience.newState;
        }
    private:
        // Private methods

        // max_a' Q(s', a') over the next state's valid actions: the greedy
        // value, never the exploring policy's choice (which would make the
        // update learn the value of the exploration policy instead of Q*).
        // StateToActionPolicy selects the action, so a DQN picks it with the
        // online network and evaluates it with the target network (Double
        // DQN). A state with no valid actions is terminal and contributes 0.
        ValueType calculateFutureQValue(const StateType state)
        {
            ActionType* pActions = reinterpret_cast<ActionType*>(&this->mActionsBuffer[0]);
            const size_t numberOfValidActions = this->mEnvironment.getValidActionsForState(state, pActions);
            ActionType action;

            if (0 == numberOfValidActions)
            {
                return 0;
            }

            action = StateToActionPolicy::selectBestActionForState(state, pActions, numberOfValidActions, this->mQValuePolicy);
            if (InvalidAction == action)
            {
                // Every valid action ties, so each one's value is the max.
                action = pActions[0];
            }

            return this->mQValuePolicy.getFutureQValue(state, action);
        }

        ActionType chooseAction(const StateType state)
        {
            ActionType* pActions = reinterpret_cast<ActionType*>(&this->mActionsBuffer[0]);
            const size_t numberOfValidActions = this->mEnvironment.getValidActionsForState(state, pActions);
            ActionType action;

            if (this->mEnvironment.shouldChooseRandomAction())
            {
                action = this->mEnvironment.chooseRandomActionFromValidActions(pActions, numberOfValidActions);
            }
            else
            {
                action = StateToActionPolicy::selectBestActionForState(state, pActions, numberOfValidActions, this->mQValuePolicy);
                if(InvalidAction == action)
                {
                    action = this->mEnvironment.chooseRandomActionFromValidActions(pActions, numberOfValidActions);
                }
            }

            return action;
        }

    private:
        EnvironmentType mEnvironment;
        QValuePolicy mQValuePolicy;
        StateType mState;
        alignas(ActionType) unsigned char mActionsBuffer[sizeof(ActionType) * NumberOfActions];

        static_assert(NumberOfStates > 1, "Invalid number of states.");
        static_assert(NumberOfActions > 1, "Invalid number of actions.");
        static_assert((tinymind::limits<StateType>::max > (NumberOfStates - 1)), "State type too small to hold number of states. Max value reserved for invalid state.");
        static_assert((tinymind::limits<ActionType>::max > (NumberOfActions - 1)), "Action type too small to hold number of actions. Max value reserved for invalid action.");
        static_assert(IsSigned<StateType>::result == false, "Must use an unsigned type to represent states.");
        static_assert(IsSigned<ActionType>::result == false, "Must use an unsigned type to represent actions.");
    };
}
