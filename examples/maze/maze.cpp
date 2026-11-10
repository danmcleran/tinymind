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

/*
Q-Learning unit test. Learn the best path out of a simple maze.

5 == Outside the maze
________________________________________________
|                       |                       |
|                       |                       |
|           0           |          1             / 5
|                       |                       |
|____________/  ________|__/  __________________|_______________________
|                       |                       |                       |
|                       |                        /                      |
|           4           |          3            |           2           |
|                        /                      |                       |
|__/  __________________|_______________________|_______________________|
    5

The paths out of the maze:

0->4->5
0->4->3->1->5
1->5
1->3->4->5
2->3->1->5
2->3->4->5
3->1->5
3->4->5
4->5
4->3->1->5
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "mazelearner.h"

extern QLearnerType qLearner;

// Every episode starts in a random room, the goal included. The goal's own
// row of the Q table is only updated in episodes that start there, and its
// self-loop (taking door 5 from outside) shrinks the error by 4% per update,
// so the table needs a few thousand episodes to reach Q*. Exploration stays
// at 100% (Q-learning is off-policy: the update bootstraps from the best next
// action whatever the behaviour policy does) and is phased out over the last
// 100 episodes.
#define TOTAL_NUMBER_OF_EPISODES 5000U
#define NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES (TOTAL_NUMBER_OF_EPISODES - 100U)
#define NUMBER_OF_TEST_EPISODES 100U

static double toDouble(const QValueType& value)
{
    return static_cast<double>(value.getValue()) / 65536.0;
}

// Q* by value iteration: Q*(s, a) = r(s, a) + gamma * max_a' Q*(s', a'),
// where the next room s' is the door a just taken. Comparing the learned
// table with it shows whether the values are right, not only whether they
// rank the actions in the right order (on this small maze a wrong table can
// still pick every shortest path).
static void reportDistanceFromOptimal(const char* const path)
{
    using namespace std;
    MazeEnvironmentType& environment = qLearner.getEnvironment();
    const double gamma = toDouble(environment.getDiscountFactor());
    double qStar[NUMBER_OF_STATES][NUMBER_OF_ACTIONS] = {};

    for (unsigned iteration = 0; iteration < 5000; ++iteration)
    {
        double next[NUMBER_OF_STATES][NUMBER_OF_ACTIONS] = {};
        for (state_t s = 0; s < NUMBER_OF_STATES; ++s)
        {
            for (action_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
            {
                if (!environment.isActionValidForState(s, a))
                {
                    continue;
                }
                double best = 0.0;
                bool any = false;
                for (action_t nextAction = 0; nextAction < NUMBER_OF_ACTIONS; ++nextAction)
                {
                    if (environment.isActionValidForState(a, nextAction) && (!any || (qStar[a][nextAction] > best)))
                    {
                        best = qStar[a][nextAction];
                        any = true;
                    }
                }
                next[s][a] = toDouble(environment.getRewardForStateAndAction(s, a)) + (gamma * best);
            }
        }
        copy(&next[0][0], &next[0][0] + (NUMBER_OF_STATES * NUMBER_OF_ACTIONS), &qStar[0][0]);
    }

    ofstream qTableFile(path);
    qTableFile << "state,action,q,q_star" << endl;
    double maxError = 0.0;
    double sumSquares = 0.0;
    unsigned count = 0;
    for (state_t s = 0; s < NUMBER_OF_STATES; ++s)
    {
        for (action_t a = 0; a < NUMBER_OF_ACTIONS; ++a)
        {
            if (!environment.isActionValidForState(s, a))
            {
                continue;
            }
            const double q = toDouble(qLearner.getQValue(s, a));
            const double error = q - qStar[s][a];
            qTableFile << static_cast<int>(s) << "," << static_cast<int>(a) << "," << q << "," << qStar[s][a] << endl;
            maxError = max(maxError, fabs(error));
            sumSquares += error * error;
            ++count;
        }
    }
    cout << "Q table vs Q*: max |error| " << maxError << ", RMS " << sqrt(sumSquares / count) << endl;
}

int main(const int argc, char *argv[])
{
    (void)argc; // suppress unused parameter warning
    (void)argv; // suppress unused parameter warning

    using namespace std;
    state_t state;
    action_t action;
    typename MazeEnvironmentType::ParentType::experience_t experience;
    string logEntry;
    ofstream logFile;
    size_t randomActionDecisionPoint;

    logFile.open("maze_training.csv");
    if(!logFile.is_open())
    {
        cerr << "Failed to open output file." << endl;
        return -1;
    }

    // Init reward table
    for(state = 0;state < NUMBER_OF_STATES;++state)
    {
        for(action = 0;action < NUMBER_OF_ACTIONS;++action)
        {
            if (qLearner.getEnvironment().isActionValidForState(state, action))
            {
                if (5 == action)
                {
                    qLearner.getEnvironment().setRewardForStateAndAction(state, action, MazeEnvironmentType::EnvironmentRewardValue);
                }
                else
                {
                    qLearner.getEnvironment().setRewardForStateAndAction(state, action, MazeEnvironmentType::EnvironmentNoRewardValue);
                }
            }
            else
            {
                qLearner.getEnvironment().setRewardForStateAndAction(state, action, MazeEnvironmentType::EnvironmentInvalidActionValue);
            }
        }
    }

    qLearner.getEnvironment().setRandomActionDecisionPoint(100);
    qLearner.getEnvironment().setGoalState(5);

    // randomly search the maze for the reward, keep updating the Q table
    for (unsigned i = 0; i < TOTAL_NUMBER_OF_EPISODES; ++i)
    {
        // after the pure random episodes, scale down the randomness on every episode
        if (i >= NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES)
        {
            randomActionDecisionPoint = qLearner.getEnvironment().getRandomActionDecisionPoint();
            if (randomActionDecisionPoint > 0)
            {
                --randomActionDecisionPoint;
                qLearner.getEnvironment().setRandomActionDecisionPoint(randomActionDecisionPoint);
            }
        }

        qLearner.startNewEpisode();

        state = rand() % NUMBER_OF_STATES;
        cout << "*** starting in state " << (int)state << " ***" << endl;
        logEntry.clear();
        logEntry += to_string(state);
        logEntry += ",";
        action = qLearner.takeAction(state);
        cout << "take action " << (int)action << endl;
        logEntry += to_string(action);
        logEntry += ",";
        experience.state = state;
        experience.action = action;
        experience.reward = qLearner.getEnvironment().getRewardForStateAndAction(experience.state, experience.action);
        experience.newState =  static_cast<state_t>(experience.action);
        qLearner.updateFromExperience(experience);

        // look until we find the cheese
        while (qLearner.getState() != qLearner.getEnvironment().getGoalState())
        {
            action = qLearner.takeAction(qLearner.getState());
            cout << "take action " << (int)action << endl;
            logEntry += to_string(action);
            logEntry += ",";
            experience.state = qLearner.getState();
            experience.action = action;
            experience.reward = qLearner.getEnvironment().getRewardForStateAndAction(experience.state, experience.action);
            experience.newState =  static_cast<state_t>(experience.action);
            qLearner.updateFromExperience(experience);
        }

        logFile << logEntry << endl;
    }

    logFile.close();

    reportDistanceFromOptimal("maze_qtable.csv");

    logFile.open("maze_test.csv");
    if(!logFile.is_open())
    {
        cerr << "Failed to open output file." << endl;
        return -1;
    }

    // Training is done. The test runs are greedy (no exploration) and learn
    // nothing (learning rate 0), so they measure the trained table as it is.
    qLearner.getEnvironment().setRandomActionDecisionPoint(0);
    qLearner.getEnvironment().setLearningRate(QValueType(0));
    for (unsigned i = 0; i < NUMBER_OF_TEST_EPISODES; ++i)
    {
        qLearner.startNewEpisode();

        state = rand() % NUMBER_OF_STATES;
        cout << "*** starting in state " << (int)state << " ***" << endl;
        logEntry.clear();
        logEntry += to_string(state);
        logEntry += ",";
        action = qLearner.takeAction(state);
        cout << "take action " << (int)action << endl;
        logEntry += to_string(action);
        logEntry += ",";
        experience.state = state;
        experience.action = action;
        experience.reward = qLearner.getEnvironment().getRewardForStateAndAction(experience.state, experience.action);
        experience.newState =  static_cast<state_t>(experience.action);
        qLearner.updateFromExperience(experience);

        // look until we find the cheese
        while (qLearner.getState() != qLearner.getEnvironment().getGoalState())
        {
            action = qLearner.takeAction(qLearner.getState());
            cout << "take action " << (int)action << endl;
            logEntry += to_string(action);
            logEntry += ",";
            experience.state = qLearner.getState();
            experience.action = action;
            experience.reward = qLearner.getEnvironment().getRewardForStateAndAction(experience.state, experience.action);
            experience.newState =  static_cast<state_t>(experience.action);
            qLearner.updateFromExperience(experience);
        }

        logFile << logEntry << endl;
    }

    if (qLearner.getState() == qLearner.getEnvironment().getGoalState())
    {
        cout << "Reached the goal state!" << endl;
    }
    else
    {
        cout << "Did not reach the goal state!" << endl;
    }

    return 0;
}