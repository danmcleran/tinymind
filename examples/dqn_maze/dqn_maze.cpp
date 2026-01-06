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

#include <iostream>
#include <fstream>
#include <string>

#include "dqn_mazelearner.h"

extern QLearnerType qLearner;

#define TOTAL_NUMBER_OF_EPISODES 10000U
#define NUMBER_OF_PURE_RANDOM_TRAINING_EPISODES (TOTAL_NUMBER_OF_EPISODES - 1000U)
#define NUMBER_OF_TEST_EPISODES 100U
#define MAX_NUMBER_OF_STEPS_PER_EPISODE 100U

int main(const int argc, char *argv[])
{
    (void)argc; // Unused parameter
    (void)argv; // Unused parameter

    using namespace std;
    state_t state;
    action_t action;
    typename MazeEnvironmentType::ParentType::experience_t experience;
    string logEntry;
    ofstream logFile;
    size_t randomActionDecisionPoint;

    logFile.open("dqn_maze_training.txt");
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
        // after 400 random iterations, scale down the randomness on every iteration
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
        unsigned stepCount = 0;
        while ((qLearner.getState() != qLearner.getEnvironment().getGoalState()) && (stepCount < MAX_NUMBER_OF_STEPS_PER_EPISODE))
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
            ++stepCount;
        }

        if (stepCount >= MAX_NUMBER_OF_STEPS_PER_EPISODE)
        {
            cerr << "Episode " << i << " reached max step count without finding goal state." << endl;
        }

        logFile << logEntry << endl;
    }

    logFile.close();
    logFile.open("dqn_maze_test.txt");
    if(!logFile.is_open())
    {
        cerr << "Failed to open output file." << endl;
        return -1;
    }

    // trainging is done, now run some test iterations
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
        unsigned stepCount = 0;
        while ((qLearner.getState() != qLearner.getEnvironment().getGoalState()) && (stepCount < MAX_NUMBER_OF_STEPS_PER_EPISODE))
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
            ++stepCount;
        }

        if (stepCount >= MAX_NUMBER_OF_STEPS_PER_EPISODE)
        {
            cerr << "Episode " << i << " reached max step count without finding goal state." << endl;
        }

        logFile << logEntry << endl;
    }
    
    if (qLearner.getState() == qLearner.getEnvironment().getGoalState())
    {
        cout << "Reached the goal state!" << endl;
    }
    else
    {
        cerr << "Did not reach the goal state!" << endl;
    }

    logFile.close();

    return 0;
}