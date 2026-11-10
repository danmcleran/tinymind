---
title: DQN Maze Solver
layout: default
parent: Getting Started
nav_order: 3
---

# DQN Maze Solver

In the [previous tutorial]({{ site.baseurl }}/getting-started/q-learning-under-1kb) I showed how to build a maze-solving Q-learner using a table-based Q-learning method. In this post, I show how to solve the same maze using DQN (Deep Q-Learning). The code for this example program is found [here](https://github.com/danmcleran/tinymind/tree/master/examples/dqn_maze).

While the table-based approach is ideal for small state spaces (869 bytes for 6 states), DQN replaces the Q-table with a neural network, enabling reinforcement learning in environments with hundreds or thousands of states where a Q-table would be prohibitively large. The entire DQN implementation here -- neural network, Q-learning logic, and lookup tables -- fits in under 16 KB. In standard frameworks like PyTorch or TensorFlow, a DQN of any size requires tens of megabytes of runtime before the first weight is allocated.

# Reinforcement Learning

In a reinforcement learning problem, an Agent interacts with an Environment by evaluating its State, taking Actions, and receiving Rewards. The goal is to learn which Actions provide the most Reward over the course of time.

![qlearn]({{ site.baseurl }}/assets/diagrams/200393565-a67a5a93-4f3f-4159-a4d8-b51a2dee7f64.png)

# The Maze Problem

As in the [previous tutorial]({{ site.baseurl }}/getting-started/q-learning-under-1kb), we're going to attempt to solve the maze problem. A mouse gets dropped randomly into 1 of 6 places on the maze and must learn to find the cheese (reward).

![maze]({{ site.baseurl }}/assets/diagrams/200393635-b3ce901b-cc58-48f3-87ce-730e3825f7ff.png)

# DQN Architecture

For this problem we will get rid of the Q-table and replace it with a neural network. We will need 6 input neurons (one per room) and 6 output neurons (one Q value per action). I've chosen to implement 1 hidden layer with 2 more neurons than output neurons. The diagram below shows the first version of this network, which had a single input neuron; the reasons it changed follow the diagram.

![dqn]({{ site.baseurl }}/assets/diagrams/200395665-9236afda-feed-440e-b29f-11e2911454b6.png)

Each output neuron holds the Q value of one action from the current state. The state is one-hot encoded: input `i` is 1.0 when the mouse is in room `i` and 0 otherwise. An earlier version fed a single input, the state divided by the maximum state, but that squeezes six rooms into six nearby points of one number, and a hidden layer this small cannot map them to six different rows of Q values:

```cpp
template<typename StateType,
         typename ActionType,
         typename ValueType,
         size_t NumberOfStates,
         size_t NumberOfActions,
         typename RewardPolicyType,
         template<typename> class QLearningPolicy = tinymind::DefaultLearningPolicy
         >
struct DQNMazeEnvironment : public tinymind::QLearningEnvironment<state_t, action_t, ValueType, NumberOfStates, NumberOfActions, DQNMazeEnvironmentRandomNumberGeneratorPolicy>
{
...
static void getInputValues(const StateType state, ValueType *pInputs)
{
    for (size_t i = 0; i < NumberOfStates; ++i)
    {
        pInputs[i] = ValueType((i == state) ? 1 : 0, 0);
    }
}
```

The output layer is linear. A Q value is the discounted sum of future rewards, and with a reward of 1 and a discount factor of 0.8 it reaches 1 / (1 - 0.8) = 5 at the goal, outside the (-1, 1) range of a tanh output.

# DQN Neural Network

The neural network is trained to predict the Q-value for every action taken from the given state. After the neural network is trained, we then choose the action which contains the highest Q-value. See the [Neural Networks]({{ site.baseurl }}/neural-networks) page and the [Under 4KB tutorial]({{ site.baseurl }}/getting-started/xor-under-4kb) for more insight into how these neural networks work.

![dqn_nn]({{ site.baseurl }}/assets/diagrams/200394536-51e966f5-268c-421d-9351-a65ada4e05d0.png)

The DQN is defined within [dqn_mazelearner.h](https://github.com/danmcleran/tinymind/blob/master/examples/dqn_maze/dqn_mazelearner.h).

# Building The Example

```bash
cd examples/dqn_maze
make
```

Run the example:

```bash
make run
```

# Visualizing Training And Testing

I have included a [Python script](https://github.com/danmcleran/tinymind/blob/master/examples/dqn_maze/dqn_mazeplot.py) to plot the training and test data.

Training data for start state == 2:

![dqn state 2]({{ site.baseurl }}/assets/diagrams/200396035-8f17e0ae-f904-4859-ace2-2dbf6201cf14.png)

After training, the Q-learner has learned a shortest path from every room. From room 2 it takes 2->3->4->5, and all 100 greedy test episodes reach the cheese:

![dqn test trajectories]({{ site.baseurl }}/assets/plots/dqn_maze.png)

# Determining The Size Of The DQN

```bash
make release
size output/dqn_mazelearner.o
```

```
   text	   data	    bss	    dec	    hex	filename
  11350	      8	   5148	  16506	   407a	output/dqn_mazelearner.o
```

The entire DQN fits within **~16.5 KB**. Pretty small as DQN goes.

# Conclusion

DQN trades out the Q-table for a neural network to learn the relationship between states, actions, and Q-values. One would want to use DQN when the state space is large and the memory consumed by the Q-table would be prohibitively large. By using DQN, we're trading memory for CPU cycles. The CPU overhead for DQN will be far larger than a Q-table. But, DQN allows us to do Q-learning while keeping our memory footprint manageable for complex environments.
