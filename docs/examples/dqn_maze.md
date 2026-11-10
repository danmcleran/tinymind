---
title: DQN Maze
parent: Examples
nav_order: 15
layout: default
---

# DQN Maze

Solves the same six-room maze as the tabular Q-learning example, but replaces the Q-table with a Deep Q-Network (DQN). The Q-values are approximated by a neural network trained from experience instead of stored explicitly.

## How it works

- Deep Q-learning: a Q16.16 fixed-point multilayer perceptron (`tinymind::MultilayerPerceptron` wrapped in a `QValueNeuralNetworkPolicy`) approximates the action-value function, with a periodically-synced target network for stable bootstrapping. The state is one-hot encoded (6 inputs, one per room), and the output layer is linear, since Q values reach 5 at the goal, beyond a tanh output's range.
- Demonstrates the function-approximation variant of reinforcement learning on the same environment as the tabular maze, so the two examples can be compared directly.
- Each update trains the taken action's output toward (1 - alpha) * Q(s, a) + alpha * (r + gamma * Q_target(s', a*)), where a* is the online network's best action in s' (Double DQN). The target network starts as a copy of the online one and is re-synced every 10 training steps.
- Trains over 10,000 episodes with a per-episode step cap. Q-learning is off-policy, so actions stay 100% random until the last 100 episodes, which phase exploration out; a long greedy phase without a replay buffer lets the network drift on the moves the greedy policy never takes. It then runs 100 greedy test episodes with learning switched off.

## Build and run

```bash
cd examples/dqn_maze
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` writes `output/dqn_maze_training.csv` (10,000 training episodes), `output/dqn_maze_test.csv` (100 greedy test episodes) and `output/dqn_maze_qtable.csv` (each of the network's Q values beside Q*, the exact optimal value from value iteration); `make plot` renders the first two via `dqn_mazeplot.py`, one trajectory per episode grouped by starting room.

## Output

![DQN maze trajectories]({{ site.baseurl }}/assets/plots/dqn_maze.png)

Each panel is one starting room and plots the greedy post-training rollouts (`dqn_maze_test.csv`). Once the value network is trained, every run drives straight to the exit (state 5) from any start room, along a shortest path: all 100 greedy test episodes reach the goal. Far more training episodes are needed than in the tabular case because the network learns the value function from scratch (the exploratory `dqn_maze_training.csv` paths are also produced by `make plot`).

The right paths alone do not show that the values were learned: on a maze this small a wrong set of Q values can still rank the actions correctly. The network's Q values are within 0.0005 RMS of Q* (largest error 0.0006, on values from 2.56 to 5). They were 0.023 RMS away (largest error 0.08) until the fixed-point momentum terms were made to round toward zero, which stopped the weights drifting by an LSB per training step, and 0.0017 RMS (largest error 0.003) until the gradient step was rounded to nearest instead of floored, which removed a downward drift of about half an LSB per step. Before two fixes to the Q-learning update, the same training left them 3.07 RMS away with every greedy path still shortest: the update bootstrapped from the exploring policy's random next action instead of the best one, and the DQN skipped training whenever the network's mean output error was within its fixed-point zero tolerance, so it stopped learning after about 200 episodes.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/dqn_maze)
