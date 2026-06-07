---
title: DQN Maze
parent: Examples
nav_order: 15
layout: default
---

# DQN Maze

Solves the same six-room maze as the tabular Q-learning example, but replaces the Q-table with a Deep Q-Network (DQN). The Q-values are approximated by a neural network trained from experience instead of stored explicitly.

## How it works

- Deep Q-learning: a Q16.16 fixed-point multilayer perceptron (`tinymind::MultilayerPerceptron` wrapped in a `QValueNeuralNetworkPolicy`) approximates the action-value function, with a periodically-synced target network for stable bootstrapping.
- Demonstrates the function-approximation variant of reinforcement learning on the same environment as the tabular maze, so the two examples can be compared directly.
- Trains over 10,000 episodes (random exploration annealed over the final 1,000) with a per-episode step cap, then runs 100 greedy test episodes against the learned network.

## Build and run

```bash
cd examples/dqn_maze
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes `output/dqn_maze_training.csv` (10,000 training episodes) and `output/dqn_maze_test.csv` (100 greedy test episodes); `make plot` renders both via `dqn_mazeplot.py`, one trajectory per episode grouped by starting room.

## Output

![DQN maze trajectories]({{ site.baseurl }}/assets/plots/dqn_maze.png)

Each panel groups episodes by starting room and shows the visited state at every step. Far more episodes are needed than in the tabular case because the network must learn the value function from scratch; once trained, the greedy test rollouts in `dqn_maze_test.csv` drive to the exit state efficiently from any start room.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/dqn_maze)
