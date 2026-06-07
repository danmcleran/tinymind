---
title: Q-Learning Maze
parent: Examples
nav_order: 14
layout: default
---

# Q-Learning Maze

Learns the shortest path out of a small six-room maze using tabular Q-learning. The agent starts in a random room and explores until it reaches the exit, updating a Q-table from each experience.

## How it works

- Classic tabular Q-learning (`tinymind::QLearner` with a `QTableRewardPolicy`) over six states (rooms 0-5, where 5 is outside the maze) and six actions; no neural network involved.
- Demonstrates the reinforcement-learning core: an environment with a reward table, epsilon-style random exploration controlled by a `RandomActionDecisionPoint`, and Q-value updates from `(state, action, reward, newState)` experiences.
- Exploration is annealed late in training (the random-action probability is scaled down after 400 of the 500 training episodes) so the agent shifts from exploring to exploiting the learned Q-table.

## Build and run

```bash
cd examples/maze
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes two CSVs: `output/maze_training.csv` (500 training episodes) and `output/maze_test.csv` (100 greedy test episodes). The `make plot` target renders both via `mazeplot.py`; each CSV row is one episode's state-by-step path.

## Output

![Q-learning maze trajectories]({{ site.baseurl }}/assets/plots/maze_trajectories.png)

Each panel groups the training trajectories by starting room, plotting the visited state at each step. The paths are long and erratic while the agent is still exploring; after the Q-table converges the greedy test episodes (in `maze_test.csv`) reach the exit state in just a few steps from any start room.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/maze)
