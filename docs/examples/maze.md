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
- Each update moves Q(s, a) toward r + gamma * max over a' of Q(s', a'): the best next action, whatever the exploring policy does next. Q-learning is off-policy, so training can explore at random throughout: actions are 100% random for 4,900 of the 5,000 training episodes, and exploration is phased out over the last 100.
- After training the example compares the learned table with Q*, the exact optimal values computed by value iteration, and writes both to `output/maze_qtable.csv`. On a maze this small a wrong table can still rank the actions correctly, so the right paths alone do not show that the values were learned correctly.

## Build and run

```bash
cd examples/maze
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` writes three CSVs: `output/maze_training.csv` (5,000 training episodes), `output/maze_test.csv` (100 greedy test episodes, run with learning switched off) and `output/maze_qtable.csv` (each learned Q value beside Q*). The `make plot` target renders the first two via `mazeplot.py`; each of their rows is one episode's state-by-step path.

## Output

![Q-learning maze trajectories]({{ site.baseurl }}/assets/plots/maze_trajectories.png)

Each panel is one starting room and plots the greedy post-training trajectories (`maze_test.csv`). All 100 test runs reach the exit (state 5) along a shortest path from their start room. Rooms 2 and 3 have two equally short ways out (through room 1 or room 4); Q* rates them the same, so either is optimal. (The exploratory `maze_training.csv` paths, also produced by `make plot`, are long and erratic by contrast.)

The learned Q values match Q* to within 0.0007 (the largest is 5 at the goal, so this is Q16.16 rounding). Before the Q-learning update was fixed to bootstrap from the best next action rather than the exploring policy's random one, the same 5,000 episodes left the table 2.1 RMS away from Q*, with every greedy path still shortest.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/maze)
