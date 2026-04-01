---
title: Q-Learning in Under 1KB
layout: default
parent: Getting Started
nav_order: 2
---

# Table-Based Q-Learning in Under 1KB

Q-learning is an algorithm in which an agent interacts with its environment and collects rewards for taking desirable actions.

![qlearn](https://user-images.githubusercontent.com/1591721/200389242-7fb3238d-9539-49c6-9b17-7cd0c767b521.png)

The simplest implementation of Q-learning is referred to as tabular or table-based Q-learning. In this article, I will describe how tinymind implements Q-learning using C++ templates and fixed-point ([Q-Format]({{ site.baseurl }}/q-format)) numbers as well as go through the [maze](https://github.com/danmcleran/tinymind/blob/master/examples/maze/maze.cpp) example in the [tinymind](https://github.com/danmcleran/tinymind) repo.

At 869 bytes total, this Q-learner fits comfortably on even the smallest microcontrollers -- an ARM Cortex-M0+ with 4 KB of RAM has room for this plus the full application. Reinforcement learning is typically associated with cloud computing and massive compute budgets, but tinymind brings it to the edge with zero runtime overhead and no floating-point requirement.

# The Maze Problem

A common table-based Q-learning problem is to train a virtual mouse to find its way out of a maze to get the cheese (reward). Tinymind contains an [example](https://github.com/danmcleran/tinymind/blob/master/examples/maze/maze.cpp) program which demonstrates how the Q-learning template library works.

![maze](https://user-images.githubusercontent.com/1591721/200389435-ee70c325-9031-41cf-982c-f4ebfbef6831.png)

In the example program, we define the maze:

```
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
```

We define all of our types in a common header so that we can separate the maze learner code from the training and file management code:

```cpp
// 6 rooms and 6 actions
#define NUMBER_OF_STATES 6
#define NUMBER_OF_ACTIONS 6

typedef uint8_t state_t;
typedef uint8_t action_t;
```

We train the mouse by dropping it into a randomly-selected room. The mouse starts off by taking a random action from a list of available actions at each step. The mouse receives a reward only when he finds the cheese (makes it to position 5 outside the maze).

# Building The Example

```bash
cd examples/maze
make
```

Run the example:

```bash
make run
```

# Visualizing Training And Testing

I have included a [Python script](https://github.com/danmcleran/tinymind/blob/master/examples/maze/mazeplot.py) to plot the training and test data:

```bash
make plot
```

Training data for start state == 2 (random exploration):

![maze_training_2](https://user-images.githubusercontent.com/1591721/200390434-22bbc626-4efc-4e8a-831d-641ffa8da8b7.png)

After training, the Q-learner has learned an optimal path: 2->3->4->5:

![maze_test_2](https://user-images.githubusercontent.com/1591721/200390470-27a5cd7f-3172-4314-a300-90cac7c9cd2d.png)

What happens when we drop the virtual mouse outside of the maze where the cheese is? During training:

![maze_training_5](https://user-images.githubusercontent.com/1591721/200390506-57b33669-b7b4-4f7c-9602-70840a917287.png)

After training, the mouse has learned to stay put and get the reward:

![maze_test_5](https://user-images.githubusercontent.com/1591721/200390529-805d50a5-8935-49ee-bf04-ebebe5a7de27.png)

# Determining The Size Of The Q-Learner

```bash
make release
size output/mazelearner.o
```

```
   text	   data	    bss	    dec	    hex	filename
    493	      8	    348	    849	    351	output/mazelearner.o
```

The total code + data footprint of the Q-learner is **869 bytes**. This should allow a table-based Q-learning implementation to fit in any embedded system available today.

# Conclusion

Table-based Q-learning can be done very efficiently using the capabilities provided within tinymind. We don't need floating point or fancy interpreted programming languages. One can instantiate a Q-learner using C++ templates and fixed point numbers. Clone the repo and try the example for yourself!
