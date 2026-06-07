---
title: Human Activity Recognition (LSTM)
parent: Examples
nav_order: 54
layout: default
---

# Human Activity Recognition (LSTM)

A recurrent human-activity classifier over tri-axial accelerometer sequences, inspired by the UCI [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240) dataset. It discriminates WALKING, WALKING_UPSTAIRS, SITTING, and STANDING from raw motion.

## How it works

- Q16.16 fixed-point **LSTM**, 3 inputs &rarr; one hidden LSTM layer of 16 tanh neurons &rarr; 4 sigmoid outputs, with `GradientClipByValue` for stable recurrent fixed-point training.
- Demonstrates TinyMind's `LstmNeuralNetwork<>` — a recurrent net trained and run entirely in `QValue` fixed-point — applied to sequence classification rather than the feed-forward MLP path.
- Each window is 32 timesteps. The LSTM is stateful across `feedForward` calls, so the cell state is reset (`resetState()`) at the start of every window; all 32 steps are supervised toward the constant window label so the net settles on its decision as evidence accumulates, and the prediction is read (argmax) at the final timestep. 45 epochs of shuffled passes.

## Build and run

```bash
cd examples/har_activity
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

By default the example synthesizes physically-motivated tri-axial accelerometer windows (deterministic, seed 7): sinusoids + harmonics + noise for the dynamic activities and near-flat gravity vectors for the static ones, normalized into Q16.16's stable range (200 windows/class train, 50/class test). To use real data, drop a long-format `har.csv` (`window,t,ax,ay,az,label`, contiguous rows per window) into `./output/` and it is loaded instead, with an 80/20 deterministic split.

## Output

![HAR training loss, 4x4 test confusion matrix, and per-activity accelerometer traces]({{ site.baseurl }}/assets/plots/har_activity.png)

The loss converges to ~0.027 average error and the confusion matrix is almost perfectly diagonal at **~97.5% test accuracy** (195/200). The only residual confusion is a few WALKING_UPSTAIRS windows read as WALKING — the genuinely harder frequency-discrimination case — while the static SITTING/STANDING classes are nailed, as the per-activity accelerometer traces make visually clear.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/har_activity)
