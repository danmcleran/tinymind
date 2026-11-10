---
title: Human Activity Recognition (LSTM)
parent: Examples
nav_order: 54
layout: default
---

# Human Activity Recognition (LSTM)

A recurrent human-activity classifier over tri-axial accelerometer sequences, inspired by the UCI [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240) dataset. It discriminates WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, and SITTING, and the synthetic task is built so that the stairs classes can only be told apart by the order of the samples.

## How it works

- Q16.16 fixed-point **LSTM**, 3 inputs &rarr; one hidden LSTM layer of 16 tanh neurons &rarr; 4 sigmoid outputs, with `GradientClipByValue`, learning rate 0.1 and no momentum (the library default of 0.25 with momentum 0.5 oscillates on this task in Q16.16).
- Each window is 32 timesteps. The cell state is reset (`resetState()`) at the start of every window; all 32 steps are supervised toward the window label, trained on every step, and the prediction is read (argmax) at the final timestep. 60 epochs of shuffled passes.
- The three walking classes share the step rate, lateral sway, gravity direction, and the per-axis mean and standard deviation. Level walking has a sinusoidal vertical acceleration; the stairs classes have a sawtooth, and WALKING_DOWNSTAIRS is WALKING_UPSTAIRS played backwards in time, so every individual sample has the same distribution in both. An LSTM that reads one timestep at a time can only separate them by remembering earlier samples.

## Build and run

```bash
cd examples/har_activity
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
./output/har_activity --readout-only   # ablation: train only the hidden-to-output weights
```

By default the example synthesizes the accelerometer windows (deterministic, seed 7; 200 windows/class train, 50/class test, generated independently). To use real data, drop a long-format `har.csv` (`window,t,ax,ay,az,label`, contiguous rows per window) into `./output/` and it is loaded instead, with an 80/20 deterministic split.

## Results

| Model | Test accuracy |
|---|---|
| LSTM, all weights trained | **76.5%** (153/200); 89.6% mean over 10 seeds |
| LSTM, `--readout-only` (input, gate and recurrent weights frozen at their random initial values) | 50.0% |
| Nearest centroid on per-axis mean + standard deviation (ignores sample order) | 52.5% |
| Chance | 25% |

The two order-blind baselines both separate SITTING and lose most of the walking classes. Training the LSTM's gates and recurrent weights is what reads the sample order: it lifts accuracy from 50% to 76.5% on the default seed. Recurrent training in Q16.16 is noisy on this task, though: over data seeds 1 to 10 (weights from seed 1) the full model scores 89.6% ± 10.4% (73.5% to 100%), never at chance, while `--readout-only` stays at 50% on every seed. The learning rate was 0.05 until SGD steps were rounded to nearest; at 0.05 the unbiased steps leave 8 of those 10 seeds stuck at 50%, and 0.1 gives the numbers above.

## Output

![HAR training loss, 4x4 test confusion matrix, and per-activity accelerometer traces]({{ site.baseurl }}/assets/plots/har_activity.png)

The right panel shows one window of each class. The stairs traces are mirror images in time on the vertical (dotted) axis; the confusion matrix shows SITTING and the stairs classes separated, with most of the remaining errors between level walking and the stairs.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/har_activity)
