# Human Activity Recognition (HAR) — recurrent LSTM

A **recurrent** human-activity classifier trained on tri-axial accelerometer
sequences, inspired by the UCI [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240)
dataset. Built with a Q16.16 fixed-point **LSTM** from TinyMind.

## Network

This is a recurrent classifier, not a feed-forward one:

- **3 inputs** — normalized accelerometer samples `(ax, ay, az)`, fed one
  timestep at a time
- **1 LSTM hidden layer of 16 tanh neurons** (`HiddenLayers<16>`)
- **4 sigmoid outputs** — a one-hot encoding of the activity; predicted class is
  argmax read at the **last timestep** of the window
- Q16.16 fixed-point throughout, with `GradientClipByValue` to keep the
  recurrent training stable in fixed point

Each window is **32 timesteps** long. For every window the network:

1. **resets its recurrent cell state** (`gNet.resetState()`) — the LSTM is
   stateful across `feedForward` calls, so the state must be cleared at the start
   of each window or windows bleed into each other
2. feeds all 32 timesteps through `feedForward`, supervising toward the
   (constant) window label at every step so the recurrent net gets a strong,
   repeated gradient and learns to *settle* on the classification as evidence
   accumulates
3. reads the prediction (argmax of `getLearnedValues`) at the final timestep

45 epochs of shuffled passes over the training split.

## Activities

| label | activity          | accelerometer signature                                            |
|-------|-------------------|--------------------------------------------------------------------|
| 0     | WALKING           | ~2 Hz periodic oscillation on all axes + harmonic, gravity on x    |
| 1     | WALKING_UPSTAIRS  | lower freq, larger amplitude, strong vertical (z) swing, forward-lean gravity split between x and z |
| 2     | SITTING           | near-flat, small noise, gravity on the y axis                      |
| 3     | STANDING          | near-flat, small noise, gravity on the z axis (different orientation) |

## Data

The real HAR raw inertial signals are far too large to bundle, so — following
the `predictive_maintenance` precedent — this example **synthesizes
physically-motivated tri-axial accelerometer windows** by default
(deterministic, `std::srand(7U)`). Each window is a sinusoid + harmonics + noise
for the dynamic activities and a near-flat gravity vector for the static ones;
accelerations are expressed in units of g and normalized by a 1.5 g divisor so
both the ~1 g static component and the motion land roughly in `[-1, 1]` (the
stable range for Q16.16 LSTM inputs). The default dataset is balanced: 200
windows/class train, 50/class test.

### Using a real CSV

If `har.csv` is present in the run directory (the example looks in `./output/`
since `make run` `cd`s there) it is loaded instead. Expected **long format**,
one row per (window, timestep):

```
window,t,ax,ay,az,label
0,0,0.66,0.01,0.00,0
0,1,0.71,0.02,0.05,0
...
0,31,0.69,-0.01,0.03,0
1,0,...
```

- `window` — integer window id (rows of the same window must be contiguous)
- `t` — timestep index in `[0, 31]`
- `ax, ay, az` — accelerations already normalized to roughly `[-1, 1]`
- `label` — activity index `0..3` (constant within a window)

The loader does an 80/20 deterministic shuffle + split. To convert the UCI raw
signals, window the per-axis `total_acc_{x,y,z}` streams into 32-step windows,
normalize by your g-scale, and emit the long format above.

## Build and run

```bash
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

## Output

- `output/har_loss.csv` — training loss curve (`epoch, avg_err` at the last step)
- `output/har_confusion.csv` — 4×4 test confusion matrix
- `output/har_samples.csv` — example accelerometer windows (long format:
  `class, t, ax, ay, az`), one representative window per activity for the trace plot
- `output/har_confusion_behavior.png` — `plot.py` renders loss + confusion +
  per-activity accelerometer traces

Expected (seed = 7): convergence to ~0.027 avg error and **~97.5% test
accuracy** (195/200). The only residual confusion is a handful of
WALKING_UPSTAIRS windows read as WALKING — the genuinely harder
frequency-discrimination case; the static SITTING / STANDING classes are nailed.

## TinyMind capability shown

`LstmNeuralNetwork<>` — a recurrent LSTM trained and run entirely in `QValue`
Q16.16 fixed-point — applied to sequence classification, with the per-window
state reset (`resetState()`) and `GradientClipByValue` that recurrent fixed-point
training needs. See `examples/lstm_sinusoid` for the same recurrent net on a
regression / auto-regressive prediction task.
