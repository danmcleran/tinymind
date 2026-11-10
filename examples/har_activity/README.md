# Human Activity Recognition (HAR) — recurrent LSTM

A **recurrent** human-activity classifier trained on tri-axial accelerometer
sequences, inspired by the UCI [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240)
dataset. Built with a Q16.16 fixed-point **LSTM** from tinymind.

## Network

This is a recurrent classifier, not a feed-forward one:

- **3 inputs** — normalized accelerometer samples `(ax, ay, az)`, fed one
  timestep at a time
- **1 LSTM hidden layer of 16 tanh neurons** (`HiddenLayers<16>`)
- **4 sigmoid outputs** — a one-hot encoding of the activity; predicted class is
  argmax read at the **last timestep** of the window
- Q16.16 fixed-point throughout, with `GradientClipByValue`, a learning rate of
  0.05 and no momentum (the library default of 0.25 with momentum 0.5
  oscillates on this task in Q16.16)

Each window is **32 timesteps** long. For every window the network:

1. **resets its recurrent cell state** (`net.resetState()`) — the LSTM is
   stateful across `feedForward` calls, so the state must be cleared at the start
   of each window or windows bleed into each other
2. feeds all 32 timesteps through `feedForward`, training toward the
   (constant) window label at every step
3. reads the prediction (argmax of `getLearnedValues`) at the final timestep

60 epochs of shuffled passes over the training split.

## Activities

| label | activity            | accelerometer signature |
|-------|---------------------|-------------------------|
| 0     | WALKING             | sinusoidal vertical (z) acceleration, three steps per window |
| 1     | WALKING_UPSTAIRS    | per step, a sharp vertical rise and then a slow decline (sawtooth) |
| 2     | WALKING_DOWNSTAIRS  | the upstairs waveform played backwards in time: a slow rise, then a sharp drop |
| 3     | SITTING             | near-flat, small noise, gravity on the y axis |

The three walking classes share the step rate, the lateral sway, the gravity
direction, and the same per-axis mean and standard deviation. UPSTAIRS and
DOWNSTAIRS are exact time reversals of each other, so every individual sample
has the same distribution in both: the only thing that separates them is the
order of the samples, which an LSTM reading one timestep at a time has to
remember.

## Data

The real HAR raw inertial signals are far too large to bundle, so this
example **synthesizes tri-axial accelerometer windows** by default
(deterministic, `std::srand(7U)`), built as described above. Accelerations are
expressed in units of g and normalized by a 1.5 g divisor so both the ~1 g
static component and the motion land roughly in `[-1, 1]` (the stable range
for Q16.16 LSTM inputs). The default dataset is balanced: 200 windows/class
train and 50/class test, generated independently.

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
- `label` — activity index `0..3` (0 walking, 1 upstairs, 2 downstairs, 3 sitting; constant within a window)

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
- `output/har_confusion.csv` — 4×4 test confusion matrix (rows: actual walking,
  upstairs, downstairs, sitting)
- `output/har_samples.csv` — example accelerometer windows (long format:
  `class, t, ax, ay, az`), one representative window per activity for the trace plot
- `output/har_confusion_behavior.png` — `plot.py` renders loss + confusion +
  per-activity accelerometer traces

## Results

| model | test accuracy |
|---|---|
| LSTM, all weights trained | **92.5%** (185/200) |
| LSTM, `--readout-only` (input, gate and recurrent weights frozen at their random initial values) | 50.0% |
| nearest centroid on per-axis mean + standard deviation (ignores sample order) | 52.5% |
| chance | 25% |

`make run` prints the full model's accuracy and the nearest-centroid and chance baselines; run
`./output/har_activity --readout-only` for the ablation. Training the LSTM's
gates and recurrent weights is what reads the sample order. Recurrent training
in Q16.16 is noisy on this task: across five weight-initialization seeds the
full model scored 71%, 73.5%, 89.5%, 92.5% and 94.5%, always well above both
order-blind baselines.

An earlier version of this example (four classes separable by gravity
direction and step frequency) scored 100%, but a nearest-centroid classifier
on per-axis mean and standard deviation also scored 100% on it, so it showed
nothing about recurrence.

## tinymind capability shown

`LstmNeuralNetwork<>` — a recurrent LSTM trained and run entirely in `QValue`
Q16.16 fixed-point — applied to sequence classification, with the per-window
state reset (`resetState()`) and `GradientClipByValue` that recurrent fixed-point
training needs. See `examples/lstm_sinusoid` for the same recurrent net on a
regression / auto-regressive prediction task.
