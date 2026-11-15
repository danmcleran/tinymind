# Air-Quality Hourly LSTM Forecaster

A recurrent (LSTM) next-hour forecaster for an hourly air-quality pollutant
series, inspired by the UCI
[Air Quality dataset](https://archive.ics.uci.edu/dataset/360). Built on a
Q16.16 fixed-point `LstmNeuralNetwork` from tinymind — the same recurrent shape
as [`examples/lstm_sinusoid`](../lstm_sinusoid), applied to a richer, daily-cycle
pollutant signal.

## Network

- 1 input — the current hour's pollutant concentration, normalized to [0, 1]
- 1 hidden LSTM layer of 16 neurons (`HiddenLayers<16>`), tanh activation
- 1 sigmoid output — the predicted next-hour concentration (target is in [0, 1])
- Q16.16 fixed-point (`QValue<16, 16, true>`) end to end
- `GradientClipByValue` for recurrent stability, learning rate 0.5
- Up to 15000 epochs over the sequential training pairs `series[t] -> series[t+1]`,
  with the recurrent state reset every 24 hours so each BPTT segment is a clean
  one-day stretch (this keeps the daily-cycle gradient from washing out over the
  2000+ step series and prevents the net from collapsing to predicting the mean)
- Training stops at the first epoch in which no sample trains. A sample whose
  error is inside `ZeroToleranceCalculator`'s dead-band (1/128 of full scale)
  is skipped, so once every sample is inside it the weights stop moving; the
  state is reset at every epoch start, so any later epoch would repeat that one
  exactly. On the synthetic series this happens at epoch 28. A tighter band
  does not help: on this seed, bands of 64, 8 and 1 LSB gave held-out MAEs of
  0.08 to 0.58 against 0.070, because the learning rate was tuned with the 1/128
  band in place

The model is trained on all but the last 200 hours; those held-out hours are then
forecast **one step ahead** (predict hour `t+1` from the true hour `t`) and
written out.

## Data

By default the example runs **fully offline** on a synthetic hourly series
(`std::srand(7U)` for determinism): a strong 24-hour daily cycle with
morning/evening rush-hour peaks, a gentle multi-day (weekly) trend, and mild
noise — ~100 days (2400 hourly samples) of a CO-like concentration. The series is
normalized to [0, 1] for training; the min/max are tracked so the forecast is
de-normalized back into real-ish concentration units for reporting and plotting.

To run on the **real** UCI data instead, drop `AirQualityUCI.csv` into the run
directory (`output/`, since `make run` cd's there). The example auto-detects it.
The real file is `;`-separated, uses a decimal comma, and marks missing readings
with `-200`; the loader reads the third column (`CO(GT)`), converts decimal
commas to dots, and skips missing rows. If at least ~300 valid samples are found
the real series is used, otherwise the example falls back to the synthetic one.

The irregular-sampling sibling of this recurrent forecaster — a closed-form
continuous-time (CfC) cell fed a varying per-step elapsed time — lives in
[`examples/cfc_sequence`](../cfc_sequence).

## Build and run

```bash
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` trains for at most 15000 epochs and stops after 28 on the synthetic
series (well under a second in release mode). You can override the epoch limit
and learning rate for quick experiments:

```bash
cd output && ./air_quality 8000 0.3    # epochs, learning rate
```

## Output

- `output/airq_loss.csv` — training loss curve (`epoch,avg_err`)
- `output/airq_forecast.csv` — one-step-ahead forecast over the held-out tail
  (`hour,true,predicted`), in de-normalized concentration units
- `output/airq_forecast_behavior.png` — `plot.py` renders a 1×2 figure: training
  loss over epochs, and the forecast vs actual over the held-out hours (annotated
  with the mean absolute error)

Expected (seed = 7, synthetic series): training stops after 28 epochs, the loss
having fallen from ~0.015 to ~0.0016, and the predicted line clearly tracks the daily
rise-and-fall of the true series (amplitude is mildly damped at the tallest
peaks, the classic RNN-on-noisy-series behavior), with a one-step-ahead MAE
around 0.070 in concentration units over a series spanning ~1.3–3.0.

## tinymind capability shown

`LstmNeuralNetwork<>` recurrent training and inference entirely in `QValue`
Q16.16 fixed-point, with windowed state resets for stable BPTT on a long
time-series — the next step up in complexity from the `lstm_sinusoid` toy, on a
signal that actually looks like a sensor stream you would deploy against on an
MCU.
