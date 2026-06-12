# Elman Network — Japanese Vowels Speaker Recognition

The full embedded deployment story for a recurrent network: **train offline in
floating point, deploy in the smallest fixed-point format that holds the
accuracy**.

## Dataset

[Japanese Vowels](https://archive.ics.uci.edu/dataset/128/japanese+vowels)
(UCI ML Repository, id 128, CC BY 4.0). Nine male speakers each utter the
Japanese vowel /ae/; every utterance is a short sequence — 7 to 29 frames — of
12 LPC cepstrum coefficients. The task is to identify the speaker from the
sequence: 270 training utterances (30 per speaker), 370 test utterances.

The task is genuinely temporal. A single 12-coefficient frame is ambiguous
between speakers; the *trajectory* of the coefficients across an utterance is
what discriminates. An [`ElmanNeuralNetwork`](../../cpp/neuralnet.hpp) walks the
frames one at a time, carrying what it has heard so far in its recurrent hidden
context. Per-frame class scores are summed over the utterance and the predicted
speaker is the argmax.

## Deployment flow

1. **Offline float training.** An `ElmanNeuralNetwork<double, 12, 16, 9>` is
   trained on the host in double precision — per-frame backprop against the
   utterance's one-hot speaker target, context reset at every utterance
   boundary.
2. **Weight transfer.** The trained weights — input, recurrent, output, and
   biases — are copied into inference-only (`IsTrainable = false`) fixed-point
   networks, saturating each weight to the target format's range.
3. **Fixed-point inference sweep.** The test split runs through every network;
   the question is the *smallest* Q-format that matches the float model.

## Results

| Format  | Weight storage | Test accuracy |
|---------|---------------:|--------------:|
| float64 |        4936 B  |        94.05% |
| Q16.16  |        2468 B  |        94.32% |
| Q8.8    |     **1234 B** |    **94.05%** |
| Q4.4    |         617 B  |        48.38% |

**Q8.8 matches double-precision accuracy exactly** with 4× smaller weights —
the entire 617-weight network fits in 1.2 KB, with no FPU required at
inference. Q4.4 (1/16 resolution, ±8 range) is below what this network needs:
none of the weights clip, but the accumulated resolution loss through the
recurrent loop collapses accuracy. The original paper's baseline on this
dataset is 94.1% (Kudo, Toyama & Shimbo, 1999) — the Q8.8 Elman network lands
on it.

For reference, the float accuracy matches the Q16.16/Q8.8 results within one
utterance (±0.27%), so the fixed-point networks are not merely "close" — they
reproduce the float model's decisions.

## Network

- 12 inputs — one LPC cepstrum frame, per-feature z-scored (train statistics)
  then divided by 3 so inputs sit in roughly [-1, 1]
- 1 recurrent (Elman) hidden layer of 16 tanh units, context depth 1
- 9 sigmoid outputs — one-hot speaker encoding, argmax decision
- 617 weights total: 12×16 input + 16 bias + 16×16 recurrent + 16×9 output + 9 bias
- Training: 60 epochs, learning rate 0.01, momentum 0.5, utterance order
  shuffled per epoch, fixed seed for reproducibility

## Run

```bash
make          # build (also copies ae.train / ae.test next to the binary)
make run      # train offline + sweep all four formats, write CSVs
make plot     # accuracy bar chart, training loss, Q8.8 confusion matrix
```

Outputs land in `output/`: `vowels_accuracy.csv` (one row per format),
`vowels_loss.csv` (per-epoch training MSE), `vowels_confusion.csv` (Q8.8
confusion matrix), and the PNGs from `make plot`.

## Data files

`ae.train` / `ae.test` are the unmodified UCI distribution files, included here
for a self-contained build. Citation: M. Kudo, J. Toyama, M. Shimbo (1999).
"Multidimensional Curve Classification Using Passing-Through Regions." Pattern
Recognition Letters 20(11-13), 1103-1111.
