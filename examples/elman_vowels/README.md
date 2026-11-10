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
   utterance's one-hot speaker target, context cleared with `resetState()`
   at every utterance boundary.
2. **Weight transfer.** The trained weights — input, recurrent, output, and
   biases — are copied into inference-only (`IsTrainable = false`) fixed-point
   networks, saturating each weight to the target format's range.
3. **Fixed-point inference sweep.** The test split runs through every network;
   the question is the *smallest* Q-format that matches the float model.

## Results

| Format  | Weight storage | Test accuracy |
|---------|---------------:|--------------:|
| float64 |        4936 B  |        92.97% |
| Q16.16  |        2468 B  |        92.97% |
| Q8.8    |     **1234 B** |    **92.97%** |
| Q4.4    |         617 B  |        58.92% |

**Q8.8 matches double-precision accuracy exactly** with 4× smaller weights —
the entire 617-weight network fits in 1.2 KB, with no FPU required at
inference. Q4.4 (1/16 resolution, ±8 range) is below what this network needs:
none of the weights clip, but the accumulated resolution loss through the
recurrent loop collapses accuracy. The original paper's baseline on this
dataset is 94.1% (Kudo, Toyama & Shimbo, 1999); this 617-weight Elman network
comes within about a point of it.

Q16.16 and Q8.8 do not merely score what the float model scores: they make the
same prediction as the float model on every one of the 370 test utterances.

Across random seeds the float model scores 90.5% to 93.2% (seed 2, shipped, is
92.97%); a 24-unit hidden layer or 120 epochs does no better. Earlier versions
of this example reported a higher accuracy, but it was inflated: the per-utterance
context reset the example did by hand stopped having any effect once the Elman
network began restoring the previous step's hidden state before each step, so
each utterance started from the previous speaker's trailing context, in
training and at test time. The example now calls `resetState()`, and each
utterance starts from an empty context.

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
