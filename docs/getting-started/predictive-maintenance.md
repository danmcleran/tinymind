---
title: Predictive Maintenance on the AI4I 2020 Dataset
layout: default
parent: Getting Started
nav_order: 5
---

# Predictive Maintenance on the AI4I 2020 Dataset

Predictive maintenance is a workload where TinyMind's fixed-point networks shine: a machine streams a handful of sensor readings, a tiny MLP riding alongside the control loop decides whether something is about to fail, and the decision has to happen inside one control cycle with no FPU in sight. The [`examples/predictive_maintenance/`](https://github.com/danmcleran/tinymind/tree/master/examples/predictive_maintenance) example trains a Q16.16 MLP on the UCI [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), reports precision/recall/F1 on a held-out test set, and runs end-to-end without requiring the CSV download.

This tutorial walks through the dataset, feature preprocessing, the 50/50 balanced-sampling trick that makes class-imbalanced training actually work, and what the numbers mean when you read the confusion matrix.

## Why This Problem?

AI4I 2020 is a good fit for TinyMind because:

- **Six features, one label.** Air temperature, process temperature, rotational speed, torque, tool wear, and a product-quality variant (L/M/H) all come in at ~Hz sample rates from sensors a PLC already reads. No computer-vision front-end required.
- **Deeply imbalanced.** Only ~3.4% of the 10 000 rows are failures. That mirrors real industrial fault data and makes it a good test of how you handle minority-class learning on a small model.
- **Five failure modes folded into one label.** The dataset ships both a binary `Machine failure` column and five independent failure-mode labels (Tool wear, Heat dissipation, Power, Overstrain, Random). Training on the binary label is the natural starting point; extending to multi-label output is a one-line template change.
- **Entirely tabular.** No FFT, no MFCC, no convolutions. This is the ["sensor -> MLP" workload]({{ site.baseurl }}/architectures) TinyMind was originally designed for.

## The Dataset

Each row looks like this (UCI CSV layout):

```
UDI,Product ID,Type,Air temperature [K],Process temperature [K],
    Rotational speed [rpm],Torque [Nm],Tool wear [min],
    Machine failure,TWF,HDF,PWF,OSF,RNF
```

The example either reads `ai4i2020.csv` from the run directory (the real UCI CSV) or, if that file is not present, synthesizes 10 000 rows following the documented generative and failure-labelling rules. The synthetic path exists so the example trains end-to-end in CI without a download; the real CSV produces the distribution the published benchmarks use.

Failure rules (from the dataset docs) that determine the binary label:

| Mode | Rule |
|------|------|
| TWF  | Tool wear in [200, 240] min (sampled 5/10000 as failures) |
| HDF  | Process-air temperature diff < 8.6 K AND rpm < 1380 |
| PWF  | torque * omega not in [3500, 9000] W (omega = rpm * 2pi / 60) |
| OSF  | tool_wear * torque > 11000 / 12000 / 13000 min*Nm for L / M / H |
| RNF  | 0.1% random |

`Machine failure` is the logical OR of the five.

## Network Architecture

```
input [7]                         5 numeric + 2 one-hot (L, M; H = [0,0])
  -> Dense 7 -> 8, ReLU
  -> Dense 8 -> 1, Sigmoid        binary failure probability
```

Declared in Q16.16 fixed-point:

```cpp
using ValueType = tinymind::QValue<16, 16, true>;

using Transfer = tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    tinymind::ReluActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>>;

using Net = tinymind::MultilayerPerceptron<
    ValueType,
    7,  // inputs
    1,  // hidden layer count
    8,  // neurons per hidden layer
    1,  // outputs
    Transfer>;
```

Why these choices:

- **Q16.16.** Q8.8 is too coarse for z-scored features that can sit at 2-3 standard deviations; Q16.16 has 16 fractional bits (resolution ~1.5e-5) and still fits in a 32-bit integer register. It is the sweet spot for numerical headroom without stepping up to 64-bit math.
- **7 inputs.** 5 numeric features plus a two-dimensional one-hot for the product variant. With three categories {L, M, H}, two dimensions are enough (H = [0, 0]); adding a third would just make the weights redundant.
- **ReLU hidden, sigmoid output.** ReLU keeps the hidden layer cheap; sigmoid on the single output produces a probability comparable to a 0.5 decision threshold.
- **8 hidden neurons.** Enough to learn a non-linear decision boundary between the five failure modes, small enough that all the weights fit in ~300 bytes.

## Feature Preprocessing

The six raw features span wildly different magnitudes (tool wear goes 0-253 min, torque hovers around 40 Nm, rpm lives around 1540). Feeding them straight into a Q16.16 network would either saturate the integer part or lose all resolution in the small features. Z-scoring based on training-set statistics pins every feature to roughly N(0, 1):

```cpp
// Fit mean + stdev on training split only (no test leakage)
FeatureStats st;
fitStats(train, st);

// Per-sample: z = (x - mean) / stdev / 3.0
// The extra /3.0 keeps typical z-values well inside Q16.16's sweet spot
const double z = (v[f] - st.mean[f]) / st.stdev[f] / 3.0;
in[f] = toQ(z);
```

The `/ 3.0` is a small safety margin: a z-score of 3.0 is already three standard deviations out, so dividing by three means 99.7% of inputs end up in [-1, +1] even before the network does anything -- no risk of saturating the Q16.16 integer part on a freak sensor reading.

Variant gets a two-dim one-hot rather than an integer code so the network does not see an ordered relationship between L, M, and H that does not exist in the data.

## Handling Class Imbalance: Balanced Sampling

The dataset is ~3.4% failures. If you train on uniform samples, the network learns the trivial majority classifier -- predict "no failure" always, get ~96.6% accuracy, learn nothing. The fix is to oversample failures during training so each batch sees ~50% positives.

```cpp
std::vector<size_t> pos, neg;
for (size_t i = 0; i < train.size(); ++i)
    (train[i].label ? pos : neg).push_back(i);

std::uniform_int_distribution<size_t> posPick(0, pos.size() - 1);
std::uniform_int_distribution<size_t> negPick(0, neg.size() - 1);
std::bernoulli_distribution coin(0.5);

for (unsigned it = 0; it < iterations; ++it)
{
    const Sample& s = coin(rng)
        ? train[pos[posPick(rng)]]
        : train[neg[negPick(rng)]];
    // ... feedForward + trainNetwork
}
```

Positives get repeated many times across the 40 000 iterations; negatives get undersampled. The network ends up with a decision boundary that cuts through the failure region rather than parking on the "always no" side.

## Training Loop

Each iteration is the standard TinyMind feed-forward / error / train pattern, no different from the XOR example:

```cpp
ValueType input[7], target[1], learned[1];

toInput(s, st, input);
target[0] = toQ(s.label ? 1.0 : 0.0);

gNet.feedForward(input);
const ValueType err = gNet.calculateError(target);
if (!Net::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(err))
{
    gNet.trainNetwork(target);
}
```

The `isWithinZeroTolerance` gate skips the backward pass when the error is already below the per-type zero tolerance, saving cycles on already-correct samples.

## Evaluation

Training optimizes binary cross-entropy, but the number that matters for predictive maintenance is **recall**: what fraction of real failures did you catch? Missing a failure is usually much more expensive than a false alarm, so the decision-boundary choice leans toward high recall even at the cost of precision. The example reports the full confusion matrix plus accuracy, precision, recall, and F1:

```cpp
for (const auto& s : test)
{
    toInput(s, st, input);
    gNet.feedForward(input);
    gNet.getLearnedValues(learned);

    const bool predFail = fromQ(learned[0]) >= 0.5;
    const bool realFail = s.label != 0;
    // ... update tp/fp/tn/fn counters
}
```

## Building and Running

```bash
cd examples/predictive_maintenance
make release
make run
```

Sample output (synthetic path, seed = 7):

```
ai4i2020.csv not found; synthesizing 10000 rows using the documented AI4I 2020 generative rules.
Train: 8000 (pos=1305, neg=6695)  Test: 2000
iter   2000   avg|err| = 0.1906
iter   4000   avg|err| = 0.1349
iter   8000   avg|err| = 0.1271
iter  16000   avg|err| = 0.1106
iter  24000   avg|err| = 0.0901
iter  32000   avg|err| = 0.0802
iter  40000   avg|err| = 0.0888

Confusion matrix (rows=actual, cols=predicted):
               pred no-fail   pred fail
  actual no-fail     1456           227
  actual fail          35           282

accuracy=0.8690  precision=0.5540  recall=0.8896  F1=0.6828
```

Reading the matrix:

- **282 / 317** real failures caught (recall 89%). 35 failures slip through undetected.
- **227 / 1683** good machines trigger false alarms (14%). That is the cost of the recall-biased threshold -- move the 0.5 threshold up to shift the tradeoff.
- **Accuracy is misleading** on a 3.4% positive class. A trivial "always no failure" classifier would score 96.6% accuracy on the real distribution. The numbers you should actually track are recall and F1.

Using the real UCI CSV (put it in `output/` before running, which is the working directory `make run` uses) typically produces **higher** precision and F1 because the real failure distribution is less noisy than the synthetic fallback.

## Footprint

Stripped binary on x86_64 host (`-O3 -Wall -Wextra -Werror -Wpedantic`):

```
   text    data     bss     dec     hex
  31581     976    2360   34917    8865
```

| Component | Approx bytes |
|---|---|
| Network weights + gradients (Q16.16) | ~300 |
| Q16.16 sigmoid lookup table | ~28 KB |
| Feature stats + program logic | ~6 KB |
| **Total static allocation** | **~35 KB** |

The sigmoid LUT dominates. That cost is fixed per Q-format: every TinyMind Q16.16 network with a sigmoid output on the same MCU shares the same LUT, so adding a second classifier is nearly free. Dropping to Q8.8 shrinks the LUT by ~16x at the cost of resolution -- a workable option when input features compress to a small integer range (eg raw ADC codes).

## Next Steps

- **Train on the real CSV.** Download `ai4i2020.csv` from [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and drop it in `examples/predictive_maintenance/output/` before `make run`. The loader auto-detects it.
- **Multi-label output.** Change `NUMBER_OF_OUTPUTS` from 1 to 5 and train against the five failure-mode columns (TWF/HDF/PWF/OSF/RNF) instead of the OR'd binary. Each output becomes an independent sigmoid -- no softmax needed for multi-label classification.
- **Calibrate the decision threshold.** 0.5 is never the right operating point for an imbalanced classifier. Sweep the threshold over the test set and pick the one that hits your target recall (say 95%) with the lowest false-positive rate.
- **Export from PyTorch.** If you already have a trained model in PyTorch, see the [PyTorch interop guide]({{ site.baseurl }}/training/pytorch-interop) for the weight-export pattern. The numeric preprocessing (z-score + /3) has to match between training and deployment.
- **Shrink further.** 8 hidden neurons is generous for a 7-input problem. 4 is often enough and halves the (already tiny) parameter count.
