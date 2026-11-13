---
title: On-Device Fine-Tuning
layout: default
parent: Training & Deployment
nav_order: 3
---

# On-Device Fine-Tuning

A model trained on the host meets data on the device that it never saw: a sensor ages, a user moves differently, a new class appears. Retraining the whole network on a microcontroller is expensive. It needs the training state for every layer and backpropagation through all of them. **Head-only fine-tuning** keeps the shipped model frozen and adapts only a small classifier on top of its embedding, from a few labelled samples, on the device.

tinymind provides two heads for this, both standalone layers that sit after any model that produces an embedding: an int8 pipeline (through the [mixed-precision bridges]({{ site.baseurl }}/architectures/mixed-precision)), a `QValue` network, or anything else.

| Head | Header | How it learns | RAM (32-value embedding, 6 classes, Q16.16) |
|---|---|---|---|
| `DenseHead` | `cpp/densehead.hpp` | SGD on a dense layer, softmax cross-entropy or squared error | 792 bytes |
| `PrototypeHead` | `cpp/prototypehead.hpp` | One running-mean embedding per class, nearest mean wins | 784 bytes |

Nothing is backpropagated through the frozen model, so there are no stored activations and no gradients for it. `DenseHead` fuses its update into the backward pass and keeps no gradient buffer, so its RAM is its parameters: (inputs&nbsp;+&nbsp;1)&nbsp;&times;&nbsp;outputs values. Neither head allocates.

## DenseHead

```cpp
#include "densehead.hpp"

typedef tinymind::QValue<16, 16, true> ValueType;
typedef tinymind::DenseHead<ValueType, 32, 6> HeadType;   // softmax cross-entropy

HeadType head;
head.loadWeights(shippedWeights, shippedBiases);  // the shipped head, kept in flash

// On the device, for each labelled sample:
backbone(sample, embedding);                       // frozen model -> 32 values
head.trainClass(embedding, label, learningRate);   // one SGD step

const size_t predicted = head.predict(embedding);
```

- **Loss policies.** `SoftmaxCrossEntropyHeadLoss` (the default) subtracts the largest logit before `exp`, so no fixed-point format overflows however large the logits grow. In fixed point it uses the exp lookup table: define `TINYMIND_USE_EXP_<I>_<F>` and link `lookupTables.cpp`. A class more than 6 behind the largest logit, beyond the table, gets probability 0, so a well-separated head stops moving instead of growing its logits until the format wraps. `SquaredErrorHeadLoss` is a linear output with squared error. It needs no lookup table and holds at `TINYMIND_ENABLE_FLOAT=0`, `TINYMIND_ENABLE_STD=0`.
- **The step.** Each parameter moves by learning rate &times; (delta &times; input), rounded to nearest by `multiplyRoundNearest` as every trainable layer in tinymind does. There is no momentum.
- **Undoing an adaptation.** `loadWeights()` restores the shipped head bit for bit. Keep the shipped parameters in flash rather than spend RAM on a copy.
- **Adapting more than the head.** `train()` and `trainClass()` take an optional `inputDeltas` buffer that receives the gradient with respect to the embedding, computed before the update; `backward()` gives the same thing on its own. The gas sensor example uses it to train the biases of the backbone's last layer.
- **Range.** `QValue` wraps on overflow by default. Keep the embedding and the logits well inside the format: z-scored inputs and a Q16.16 head are the working recipe, and Q8.8 is too narrow for a wide head.

## PrototypeHead

```cpp
#include "prototypehead.hpp"

typedef tinymind::PrototypeHead<ValueType, 32, 6> PrototypeType;  // squared Euclidean

PrototypeType head;
backbone(sample, embedding);
head.learn(embedding, label);                     // fold into the class mean
const size_t predicted = head.predict(embedding);  // nearest prototype
```

- There is no learning rate and no epoch count. One pass over the labelled samples is the whole training.
- A class can be added on the device at any time, and a class never learned is never predicted. `predict()` returns the number of classes when nothing has been learned.
- `setMaximumCount(n)` caps each class's count. Past the cap the prototype becomes an exponential moving average with weight 1/n, so it keeps tracking a slowly drifting input instead of freezing on old samples.
- `setPrototype()` loads a shipped prototype with the weight it carries against new samples. `forget()` removes one class.
- `ManhattanDistance` replaces the default `SquaredEuclideanDistance` when multiplies are expensive or the squared distance would overflow the format.

## Which one, measured

The [gas sensor drift example]({{ site.baseurl }}/examples/gas_sensor_drift) adapts a Q16.16 128&nbsp;&rarr;&nbsp;32&nbsp;&rarr;&nbsp;6 classifier to a drifted batch from 1 to 20 labelled samples per class. On its synthetic data, batch 10 accuracy over 10 data seeds:

| Method | 1 per class | 5 per class |
|---|---|---|
| No adaptation | 0.766 &plusmn; 0.097 | 0.765 &plusmn; 0.097 |
| `PrototypeHead` | 0.985 &plusmn; 0.008 | 0.998 &plusmn; 0.002 |
| `DenseHead` | 0.936 &plusmn; 0.025 | 0.988 &plusmn; 0.005 |
| Backbone biases only | 0.876 &plusmn; 0.039 | 0.969 &plusmn; 0.015 |
| Biases + `DenseHead` | 0.940 &plusmn; 0.026 | 0.989 &plusmn; 0.006 |
| Re-normalized inputs (whole batch, unlabelled) | 0.924 &plusmn; 0.052 | 0.924 &plusmn; 0.052 |

How to choose:

- **Start with `PrototypeHead`** when the frozen model's embedding still separates the classes and they have only moved: drift, a new user, few-shot classes. It is the cheapest to run and was the most accurate here.
- **Use `DenseHead`** when the head must keep what it knew while it learns something new. After adapting to the drifted batch it still classified the original batch perfectly; the `PrototypeHead`, which moves its class means, did not (0.70 to 0.75 over the 10 seeds).
- **Adapt more than the head** when the embedding itself stops separating the classes. On this data it never did, so the biases added almost nothing. Published work on real data (TinyTL, MCUNetV3, BioTrain) finds that updating more than the last layer gives higher accuracy than the last layer alone.

## Memory, against training the whole network

For the gas example's network in Q16.16:

| Configuration | Bytes |
|---|---|
| Trainable `MultilayerPerceptron` 128&nbsp;&rarr;&nbsp;32&nbsp;&rarr;&nbsp;6 | 73,400 |
| Inference-only `MultilayerPerceptron` 128&nbsp;&rarr;&nbsp;32&nbsp;&rarr;&nbsp;6 | 20,032 |
| `DenseHead<Q16.16, 32, 6>` | 792 |
| `PrototypeHead<Q16.16, 32, 6>` | 784 |

An inference-only model plus a head adapts in 28% of the RAM that training the whole network takes. With an int8 backbone whose weights stay in flash, the head is most of the RAM that adaptation needs.

## What the application still owns

- **Saving the adapted head.** Writing it to flash, how often, and the wear that causes are the application's decisions; the heads only expose their parameters.
- **Forgetting.** A few samples can overwrite what the shipped head knew. Keep a handful of samples from the original classes and train on them alongside the new ones, or reload the shipped head when adaptation goes wrong.
- **Label quality.** Every method here trusts its labels; a wrong label on the device is a wrong update.
