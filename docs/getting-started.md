---
title: Getting Started
layout: default
nav_order: 5
has_children: true
---

# Getting Started

These tutorials walk through complete, working examples that demonstrate TinyMind's capabilities on real problems. Each tutorial includes source code, build instructions, and size analysis.

| Tutorial | What You'll Learn | Final Size |
|---|---|---|
| [Neural Network in Under 4KB]({{ site.baseurl }}/getting-started/xor-under-4kb) | Feed-forward NN with fixed-point, XOR prediction | 3,892 bytes |
| [Q-Learning in Under 1KB]({{ site.baseurl }}/getting-started/q-learning-under-1kb) | Tabular Q-learning, maze solving | 869 bytes |
| [DQN Maze Solver]({{ site.baseurl }}/getting-started/dqn-maze-solver) | Deep Q-Network with neural network function approximation | ~16 KB |
| [Keyword Spotting CNN on a Cortex-M]({{ site.baseurl }}/getting-started/keyword-spotting-cnn) | Depthwise-separable 2D CNN, bench harness, MCU porting | ~19 KB static |
| [Predictive Maintenance on AI4I 2020]({{ site.baseurl }}/getting-started/predictive-maintenance) | Q16.16 MLP, imbalanced binary classification, confusion matrix | ~35 KB static |
| [PyTorch -> TinyMind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) | End-to-end post-training int8 quantization: PyTorch float training, per-tensor calibration, pure-integer C++ inference | Tiny |
| [PyTorch -> TinyMind int8 (importer)]({{ site.baseurl }}/getting-started/pytorch-importer) | Production flow: `torch.state_dict` -> `tinymind_import.py` -> `weights.hpp`. `PercentileObserver` / `KLDivergenceObserver` / cross-layer equalization | Tiny |
| [Keyword Spotting CNN (int8)]({{ site.baseurl }}/getting-started/keyword-spotting-int8) | int8 quantized depthwise-separable CNN, per-channel depthwise, CSV cycle/byte report vs float | ~5 KB static |
| [MobileNetV2-shaped int8]({{ site.baseurl }}/getting-started/mobilenetv2-int8) | int8 exemplar: stride-2 stem + inverted-residual blocks + GAP + dense, linear-bottleneck convention, golden-byte regression | Compact |
