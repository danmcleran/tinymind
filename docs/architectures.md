---
title: Architectures
layout: default
nav_order: 6
has_children: true
---

# Network Architectures

TinyMind provides a range of neural network architectures, all as header-only C++ templates with both fixed-point and floating-point support.

| Architecture | Memory (Q8.8, trainable) | Key Advantage |
|---|---|---|
| [LSTM & GRU]({{ site.baseurl }}/architectures/lstm-gru) | 952 / 808 bytes | Sequential data, temporal patterns |
| [Kolmogorov-Arnold Networks]({{ site.baseurl }}/architectures/kan) | 1,192 bytes | Learnable activation functions |
| [Conv & Pooling Layers]({{ site.baseurl }}/architectures/conv-pooling) | 1,825 bytes (pipeline) | Time-series feature extraction |
| [Linear Self-Attention]({{ site.baseurl }}/architectures/self-attention) | ~6 KB (mid-range) | Sequence dependency modeling without O(N^2) |
| [Quantized Networks]({{ site.baseurl }}/architectures/quantized-networks) | 128 bytes (packed binary) | 32-64x weight compression |
