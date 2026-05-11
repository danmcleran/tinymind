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
| [Conv & Pooling Layers]({{ site.baseurl }}/architectures/conv-pooling) | 1,825 bytes (1D pipeline) | 1D time-series + 2D spectrogram/image feature extraction, MobileNet-style separable blocks |
| [Linear Self-Attention]({{ site.baseurl }}/architectures/self-attention) | ~6 KB (mid-range) | Sequence dependency modeling without O(N^2) |
| [FFT Layer]({{ site.baseurl }}/architectures/fft) | 768 bytes (64-pt Q8.8) | Frequency-domain feature extraction |
| [Quantized Networks]({{ site.baseurl }}/architectures/quantized-networks) | 128 bytes (packed binary) | 32-64x weight compression |
| [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) | int8 weights + int32 accum | TFLite/CMSIS-NN style post-training int8 across dense/conv/pool/BN/LN/softmax/RNN/attention/FFT |
| [Mixed Precision]({{ site.baseurl }}/architectures/mixed-precision) | int8 + fp16 + bf16 bridges | Phase 9 qbridge converters between int8 affine / `QValue` Q-format / float / fp16 / bf16 |
| [SIMD Backends]({{ site.baseurl }}/architectures/simd-backends) | n/a (perf, not capacity) | Phase 14 ISA-capability gates: NEON / SVE / Helium / AVX2 / AVX-512, byte-identical to scalar |
