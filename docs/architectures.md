---
title: Architectures
layout: default
nav_order: 7
has_children: true
---

# Network Architectures

TinyMind provides a range of neural network architectures, all as header-only C++ templates with both fixed-point and floating-point support.

| Architecture | Memory (Q8.8, trainable) | Key Advantage |
|---|---|---|
| [LSTM, GRU & Liquid (LTC/CfC)]({{ site.baseurl }}/architectures/lstm-gru) | 952 / 808 bytes | Sequential data, temporal patterns; continuous-time cells for irregular sampling |
| [Kolmogorov-Arnold Networks]({{ site.baseurl }}/architectures/kan) | 1,192 bytes | Learnable activation functions |
| [Conv & Pooling Layers]({{ site.baseurl }}/architectures/conv-pooling) | 1,825 bytes (1D pipeline) | 1D time-series + 2D spectrogram/image feature extraction, MobileNet-style separable blocks |
| [Linear Self-Attention]({{ site.baseurl }}/architectures/self-attention) | ~6 KB (mid-range) | Sequence dependency modeling without O(N^2) |
| [FFT Layer]({{ site.baseurl }}/architectures/fft) | 768 bytes (64-pt Q8.8) | Frequency-domain feature extraction |
| [Quantized Networks]({{ site.baseurl }}/architectures/quantized-networks) | 128 bytes (packed binary) | 32-64x weight compression |
| [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) | int8 weights + int32 accum | TFLite/CMSIS-NN style post-training int8 across dense/conv/pool/BN/LN/softmax/RNN/attention/FFT |
| [Mixed Precision]({{ site.baseurl }}/architectures/mixed-precision) | int8 + fp16 + bf16 bridges | `qbridge` converters between int8 affine / `QValue` Q-format / float / fp16 / bf16 |
| [SIMD Backends]({{ site.baseurl }}/architectures/simd-backends) | n/a (perf, not capacity) | ISA-capability gates: NEON / SVE / Helium / AVX2 / AVX-512, byte-identical to scalar |
| [Mixture of Experts]({{ site.baseurl }}/architectures/mixture-of-experts) | router + N resident experts | Decouples compute from capacity: 1 of N experts runs per inference (top-1), or k-of-N softmax-blended |
