"""
Copyright (c) 2026 Dan McLeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
PyTorch -> TinyMind int8 importer (Phase 15).

Generalizes the pattern from examples/pytorch_quant/xor/xor_quant.py. The
caller assembles a sequenced list of `Layer` descriptors that mirror the
TinyMind q*.hpp shapes (QDense / QConv2D / QBatchNorm2D / QReLU / QSigmoid /
QTanh / QSoftmax1D) and supplies a calibration callback that streams a
representative dataset through the float reference model. The importer:

  1. Detects Conv2D-followed-by-BatchNorm and folds the BN parameters into
     the conv weights/bias via foldBatchNorm semantics.
  2. Observes per-tensor activation ranges using one of:
       - MinMaxObserver       (= cpp RangeObserver)
       - PercentileObserver   (e.g. 0.05 / 99.95)
       - KLDivergenceObserver (TensorRT-style)
     The user picks per layer; default is MinMax.
  3. Symmetric per-tensor weight quantization (zero_point = 0, qmax = 127).
  4. Asymmetric activation quantization (zero_point in [-128, 127]).
  5. Emits a TinyMind-format weights.hpp.

The C++ side rebuilds Requantizers at startup via buildRequantizer from
qcalibration.hpp; the deployable MCU shape bakes the integer
(multiplier, shift, zero_point) triples in directly.

Rounding convention is std::lround (ties away from zero) so the metadata
emitted here is bit-identical to what the C++ Requantizer expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Rounding / quantization primitives (mirror cpp/include/qcalibration.hpp).
# ----------------------------------------------------------------------------


def lround(x: float) -> int:
    """C++ std::lround semantics: ties round away from zero."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def affine_asymmetric(fmin: float, fmax: float,
                      qmin: int = -128, qmax: int = 127) -> tuple[float, int]:
    """Mirror cpp/include/qcalibration.hpp::computeAffineParamsAsymmetric."""
    if fmin > 0.0:
        fmin = 0.0
    if fmax < 0.0:
        fmax = 0.0
    qrange = float(qmax - qmin)
    if fmax == fmin or qrange == 0.0:
        return 1.0, qmin
    scale = (fmax - fmin) / qrange
    zp = lround(qmin - fmin / scale)
    zp = max(qmin, min(qmax, zp))
    return scale, int(zp)


def symmetric_weight_scale(w: np.ndarray, qmax_signed: int = 127) -> float:
    absmax = float(np.max(np.abs(w))) if w.size else 0.0
    if absmax == 0.0:
        return 1.0
    return absmax / float(qmax_signed)


def quantize_symmetric_weights(w: np.ndarray, scale: float) -> np.ndarray:
    flat = w.reshape(-1).tolist()
    q = [max(-127, min(127, lround(v / scale))) for v in flat]
    return np.array(q, dtype=np.int8).reshape(w.shape)


def quantize_bias_int32(b: np.ndarray, bias_scale: float) -> np.ndarray:
    qmin = -(1 << 31)
    qmax = (1 << 31) - 1
    q = [max(qmin, min(qmax, lround(v / bias_scale))) for v in b.tolist()]
    return np.array(q, dtype=np.int32)


def quantize_multiplier(ratio: float) -> tuple[int, int]:
    """Mirror cpp/qaffine.hpp::quantizeMultiplier.

    Decomposes a positive real ratio into a Q0.31 (multiplier, shift) pair
    consumed by cpp/qaffine.hpp::multiplyByQuantizedMultiplier. Shift > 0
    right-shifts after the multiply; shift < 0 left-shifts before it.
    """
    if ratio == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(ratio)
    q = lround(mantissa * float(1 << 31))
    if q == (1 << 31):
        q >>= 1
        exponent += 1
    return int(q), -int(exponent)


def quantize_qformat_weights(w: np.ndarray, fractional_bits: int,
                             raw_width_bits: int, signed: bool) -> np.ndarray:
    """Convert float weights to raw QValue integers.

    The raw integer fits in `raw_width_bits` so the caller can choose an
    int8/int16/int32/int64 storage cell. Rounding is round-half-away-from-zero
    matching cpp/qformat.hpp::floatToQValue.
    """
    scale = float(1 << fractional_bits)
    if signed:
        qmin = -(1 << (raw_width_bits - 1))
        qmax = (1 << (raw_width_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << raw_width_bits) - 1
    flat = w.reshape(-1).tolist()
    q = [max(qmin, min(qmax, lround(v * scale))) for v in flat]
    if raw_width_bits <= 8:
        dt = np.int8 if signed else np.uint8
    elif raw_width_bits <= 16:
        dt = np.int16 if signed else np.uint16
    elif raw_width_bits <= 32:
        dt = np.int32 if signed else np.uint32
    else:
        dt = np.int64 if signed else np.uint64
    return np.array(q, dtype=dt).reshape(w.shape)


# ----------------------------------------------------------------------------
# Observers.
# ----------------------------------------------------------------------------


class MinMaxObserver:
    """Streaming min/max range observer."""

    def __init__(self) -> None:
        self.min = math.inf
        self.max = -math.inf
        self.has_data = False

    def observe(self, x: np.ndarray) -> None:
        flat = x.reshape(-1)
        if flat.size == 0:
            return
        lo = float(flat.min())
        hi = float(flat.max())
        if lo < self.min:
            self.min = lo
        if hi > self.max:
            self.max = hi
        self.has_data = True

    def range(self) -> tuple[float, float]:
        if not self.has_data:
            return 0.0, 0.0
        return self.min, self.max


class PercentileObserver:
    """Percentile clipping observer. Stores every sample."""

    def __init__(self, lower: float = 0.05, upper: float = 99.95) -> None:
        self.lower = lower
        self.upper = upper
        self._samples: list[np.ndarray] = []
        self.has_data = False

    def observe(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._samples.append(x.reshape(-1).astype(np.float32))
        self.has_data = True

    def range(self) -> tuple[float, float]:
        if not self.has_data:
            return 0.0, 0.0
        cat = np.concatenate(self._samples)
        cat.sort()
        n = cat.shape[0]
        lo_idx = max(0, min(n - 1, int(round((self.lower / 100.0) * (n - 1)))))
        hi_idx = max(0, min(n - 1, int(round((self.upper / 100.0) * (n - 1)))))
        return float(cat[lo_idx]), float(cat[hi_idx])


class KLDivergenceObserver:
    """TensorRT-style symmetric calibration via KL divergence."""

    NUM_BINS = 2048
    TARGET_BINS = 128

    def __init__(self) -> None:
        self.absmax = 0.0
        self._samples: list[np.ndarray] = []
        self.has_data = False

    def observe(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._samples.append(x.reshape(-1).astype(np.float32))
        am = float(np.max(np.abs(x)))
        if am > self.absmax:
            self.absmax = am
        self.has_data = True

    def range(self) -> tuple[float, float]:
        if not self.has_data or self.absmax <= 0.0:
            return 0.0, 0.0
        cat = np.abs(np.concatenate(self._samples))
        bins = self.NUM_BINS
        target = self.TARGET_BINS
        hist, _ = np.histogram(cat, bins=bins, range=(0.0, self.absmax))
        best_kl = math.inf
        best_T = target
        for T in range(target, bins + 1):
            P = hist[:T].astype(np.float64).copy()
            outliers = float(hist[T:].sum())
            P[-1] += outliers
            Q = np.zeros(T, dtype=np.float64)
            for k in range(target):
                start = int((k * T) / target)
                end = int(((k + 1) * T) / target)
                end = min(end, T)
                if end <= start:
                    continue
                seg = P[start:end]
                nz = (hist[start:end] > 0)
                if not nz.any():
                    continue
                avg = seg.sum() / float(nz.sum())
                Q[start:end] = np.where(nz, avg, 0.0)
            ps = P.sum()
            qs = Q.sum()
            if ps <= 0.0 or qs <= 0.0:
                continue
            p_norm = P / ps
            q_norm = Q / qs
            mask = (p_norm > 0.0) & (q_norm > 0.0)
            kl = float(np.sum(p_norm[mask] * np.log(p_norm[mask] / q_norm[mask])))
            if kl < best_kl:
                best_kl = kl
                best_T = T
        threshold = self.absmax * (best_T / bins)
        return -threshold, threshold


# ----------------------------------------------------------------------------
# Layer descriptors.
# ----------------------------------------------------------------------------


@dataclass
class Layer:
    """Base layer descriptor.

    name : unique tag used to label the activation tensor flowing OUT of this
           layer (its 'output_scale' / 'output_zp' will be emitted under this
           name in weights.hpp).
    observer : range observer for the output tensor. Defaults to MinMax.
    """

    name: str
    observer: object = field(default_factory=MinMaxObserver)


@dataclass
class Dense(Layer):
    weight: np.ndarray = None       # [out, in]
    bias: Optional[np.ndarray] = None
    input_name: str = ""            # which tensor feeds this layer
    forward: Optional[Callable[[np.ndarray], np.ndarray]] = None
    # Filled by importer:
    weight_scale: float = 0.0
    q_weight: np.ndarray = None
    q_bias: np.ndarray = None


@dataclass
class Conv2D(Layer):
    """Conv2D with OHWI weight layout [num_filters][kh][kw][in_channels]."""
    weight: np.ndarray = None
    bias: Optional[np.ndarray] = None
    input_name: str = ""
    forward: Optional[Callable[[np.ndarray], np.ndarray]] = None
    weight_scale: float = 0.0
    q_weight: np.ndarray = None
    q_bias: np.ndarray = None


@dataclass
class BatchNorm2D(Layer):
    """BN that may fuse into a preceding Conv2D."""
    gamma: np.ndarray = None
    beta: np.ndarray = None
    mean: np.ndarray = None
    variance: np.ndarray = None
    eps: float = 1.0e-5
    input_name: str = ""


@dataclass
class ReLU(Layer):
    input_name: str = ""


@dataclass
class Sigmoid(Layer):
    input_name: str = ""


@dataclass
class Tanh(Layer):
    input_name: str = ""


@dataclass
class Softmax(Layer):
    input_name: str = ""
    axis: int = -1


@dataclass
class QFormatDense(Layer):
    """Q-format dense layer (TinyMind QValue pipeline).

    Drops alongside the int8 affine layers so a hybrid model can mix
    QDense (int8) with QFormatDense (Q-format) inside the same weights.hpp.
    The weights/bias are emitted as raw QValue integers — no scale or
    zero_point metadata at runtime. Boundaries with int8 layers should be
    declared via HybridBoundary so the emitter writes the precomputed
    integer-bridge params for cpp/qbridge.hpp::affineToQValueInt /
    qValueToAffineInt.
    """
    weight: np.ndarray = None          # [out, in]
    bias: Optional[np.ndarray] = None
    input_name: str = ""
    forward: Optional[Callable[[np.ndarray], np.ndarray]] = None
    fractional_bits: int = 8
    fixed_bits: int = 8
    signed: bool = True
    # Filled by importer:
    q_weight: np.ndarray = None
    q_bias: np.ndarray = None

    @property
    def raw_width_bits(self) -> int:
        return self.fixed_bits + self.fractional_bits

    @property
    def qformat_tag(self) -> str:
        s = "" if self.signed else "u"
        return f"Q{self.fixed_bits}_{self.fractional_bits}{s}"


@dataclass
class HybridBoundary:
    """Precision-tier transition between two adjacent layers.

    `from_name` produces the tensor; `to_name` consumes it. `kind` is one of
    "affine_to_qvalue" (int8 -> QValue) or "qvalue_to_affine" (QValue -> int8).
    `qformat` is the QFormatDense the bridge talks to, used to fold the
    fractional-bits factor into the integer multiplier/shift. The emitter
    writes one C++ struct literal per boundary.
    """
    from_name: str
    to_name: str
    kind: str                          # "affine_to_qvalue" | "qvalue_to_affine"
    qformat: "QFormatDense"            # carries fractional_bits / signed
    qmin: int = -128
    qmax: int = 127


# ----------------------------------------------------------------------------
# Fusion pass.
# ----------------------------------------------------------------------------


def fold_conv_bn(conv: Conv2D, bn: BatchNorm2D) -> Conv2D:
    """Fold BN into a preceding Conv2D, returning the fused Conv2D.

    Implements the same math as cpp/include/qcalibration.hpp::foldBatchNorm.
    Weight layout is [F, KH, KW, IC]; sigma_eff and bias correction apply
    per output filter.
    """
    sigma_eff = bn.gamma / np.sqrt(bn.variance + bn.eps)
    fused_w = conv.weight * sigma_eff.reshape(-1, 1, 1, 1)
    if conv.bias is not None:
        fused_b = (conv.bias - bn.mean) * sigma_eff + bn.beta
    else:
        fused_b = -bn.mean * sigma_eff + bn.beta
    fused = Conv2D(
        name=bn.name,
        observer=bn.observer,
        weight=fused_w,
        bias=fused_b,
        input_name=conv.input_name,
        forward=conv.forward,
    )
    return fused


def fuse_layers(layers: Sequence[Layer]) -> list[Layer]:
    """Detect Conv2D-then-BatchNorm2D pairs and fuse them in-place.

    A BatchNorm immediately downstream of a Conv2D collapses into the
    conv weights/bias. The fused layer inherits the BN's observer (so the
    activation observed at runtime is the fused-conv output) and the BN's
    output name (downstream layers referencing the BN keep working).
    """
    out: list[Layer] = []
    i = 0
    while i < len(layers):
        a = layers[i]
        if (i + 1 < len(layers)
                and isinstance(a, Conv2D)
                and isinstance(layers[i + 1], BatchNorm2D)
                and layers[i + 1].input_name == a.name):
            fused = fold_conv_bn(a, layers[i + 1])
            out.append(fused)
            i += 2
            continue
        out.append(a)
        i += 1
    return out


# ----------------------------------------------------------------------------
# Calibration driver.
# ----------------------------------------------------------------------------


def calibrate(layers: Sequence[Layer],
              input_name: str,
              input_observer: object,
              dataset: Iterable[np.ndarray]) -> dict[str, tuple[float, int]]:
    """Stream `dataset` through the float reference model, observe ranges.

    Each layer must carry a `forward` callable for layers that produce a
    new tensor (Dense / Conv2D). Activation-only layers (ReLU / Sigmoid /
    Tanh / Softmax / BatchNorm) compute their output from the parent
    tensor via numpy directly.

    Returns a dict of tensor_name -> (scale, zero_point).
    """
    activations: dict[str, np.ndarray] = {}
    for x in dataset:
        x_f = x.astype(np.float32)
        input_observer.observe(x_f)
        activations[input_name] = x_f
        # Reset for-this-sample storage.
        per_sample: dict[str, np.ndarray] = {input_name: x_f}
        for layer in layers:
            parent = per_sample[layer.input_name]
            if isinstance(layer, (Dense, Conv2D, QFormatDense)):
                y = layer.forward(parent)
            elif isinstance(layer, BatchNorm2D):
                sigma_eff = layer.gamma / np.sqrt(layer.variance + layer.eps)
                y = (parent - layer.mean) * sigma_eff + layer.beta
            elif isinstance(layer, ReLU):
                y = np.maximum(parent, 0.0)
            elif isinstance(layer, Sigmoid):
                y = 1.0 / (1.0 + np.exp(-parent))
            elif isinstance(layer, Tanh):
                y = np.tanh(parent)
            elif isinstance(layer, Softmax):
                shifted = parent - np.max(parent, axis=layer.axis, keepdims=True)
                ex = np.exp(shifted)
                y = ex / np.sum(ex, axis=layer.axis, keepdims=True)
            else:
                raise NotImplementedError(type(layer).__name__)
            layer.observer.observe(y)
            per_sample[layer.name] = y

    ranges: dict[str, tuple[float, int]] = {}
    ranges[input_name] = affine_asymmetric(*input_observer.range())
    for layer in layers:
        if isinstance(layer, Softmax):
            # TFLite-fixed: scale 1/256, zero_point -128.
            ranges[layer.name] = (1.0 / 256.0, -128)
            continue
        if isinstance(layer, Sigmoid):
            ranges[layer.name] = (1.0 / 256.0, -128)
            continue
        if isinstance(layer, Tanh):
            ranges[layer.name] = (1.0 / 128.0, 0)
            continue
        ranges[layer.name] = affine_asymmetric(*layer.observer.range())
    return ranges


# ----------------------------------------------------------------------------
# Weight quantization pass.
# ----------------------------------------------------------------------------


def quantize_weights(layers: Sequence[Layer],
                     ranges: dict[str, tuple[float, int]]) -> None:
    """Symmetric per-tensor weight quantization. Mutates each layer."""
    for layer in layers:
        if isinstance(layer, (Dense, Conv2D)):
            layer.weight_scale = symmetric_weight_scale(layer.weight)
            layer.q_weight = quantize_symmetric_weights(
                layer.weight, layer.weight_scale)
            if layer.bias is not None:
                input_scale = ranges[layer.input_name][0]
                bias_scale = input_scale * layer.weight_scale
                layer.q_bias = quantize_bias_int32(layer.bias, bias_scale)
        elif isinstance(layer, QFormatDense):
            layer.q_weight = quantize_qformat_weights(
                layer.weight, layer.fractional_bits,
                layer.raw_width_bits, layer.signed)
            if layer.bias is not None:
                layer.q_bias = quantize_qformat_weights(
                    layer.bias, layer.fractional_bits,
                    layer.raw_width_bits, layer.signed)


# ----------------------------------------------------------------------------
# Emitter.
# ----------------------------------------------------------------------------


def _fmt_int_array(name: str, values: Iterable[int], cpp_type: str,
                   per_line: int = 12) -> str:
    vs = list(values)
    lines: list[str] = []
    for i in range(0, len(vs), per_line):
        chunk = ", ".join(f"{v:>5d}" for v in vs[i:i + per_line])
        lines.append("        " + chunk + ("," if i + per_line < len(vs) else ""))
    body = "\n".join(lines)
    return f"    constexpr {cpp_type} {name}[{len(vs)}] = {{\n{body}\n    }};\n"


COPYRIGHT_BLOCK = """/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import.
*/"""


def _qformat_cpp_storage_type(width_bits: int, signed: bool) -> str:
    if width_bits <= 8:
        return "int8_t" if signed else "uint8_t"
    if width_bits <= 16:
        return "int16_t" if signed else "uint16_t"
    if width_bits <= 32:
        return "int32_t" if signed else "uint32_t"
    return "int64_t" if signed else "uint64_t"


def _emit_bridge_params(boundary: HybridBoundary,
                        ranges: dict[str, tuple[float, int]]) -> str:
    """Emit one bridge-param struct literal.

    affine_to_qvalue: encode ratio = upstream_scale * 2^F
                      (cpp/qbridge.hpp::buildAffineToQValueIntParams).
    qvalue_to_affine: encode ratio = 1 / (downstream_scale * 2^F)
                      (cpp/qbridge.hpp::buildQValueToAffineIntParams).
    """
    q = boundary.qformat
    frac = float(1 << q.fractional_bits)
    if boundary.kind == "affine_to_qvalue":
        upstream_scale, upstream_zp = ranges[boundary.from_name]
        ratio = float(upstream_scale) * frac
        m, s = quantize_multiplier(ratio)
        nm = f"k{boundary.from_name}_to_{boundary.to_name}_Bridge"
        return (
            f"    // affine ({boundary.from_name}) -> QValue ({boundary.to_name}).\n"
            f"    constexpr int32_t {nm}_Multiplier = {m};\n"
            f"    constexpr int32_t {nm}_Shift      = {s};\n"
            f"    constexpr int32_t {nm}_ZeroPoint  = {int(upstream_zp)};\n"
        )
    if boundary.kind == "qvalue_to_affine":
        downstream_scale, downstream_zp = ranges[boundary.to_name]
        ratio = 1.0 / (float(downstream_scale) * frac)
        m, s = quantize_multiplier(ratio)
        nm = f"k{boundary.from_name}_to_{boundary.to_name}_Bridge"
        return (
            f"    // QValue ({boundary.from_name}) -> affine ({boundary.to_name}).\n"
            f"    constexpr int32_t {nm}_Multiplier = {m};\n"
            f"    constexpr int32_t {nm}_Shift      = {s};\n"
            f"    constexpr int32_t {nm}_ZeroPoint  = {int(downstream_zp)};\n"
            f"    constexpr int32_t {nm}_QMin       = {int(boundary.qmin)};\n"
            f"    constexpr int32_t {nm}_QMax       = {int(boundary.qmax)};\n"
        )
    raise ValueError(f"unknown boundary kind: {boundary.kind}")


def emit_weights_header(path: str,
                        namespace: str,
                        layers: Sequence[Layer],
                        ranges: dict[str, tuple[float, int]],
                        input_name: str,
                        meta: Optional[dict[str, int]] = None,
                        boundaries: Sequence[HybridBoundary] = ()) -> None:
    """Write a TinyMind-format weights.hpp.

    namespace  : C++ namespace for the constants.
    meta       : optional dict of NameInCpp -> integer literal (e.g.
                 {"NumInputs": 2, "HiddenSize": 4}).
    boundaries : optional precision-tier transitions. For each entry the
                 emitter writes the precomputed integer-bridge triple consumed
                 by cpp/qbridge.hpp::affineToQValueInt / qValueToAffineInt so
                 the deployable freestanding shape needs no host-side
                 buildAffineToQValueIntParams call at startup.
    """
    parts: list[str] = []
    parts.append(COPYRIGHT_BLOCK)
    parts.append("\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n")
    parts.append(f"namespace {namespace} {{\n")
    if meta is not None:
        for k, v in meta.items():
            parts.append(f"    constexpr std::size_t {k} = {v};\n")
        parts.append("\n")

    parts.append("    // Activation calibration.\n")
    for name, (scale, zp) in ranges.items():
        cpp_name = name.replace("/", "_").replace("-", "_")
        parts.append(f"    constexpr float   k{cpp_name}_Scale     = {scale!r}f;\n")
        parts.append(f"    constexpr int32_t k{cpp_name}_ZeroPoint = {zp};\n")
    parts.append("\n    // Weight calibration + int8 weights + int32 biases.\n")
    for layer in layers:
        if not isinstance(layer, (Dense, Conv2D)):
            continue
        nm = layer.name.replace("/", "_").replace("-", "_")
        parts.append(f"    constexpr float k{nm}_WeightScale = {layer.weight_scale!r}f;\n")
        parts.append(_fmt_int_array(f"k{nm}_Weights",
                                    layer.q_weight.reshape(-1).tolist(),
                                    "int8_t"))
        if layer.q_bias is not None:
            parts.append(_fmt_int_array(f"k{nm}_Biases",
                                        layer.q_bias.tolist(),
                                        "int32_t"))

    if any(isinstance(layer, QFormatDense) for layer in layers):
        parts.append("\n    // Q-format weights (TinyMind QValue pipeline).\n")
        for layer in layers:
            if not isinstance(layer, QFormatDense):
                continue
            nm = layer.name.replace("/", "_").replace("-", "_")
            storage = _qformat_cpp_storage_type(layer.raw_width_bits, layer.signed)
            parts.append(f"    // {nm}: {layer.qformat_tag} "
                         f"(fixed={layer.fixed_bits}, frac={layer.fractional_bits})\n")
            parts.append(f"    constexpr std::size_t k{nm}_FractionalBits = "
                         f"{layer.fractional_bits};\n")
            parts.append(f"    constexpr std::size_t k{nm}_FixedBits      = "
                         f"{layer.fixed_bits};\n")
            parts.append(_fmt_int_array(f"k{nm}_Weights",
                                        layer.q_weight.reshape(-1).tolist(),
                                        storage))
            if layer.q_bias is not None:
                parts.append(_fmt_int_array(f"k{nm}_Biases",
                                            layer.q_bias.reshape(-1).tolist(),
                                            storage))

    if boundaries:
        parts.append("\n    // Hybrid precision-tier bridges (cpp/qbridge.hpp).\n")
        for b in boundaries:
            parts.append(_emit_bridge_params(b, ranges))

    parts.append(f"}} // namespace {namespace}\n")
    with open(path, "w") as f:
        f.write("".join(parts))


# ----------------------------------------------------------------------------
# Top-level convenience.
# ----------------------------------------------------------------------------


def import_pytorch_model(layers: Sequence[Layer],
                         input_name: str,
                         input_observer: object,
                         dataset: Iterable[np.ndarray],
                         output_path: str,
                         namespace: str,
                         meta: Optional[dict[str, int]] = None,
                         boundaries: Sequence[HybridBoundary] = ()
                         ) -> dict[str, tuple[float, int]]:
    """End-to-end: fuse -> calibrate -> quantize -> emit weights.hpp.

    boundaries declares any int8 <-> QFormatDense transitions; the emitter
    writes the precomputed integer-bridge triples alongside the rest of the
    weights so the deployable freestanding shape FLOAT=0 STD=0 QUANT=1
    consumes them as data.

    Returns the activation ranges dict (useful for diagnostics / parity tests).
    """
    fused = fuse_layers(layers)
    ranges = calibrate(fused, input_name, input_observer, dataset)
    quantize_weights(fused, ranges)
    emit_weights_header(output_path, namespace, fused, ranges, input_name,
                        meta, boundaries)
    return ranges
