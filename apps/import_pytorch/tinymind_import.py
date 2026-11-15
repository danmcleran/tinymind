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

from __future__ import annotations

"""
PyTorch -> tinymind int8 importer (Phase 15).

Generalizes the pattern from examples/pytorch_quant/xor/xor_quant.py. The
caller assembles a sequenced list of `Layer` descriptors that mirror the
tinymind q*.hpp shapes (QDense / QConv2D / QBatchNorm2D / QReLU / QSigmoid /
QTanh / QSoftmax1D, and QLSTMCell for a model that starts with one nn.LSTM)
and supplies a calibration callback that streams a representative dataset
through the float reference model. The importer:

  1. Detects Conv2D-followed-by-BatchNorm and folds the BN parameters into
     the conv weights/bias via foldBatchNorm semantics.
  2. Observes per-tensor activation ranges using one of:
       - MinMaxObserver       (= cpp RangeObserver)
       - PercentileObserver   (e.g. 0.05 / 99.95)
       - KLDivergenceObserver (TensorRT-style)
     The user picks per layer; default is MinMax.
  3. Symmetric per-tensor weight quantization (zero_point = 0, qmax = 127).
  4. Asymmetric activation quantization (zero_point in [-128, 127]).
  5. Emits a tinymind-format weights.hpp.

The C++ side rebuilds Requantizers at startup via buildRequantizer from
qcalibration.hpp; the deployable MCU shape bakes the integer
(multiplier, shift, zero_point) triples in directly.

Rounding convention is std::lround (ties away from zero) so the metadata
emitted here is bit-identical to what the C++ Requantizer expects.
"""

import math
import re
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
    shift = -int(exponent)
    # Same clamp as the C++: keep the pair inside Requantizer's contract,
    # -31 <= shift <= 31, without changing any result. A ratio below 2^-31
    # rounds every int32 accumulator to 0, which multiplier 0 also says; a
    # left shift of 31 or more saturates every non-zero accumulator.
    if shift > 31:
        return 0, 0
    return int(q), max(shift, -31)


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
        # Mirrors KLDivergenceObserver::findBestThresholdBin in
        # cpp/include/qcalibration.hpp. Q is built from the *unfolded*
        # histogram: building it from the tail-folded P makes Q == P at
        # T == target and pins every result to absmax / 16. Empty bins are
        # epsilon-smoothed so a bin where P has mass and Q has none is
        # penalized rather than skipped (skipping drives the sum negative).
        eps = 1e-4
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
                seg = hist[start:end].astype(np.float64)
                nz = (P[start:end] > 0)
                if not nz.any():
                    continue
                avg = seg.sum() / float(nz.sum())
                Q[start:end] = np.where(nz, avg, 0.0)
            if not (P > 0.0).any() or not (Q > 0.0).any():
                continue
            P = np.where(P > 0.0, P, eps)
            Q = np.where(Q > 0.0, Q, eps)
            p_norm = P / P.sum()
            q_norm = Q / Q.sum()
            kl = float(np.sum(p_norm * np.log(p_norm / q_norm)))
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
class MoE(Layer):
    """Mixture-of-Experts layer with top-1 routing (cpp/qmoe.hpp).

    A linear router (NumExperts x NumInputs) scores the input; the argmax
    expert runs and produces the output. Experts are linear QDense layers
    (NumOutputs x NumInputs). All experts are resident; one runs per sample.

    Quantization shape:
      * Router weights: symmetric per-tensor int8. The router is NOT
        requantized at runtime -- the C++ layer argmaxes over the raw int32
        logits -- so the router weight scale only sets the bias scale and never
        affects the decision (argmax is invariant to a shared positive scale).
      * Router bias (optional): int32 at scale input_scale * router_weight_scale.
      * Each expert: its OWN symmetric per-tensor weight scale (per-expert
        scale), so every expert uses the full int8 range for its own weights.
      * Output activation: a SINGLE shared scale/zero_point observed over the
        realized MoE output (whichever expert fired). Each expert's Requantizer
        maps its int32 accumulator into this shared output scale, so a
        downstream layer sees one consistent quantization regardless of which
        expert produced the value.

    The float forward used for calibration is computed directly from the
    supplied float weights (experts are linear), so no `forward` callable is
    required.
    """
    router_weight: np.ndarray = None              # [num_experts, in]
    router_bias: Optional[np.ndarray] = None      # [num_experts]
    expert_weights: Sequence[np.ndarray] = None   # each [out, in]
    expert_biases: Optional[Sequence[np.ndarray]] = None  # each [out] or None
    input_name: str = ""
    # Filled by importer:
    router_weight_scale: float = 0.0
    q_router_weight: np.ndarray = None
    q_router_bias: np.ndarray = None
    expert_weight_scales: Sequence[float] = None
    q_expert_weights: Sequence[np.ndarray] = None
    q_expert_biases: Sequence[np.ndarray] = None

    @property
    def num_experts(self) -> int:
        return len(self.expert_weights)


def moe_float_forward(layer: "MoE", parent: np.ndarray) -> np.ndarray:
    """Top-1 float forward: argmax router logits, run that expert.

    parent is a single sample [in]. Mirrors cpp/qmoe.hpp routing: ties resolve
    to the lowest index (np.argmax convention).
    """
    logits = parent @ layer.router_weight.T
    if layer.router_bias is not None:
        logits = logits + layer.router_bias
    e = int(np.argmax(logits))
    y = parent @ layer.expert_weights[e].T
    if layer.expert_biases is not None and layer.expert_biases[e] is not None:
        y = y + layer.expert_biases[e]
    return y


# The LSTM gates saturate beyond +-8, so the gate pre-activations (and the
# LUTs that read them) use a fixed int8 grid of 8/127, as the QLSTMCell unit
# tests and TFLite's integer LSTM (Q3.12 gates) do. Not calibrated.
LSTM_LUT_INPUT_SCALE = 8.0 / 127.0


@dataclass
class LSTM(Layer):
    """One LSTM layer (PyTorch nn.LSTM, num_layers=1, unidirectional) mapped
    onto cpp/qlstm.hpp::QLSTMCell.

    The layer's input tensor is a whole sequence, [step][feature]; the cell
    runs it from zero hidden and cell state, and the tensor named `name` is
    the LAST hidden state, so the layers after it see one vector of
    `hidden_size` values. The LSTM must be the model's first layer.

    Gate order is PyTorch's (i, f, g, o), which is QLSTMCell's, so weight_ih
    and weight_hh are taken as they come from the state dict:
      weight_ih [4 * hidden, in]      (weight_ih_l0)
      weight_hh [4 * hidden, hidden]  (weight_hh_l0)
      bias_ih, bias_hh [4 * hidden]   (bias_ih_l0, bias_hh_l0; either may be
                                       None). The cell has one bias per gate,
                                       their sum.

    Quantization shape:
      * `observer` sees every hidden state of every calibration sequence and
        `cell_observer` every cell state. Both get symmetric scales with zero
        point 0, so the zero initial state is exact; the hidden scale is also
        the grid of the layer's output.
      * The cell state is int16 (`cell_bits=16`, the default: finer, for
        long sequences) or int8 (`cell_bits=8`).
      * Weights: symmetric int8 per gate, input and recurrent weights each
        with their own scale (8 scales).
      * Biases: int32 in the LUT input grid (quantizeQLSTMBiases).
    """
    weight_ih: np.ndarray = None
    weight_hh: np.ndarray = None
    bias_ih: Optional[np.ndarray] = None
    bias_hh: Optional[np.ndarray] = None
    input_name: str = ""
    sequence_length: Optional[int] = None
    cell_bits: int = 16
    cell_observer: object = field(default_factory=MinMaxObserver)
    # Filled by importer:
    cell_scale: float = 0.0
    input_weight_scales: Sequence[float] = None
    recurrent_weight_scales: Sequence[float] = None
    q_weight_ih: np.ndarray = None   # [4][hidden][in]
    q_weight_hh: np.ndarray = None   # [4][hidden][hidden]
    q_bias: np.ndarray = None        # [4][hidden], int32 in the LUT grid

    def __post_init__(self) -> None:
        if self.weight_ih is None or self.weight_hh is None:
            raise ValueError(f"LSTM {self.name}: weight_ih and weight_hh are required")
        self.weight_ih = np.asarray(self.weight_ih, dtype=np.float32)
        self.weight_hh = np.asarray(self.weight_hh, dtype=np.float32)
        if self.weight_hh.ndim != 2 or self.weight_hh.shape[0] != 4 * self.weight_hh.shape[1]:
            raise ValueError(f"LSTM {self.name}: weight_hh must be [4 * hidden, hidden], "
                             f"got {self.weight_hh.shape}")
        hid = self.weight_hh.shape[1]
        if self.weight_ih.ndim != 2 or self.weight_ih.shape[0] != 4 * hid:
            raise ValueError(f"LSTM {self.name}: weight_ih must be [4 * hidden, in] "
                             f"= [{4 * hid}, in], got {self.weight_ih.shape}")
        for nm in ("bias_ih", "bias_hh"):
            b = getattr(self, nm)
            if b is not None:
                b = np.asarray(b, dtype=np.float32).reshape(-1)
                if b.shape[0] != 4 * hid:
                    raise ValueError(f"LSTM {self.name}: {nm} must have {4 * hid} "
                                     f"values, got {b.shape[0]}")
                setattr(self, nm, b)
        if self.cell_bits not in (8, 16):
            raise ValueError(f"LSTM {self.name}: cell_bits must be 8 or 16, "
                             f"got {self.cell_bits}")

    @classmethod
    def from_state_dict(cls, name: str, state: dict, input_name: str,
                        **kwargs) -> "LSTM":
        """Build from an nn.LSTM's state_dict(), converted to numpy
        ({k: v.detach().numpy() for k, v in lstm.state_dict().items()}).

        Rejects what QLSTMCell cannot run: stacked layers (weight_ih_l1),
        bidirectional (_reverse) and projections (weight_hr_l0)."""
        for k in state:
            if k.endswith("_reverse"):
                raise ValueError(f"LSTM {name}: bidirectional LSTMs are not supported "
                                 f"({k}); QLSTMCell runs one direction")
            if k.startswith("weight_hr"):
                raise ValueError(f"LSTM {name}: proj_size is not supported ({k})")
            m = re.search(r"_l(\d+)$", k)
            if m is not None and int(m.group(1)) > 0:
                raise ValueError(f"LSTM {name}: stacked LSTMs (num_layers > 1) are "
                                 f"not supported ({k})")
        return cls(name=name, input_name=input_name,
                   weight_ih=state["weight_ih_l0"],
                   weight_hh=state["weight_hh_l0"],
                   bias_ih=state.get("bias_ih_l0"),
                   bias_hh=state.get("bias_hh_l0"),
                   **kwargs)

    @property
    def hidden_size(self) -> int:
        return int(self.weight_hh.shape[1])

    @property
    def input_size(self) -> int:
        return int(self.weight_ih.shape[1])

    @property
    def cell_qmin(self) -> int:
        return -(1 << (self.cell_bits - 1))

    @property
    def cell_qmax(self) -> int:
        return (1 << (self.cell_bits - 1)) - 1

    @property
    def bias(self) -> np.ndarray:
        """bias_ih + bias_hh, summed in double and stored as float32 (the
        float the C++ quantizeQLSTMBiases reads)."""
        total = np.zeros(4 * self.hidden_size, dtype=np.float64)
        for b in (self.bias_ih, self.bias_hh):
            if b is not None:
                total = total + b.astype(np.float64)
        return total.astype(np.float32)


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def _matvec_f32(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x @ w.T accumulated in float64 without BLAS, rounded to float32, so
    the result does not depend on the CPU's BLAS or SIMD width."""
    y = np.einsum("i,oi->o", x.astype(np.float64), w.astype(np.float64),
                  optimize=False)
    return y.astype(np.float32)


def lstm_float_forward(layer: "LSTM", seq: np.ndarray
                       ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Float reference of the LSTM over one sequence [step][feature] from
    zero state. Returns (hidden states, cell states), one float32 vector per
    step. PyTorch nn.LSTM semantics:

        i, f, g, o = sigmoid, sigmoid, tanh, sigmoid of
                     W_ih x + b_ih + W_hh h + b_hh   (gate blocks i, f, g, o)
        c = f * c + i * g
        h = o * tanh(c)

    Gates use Python's math.exp / math.tanh (the C library), and h and c are
    stored as float32 every step, so the reference is CPU-independent.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != layer.input_size:
        raise ValueError(f"LSTM {layer.name}: a sample must be [step][{layer.input_size}], "
                         f"got {seq.shape}")
    if layer.sequence_length is not None and seq.shape[0] != layer.sequence_length:
        raise ValueError(f"LSTM {layer.name}: sequence_length is {layer.sequence_length}, "
                         f"a sample has {seq.shape[0]} steps")
    hid = layer.hidden_size
    bias = layer.bias.astype(np.float64)
    h = np.zeros(hid, np.float32)
    c = np.zeros(hid, np.float32)
    hs: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    for x in seq:
        pre = (_matvec_f32(x, layer.weight_ih).astype(np.float64) + bias
               + _matvec_f32(h, layer.weight_hh).astype(np.float64))
        nh = np.empty(hid, np.float32)
        nc = np.empty(hid, np.float32)
        for u in range(hid):
            i = _sigmoid(pre[u])
            f = _sigmoid(pre[hid + u])
            g = math.tanh(pre[2 * hid + u])
            o = _sigmoid(pre[3 * hid + u])
            cu = np.float32(f * float(c[u]) + i * g)
            nc[u] = cu
            nh[u] = np.float32(o * math.tanh(float(cu)))
        h, c = nh, nc
        hs.append(h)
        cs.append(c)
    return hs, cs


def symmetric_activation_scale(observer: object, qmax: int) -> float:
    """max(|min|, |max|) / qmax of an observed range, zero point 0. An
    all-zero range gets 1 / qmax so the scale is never zero."""
    lo, hi = observer.range()
    m = max(abs(lo), abs(hi))
    return m / qmax if m > 0.0 else 1.0 / qmax


def quantize_lstm_biases(bias: np.ndarray, lut_input_scale: float) -> np.ndarray:
    """Mirror cpp/include/qcalibration.hpp::quantizeQLSTMBiases on the float
    values the C++ reads: lround(double(b) / double(lut_input_scale)) with
    both operands float32. Raises rather than wrapping when a bias does not
    fit int32 (the C++ conversion would wrap it)."""
    lut = float(np.float32(lut_input_scale))
    out = []
    for b in np.asarray(bias, dtype=np.float32).tolist():
        q = lround(float(b) / lut)
        if q < -(1 << 31) or q > (1 << 31) - 1:
            raise ValueError(f"LSTM bias {b} does not fit int32 at LUT scale {lut}")
        out.append(q)
    return np.array(out, dtype=np.int32)


@dataclass
class QFormatDense(Layer):
    """Q-format dense layer (tinymind QValue pipeline).

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
    # Calibration observes the fused layer's output through `forward`, and
    # the int8 layer computes conv-then-BN, so the float forward must too.
    # conv.forward alone would fit the output scale to the range before BN.
    forward = None
    if conv.forward is not None:
        conv_forward = conv.forward

        def forward(x: np.ndarray) -> np.ndarray:
            return (conv_forward(x) - bn.mean) * sigma_eff + bn.beta
    fused = Conv2D(
        name=bn.name,
        observer=bn.observer,
        weight=fused_w,
        bias=fused_b,
        input_name=conv.input_name,
        forward=forward,
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

    An LSTM (only as the first layer, reading `input_name`) takes each
    sample as a sequence [step][feature]; its hidden and cell observers see
    every step, and the tensor it passes on is the last hidden state.

    Returns a dict of tensor_name -> (scale, zero_point).
    """
    _check_lstm_placement(layers, input_name)
    for x in dataset:
        x_f = x.astype(np.float32)
        input_observer.observe(x_f)
        # Reset for-this-sample storage.
        per_sample: dict[str, np.ndarray] = {input_name: x_f}
        for layer in layers:
            parent = per_sample[layer.input_name]
            if isinstance(layer, LSTM):
                hs, cs = lstm_float_forward(layer, parent)
                layer.observer.observe(np.stack(hs))
                layer.cell_observer.observe(np.stack(cs))
                per_sample[layer.name] = hs[-1]
                continue
            if isinstance(layer, (Dense, Conv2D, QFormatDense)):
                y = layer.forward(parent)
            elif isinstance(layer, MoE):
                y = moe_float_forward(layer, parent)
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
        if isinstance(layer, LSTM):
            # Symmetric hidden and cell states, zero point 0 (the zero
            # initial state is exact); the hidden grid is the layer output's.
            ranges[layer.name] = (symmetric_activation_scale(layer.observer, 127), 0)
            layer.cell_scale = symmetric_activation_scale(layer.cell_observer,
                                                          layer.cell_qmax)
            continue
        ranges[layer.name] = affine_asymmetric(*layer.observer.range())
    return ranges


def _check_lstm_placement(layers: Sequence[Layer], input_name: str) -> None:
    """QLSTMCell consumes the model input sequence and hands one hidden
    state on, so an LSTM is accepted only as the first layer."""
    for idx, layer in enumerate(layers):
        if not isinstance(layer, LSTM):
            continue
        if idx != 0 or layer.input_name != input_name:
            raise ValueError(
                f"LSTM {layer.name}: an LSTM must be the first layer and read the "
                f"model input '{input_name}' (it reads '{layer.input_name}' at "
                f"position {idx}); an LSTM after another layer is not supported")


# ----------------------------------------------------------------------------
# Weight quantization pass.
# ----------------------------------------------------------------------------


def quantize_weights(layers: Sequence[Layer],
                     ranges: dict[str, tuple[float, int]]) -> None:
    """Symmetric per-tensor weight quantization. Mutates each layer."""
    for layer in layers:
        if isinstance(layer, LSTM):
            # Symmetric per gate, input and recurrent weights separately;
            # biases in the fixed LUT input grid.
            hid = layer.hidden_size
            w_in = layer.weight_ih.reshape(4, hid, layer.input_size)
            w_rec = layer.weight_hh.reshape(4, hid, hid)
            layer.input_weight_scales = [symmetric_weight_scale(w_in[g])
                                         for g in range(4)]
            layer.recurrent_weight_scales = [symmetric_weight_scale(w_rec[g])
                                             for g in range(4)]
            layer.q_weight_ih = np.stack([
                quantize_symmetric_weights(w_in[g], layer.input_weight_scales[g])
                for g in range(4)])
            layer.q_weight_hh = np.stack([
                quantize_symmetric_weights(w_rec[g], layer.recurrent_weight_scales[g])
                for g in range(4)])
            layer.q_bias = quantize_lstm_biases(
                layer.bias, LSTM_LUT_INPUT_SCALE).reshape(4, hid)
        elif isinstance(layer, (Dense, Conv2D)):
            layer.weight_scale = symmetric_weight_scale(layer.weight)
            layer.q_weight = quantize_symmetric_weights(
                layer.weight, layer.weight_scale)
            if layer.bias is not None:
                input_scale = ranges[layer.input_name][0]
                bias_scale = input_scale * layer.weight_scale
                layer.q_bias = quantize_bias_int32(layer.bias, bias_scale)
        elif isinstance(layer, MoE):
            input_scale = ranges[layer.input_name][0]
            # Router: symmetric per-tensor weights; bias at input*router scale.
            layer.router_weight_scale = symmetric_weight_scale(layer.router_weight)
            layer.q_router_weight = quantize_symmetric_weights(
                layer.router_weight, layer.router_weight_scale)
            if layer.router_bias is not None:
                router_bias_scale = input_scale * layer.router_weight_scale
                layer.q_router_bias = quantize_bias_int32(
                    layer.router_bias, router_bias_scale)
            # Experts: per-expert weight scale; bias at input*expert scale.
            scales: list[float] = []
            qws: list[np.ndarray] = []
            qbs: list[Optional[np.ndarray]] = []
            for e in range(layer.num_experts):
                w = layer.expert_weights[e]
                ws = symmetric_weight_scale(w)
                scales.append(ws)
                qws.append(quantize_symmetric_weights(w, ws))
                b = (layer.expert_biases[e]
                     if layer.expert_biases is not None else None)
                if b is not None:
                    qbs.append(quantize_bias_int32(b, input_scale * ws))
                else:
                    qbs.append(None)
            layer.expert_weight_scales = scales
            layer.q_expert_weights = qws
            layer.q_expert_biases = qbs
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


def _fmt_float_array(name: str, values: Iterable[float]) -> str:
    vs = list(values)
    body = ", ".join(f"{float(v)!r}f" for v in vs)
    return f"    constexpr float {name}[{len(vs)}] = {{ {body} }};\n"


def _emit_lstm(layer: "LSTM") -> str:
    """Emit one LSTM's QLSTMCell inputs.

    The hidden state's scale and zero point are the layer's activation
    calibration entry (k<name>_Scale / k<name>_ZeroPoint). The C++ side fills
    tinymind::QLSTMScales from these floats and calls buildQLSTMParams, and
    builds the sigmoid / tanh LUTs on k<name>_LutInputScale.
    """
    nm = layer.name.replace("/", "_").replace("-", "_")
    hid, nin = layer.hidden_size, layer.input_size
    cell_type = "int16_t" if layer.cell_bits == 16 else "int8_t"
    out = [
        f"\n    // LSTM {nm} (cpp/qlstm.hpp QLSTMCell): gates i, f, g, o; weights\n"
        f"    // gate-major [gate][hidden][input]; runs from zero state. k{nm}_Scale /\n"
        f"    // k{nm}_ZeroPoint above are the hidden state's grid.\n",
        f"    constexpr std::size_t k{nm}_NumInputs = {nin};\n",
        f"    constexpr std::size_t k{nm}_NumHidden = {hid};\n",
    ]
    if layer.sequence_length is not None:
        out.append(f"    constexpr std::size_t k{nm}_SequenceLength = "
                   f"{layer.sequence_length};\n")
    out += [
        f"    typedef {cell_type} {nm}_CellType;\n",
        f"    constexpr float   k{nm}_CellScale     = {layer.cell_scale!r}f;\n",
        f"    constexpr int32_t k{nm}_CellZeroPoint = 0;\n",
        f"    constexpr int32_t k{nm}_CellQMin      = {layer.cell_qmin};\n",
        f"    constexpr int32_t k{nm}_CellQMax      = {layer.cell_qmax};\n",
        f"    constexpr float   k{nm}_LutInputScale = {LSTM_LUT_INPUT_SCALE!r}f;\n",
        _fmt_float_array(f"k{nm}_InputWeightScales", layer.input_weight_scales),
        _fmt_float_array(f"k{nm}_RecurrentWeightScales",
                         layer.recurrent_weight_scales),
        _fmt_int_array(f"k{nm}_InputWeights",
                       layer.q_weight_ih.reshape(-1).tolist(), "int8_t"),
        _fmt_int_array(f"k{nm}_RecurrentWeights",
                       layer.q_weight_hh.reshape(-1).tolist(), "int8_t"),
        _fmt_int_array(f"k{nm}_Biases", layer.q_bias.reshape(-1).tolist(),
                       "int32_t"),
    ]
    return "".join(out)


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
    """Write a tinymind-format weights.hpp.

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
        parts.append("\n    // Q-format weights (tinymind QValue pipeline).\n")
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

    for layer in layers:
        if isinstance(layer, LSTM):
            parts.append(_emit_lstm(layer))

    if any(isinstance(layer, MoE) for layer in layers):
        parts.append("\n    // Mixture-of-Experts (cpp/qmoe.hpp): top-1 router "
                     "+ per-expert int8 experts.\n")
        for layer in layers:
            if not isinstance(layer, MoE):
                continue
            nm = layer.name.replace("/", "_").replace("-", "_")
            parts.append(f"    constexpr std::size_t k{nm}_NumExperts = "
                         f"{layer.num_experts};\n")
            # Router (no requantizer: argmax over raw int32 logits).
            parts.append(f"    constexpr float k{nm}_Router_WeightScale = "
                         f"{layer.router_weight_scale!r}f;\n")
            parts.append(_fmt_int_array(f"k{nm}_Router_Weights",
                                        layer.q_router_weight.reshape(-1).tolist(),
                                        "int8_t"))
            if layer.q_router_bias is not None:
                parts.append(_fmt_int_array(f"k{nm}_Router_Biases",
                                            layer.q_router_bias.tolist(),
                                            "int32_t"))
            # Experts: each its own weight scale; all share the layer output
            # scale (k{nm}_Scale / k{nm}_ZeroPoint above).
            for e in range(layer.num_experts):
                parts.append(f"    constexpr float k{nm}_Expert{e}_WeightScale = "
                             f"{layer.expert_weight_scales[e]!r}f;\n")
                parts.append(_fmt_int_array(
                    f"k{nm}_Expert{e}_Weights",
                    layer.q_expert_weights[e].reshape(-1).tolist(),
                    "int8_t"))
                if layer.q_expert_biases[e] is not None:
                    parts.append(_fmt_int_array(
                        f"k{nm}_Expert{e}_Biases",
                        layer.q_expert_biases[e].tolist(),
                        "int32_t"))

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
