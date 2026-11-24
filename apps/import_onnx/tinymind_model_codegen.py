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

Whole-model code generation and static activation memory planning for the
records tinymind_import_onnx parses.

The weights header the importer writes holds every constant a layer needs, but
leaves the wiring to the caller: which layer type each record becomes, which
buffer it reads, and where its output lives. This module writes that wiring as
a second header, model.hpp, holding

  * one constexpr layer object per record (QConv2D, QConv2DPerChannel,
    QDepthwiseConv2D, QDense, QMaxPool2D, QAdd, QSoftmaxFixedPoint1D) with
    its weights, biases and requantizers bound at compile time, so the object
    sits in flash and the compiler folds its fields into the kernels;
  * the 256-entry lookup tables for Sigmoid and Tanh records and the QAdd and
    softmax parameters, computed here rather than by the host-only builders in
    qcalibration.hpp / qactivations.hpp, so the generated model compiles with
    TINYMIND_ENABLE_FLOAT=0 and TINYMIND_ENABLE_STD=0;
  * a memory plan: every intermediate activation gets a fixed offset into one
    caller-owned arena, and two activations share bytes when no layer needs
    both at once (see plan_memory);
  * forward(input, output, arena), which runs the layers in graph order.

The model input and output are caller-owned and are not part of the arena: the
first layers read `input` directly and the last layer writes `output`. The
arena must not overlap either of them.

Everything computed here mirrors a C++ host builder, and the tests compare
the two: quantize_multiplier mirrors quantizeMultiplier, qadd_params
mirrors buildQAddParams, softmax_params mirrors
buildQSoftmaxFixedPointParams and activation_lut mirrors buildQSigmoidLUT /
buildQTanhLUT.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import tinymind_import_onnx as ti


# ----------------------------------------------------------------------------
# Static memory planning.
# ----------------------------------------------------------------------------


@dataclass
class PlannedBuffer:
    """One arena region. `layers` lists the records whose output lives in it:
    more than one when an element-wise layer writes over its input."""

    size: int
    first: int
    last: int
    layers: list = field(default_factory=list)
    offset: int = -1


@dataclass
class MemoryPlan:
    buffers: list
    offsets: dict            # layer index -> arena offset of its output
    arena_size: int          # bytes the planned arena needs
    unshared_size: int       # bytes if every output had its own buffer
    lower_bound: int         # bytes live at the busiest layer; no plan beats it
    alignment: int


def _align(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


# Records whose kernel reads element j before it writes element j and touches
# no other element, so the output may overwrite an input of the same size.
IN_PLACE_OPS = ("Relu", "Sigmoid", "Tanh", "Add")


def plan_memory(layers, sizes: dict, alignment: int = 4, in_place: bool = True) -> MemoryPlan:
    """Give every intermediate output an offset into one arena.

    A layer's output is live from the layer that writes it to the last layer
    that reads it, both included, so a layer's inputs and output never share
    bytes unless the layer is element-wise and its input dies there (then the
    output takes over the input's buffer, which IN_PLACE_OPS lists). The last
    record writes the caller's output buffer and is not planned.

    Placement is TFLite Micro's greedy-by-size: the largest buffer first,
    each at the lowest aligned offset that overlaps no buffer it is live
    beside. The result is never above the unshared total and is usually at
    or near lower_bound, the bytes live during the busiest layer.
    """
    n = len(layers)
    last_use = {i: i for i in range(n)}
    for i, layer in enumerate(layers):
        for j in layer.sources:
            if j >= 0:
                last_use[j] = max(last_use[j], i)

    owner: dict[int, PlannedBuffer] = {}
    buffers: list[PlannedBuffer] = []
    for i, layer in enumerate(layers[:-1]):
        target = None
        if in_place and layer.op_type in IN_PLACE_OPS:
            for j in layer.sources:
                if j < 0 or j not in owner:
                    continue
                buf = owner[j]
                if buf.last == i and sizes[j] == sizes[i]:
                    target = buf
                    break
        if target is None:
            target = PlannedBuffer(size=_align(sizes[i], alignment), first=i, last=last_use[i])
            buffers.append(target)
        else:
            target.last = last_use[i]
        target.layers.append(i)
        owner[i] = target

    placed: list[PlannedBuffer] = []
    for buf in sorted(buffers, key=lambda b: (-b.size, b.first)):
        clashes = sorted((p for p in placed if p.first <= buf.last and buf.first <= p.last),
                         key=lambda p: p.offset)
        offset = 0
        for p in clashes:
            if offset + buf.size <= p.offset:
                break
            offset = max(offset, p.offset + p.size)
        buf.offset = offset
        placed.append(buf)

    arena = max((b.offset + b.size for b in buffers), default=0)
    lower = max((sum(b.size for b in buffers if b.first <= t <= b.last) for t in range(n)), default=0)
    unshared = sum(_align(sizes[i], alignment) for i in range(n - 1))
    offsets = {i: owner[i].offset for i in owner}
    return MemoryPlan(buffers=buffers, offsets=offsets, arena_size=arena,
                      unshared_size=unshared, lower_bound=lower, alignment=alignment)


def check_plan(plan: MemoryPlan) -> None:
    """Raise AssertionError if two buffers that are live together overlap."""
    for a in plan.buffers:
        assert a.offset >= 0 and a.offset % plan.alignment == 0, a
        for b in plan.buffers:
            if a is b:
                continue
            live_together = a.first <= b.last and b.first <= a.last
            disjoint = a.offset + a.size <= b.offset or b.offset + b.size <= a.offset
            assert not live_together or disjoint, (a, b)


# ----------------------------------------------------------------------------
# Mirrors of the host-only C++ builders.
# ----------------------------------------------------------------------------


def _f32(x: float) -> float:
    """Round a double to the nearest float, as a C++ float variable holds it."""
    import numpy as np
    return float(np.float32(x))


def qadd_params(a_scale: float, b_scale: float, out_scale: float, left_shift: int = 20) -> dict:
    """Mirror qcalibration.hpp::buildQAddParams (float scales in)."""
    a, b, o = _f32(a_scale), _f32(b_scale), _f32(out_scale)
    twice_max = 2.0 * (a if a > b else b)
    real_out = twice_max / (float(1 << left_shift) * o)
    am, ash = ti.quantize_multiplier(a / twice_max)
    bm, bsh = ti.quantize_multiplier(b / twice_max)
    om, osh = ti.quantize_multiplier(real_out)
    return dict(left_shift=left_shift, input_a_multiplier=am, input_a_shift=ash,
                input_b_multiplier=bm, input_b_shift=bsh, output_multiplier=om, output_shift=osh)


def softmax_params(input_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    """Mirror qcalibration.hpp::buildQSoftmaxFixedPointParams for a float
    input scale; returns (input_multiplier, input_left_shift, diff_min)."""
    integer_bits = 5
    real = beta * _f32(input_scale) * float(1 << (31 - integer_bits))
    real = min(real, float((1 << 31) - 1))
    mult, shift = ti.quantize_multiplier(real)
    left = -shift
    max_rescaled = float((1 << integer_bits) - 1) * float(1 << (31 - integer_bits)) / float(1 << left)
    return mult, left, -int(math.floor(max_rescaled))


def activation_lut(kind: str, in_scale: float, in_zp: int, out_scale: float, out_zp: int) -> list:
    """Mirror qactivations.hpp::buildQSigmoidLUT / buildQTanhLUT: entry idx
    holds the output for the int8 input idx - 128. The builders compute in
    float; the products and quotients are rounded to float here in the same
    order, and exp / tanh are taken in double and rounded to float."""
    import numpy as np
    f = np.float32
    s_in, s_out, zp_out = f(in_scale), f(out_scale), f(out_zp)
    lut = []
    for idx in range(256):
        x = f(s_in * f(idx - 128 - in_zp))
        if kind == "Sigmoid":
            e = f(math.exp(-float(x)))
            y = f(f(1.0) / f(f(1.0) + e))
        elif kind == "Tanh":
            y = f(math.tanh(float(x)))
        else:
            raise ValueError(kind)
        q = f(zp_out + f(y / s_out))
        lut.append(max(-128, min(127, ti.lround(float(q)))))
    return lut


def round_half_away(x: float) -> int:
    """std::round semantics on a double (ties away from zero), exactly."""
    if x < 0.0:
        return -round_half_away(-x)
    r = math.floor(x)
    return int(r) + 1 if x - r >= 0.5 else int(r)


def activation_lut16(kind: str, in_scale: float, in_zp: int, out_scale: float) -> list:
    """Mirror qactivations.hpp::buildQLUT16 (buildQSigmoidLUT16 /
    buildQTanhLUT16): 513 int16 entries, entry i the function at int16 input
    -32768 + 128 i less half the interval's midpoint interpolation error,
    on the symmetric output grid of out_scale. All in double, as there."""
    if kind == "Sigmoid":
        def fn(x):
            return 1.0 / (1.0 + math.exp(-x))
    elif kind == "Tanh":
        fn = math.tanh
    else:
        raise ValueError(kind)
    inv = 1.0 / out_scale
    lut = []
    for i in range(513):
        x = in_scale * float(-32768 + 128 * i - in_zp)
        sample = float(round_half_away(fn(x) * inv))
        v = sample
        if i < 512:
            x_next = in_scale * float(-32768 + 128 * (i + 1) - in_zp)
            mid_interp = float(round_half_away((fn(x_next) * inv + sample) / 2))
            mid = float(round_half_away(fn((x + x_next) / 2) * inv))
            v = sample - float(round_half_away((mid_interp - mid) / 2))
        lut.append(int(max(-32768.0, min(32767.0, v))))
    return lut


@dataclass
class Grid16:
    """An int8 activation grid widened to int16: symmetric (zero point 0)
    over the same real range, with the int8 range's ends as clamp bounds."""
    scale: float
    qmin: int
    qmax: int


def int16_grid(scale: float, zp: int) -> Grid16:
    """The int16 grid for an int8 tensor of (scale, zp). The int8 tensor
    holds reals in [lo, hi] = [(-128 - zp) * scale, (127 - zp) * scale]; the
    int16 grid is symmetric with scale max(|lo|, |hi|) / 32767, and a layer
    writing the tensor clamps to [lo, hi] on it. The clamp keeps whatever
    the int8 grid implied: a ReLU fused into the requantizer (zero point
    -128, so lo = 0) still clamps at zero."""
    lo = (-128 - zp) * scale
    hi = (127 - zp) * scale
    s16 = max(abs(lo), abs(hi)) / 32767.0
    return Grid16(scale=s16, qmin=max(-32768, round_half_away(lo / s16)),
                  qmax=min(32767, round_half_away(hi / s16)))


@dataclass
class Wide16x8:
    """A weight layer moved to int16 activations: int64 biases at the int16
    input scale, and one 64-bit requantizer per output channel."""
    biases: Optional[list]
    multipliers: list
    shifts: list
    qmin: int
    qmax: int


def requantize_16x8(layer) -> Wide16x8:
    """Move a weight record (Conv, DepthwiseConv, AveragePool, Dense) to
    int16 activations, keeping its int8 weights: the bias b (at input_scale *
    weight_scale) becomes round(b * input_scale / s_in16) at s_in16 *
    weight_scale, and each requantizer's ratio becomes s_in16 *
    weight_scale / s_out16. Raises ValueError for a ratio of 2^7 or more,
    beyond the 64-bit Requantizer's (TFLite's) shift contract."""
    gin = int16_grid(layer.input_scale, layer.input_zp)
    gout = int16_grid(layer.output_scale, layer.output_zp)
    multipliers, shifts = [], []
    for ws in layer.weight_scales:
        m, sh = ti.quantize_multiplier(gin.scale * float(ws) / gout.scale)
        if sh < -7:
            raise ValueError(f"{layer.name}: scale ratio {gin.scale * float(ws) / gout.scale:g} is 2^7 or more, "
                             "beyond the int64 requantizer's range")
        multipliers.append(m)
        shifts.append(sh)
    biases = None
    if layer.bias is not None:
        ratio = layer.input_scale / gin.scale
        biases = [round_half_away(float(b) * ratio) for b in layer.bias.tolist()]
    return Wide16x8(biases=biases, multipliers=multipliers, shifts=shifts, qmin=gout.qmin, qmax=gout.qmax)


def float_reference(layers, x):
    """Run parsed records in float: dequantized int8 weights and biases,
    real activations, every layer's output clipped to the real range its
    int8 grid holds, and no other rounding. Both the int8 and the 16x8 model
    approximate exactly this, so their distance from it measures what their
    activation grids cost. x is the model input (reals, flattened NHWC);
    returns the output, flattened. Softmax records are not supported."""
    import numpy as np
    vals = {-1: np.asarray(x, dtype=np.float64)}
    for i, l in enumerate(layers):
        g = l.geometry
        src = vals[l.sources[0]]
        op = l.op_type
        if l.weight is not None:
            out_ch = l.weight.shape[0]
            ws = np.array(l.weight_scales, dtype=np.float64)
            ws = np.repeat(ws, out_ch) if ws.size == 1 else ws
            w = l.weight.astype(np.float64) * ws.reshape((-1,) + (1,) * (l.weight.ndim - 1))
            b = np.zeros(out_ch) if l.bias is None else l.bias.astype(np.float64) * l.input_scale * ws
        if op == "Dense":
            y = w @ src + b
        elif op in ("Conv", "DepthwiseConv", "AveragePool", "MaxPool"):
            H, W, C = g["InputHeight"], g["InputWidth"], g["InputChannels"]
            KH, KW, SH, SW = g["KernelHeight"], g["KernelWidth"], g["StrideH"], g["StrideW"]
            pads = ((g.get("PadTop", 0), g.get("PadBottom", 0)), (g.get("PadLeft", 0), g.get("PadRight", 0)), (0, 0))
            xp = np.pad(src.reshape(H, W, C), pads)
            OH, OW = g["OutputHeight"], g["OutputWidth"]
            y = np.zeros((OH, OW, g.get("NumFilters", C)))
            for oh in range(OH):
                for ow in range(OW):
                    patch = xp[oh * SH:oh * SH + KH, ow * SW:ow * SW + KW, :]
                    if op == "Conv":
                        y[oh, ow] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2])) + b
                    elif op == "MaxPool":
                        y[oh, ow] = patch.max(axis=(0, 1))
                    else:
                        y[oh, ow] = np.einsum("hwc,chw->c", patch, w.reshape(C, KH, KW)) + b
        elif op == "Add":
            y = src + vals[l.sources[1]]
        elif op == "Relu":
            y = np.maximum(src, 0.0)
        elif op == "Sigmoid":
            y = 1.0 / (1.0 + np.exp(-src))
        elif op == "Tanh":
            y = np.tanh(src)
        else:
            raise ValueError(op)
        lo = (-128 - l.output_zp) * l.output_scale
        hi = (127 - l.output_zp) * l.output_scale
        vals[i] = np.clip(np.asarray(y, dtype=np.float64).reshape(-1), lo, hi)
    return vals[len(layers) - 1]


# ----------------------------------------------------------------------------
# Emitter.
# ----------------------------------------------------------------------------


SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128


def output_size(layer, sizes: dict) -> int:
    g = layer.geometry
    if layer.op_type in ("Conv", "DepthwiseConv", "AveragePool"):
        return g["OutputHeight"] * g["OutputWidth"] * g["NumFilters"]
    if layer.op_type == "MaxPool":
        return g["OutputHeight"] * g["OutputWidth"] * g["InputChannels"]
    if layer.op_type == "Dense":
        return g["NumOutputs"]
    return sizes[layer.sources[0]]


def input_size(layer) -> Optional[int]:
    g = layer.geometry
    if layer.op_type == "Dense":
        return g["NumInputs"]
    if "InputHeight" in g:
        return g["InputHeight"] * g["InputWidth"] * g["InputChannels"]
    return None


def _rq(m: int, s: int, zp: int) -> str:
    return f"{{{m}, {s}, {zp}, -128, 127}}"


@dataclass
class Int4Dense:
    """A dense layer's weights requantized to per-channel int4."""
    codes: object          # int8 numpy array [out][in], values -7..7
    scales: list           # per output channel
    biases: object         # int32 numpy array, or None
    multipliers: list
    shifts: list


def requantize_dense_int4(layer) -> Int4Dense:
    """Requantize a Dense record's int8 weights to symmetric per-channel
    int4: the float weights w = q8 * s8 get scale max |w| / 7 per output
    channel and codes round(w / s4) in -7..7 (ties away from zero, as
    quantizeInt4PerChannel). The bias moves to the new accumulator scale
    input_scale * s4, and each requantizer to input_scale * s4 /
    output_scale."""
    import numpy as np
    w8 = layer.weight.astype(np.float64)
    out_ch = w8.shape[0]
    s8 = layer.weight_scales if layer.per_channel else list(layer.weight_scales) * out_ch
    codes = np.zeros(w8.shape, dtype=np.int8)
    scales, multipliers, shifts = [], [], []
    biases = None if layer.bias is None else np.zeros(out_ch, dtype=np.int64)
    for o in range(out_ch):
        wf = w8[o] * float(s8[o])
        largest = float(np.max(np.abs(wf)))
        s4 = largest / 7.0 if largest > 0.0 else float(s8[o])
        scales.append(s4)
        codes[o] = [max(-7, min(7, ti.lround(v / s4))) for v in wf.tolist()]
        if biases is not None:
            biases[o] = max(-(1 << 31), min((1 << 31) - 1, ti.lround(float(layer.bias[o]) * float(s8[o]) / s4)))
        m, sh = ti.quantize_multiplier(layer.input_scale * s4 / layer.output_scale)
        multipliers.append(m)
        shifts.append(sh)
    return Int4Dense(codes=codes, scales=scales, biases=biases, multipliers=multipliers, shifts=shifts)


def pack_int4(values) -> list:
    """TFLite / ONNX int4 packing: element k in the low nibble of byte k / 2
    when k is even, the high nibble when odd (as packInt4)."""
    flat = [int(v) & 15 for v in values]
    out = []
    for b in range((len(flat) + 1) // 2):
        low = flat[2 * b]
        high = flat[2 * b + 1] if 2 * b + 1 < len(flat) else 0
        out.append(low | (high << 4))
    return out


@dataclass
class GeneratedModel:
    plan: MemoryPlan
    sizes: dict
    input_size: int
    output_size: int
    weight_bytes: int = 0      # int8 / int4 weight bytes the layers read
    int4_layers: list = field(default_factory=list)


def _rows(values, per_row: int, fmt=str) -> str:
    rows = [", ".join(fmt(v) for v in values[r:r + per_row]) for r in range(0, len(values), per_row)]
    return ",\n            ".join(rows)


def _emit_layer_16x8(p: list, body: list, layer, nm: str, k: str, src: str, dst: str, n: int, ptr) -> int:
    """One record of a 16x8 model (see emit_model_header's activation_bits).
    Appends its constants to p and its call to body; returns the int8 weight
    bytes it reads."""
    g = layer.geometry
    op = layer.op_type
    if op in ("Conv", "DepthwiseConv", "AveragePool", "Dense"):
        q = requantize_16x8(layer)
        bias = "nullptr"
        if q.biases is not None:
            p.append(f"        constexpr int64_t {nm}_biases16[{len(q.biases)}] = {{\n            "
                     + _rows(q.biases, 6) + "\n        };\n")
            bias = f"{nm}_biases16"
        entries = [f"{{{m}, {sh}, 0, {q.qmin}, {q.qmax}}}" for m, sh in zip(q.multipliers, q.shifts)]
        per_channel = layer.per_channel or op in ("DepthwiseConv", "AveragePool")
        if per_channel:
            count = g["NumFilters"] if op != "Dense" else g["NumOutputs"]
            if len(entries) == 1:
                entries = entries * count
            p.append(f"        constexpr tinymind::Requantizer<int64_t, int16_t> {nm}_requantizers[{count}] = {{\n"
                     "            " + _rows(entries, 3) + "\n        };\n")
        types = "int16_t, int8_t, int64_t, int16_t"
        if op == "Dense" and not layer.per_channel:
            p.append(f"        typedef tinymind::QDense<{types}, {g['NumInputs']}, {g['NumOutputs']}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, 0, {entries[0]}}};\n")
            p.append(f"        static_assert({nm}_type::OutputLength == {n}, \"{nm}: output size\");\n")
        elif op in ("DepthwiseConv", "AveragePool"):
            dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                    f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}, "
                    f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}")
            p.append(f"        typedef tinymind::QDepthwiseConv2D<{types}, {dims}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, 0, {nm}_requantizers}};\n")
            p.append(f"        static_assert({nm}_type::OutputSize == {n}, \"{nm}: output size\");\n")
        else:
            if op == "Conv":
                dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                        f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}, {g['NumFilters']}, "
                        f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}")
            else:
                dims = f"1, 1, {g['NumInputs']}, 1, 1, 1, 1, {g['NumOutputs']}"
            if per_channel:
                p.append(f"        typedef tinymind::QConv2DPerChannel<{types}, {dims}> {nm}_type;\n")
                p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, 0, {nm}_requantizers}};\n")
            else:
                p.append(f"        typedef tinymind::QConv2D<{types}, {dims}> {nm}_type;\n")
                p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, 0, {entries[0]}}};\n")
            p.append(f"        static_assert({nm}_type::OutputSize == {n}, \"{nm}: output size\");\n")
        body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        return int(layer.weight.size)
    if op == "MaxPool":
        p.append(f"        typedef tinymind::QMaxPool2D<int16_t, {g['InputHeight']}, {g['InputWidth']}, "
                 f"{g['InputChannels']}, {g['KernelHeight']}, {g['KernelWidth']}, {g['StrideH']}, "
                 f"{g['StrideW']}> {nm}_type;\n")
        p.append(f"        constexpr {nm}_type {nm}_layer = {{}};\n")
        p.append(f"        static_assert({nm}_type::OutputSize == {n}, \"{nm}: output size\");\n")
        body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        return 0
    if op == "Add":
        ga = int16_grid(layer.input_scale, layer.input_zp)
        gb = int16_grid(layer.input_b_scale, layer.input_b_zp)
        go = int16_grid(layer.output_scale, layer.output_zp)
        a = qadd_params(ga.scale, gb.scale, go.scale, left_shift=15)
        p.append(f"        typedef tinymind::QAdd<int16_t, int16_t, int16_t, {n}> {nm}_type;\n")
        p.append(f"        constexpr {nm}_type {nm}_layer = {{0, 0, 15, "
                 f"{a['input_a_multiplier']}, {a['input_a_shift']}, "
                 f"{a['input_b_multiplier']}, {a['input_b_shift']}, "
                 f"{{{a['output_multiplier']}, {a['output_shift']}, 0, {go.qmin}, {go.qmax}}}}};\n")
        body.append(f"        model_detail::{nm}_layer.forward({src}, {ptr(layer.sources[1])}, {dst});\n")
        return 0
    if op in ("Sigmoid", "Tanh"):
        gin = int16_grid(layer.input_scale, layer.input_zp)
        gout = int16_grid(layer.output_scale, layer.output_zp)
        lut = activation_lut16(op, gin.scale, 0, gout.scale)
        lut = [max(gout.qmin, min(gout.qmax, v)) for v in lut]
        p.append(f"        constexpr int16_t {nm}_lut16[513] = {{\n            "
                 + _rows(lut, 12, lambda v: f"{v:6d}") + "\n        };\n")
        body.append(f"        for (std::size_t j = 0; j < {n}; ++j)\n        {{\n"
                    f"            {dst}[j] = tinymind::qApplyLUT16({src}[j], model_detail::{nm}_lut16);\n"
                    f"        }}\n")
        return 0
    if op == "Relu":
        body.append(f"        for (std::size_t j = 0; j < {n}; ++j)\n        {{\n"
                    f"            {dst}[j] = tinymind::qrelu<int16_t>({src}[j], 0);\n"
                    f"        }}\n")
        return 0
    raise ValueError(f"{layer.name}: no 16x8 code generation for {op}")


def emit_model_header(path: str, namespace: str, layers, weights_header: str,
                      alignment: int = 4, in_place: bool = True, int4_dense: bool = False,
                      nested_namespace: Optional[str] = None, activation_bits: int = 8) -> GeneratedModel:
    """Write model.hpp for parsed records whose constants weights_header
    (written by emit_weights_header with the same namespace) holds.

    int4_dense: run every Dense record as a QDenseInt4 over packed,
    per-channel int4 weights (see requantize_dense_int4()), stored in model.hpp; the
    int8 weights in weights_header then go unused.
    nested_namespace: put the model inside namespace::nested_namespace, so
    several generated variants can share one weights header.
    activation_bits: 16 runs the model with int16 activations and its int8
    weights (TFLite's 16x8 scheme): every activation grid is widened to a
    symmetric int16 grid over the same real range (int16_grid()), weight
    layers take int64 biases and 64-bit requantizers (requantize_16x8()),
    Add uses TFLite's int16 parameters (left_shift 15), and Sigmoid / Tanh
    use 513-entry interpolated int16 tables. The weights header is shared
    with the int8 model; the biases and requantizers live in model.hpp.
    Softmax is not supported at 16 bits.

    Raises ValueError for a graph this generator cannot run as planned: a
    Relu whose input and output grids differ (no record carries the
    rescale) or a layer that reads a softmax's output (tinymind's softmax
    writes the 1/256 grid whatever the graph calibrated)."""
    if not layers:
        raise ValueError("no layers to generate")
    if activation_bits not in (8, 16):
        raise ValueError(f"activation_bits must be 8 or 16, not {activation_bits}")
    wide = activation_bits == 16
    act = "int16_t" if wide else "int8_t"
    elem = 2 if wide else 1
    if wide and int4_dense:
        raise ValueError("int4 dense layers take int8 activations; int4_dense needs activation_bits=8")
    if wide and alignment % elem != 0:
        raise ValueError("int16 activations need an even arena alignment")
    for i, layer in enumerate(layers):
        if any(j >= i for j in layer.sources):
            raise ValueError(f"{layer.name}: reads a layer that runs after it")

    in_size = None
    for layer in layers:
        if -1 in layer.sources and input_size(layer) is not None:
            in_size = input_size(layer)
            break
    if in_size is None:
        raise ValueError("no layer that reads the model's input gives its size")

    sizes = {-1: in_size}
    for i, layer in enumerate(layers):
        sizes[i] = output_size(layer, sizes)
        if layer.op_type == "Softmax" and wide:
            raise ValueError(f"{layer.name}: no int16 softmax; generate this model with activation_bits=8")
        if layer.op_type == "Softmax":
            for k, later in enumerate(layers[i + 1:], start=i + 1):
                if i in later.sources:
                    raise ValueError(f"{later.name}: reads softmax {layer.name}'s output; tinymind's softmax "
                                     "writes the 1/256 grid, so only the model's output may be a softmax")
        if layer.op_type == "Relu" and (_f32(layer.input_scale) != _f32(layer.output_scale)
                                        or layer.input_zp != layer.output_zp):
            raise ValueError(f"{layer.name}: a Relu whose output grid differs from its input's "
                             "needs a rescale no record carries")

    plan = plan_memory(layers, {j: v * elem for j, v in sizes.items()}, alignment=alignment, in_place=in_place)
    check_plan(plan)

    last = len(layers) - 1
    out_layer = layers[last]
    if out_layer.op_type == "Softmax":
        out_scale, out_zp = SOFTMAX_OUTPUT_SCALE, SOFTMAX_OUTPUT_ZERO_POINT
    else:
        out_scale, out_zp = out_layer.output_scale, out_layer.output_zp
    first_reader = next(l for l in layers if -1 in l.sources)
    in_scale, in_zp = first_reader.input_scale, first_reader.input_zp
    if wide:
        in_scale, in_zp = int16_grid(in_scale, in_zp).scale, 0
        out_scale, out_zp = int16_grid(out_scale, out_zp).scale, 0

    ops = sorted({l.op_type for l in layers})
    includes = ["qaffine.hpp"]
    for op, hdr in (("Conv", "qconv2d.hpp"), ("Dense", "qdense.hpp"), ("DepthwiseConv", "qdepthwiseconv2d.hpp"),
                    ("AveragePool", "qdepthwiseconv2d.hpp"), ("MaxPool", "qpool2d.hpp"), ("Add", "qadd.hpp"),
                    ("Softmax", "qsoftmax_fixedpoint.hpp"), ("Relu", "qactivations.hpp"),
                    ("Sigmoid", "qactivations.hpp"), ("Tanh", "qactivations.hpp")):
        if op in ops and hdr not in includes:
            includes.append(hdr)
    if any(l.op_type == "Dense" and l.per_channel for l in layers) and "qconv2d.hpp" not in includes:
        includes.append("qconv2d.hpp")
    if int4_dense and "Dense" in ops:
        includes.append("qint4.hpp")

    p: list[str] = []
    p.append(ti.COPYRIGHT_BLOCK.replace("tinymind_import_onnx.", "tinymind_import_onnx (model code generation)."))
    p.append("\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n")
    for hdr in includes:
        p.append(f'#include "{hdr}"\n')
    p.append(f'#include "{weights_header}"\n\n')
    p.append("#if !TINYMIND_ENABLE_QUANTIZATION\n"
             "#error \"a generated tinymind model needs TINYMIND_ENABLE_QUANTIZATION=1\"\n"
             "#endif\n\n")

    p.append(f"namespace {namespace} {{\n")
    if nested_namespace:
        p.append(f"namespace {nested_namespace} {{\n")
    p.append(f"    // Model interface. input and output are caller-owned {act} buffers in\n"
             "    // NHWC order; arena is caller-owned scratch of kModelArenaSize bytes,\n"
             "    // aligned to kModelArenaAlignment, overlapping neither of them.\n")
    p.append(f"    typedef {act} ModelActivationType;\n")
    p.append(f"    constexpr std::size_t kModelInputSize      = {in_size};\n")
    p.append(f"    constexpr std::size_t kModelOutputSize     = {sizes[last]};\n")
    p.append(f"    constexpr std::size_t kModelArenaSize      = {plan.arena_size};\n")
    p.append(f"    constexpr std::size_t kModelArenaAlignment = {alignment};\n")
    p.append(f"    constexpr float   kModelInputScale      = {_f32(in_scale)!r}f;\n")
    p.append(f"    constexpr int32_t kModelInputZeroPoint  = {in_zp};\n")
    p.append(f"    constexpr float   kModelOutputScale     = {_f32(out_scale)!r}f;\n")
    p.append(f"    constexpr int32_t kModelOutputZeroPoint = {out_zp};\n\n")

    p.append(f"    // Memory plan: {len(plan.buffers)} arena buffers in {plan.arena_size} bytes "
             f"(unshared: {plan.unshared_size}; busiest layer: {plan.lower_bound}).\n")
    p.append("    //   offset    bytes  live      outputs\n")
    for buf in sorted(plan.buffers, key=lambda b: (b.offset, b.first)):
        names = ", ".join(ti.sanitize(layers[i].name) for i in buf.layers)
        p.append(f"    //   {buf.offset:6d}  {buf.size:7d}  {buf.first:3d}-{buf.last:<3d}  {names}\n")
    p.append(f"    constexpr std::size_t kModelUnsharedSize   = {plan.unshared_size};\n")
    p.append(f"    constexpr std::size_t kModelPeakLiveSize   = {plan.lower_bound};\n")
    p.append(f"    constexpr std::size_t kModelLayerCount     = {len(layers)};\n")
    p.append(f"    constexpr std::size_t kModelBufferCount    = {len(plan.buffers)};\n")
    names = ", ".join(f'"{ti.sanitize(l.name)}"' for l in layers)
    p.append(f"    constexpr const char* kModelLayerNames[{len(layers)}] = {{{names}}};\n")
    if plan.buffers:
        p.append("    struct ModelBuffer\n    {\n        std::size_t offset;\n        std::size_t size;\n"
                 "        std::size_t first_layer;\n        std::size_t last_layer;\n"
                 "        const char* outputs; // the layers whose output lives here, in order\n    };\n")
        rows = ",\n        ".join(
            f"{{{b.offset}, {b.size}, {b.first}, {b.last}, "
            f"\"{' + '.join(ti.sanitize(layers[i].name) for i in b.layers)}\"}}"
            for b in sorted(plan.buffers, key=lambda b: (b.offset, b.first)))
        p.append(f"    constexpr ModelBuffer kModelBuffers[{len(plan.buffers)}] = {{\n        {rows}\n    }};\n")
    p.append("\n    namespace model_detail {\n")

    body: list[str] = []
    weight_bytes = 0
    int4_dense_records: dict = {}

    def ptr(j: int) -> str:
        if j < 0:
            return "input"
        if j == last:
            return "output"
        return f"(arena + {plan.offsets[j] // elem})"

    for i, layer in enumerate(layers):
        nm = ti.sanitize(layer.name)
        k = f"k{nm}"
        g = layer.geometry
        src = ptr(layer.sources[0])
        dst = ptr(i)
        n = sizes[i]
        bias = f"{k}_Biases" if layer.bias is not None else "nullptr"
        reads = ", ".join("input" if j < 0 else ti.sanitize(layers[j].name) for j in layer.sources)
        p.append(f"\n        // [{i}] {layer.op_type} {nm} <- {reads}\n")

        if wide:
            weight_bytes += _emit_layer_16x8(p, body, layer, nm, k, src, dst, n, ptr)
            continue

        def rq_array(count: int) -> None:
            if layer.per_channel:
                entries = [_rq(m, s, layer.output_zp) for m, s in zip(layer.multipliers, layer.shifts)]
            else:
                entries = [_rq(layer.multipliers[0], layer.shifts[0], layer.output_zp)] * count
            rows = [", ".join(entries[r:r + 3]) for r in range(0, len(entries), 3)]
            p.append(f"        constexpr tinymind::Requantizer<int32_t, int8_t> {nm}_requantizers[{count}] = {{\n"
                     "            " + ",\n            ".join(rows) + "\n        };\n")

        if layer.op_type == "Dense" and int4_dense:
            q4 = int4_dense_records.setdefault(i, requantize_dense_int4(layer))
            packed = pack_int4(q4.codes.reshape(-1).tolist())
            weight_bytes += len(packed)
            p.append(f"        // int4 per channel: {len(packed)} bytes, against {layer.weight.size} as int8.\n")
            prow = [", ".join(f"{v:3d}" for v in packed[r:r + 16]) for r in range(0, len(packed), 16)]
            p.append(f"        constexpr uint8_t {nm}_weights_int4[{len(packed)}] = {{\n            "
                     + ",\n            ".join(prow) + "\n        };\n")
            bias4 = "nullptr"
            if q4.biases is not None:
                brow = [", ".join(str(int(v)) for v in q4.biases[r:r + 8]) for r in range(0, len(q4.biases), 8)]
                p.append(f"        constexpr int32_t {nm}_biases_int4[{len(q4.biases)}] = {{\n            "
                         + ",\n            ".join(brow) + "\n        };\n")
                bias4 = f"{nm}_biases_int4"
            entries = [_rq(m, sh, layer.output_zp) for m, sh in zip(q4.multipliers, q4.shifts)]
            rows = [", ".join(entries[r:r + 3]) for r in range(0, len(entries), 3)]
            p.append(f"        constexpr tinymind::Requantizer<int32_t, int8_t> {nm}_requantizers[{len(entries)}] = {{\n"
                     "            " + ",\n            ".join(rows) + "\n        };\n")
            p.append(f"        typedef tinymind::QDenseInt4<int8_t, int32_t, int8_t, {g['NumInputs']}, "
                     f"{g['NumOutputs']}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{nm}_weights_int4, {bias4}, {layer.input_zp}, "
                     f"{nm}_requantizers}};\n")
            p.append(f"        static_assert({nm}_type::OutputLength == {n}, \"{nm}: output size\");\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
            continue
        if layer.weight is not None:
            weight_bytes += int(layer.weight.size)
        if layer.op_type == "Conv" or (layer.op_type == "Dense" and layer.per_channel):
            if layer.op_type == "Conv":
                dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                        f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}, {g['NumFilters']}, "
                        f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}")
                filters = g["NumFilters"]
            else:
                dims = f"1, 1, {g['NumInputs']}, 1, 1, 1, 1, {g['NumOutputs']}"
                filters = g["NumOutputs"]
            if layer.per_channel:
                rq_array(filters)
                p.append(f"        typedef tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, {dims}> {nm}_type;\n")
                p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, {layer.input_zp}, "
                         f"{nm}_requantizers}};\n")
            else:
                p.append(f"        typedef tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, {dims}> {nm}_type;\n")
                p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, {layer.input_zp}, "
                         f"{_rq(layer.multipliers[0], layer.shifts[0], layer.output_zp)}}};\n")
            p.append(f"        static_assert({nm}_type::OutputSize == {n}, \"{nm}: output size\");\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        elif layer.op_type == "Dense":
            p.append(f"        typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, "
                     f"{g['NumInputs']}, {g['NumOutputs']}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, {layer.input_zp}, "
                     f"{_rq(layer.multipliers[0], layer.shifts[0], layer.output_zp)}}};\n")
            p.append(f"        static_assert({nm}_type::OutputLength == {n}, \"{nm}: output size\");\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        elif layer.op_type in ("DepthwiseConv", "AveragePool"):
            dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                    f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}, "
                    f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}")
            rq_array(g["NumFilters"])
            p.append(f"        typedef tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, {dims}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{k}_Weights, {bias}, {layer.input_zp}, "
                     f"{nm}_requantizers}};\n")
            p.append(f"        static_assert({nm}_type::OutputSize == {n}, \"{nm}: output size\");\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        elif layer.op_type == "MaxPool":
            p.append(f"        typedef tinymind::QMaxPool2D<int8_t, {g['InputHeight']}, {g['InputWidth']}, "
                     f"{g['InputChannels']}, {g['KernelHeight']}, {g['KernelWidth']}, {g['StrideH']}, "
                     f"{g['StrideW']}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{}};\n")
            p.append(f"        static_assert({nm}_type::OutputSize == {n}, \"{nm}: output size\");\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        elif layer.op_type == "Add":
            a = qadd_params(layer.input_scale, layer.input_b_scale, layer.output_scale)
            p.append(f"        typedef tinymind::QAdd<int8_t, int8_t, int8_t, {n}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{layer.input_zp}, {layer.input_b_zp}, "
                     f"{a['left_shift']}, {a['input_a_multiplier']}, {a['input_a_shift']}, "
                     f"{a['input_b_multiplier']}, {a['input_b_shift']}, "
                     f"{_rq(a['output_multiplier'], a['output_shift'], layer.output_zp)}}};\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {ptr(layer.sources[1])}, {dst});\n")
        elif layer.op_type == "Softmax":
            mult, left, diff = softmax_params(layer.input_scale)
            p.append(f"        typedef tinymind::QSoftmaxFixedPoint1D<1, {n}> {nm}_type;\n")
            p.append(f"        constexpr {nm}_type {nm}_layer = {{{mult}, {left}, {diff}}};\n")
            body.append(f"        model_detail::{nm}_layer.forward({src}, {dst});\n")
        elif layer.op_type in ("Sigmoid", "Tanh"):
            lut = activation_lut(layer.op_type, layer.input_scale, layer.input_zp,
                                 layer.output_scale, layer.output_zp)
            rows = [", ".join(f"{v:4d}" for v in lut[r:r + 16]) for r in range(0, 256, 16)]
            p.append(f"        constexpr int8_t {nm}_lut[256] = {{\n            "
                     + ",\n            ".join(rows) + "\n        };\n")
            body.append(f"        for (std::size_t j = 0; j < {n}; ++j)\n        {{\n"
                        f"            {dst}[j] = tinymind::qApplyLUT({src}[j], model_detail::{nm}_lut);\n"
                        f"        }}\n")
        elif layer.op_type == "Relu":
            body.append(f"        for (std::size_t j = 0; j < {n}; ++j)\n        {{\n"
                        f"            {dst}[j] = tinymind::qrelu<int8_t>({src}[j], {layer.output_zp});\n"
                        f"        }}\n")
        else:
            raise ValueError(f"{layer.name}: no code generation for {layer.op_type}")

    p.append("    } // namespace model_detail\n\n")
    p.append("    /**\n"
             f"     * Run the model: input (kModelInputSize {act[:-2]} values on the input grid)\n"
             "     * to output (kModelOutputSize values on the output grid). arena\n"
             "     * holds every intermediate activation; its contents need not be\n"
             "     * kept between calls.\n"
             "     */\n")
    unused_arena = plan.arena_size == 0
    p.append(f"    inline void forward(const {act}* input, {act}* output, {act}* arena)\n    {{\n")
    if unused_arena:
        p.append("        (void)arena;\n")
    p.append("".join(body))
    p.append("    }\n")
    p.append(f"    constexpr std::size_t kModelWeightBytes = {weight_bytes};\n")
    if nested_namespace:
        p.append(f"}} // namespace {nested_namespace}\n")
    p.append(f"}} // namespace {namespace}\n")

    with open(path, "w") as f:
        f.write("".join(p))
    return GeneratedModel(plan=plan, sizes=sizes, input_size=in_size, output_size=sizes[last],
                          weight_bytes=weight_bytes,
                          int4_layers=[int4_dense_records[i] for i in sorted(int4_dense_records)])


def import_onnx_model_with_code(model_path: str, weights_path: str, model_header_path: str,
                                namespace: str, alignment: int = 4, in_place: bool = True,
                                int4_dense: bool = False, nested_namespace: Optional[str] = None,
                                activation_bits: int = 8):
    """Parse an ONNX model and write both the weights header and model.hpp.
    Returns (layers, GeneratedModel)."""
    layers = ti.import_onnx_model(model_path, weights_path, namespace)
    generated = emit_model_header(model_header_path, namespace, layers,
                                  os.path.basename(weights_path), alignment=alignment, in_place=in_place,
                                  int4_dense=int4_dense, nested_namespace=nested_namespace,
                                  activation_bits=activation_bits)
    return layers, generated
