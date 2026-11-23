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
class GeneratedModel:
    plan: MemoryPlan
    sizes: dict
    input_size: int
    output_size: int


def emit_model_header(path: str, namespace: str, layers, weights_header: str,
                      alignment: int = 4, in_place: bool = True) -> GeneratedModel:
    """Write model.hpp for parsed records whose constants weights_header
    (written by emit_weights_header with the same namespace) holds.

    Raises ValueError for a graph this generator cannot run as planned: a
    Relu whose input and output grids differ (no record carries the
    rescale) or a layer that reads a softmax's output (tinymind's softmax
    writes the 1/256 grid whatever the graph calibrated)."""
    if not layers:
        raise ValueError("no layers to generate")
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
        if layer.op_type == "Softmax":
            for k, later in enumerate(layers[i + 1:], start=i + 1):
                if i in later.sources:
                    raise ValueError(f"{later.name}: reads softmax {layer.name}'s output; tinymind's softmax "
                                     "writes the 1/256 grid, so only the model's output may be a softmax")
        if layer.op_type == "Relu" and (_f32(layer.input_scale) != _f32(layer.output_scale)
                                        or layer.input_zp != layer.output_zp):
            raise ValueError(f"{layer.name}: a Relu whose output grid differs from its input's "
                             "needs a rescale no record carries")

    plan = plan_memory(layers, sizes, alignment=alignment, in_place=in_place)
    check_plan(plan)

    last = len(layers) - 1
    out_layer = layers[last]
    if out_layer.op_type == "Softmax":
        out_scale, out_zp = SOFTMAX_OUTPUT_SCALE, SOFTMAX_OUTPUT_ZERO_POINT
    else:
        out_scale, out_zp = out_layer.output_scale, out_layer.output_zp
    first_reader = next(l for l in layers if -1 in l.sources)

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
    p.append("    // Model interface. input and output are caller-owned int8 buffers in\n"
             "    // NHWC order; arena is caller-owned scratch of kModelArenaSize bytes,\n"
             "    // aligned to kModelArenaAlignment, overlapping neither of them.\n")
    p.append(f"    constexpr std::size_t kModelInputSize      = {in_size};\n")
    p.append(f"    constexpr std::size_t kModelOutputSize     = {sizes[last]};\n")
    p.append(f"    constexpr std::size_t kModelArenaSize      = {plan.arena_size};\n")
    p.append(f"    constexpr std::size_t kModelArenaAlignment = {alignment};\n")
    p.append(f"    constexpr float   kModelInputScale      = {first_reader.input_scale!r}f;\n")
    p.append(f"    constexpr int32_t kModelInputZeroPoint  = {first_reader.input_zp};\n")
    p.append(f"    constexpr float   kModelOutputScale     = {out_scale!r}f;\n")
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

    def ptr(j: int) -> str:
        if j < 0:
            return "input"
        if j == last:
            return "output"
        return f"(arena + {plan.offsets[j]})"

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

        def rq_array(count: int) -> None:
            if layer.per_channel:
                entries = [_rq(m, s, layer.output_zp) for m, s in zip(layer.multipliers, layer.shifts)]
            else:
                entries = [_rq(layer.multipliers[0], layer.shifts[0], layer.output_zp)] * count
            rows = [", ".join(entries[r:r + 3]) for r in range(0, len(entries), 3)]
            p.append(f"        constexpr tinymind::Requantizer<int32_t, int8_t> {nm}_requantizers[{count}] = {{\n"
                     "            " + ",\n            ".join(rows) + "\n        };\n")

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
             "     * Run the model: input (kModelInputSize int8 values on the input grid)\n"
             "     * to output (kModelOutputSize values on the output grid). arena\n"
             "     * holds every intermediate activation; its contents need not be\n"
             "     * kept between calls.\n"
             "     */\n")
    unused_arena = plan.arena_size == 0
    p.append("    inline void forward(const int8_t* input, int8_t* output, int8_t* arena)\n    {\n")
    if unused_arena:
        p.append("        (void)arena;\n")
    p.append("".join(body))
    p.append("    }\n")
    p.append(f"}} // namespace {namespace}\n")

    with open(path, "w") as f:
        f.write("".join(p))
    return GeneratedModel(plan=plan, sizes=sizes, input_size=in_size, output_size=sizes[last])


def import_onnx_model_with_code(model_path: str, weights_path: str, model_header_path: str,
                                namespace: str, alignment: int = 4, in_place: bool = True):
    """Parse an ONNX model and write both the weights header and model.hpp.
    Returns (layers, GeneratedModel)."""
    layers = ti.import_onnx_model(model_path, weights_path, namespace)
    generated = emit_model_header(model_header_path, namespace, layers,
                                  os.path.basename(weights_path), alignment=alignment, in_place=in_place)
    return layers, generated
