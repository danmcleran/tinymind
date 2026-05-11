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
ONNX-runtime PTQ -> TinyMind int8 importer (Phase 15).

Reads a quantized ONNX model produced by onnxruntime.quantization.quantize_static
(QDQ format: every quantized tensor is sandwiched between QuantizeLinear /
DequantizeLinear nodes carrying scale + zero_point attributes) and emits a
TinyMind-format weights.hpp identical in shape to the one produced by
apps/import_pytorch/tinymind_import.py.

Supported ops (mapped to TinyMind layer family):

    QLinearConv                  -> QConv2D            (per-tensor weights)
    QGemm / QLinearMatMul        -> QDense
    QuantizeLinear / DequantizeLinear -> consumed to extract (scale, zp)
    Relu (between QDQ pairs)     -> emitted as a separate activation
    Sigmoid / Tanh / Softmax     -> emitted with TFLite-fixed (1/256, 1/128, 1/256) scales

Heavier transformer / norm coverage left to the PyTorch importer path; the
ONNX side targets canonical CNN topologies (Conv / ReLU / Dense classifier).

This module avoids hard-dependency on onnx / onnxruntime at module import
time so the test surface can run without the heavyweight packages -- the
parse_onnx_model() entry point imports onnx lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# Shared helpers (mirror tinymind_import.py / cpp/include/qcalibration.hpp).
# ----------------------------------------------------------------------------


def lround(x: float) -> int:
    """std::lround semantics: ties round away from zero."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


COPYRIGHT_BLOCK = """/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via tinymind_import_onnx.
*/"""


# ----------------------------------------------------------------------------
# Parsed model representation.
# ----------------------------------------------------------------------------


@dataclass
class OnnxQuantizedLayer:
    """A single quantized op pulled out of a QDQ-format ONNX graph.

    op_type     : "Conv" / "Gemm" / "Relu" / "Sigmoid" / "Tanh" / "Softmax".
    name        : human-readable layer name (output tensor name).
    input_name  : name of the activation tensor feeding this layer.
    input_scale / input_zp : per-tensor activation calibration for the input.
    output_scale / output_zp : per-tensor activation calibration for the output.
    weight (np.int8) / weight_scale : per-tensor symmetric weights (if any).
    bias (np.int32) / bias_scale : int32 bias and its effective scale.

    Activation-only ops (Relu / Sigmoid / Tanh / Softmax) leave the weight /
    bias fields as None.
    """

    op_type: str
    name: str
    input_name: str
    input_scale: float
    input_zp: int
    output_scale: float
    output_zp: int
    weight: object = None         # numpy int8 array
    weight_scale: float = 0.0
    bias: object = None           # numpy int32 array
    bias_scale: float = 0.0


def parse_onnx_model(model_path: str) -> list[OnnxQuantizedLayer]:
    """Parse a QDQ-format quantized ONNX model into TinyMind layer specs.

    Walks the topologically-sorted node list. For each Conv / Gemm node
    found between QuantizeLinear / DequantizeLinear pairs, extracts the
    int8 weight initializer, int32 bias initializer, per-tensor scales
    and zero points from the surrounding Q/DQ nodes, and emits an
    `OnnxQuantizedLayer` record. Relu / Sigmoid / Tanh / Softmax that
    follow are emitted as activation-only layers with their own
    input / output calibration.

    Layer ordering in the returned list matches graph execution order so
    the importer can stream it directly through quantize_weights / emit.

    Requires the `onnx` package (loaded lazily here so the rest of this
    module is importable in environments without it).
    """
    import onnx
    import numpy as np

    model = onnx.load(model_path)
    initializers = {init.name: init for init in model.graph.initializer}

    def _as_numpy(init):
        return onnx.numpy_helper.to_array(init)

    # Walk nodes, recording per-tensor (scale, zp) seen on QuantizeLinear
    # outputs so we can look them up when the next compute node consumes
    # them.
    tensor_qparams: dict[str, tuple[float, int]] = {}
    layers: list[OnnxQuantizedLayer] = []

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            x_in, s_in, zp_in = node.input[0], node.input[1], node.input[2]
            scale = float(_as_numpy(initializers[s_in]).reshape(-1)[0])
            zp = int(_as_numpy(initializers[zp_in]).reshape(-1)[0])
            tensor_qparams[node.output[0]] = (scale, zp)
            continue
        if node.op_type == "DequantizeLinear":
            x_in, s_in, zp_in = node.input[0], node.input[1], node.input[2]
            scale = float(_as_numpy(initializers[s_in]).reshape(-1)[0])
            zp = int(_as_numpy(initializers[zp_in]).reshape(-1)[0])
            tensor_qparams[node.output[0]] = (scale, zp)
            tensor_qparams[node.input[0]] = (scale, zp)
            continue
        if node.op_type in ("QLinearConv", "QLinearMatMul"):
            x_in    = node.input[0]
            x_scale = float(_as_numpy(initializers[node.input[1]]).reshape(-1)[0])
            x_zp    = int(_as_numpy(initializers[node.input[2]]).reshape(-1)[0])
            w_q     = _as_numpy(initializers[node.input[3]])
            w_scale = float(_as_numpy(initializers[node.input[4]]).reshape(-1)[0])
            y_scale = float(_as_numpy(initializers[node.input[6]]).reshape(-1)[0])
            y_zp    = int(_as_numpy(initializers[node.input[7]]).reshape(-1)[0])
            b_q = None
            bias_scale = x_scale * w_scale
            if node.op_type == "QLinearConv" and len(node.input) >= 9:
                b_q = _as_numpy(initializers[node.input[8]])
            layer_kind = "Conv" if node.op_type == "QLinearConv" else "Gemm"
            layers.append(OnnxQuantizedLayer(
                op_type=layer_kind,
                name=node.output[0],
                input_name=x_in,
                input_scale=x_scale, input_zp=x_zp,
                output_scale=y_scale, output_zp=y_zp,
                weight=w_q.astype(np.int8),
                weight_scale=w_scale,
                bias=b_q.astype(np.int32) if b_q is not None else None,
                bias_scale=bias_scale,
            ))
            tensor_qparams[node.output[0]] = (y_scale, y_zp)
            continue
        if node.op_type in ("Relu", "Sigmoid", "Tanh", "Softmax"):
            x_in = node.input[0]
            in_scale, in_zp = tensor_qparams.get(x_in, (1.0, 0))
            if node.op_type == "Sigmoid":
                out_scale, out_zp = 1.0 / 256.0, -128
            elif node.op_type == "Tanh":
                out_scale, out_zp = 1.0 / 128.0, 0
            elif node.op_type == "Softmax":
                out_scale, out_zp = 1.0 / 256.0, -128
            else:
                out_scale, out_zp = tensor_qparams.get(node.output[0], (in_scale, in_zp))
            layers.append(OnnxQuantizedLayer(
                op_type=node.op_type,
                name=node.output[0],
                input_name=x_in,
                input_scale=in_scale, input_zp=in_zp,
                output_scale=out_scale, output_zp=out_zp,
            ))
            tensor_qparams[node.output[0]] = (out_scale, out_zp)
    return layers


# ----------------------------------------------------------------------------
# Emitter.
# ----------------------------------------------------------------------------


def _fmt_int_array(name: str, values, cpp_type: str, per_line: int = 12) -> str:
    vs = list(values)
    lines: list[str] = []
    for i in range(0, len(vs), per_line):
        chunk = ", ".join(f"{v:>5d}" for v in vs[i:i + per_line])
        lines.append("        " + chunk + ("," if i + per_line < len(vs) else ""))
    body = "\n".join(lines)
    return f"    constexpr {cpp_type} {name}[{len(vs)}] = {{\n{body}\n    }};\n"


def _sanitize(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def emit_weights_header(path: str,
                        namespace: str,
                        layers: list[OnnxQuantizedLayer],
                        meta: Optional[dict[str, int]] = None) -> None:
    """Write a TinyMind-format weights.hpp from parsed ONNX layers."""
    parts: list[str] = []
    parts.append(COPYRIGHT_BLOCK)
    parts.append("\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n")
    parts.append(f"namespace {namespace} {{\n")
    if meta is not None:
        for k, v in meta.items():
            parts.append(f"    constexpr std::size_t {k} = {v};\n")
        parts.append("\n")

    parts.append("    // Activation calibration (input + per-layer output).\n")
    seen: set[str] = set()
    for layer in layers:
        nm_in = _sanitize(layer.input_name)
        if nm_in not in seen:
            parts.append(f"    constexpr float   k{nm_in}_Scale     = {layer.input_scale!r}f;\n")
            parts.append(f"    constexpr int32_t k{nm_in}_ZeroPoint = {layer.input_zp};\n")
            seen.add(nm_in)
        nm = _sanitize(layer.name)
        if nm not in seen:
            parts.append(f"    constexpr float   k{nm}_Scale     = {layer.output_scale!r}f;\n")
            parts.append(f"    constexpr int32_t k{nm}_ZeroPoint = {layer.output_zp};\n")
            seen.add(nm)

    parts.append("\n    // Weights + biases.\n")
    for layer in layers:
        if layer.weight is None:
            continue
        nm = _sanitize(layer.name)
        parts.append(f"    constexpr float k{nm}_WeightScale = {layer.weight_scale!r}f;\n")
        parts.append(_fmt_int_array(f"k{nm}_Weights",
                                    layer.weight.reshape(-1).tolist(),
                                    "int8_t"))
        if layer.bias is not None:
            parts.append(_fmt_int_array(f"k{nm}_Biases",
                                        layer.bias.tolist(),
                                        "int32_t"))
    parts.append(f"}} // namespace {namespace}\n")
    with open(path, "w") as f:
        f.write("".join(parts))


def import_onnx_model(model_path: str,
                      output_path: str,
                      namespace: str,
                      meta: Optional[dict[str, int]] = None) -> list[OnnxQuantizedLayer]:
    """Top-level: parse the ONNX model, write the TinyMind weights.hpp."""
    layers = parse_onnx_model(model_path)
    emit_weights_header(output_path, namespace, layers, meta)
    return layers
