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

ONNX Runtime PTQ -> tinymind int8 importer (Phase 15).

Reads a model quantized by onnxruntime.quantization.quantize_static and emits
a tinymind-format weights.hpp. Both of ONNX Runtime's output formats are read:

  QDQ (ONNX Runtime's default): float Conv / Gemm / MatMul nodes whose inputs
      come from DequantizeLinear and whose outputs feed QuantizeLinear. The
      int8 weights, int32 biases and every (scale, zero_point) are read from
      those Q/DQ nodes.
  QOperator: QLinearConv / QGemm / QLinearMatMul (+ QLinearAdd bias) /
      QLinearSigmoid nodes that carry the same values as inputs.

Mapping onto tinymind's int8 layers:

    Conv (group 1, dilation 1)  -> QConv2D, or QConv2DPerChannel when the
                                   weights are quantized per output channel
    Gemm / MatMul [+ Add bias]  -> QDense, or (per channel) QConv2DPerChannel
                                   with a 1x1 kernel over a 1x1xN input,
                                   which is a dense layer with one
                                   requantizer per output
    Relu / Sigmoid / Tanh / Softmax -> activation-only records carrying the
                                   input and output (scale, zero_point) the
                                   graph uses (build the int8 LUTs from them)
    Flatten / Reshape-to-2D / Identity -> no layer; a dense layer fed by a
                                   flattened NCHW tensor gets its weight
                                   columns permuted to tinymind's NHWC order

Layouts emitted: conv weights OHWI [F][KH][KW][C] (ONNX stores OIHW), dense
weights [out][in] (MatMul stores [in][out]), int32 biases at scale
input_scale * weight_scale. Feed the network NHWC input (ONNX models take
NCHW). ONNX Runtime fuses a Relu that follows a Conv or Gemm into that
layer's output range: the output zero point sits at the bottom of the int8
range, so the requantizer's clamp applies the Relu and no separate layer is
emitted.

Anything else is refused with ValueError rather than skipped: an op this
module does not map, a weight zero point other than 0 (tinymind weights are
symmetric), uint8 tensors, grouped or dilated convolutions, Gemm with
transA / alpha / beta other than the defaults, auto_pad SAME_*, and a
compute layer whose output is not quantized.

The onnx package is imported lazily inside parse_onnx_model, so the emitter
half of this module works without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# Shared helpers (mirror tinymind_import.py / cpp/qaffine.hpp).
# ----------------------------------------------------------------------------


def lround(x: float) -> int:
    """std::lround semantics: ties round away from zero."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def quantize_multiplier(ratio: float) -> tuple[int, int]:
    """Mirror cpp/qaffine.hpp::quantizeMultiplier (and tinymind_import.py).

    Decomposes a positive ratio into the Q0.31 (multiplier, shift) pair a
    Requantizer takes. shift > 0 right-shifts after the multiply; shift < 0
    left-shifts before it. The pair always satisfies -31 <= shift <= 31.
    """
    if ratio == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(ratio)
    q = lround(mantissa * float(1 << 31))
    if q == (1 << 31):
        q >>= 1
        exponent += 1
    shift = -int(exponent)
    if shift > 31:
        return 0, 0
    return int(q), max(shift, -31)


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
    """One tinymind layer recovered from a quantized ONNX graph.

    op_type      : "Conv" / "Dense" for weight layers; "Relu" / "Sigmoid" /
                   "Tanh" / "Softmax" for activation-only records.
    name         : C++-safe layer name (from the ONNX node name).
    input_name / output_name : ONNX tensor names of the quantized input and
                   output.
    input_scale / input_zp / output_scale / output_zp : per-tensor
                   activation quantization.
    weight       : int8 numpy array in tinymind layout -- OHWI
                   [F][KH][KW][C] for Conv, [out][in] for Dense.
    weight_scales: one scale per output channel when per_channel, else a
                   single-element list. weight_scale is weight_scales[0].
    bias         : int32 numpy array at scale input_scale * weight_scales[o],
                   or None.
    multipliers / shifts : the Requantizer (multiplier, shift) per output
                   channel (one entry when per-tensor), for ratio
                   input_scale * weight_scale / output_scale.
    geometry     : Conv: H, W, C, KH, KW, StrideH, StrideW, F, PadTop,
                   PadBottom, PadLeft, PadRight, OutH, OutW.
                   Dense: NumInputs, NumOutputs.
    """

    op_type: str
    name: str
    input_name: str
    input_scale: float
    input_zp: int
    output_scale: float
    output_zp: int
    output_name: str = ""
    weight: object = None
    weight_scale: float = 0.0
    weight_scales: list = field(default_factory=list)
    per_channel: bool = False
    bias: object = None
    bias_scale: float = 0.0
    multipliers: list = field(default_factory=list)
    shifts: list = field(default_factory=list)
    geometry: dict = field(default_factory=dict)


def sanitize(name: str) -> str:
    """Map an ONNX name to a C++ identifier fragment."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    text = "".join(out).strip("_")
    return text if text else "layer"


def _finish_weight_layer(op_type, name, input_name, input_scale, input_zp,
                         output_name, output_scale, output_zp,
                         weight_q, weight_scales, bias_float, geometry):
    """Common tail: quantize the bias, build requantizers, fill the record."""
    import numpy as np

    out_channels = weight_q.shape[0]
    scales = [float(s) for s in weight_scales]
    per_channel = len(scales) > 1
    if per_channel and len(scales) != out_channels:
        raise ValueError(f"{name}: {len(scales)} weight scales for {out_channels} output channels")
    channel_scales = scales if per_channel else scales * out_channels

    bias_q = None
    if bias_float is not None:
        bias_float = np.asarray(bias_float, dtype=np.float64).reshape(-1)
        if bias_float.shape[0] != out_channels:
            raise ValueError(f"{name}: bias has {bias_float.shape[0]} values for {out_channels} outputs")
        qmin, qmax = -(1 << 31), (1 << 31) - 1
        bias_q = np.array([max(qmin, min(qmax, lround(b / (input_scale * s))))
                           for b, s in zip(bias_float.tolist(), channel_scales)], dtype=np.int32)

    multipliers, shifts = [], []
    for s in (scales if per_channel else scales[:1]):
        m, sh = quantize_multiplier(input_scale * s / output_scale)
        multipliers.append(m)
        shifts.append(sh)

    return OnnxQuantizedLayer(
        op_type=op_type, name=name, input_name=input_name,
        input_scale=input_scale, input_zp=input_zp,
        output_scale=output_scale, output_zp=output_zp, output_name=output_name,
        weight=weight_q.astype(np.int8), weight_scale=scales[0], weight_scales=scales,
        per_channel=per_channel, bias=bias_q, bias_scale=input_scale * scales[0],
        multipliers=multipliers, shifts=shifts, geometry=geometry)


def parse_onnx_model(model_path: str) -> list[OnnxQuantizedLayer]:
    """Parse a QDQ- or QOperator-format quantized ONNX model.

    Returns the tinymind layers in execution order. Raises ValueError for
    anything that cannot be mapped exactly (see the module docstring); a
    layer is never dropped silently.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    graph = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    inits[node.output[0]] = numpy_helper.to_array(a.t)

    producer = {}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    def attrs(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def scalar_scale(tensor, what):
        arr = np.asarray(inits[tensor], dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"{what}: per-channel activation scale is not supported")
        return float(arr[0])

    def zero_point(tensor, what, allow_int32=False):
        if tensor is None or tensor == "":
            return np.zeros(1, dtype=np.int64)
        arr = np.asarray(inits[tensor])
        if arr.dtype == np.uint8:
            raise ValueError(f"{what}: uint8 tensors are not supported; quantize with "
                             "activation_type=QuantType.QInt8 and weight_type=QuantType.QInt8")
        if arr.dtype != np.int8 and not (allow_int32 and arr.dtype == np.int32):
            raise ValueError(f"{what}: unsupported zero-point type {arr.dtype}")
        return arr.astype(np.int64).reshape(-1)

    def node_label(node):
        return sanitize(node.name if node.name else node.output[0])

    # Shapes, tracked in ONNX's NCHW convention. flat_chw[t] = (C, H, W) when
    # tensor t is a flattened 4D tensor, so the dense layer it feeds can
    # permute its weight columns to tinymind's NHWC flattening order.
    shapes: dict[str, list] = {}
    flat_chw: dict[str, tuple] = {}
    for vi in graph.input:
        if vi.name in inits:
            continue
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1)
        shapes[vi.name] = dims

    def same_shape(dst, src):
        if src in shapes:
            shapes[dst] = list(shapes[src])
        if src in flat_chw:
            flat_chw[dst] = flat_chw[src]

    def flatten_to_2d(dst, src):
        shape = shapes.get(src)
        if shape is None:
            raise ValueError(f"cannot flatten {src}: its shape is unknown")
        if len(shape) == 4:
            flat_chw[dst] = (shape[1], shape[2], shape[3])
        elif src in flat_chw:
            flat_chw[dst] = flat_chw[src]
        shapes[dst] = [shape[0], int(np.prod(shape[1:]))]

    def conv_geometry(node, x_name, weight_oihw):
        a = attrs(node)
        if a.get("group", 1) != 1:
            raise ValueError(f"{node.name}: grouped/depthwise Conv (group={a['group']}) is not supported")
        if any(d != 1 for d in a.get("dilations", [1, 1])):
            raise ValueError(f"{node.name}: dilated Conv is not supported")
        auto_pad = a.get("auto_pad", b"NOTSET")
        auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
        if auto_pad not in ("NOTSET", "VALID"):
            raise ValueError(f"{node.name}: auto_pad={auto_pad} is not supported; export explicit pads")
        pads = a.get("pads", [0, 0, 0, 0]) if auto_pad == "NOTSET" else [0, 0, 0, 0]
        strides = a.get("strides", [1, 1])
        x_shape = shapes.get(x_name)
        if x_shape is None or len(x_shape) != 4:
            raise ValueError(f"{node.name}: Conv input shape is unknown or not 4D")
        _, c, h, w = x_shape
        f, wc, kh, kw = weight_oihw.shape
        if wc != c:
            raise ValueError(f"{node.name}: weight has {wc} input channels, input has {c}")
        oh = (h + pads[0] + pads[2] - kh) // strides[0] + 1
        ow = (w + pads[1] + pads[3] - kw) // strides[1] + 1
        return {"InputHeight": h, "InputWidth": w, "InputChannels": c,
                "KernelHeight": kh, "KernelWidth": kw,
                "StrideH": strides[0], "StrideW": strides[1], "NumFilters": f,
                "PadTop": pads[0], "PadLeft": pads[1], "PadBottom": pads[2], "PadRight": pads[3],
                "OutputHeight": oh, "OutputWidth": ow}, [x_shape[0], f, oh, ow]

    def oihw_to_ohwi(weight):
        return np.ascontiguousarray(np.transpose(weight, (0, 2, 3, 1)))

    def dense_from_out_in(weight_out_in, x_name, label):
        """Permute the input columns of a [out][in] dense weight from NCHW
        flattening order to NHWC when the input is a flattened 4D tensor."""
        x_shape = shapes.get(x_name)
        n_in = weight_out_in.shape[1]
        if x_shape is not None and int(np.prod(x_shape[1:])) != n_in:
            raise ValueError(f"{label}: dense layer expects {n_in} inputs, its input has shape {x_shape}")
        if x_name in flat_chw:
            c, h, w = flat_chw[x_name]
            if c * h * w != n_in:
                raise ValueError(f"{label}: flattened input {c}x{h}x{w} does not match {n_in} weights")
            # tinymind index (y*W + x)*C + ch  <-  ONNX index ch*H*W + y*W + x
            perm = np.arange(n_in).reshape(c, h, w).transpose(1, 2, 0).reshape(-1)
            weight_out_in = weight_out_in[:, perm]
        return np.ascontiguousarray(weight_out_in)

    def check_weight_zero_points(zp, label):
        if np.any(zp != 0):
            raise ValueError(f"{label}: weight zero point {zp.tolist()} is not 0; tinymind's int8 "
                             "layers use symmetric weights (quantize with weight_type=QuantType.QInt8, "
                             "which is symmetric)")

    def weight_scales_for(scales, axis, n_out, out_axis, label):
        scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        if scales.size == 1:
            return [float(scales[0])]
        if axis != out_axis or scales.size != n_out:
            raise ValueError(f"{label}: per-channel weight scales must run along the output channels")
        return [float(s) for s in scales]

    layers: list[OnnxQuantizedLayer] = []

    # ---------------------------------------------------------------- QDQ
    def dq_source(tensor):
        """(quantized tensor, scale tensor, zp tensor, axis) for a DQ output."""
        node = producer.get(tensor)
        if node is None or node.op_type != "DequantizeLinear":
            return None
        zp = node.input[2] if len(node.input) > 2 else ""
        return node.input[0], node.input[1], zp, attrs(node).get("axis", 1)

    def dq_activation(tensor, label):
        src = dq_source(tensor)
        if src is None or src[0] in inits:
            raise ValueError(f"{label}: input {tensor} is not a dequantized activation")
        return src[0], scalar_scale(src[1], label), int(zero_point(src[2], label)[0])

    def dq_constant(tensor, label, allow_int32=False):
        """Float value of a constant input: an initializer or the DQ of one.
        Returns (float array, raw quantized array or None, scales, zp, axis)."""
        if tensor in inits:
            return np.asarray(inits[tensor], dtype=np.float64), None, None, None, None
        src = dq_source(tensor)
        if src is None or src[0] not in inits:
            raise ValueError(f"{label}: {tensor} is not a constant")
        q = np.asarray(inits[src[0]])
        scales = np.asarray(inits[src[1]], dtype=np.float64)
        zp = zero_point(src[2], label, allow_int32=allow_int32)
        axis = src[3]
        if scales.size == 1:
            real = (q.astype(np.float64) - zp[0]) * scales.reshape(-1)[0]
        else:
            shape = [1] * q.ndim
            shape[axis % q.ndim] = -1
            real = (q.astype(np.float64) - zp.reshape(shape)) * scales.reshape(shape)
        return real, q, scales, zp, axis

    def quantized_output(tensor, label):
        """(scale, zp, quantized tensor name) of the QuantizeLinear consuming tensor."""
        users = consumers.get(tensor, [])
        qs = [n for n in users if n.op_type == "QuantizeLinear"]
        if len(users) != 1 or len(qs) != 1:
            return None
        q = qs[0]
        return scalar_scale(q.input[1], label), int(zero_point(q.input[2], label)[0]), q.output[0]

    skip_nodes: set[int] = set()

    def is_constant(tensor):
        if tensor in inits:
            return True
        src = dq_source(tensor)
        return src is not None and src[0] in inits

    def bias_add_after(tensor):
        """(Add node, constant operand) when tensor feeds a bias Add, directly
        or through a QuantizeLinear -> DequantizeLinear pair; else (None, None)."""
        hop = tensor
        for _ in range(2):
            users = consumers.get(hop, [])
            if len(users) != 1:
                return None, None
            user = users[0]
            if user.op_type == "Add":
                other = user.input[1] if user.input[0] == hop else user.input[0]
                if is_constant(other):
                    return user, other
                return None, None
            if user.op_type != "QuantizeLinear":
                return None, None
            dq_users = consumers.get(user.output[0], [])
            if len(dq_users) != 1 or dq_users[0].op_type != "DequantizeLinear":
                return None, None
            hop = dq_users[0].output[0]
        return None, None

    def parse_qdq_weight_node(node):
        label = node_label(node)
        x_q, x_scale, x_zp = dq_activation(node.input[0], label)
        _, w_q, w_scales, w_zp, w_axis = dq_constant(node.input[1], label)
        if w_q is None:
            raise ValueError(f"{label}: weights are float, not quantized")
        if w_q.dtype != np.int8:
            raise ValueError(f"{label}: weights must be int8, got {w_q.dtype}")
        check_weight_zero_points(w_zp, label)
        bias = None
        if node.op_type in ("Conv", "Gemm") and len(node.input) > 2 and node.input[2]:
            bias = dq_constant(node.input[2], label, allow_int32=True)[0].reshape(-1)
        a = attrs(node)
        out_tensor = node.output[0]
        if node.op_type == "Conv":
            geometry, out_shape = conv_geometry(node, x_q, w_q)
            scales = weight_scales_for(w_scales, w_axis, w_q.shape[0], 0, label)
            weight = oihw_to_ohwi(w_q)
            kind = "Conv"
        else:
            if node.op_type == "Gemm":
                if a.get("transA", 0) != 0 or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                    raise ValueError(f"{label}: Gemm with transA/alpha/beta other than 0/1/1 is not supported")
                trans_b = a.get("transB", 0)
            else:
                trans_b = 0
            if trans_b:
                w_out_in, out_axis = w_q, 0
            else:
                w_out_in, out_axis = w_q.T, 1
            scales = weight_scales_for(w_scales, w_axis, w_out_in.shape[0], out_axis, label)
            weight = dense_from_out_in(w_out_in, x_q, label)
            if node.op_type == "MatMul":
                # A bias Add after the MatMul (Linear on a 3D input, or a
                # hand-built graph) folds into this layer's bias. ONNX Runtime
                # may quantize the MatMul output before the Add (Q -> DQ ->
                # Add); the folded layer then rounds once where ONNX Runtime
                # rounds twice, so it can differ from it by one more LSB.
                add, other = bias_add_after(out_tensor)
                if add is not None:
                    bias = dq_constant(other, label, allow_int32=True)[0].reshape(-1)
                    out_tensor = add.output[0]
                    skip_nodes.add(id(add))
            geometry = {"NumInputs": weight.shape[1], "NumOutputs": weight.shape[0]}
            out_shape = [shapes.get(x_q, [1])[0], weight.shape[0]]
            kind = "Dense"
        q_out = quantized_output(out_tensor, label)
        if q_out is None:
            raise ValueError(f"{label}: output {out_tensor} is not quantized by a single QuantizeLinear")
        y_scale, y_zp, y_name = q_out
        shapes[y_name] = out_shape
        layers.append(_finish_weight_layer(kind, label, x_q, x_scale, x_zp, y_name, y_scale, y_zp,
                                           weight, scales, bias, geometry))

    def parse_qdq_activation(node):
        label = node_label(node)
        x_q, x_scale, x_zp = dq_activation(node.input[0], label)
        q_out = quantized_output(node.output[0], label)
        if q_out is None:
            raise ValueError(f"{label}: {node.op_type} output is not quantized by a single QuantizeLinear")
        y_scale, y_zp, y_name = q_out
        same_shape(y_name, x_q)
        layers.append(OnnxQuantizedLayer(op_type=node.op_type, name=label, input_name=x_q,
                                         input_scale=x_scale, input_zp=x_zp, output_name=y_name,
                                         output_scale=y_scale, output_zp=y_zp))

    # ---------------------------------------------------------- QOperator
    def qop_activation_params(node, s_idx, z_idx, label):
        return scalar_scale(node.input[s_idx], label), int(zero_point(node.input[z_idx], label)[0])

    def parse_qop_weight_node(node):
        label = node_label(node)
        if node.op_type == "QLinearConv":
            x, xs, xz, w, ws, wz, ys, yz = node.input[:8]
            b = node.input[8] if len(node.input) > 8 and node.input[8] else None
        elif node.op_type == "QGemm":
            x, xs, xz, w, ws, wz = node.input[:6]
            b = node.input[6] if len(node.input) > 6 and node.input[6] else None
            if len(node.input) < 9 or not node.input[7]:
                raise ValueError(f"{label}: QGemm with a float output is not supported")
            ys, yz = node.input[7], node.input[8]
        else:  # QLinearMatMul
            x, xs, xz, w, ws, wz, ys, yz = node.input[:8]
            b = None
        if w not in inits:
            raise ValueError(f"{label}: weight input {w} is not a constant")
        x_scale = scalar_scale(xs, label)
        x_zp = int(zero_point(xz, label)[0])
        w_q = np.asarray(inits[w])
        if w_q.dtype != np.int8:
            raise ValueError(f"{label}: weights must be int8, got {w_q.dtype}")
        check_weight_zero_points(zero_point(wz, label), label)
        w_scales = np.asarray(inits[ws], dtype=np.float64).reshape(-1)
        y_scale = scalar_scale(ys, label)
        y_zp = int(zero_point(yz, label)[0])
        y_name = node.output[0]
        a = attrs(node)
        if node.op_type == "QLinearConv":
            geometry, out_shape = conv_geometry(node, x, w_q)
            scales = weight_scales_for(w_scales, 0, w_q.shape[0], 0, label)
            bias = None
            if b is not None:
                bq = np.asarray(inits[b], dtype=np.float64).reshape(-1)
                bias = bq * x_scale * np.asarray(scales if len(scales) > 1 else scales * len(bq))
            weight = oihw_to_ohwi(w_q)
            kind = "Conv"
        else:
            if node.op_type == "QGemm":
                if a.get("transA", 0) != 0 or a.get("alpha", 1.0) != 1.0:
                    raise ValueError(f"{label}: QGemm with transA/alpha other than 0/1 is not supported")
                trans_b = a.get("transB", 0)
            else:
                trans_b = 0
            if trans_b:
                w_out_in, out_axis = w_q, 0
            else:
                w_out_in, out_axis = w_q.T, 1
            n_out = w_out_in.shape[0]
            # QOperator weight scales carry no axis attribute; per-channel
            # scales always run along the output channels.
            scales = weight_scales_for(w_scales, out_axis, n_out, out_axis, label)
            weight = dense_from_out_in(w_out_in, x, label)
            channel_scales = np.asarray(scales if len(scales) > 1 else scales * n_out)
            bias = None
            if b is not None:
                bias = np.asarray(inits[b], dtype=np.float64).reshape(-1) * x_scale * channel_scales
            if node.op_type == "QLinearMatMul":
                users = consumers.get(y_name, [])
                if len(users) == 1 and users[0].op_type == "QLinearAdd":
                    add = users[0]
                    # QLinearAdd(A, As, Az, B, Bs, Bz, Cs, Cz)
                    if add.input[0] == y_name:
                        c_q, c_s, c_z = add.input[3], add.input[4], add.input[5]
                    else:
                        c_q, c_s, c_z = add.input[0], add.input[1], add.input[2]
                    if c_q not in inits:
                        raise ValueError(f"{label}: QLinearAdd after QLinearMatMul is not a constant bias")
                    bias = ((np.asarray(inits[c_q], dtype=np.float64).reshape(-1)
                             - zero_point(c_z, label)[0]) * scalar_scale(c_s, label))
                    y_scale = scalar_scale(add.input[6], label)
                    y_zp = int(zero_point(add.input[7], label)[0])
                    y_name = add.output[0]
            geometry = {"NumInputs": weight.shape[1], "NumOutputs": weight.shape[0]}
            out_shape = [shapes.get(x, [1])[0], weight.shape[0]]
            kind = "Dense"
        shapes[y_name] = out_shape
        layers.append(_finish_weight_layer(kind, label, x, x_scale, x_zp, y_name, y_scale, y_zp,
                                           weight, scales, bias, geometry))
        return y_name

    # ------------------------------------------------------------- walk
    for node in graph.node:
        if id(node) in skip_nodes:
            continue
        op = node.op_type
        if op in ("QuantizeLinear", "DequantizeLinear"):
            same_shape(node.output[0], node.input[0])
            continue
        if op == "Constant":
            continue
        if op in ("Identity",):
            same_shape(node.output[0], node.input[0])
            continue
        if op == "Flatten":
            if attrs(node).get("axis", 1) != 1:
                raise ValueError(f"{node_label(node)}: Flatten with axis != 1 is not supported")
            flatten_to_2d(node.output[0], node.input[0])
            continue
        if op == "Reshape":
            target = np.asarray(inits.get(node.input[1], [])).reshape(-1)
            if target.size != 2:
                raise ValueError(f"{node_label(node)}: only a Reshape to 2D (a flatten) is supported")
            flatten_to_2d(node.output[0], node.input[0])
            continue
        if op in ("Conv", "Gemm", "MatMul"):
            parse_qdq_weight_node(node)
            continue
        if op in ("QLinearConv", "QGemm", "QLinearMatMul"):
            parse_qop_weight_node(node)
            if op == "QLinearMatMul":
                users = consumers.get(node.output[0], [])
                if len(users) == 1 and users[0].op_type == "QLinearAdd":
                    skip_nodes.add(id(users[0]))
            continue
        if op in ("Relu", "Sigmoid", "Tanh", "Softmax"):
            parse_qdq_activation(node)
            continue
        if op == "QLinearSigmoid":
            label = node_label(node)
            xs, xz = qop_activation_params(node, 1, 2, label)
            ys, yz = qop_activation_params(node, 3, 4, label)
            same_shape(node.output[0], node.input[0])
            layers.append(OnnxQuantizedLayer(op_type="Sigmoid", name=label, input_name=node.input[0],
                                             input_scale=xs, input_zp=xz, output_name=node.output[0],
                                             output_scale=ys, output_zp=yz))
            continue
        raise ValueError(f"unsupported ONNX op {op} (node {node.name or node.output[0]}); "
                         "tinymind_import_onnx maps Conv, Gemm, MatMul(+Add), Relu, Sigmoid, Tanh, "
                         "Softmax, Flatten, Reshape-to-2D and their QOperator forms")
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


def _fmt_float_array(name: str, values) -> str:
    body = ", ".join(f"{float(v)!r}f" for v in values)
    return f"    constexpr float {name}[{len(values)}] = {{ {body} }};\n"


def _sanitize(name: str) -> str:
    return sanitize(name)


def emit_weights_header(path: str,
                        namespace: str,
                        layers: list[OnnxQuantizedLayer],
                        meta: Optional[dict[str, int]] = None) -> None:
    """Write a tinymind-format weights.hpp from parsed ONNX layers.

    Per layer `k<name>_...`:
      InputScale / InputZeroPoint / OutputScale / OutputZeroPoint
      Conv geometry (InputHeight ... OutputWidth) or NumInputs / NumOutputs
      PerChannel, WeightScale(s), Multiplier(s) / Shift(s) for the
      Requantizer, int8 Weights (OHWI or [out][in]), int32 Biases + HasBias.
    The per-tensor k<tensor>_Scale / k<tensor>_ZeroPoint block is kept for
    callers of the earlier header layout.
    """
    parts: list[str] = []
    parts.append(COPYRIGHT_BLOCK)
    parts.append("\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n")
    parts.append(f"namespace {namespace} {{\n")
    if meta is not None:
        for k, v in meta.items():
            parts.append(f"    constexpr std::size_t {k} = {v};\n")
        parts.append("\n")

    parts.append("    // Activation calibration per tensor.\n")
    seen: set[str] = set()
    for layer in layers:
        for tensor, scale, zp in ((layer.input_name, layer.input_scale, layer.input_zp),
                                  (layer.output_name or layer.name, layer.output_scale, layer.output_zp)):
            nm = sanitize(tensor)
            if nm in seen:
                continue
            seen.add(nm)
            parts.append(f"    constexpr float   k{nm}_Scale     = {scale!r}f;\n")
            parts.append(f"    constexpr int32_t k{nm}_ZeroPoint = {zp};\n")

    for layer in layers:
        nm = sanitize(layer.name)
        parts.append(f"\n    // {layer.op_type} {nm}\n")
        parts.append(f"    constexpr float   k{nm}_InputScale      = {layer.input_scale!r}f;\n")
        parts.append(f"    constexpr int32_t k{nm}_InputZeroPoint  = {layer.input_zp};\n")
        parts.append(f"    constexpr float   k{nm}_OutputScale     = {layer.output_scale!r}f;\n")
        parts.append(f"    constexpr int32_t k{nm}_OutputZeroPoint = {layer.output_zp};\n")
        if layer.weight is None:
            continue
        for key, value in layer.geometry.items():
            parts.append(f"    constexpr std::size_t k{nm}_{key} = {value};\n")
        parts.append(f"    constexpr bool k{nm}_PerChannel = {'true' if layer.per_channel else 'false'};\n")
        if layer.per_channel:
            parts.append(_fmt_float_array(f"k{nm}_WeightScales", layer.weight_scales))
            parts.append(_fmt_int_array(f"k{nm}_Multipliers", layer.multipliers, "int32_t", per_line=6))
            parts.append(_fmt_int_array(f"k{nm}_Shifts", layer.shifts, "int32_t"))
        else:
            parts.append(f"    constexpr float   k{nm}_WeightScale = {layer.weight_scale!r}f;\n")
            parts.append(f"    constexpr int32_t k{nm}_Multiplier  = {layer.multipliers[0]};\n")
            parts.append(f"    constexpr int32_t k{nm}_Shift       = {layer.shifts[0]};\n")
        parts.append(_fmt_int_array(f"k{nm}_Weights", layer.weight.reshape(-1).tolist(), "int8_t"))
        parts.append(f"    constexpr bool k{nm}_HasBias = {'true' if layer.bias is not None else 'false'};\n")
        if layer.bias is not None:
            parts.append(_fmt_int_array(f"k{nm}_Biases", layer.bias.tolist(), "int32_t"))
    parts.append(f"}} // namespace {namespace}\n")
    with open(path, "w") as f:
        f.write("".join(parts))


def import_onnx_model(model_path: str,
                      output_path: str,
                      namespace: str,
                      meta: Optional[dict[str, int]] = None) -> list[OnnxQuantizedLayer]:
    """Top-level: parse the ONNX model, write the tinymind weights.hpp."""
    layers = parse_onnx_model(model_path)
    if not layers:
        raise ValueError(f"{model_path}: no quantized layers found")
    emit_weights_header(output_path, namespace, layers, meta)
    return layers
