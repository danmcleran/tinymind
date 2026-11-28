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

TensorFlow Lite (.tflite) importer: an int8 model, as TensorFlow's
TFLiteConverter writes it with full-integer quantization, becomes the same
layer records the ONNX importer produces, so tinymind_model_codegen writes its
weights.hpp and model.hpp (constexpr layers, a static arena plan and
forward()).

The generated model computes what TensorFlow Lite for Microcontrollers
computes, bit for bit: every constant is derived the way TFLM's kernels derive
it (requantizer multipliers from the float scales in double, the fused
activation clamps, MEAN's 1/area multiplier, the fixed-point LOGISTIC and TANH
evaluated into 256-entry tables), and the tests compare the generated program
with TFLM's own interpreter on every op below.

Reading the model needs only numpy (tflite_flatbuffer.py decodes the
flatbuffer); TensorFlow is needed only to produce models and to run the tests.

Supported operators (int8 activations, int8 weights with zero point 0, int32
biases, batch 1):

    CONV_2D             -> QConv2D / QConv2DPerChannel (SAME or VALID padding,
                           strides, fused NONE / RELU / RELU6 / RELU_N1_TO_1)
    DEPTHWISE_CONV_2D   -> QDepthwiseConv2D (depth multiplier 1)
    FULLY_CONNECTED     -> QDense, or QConv2DPerChannel 1x1 for per-channel
                           weights; the input is flattened in NHWC order, as
                           TFLite flattens it, so no column permutation
    ADD                 -> QAdd (same shapes, no broadcasting)
    MAX_POOL_2D         -> QMaxPool2D (no padding)
    AVERAGE_POOL_2D     -> QAvgPool2D / QGlobalAvgPool2D (no padding)
    MEAN over H and W   -> TFLM's integer mean (a window sum and one
                           requantizer), what Keras' GlobalAveragePooling2D
                           converts to
    SOFTMAX             -> QSoftmaxFixedPoint1D (the model's output only)
    LOGISTIC / TANH     -> 256-entry tables holding TFLM's fixed-point results
    RELU                -> qrelu, when input and output share a grid
    RESHAPE / SQUEEZE / EXPAND_DIMS -> no layer; the output aliases the input
    QUANTIZE / DEQUANTIZE at the model's input / output
                        -> dropped: a float model boundary becomes the int8
                           tensor behind it, and model.hpp's kModelInputScale /
                           kModelInputZeroPoint say how to quantize the input
    SHAPE / STRIDED_SLICE / PACK feeding a RESHAPE's shape operand
                        -> ignored (Keras' Flatten computes its shape at run
                           time; the reshape is a no-op on a batch of one)

Anything else raises ValueError naming the operator: int16 activations
(TFLite's 16x8 models), uint8 tensors, dilated convolutions, depth
multipliers above 1, padded pooling, broadcasting ADD, a softmax that is not
the model's output, ops on more than one subgraph, and every operator not
listed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "import_onnx"))

import tflite_flatbuffer as fb  # noqa: E402
import tinymind_import_onnx as ti  # noqa: E402
import tinymind_model_codegen as cg  # noqa: E402

GENERATOR = "tinymind_import_tflite"

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


# ----------------------------------------------------------------------------
# TFLite (TFLM) integer arithmetic, mirrored exactly.
# ----------------------------------------------------------------------------


def tflite_round(x: float) -> int:
    """TfLiteRound / std::round: ties away from zero."""
    return ti.lround(x)


def quantize_multiplier_tflite(real: float) -> tuple[int, int]:
    """TFLM QuantizeMultiplier: real = q * 2^(shift - 31), shift > 0 = left.
    Returns (q, shift) with TFLite's sign convention."""
    if real == 0.0:
        return 0, 0
    frac, shift = math.frexp(real)
    q = tflite_round(frac * float(1 << 31))
    if q == (1 << 31):
        q //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if shift > 30:
        return INT32_MAX, 30
    return int(q), int(shift)


def _wrap32(x: int) -> int:
    return ((x + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def _srdhm(a: int, b: int) -> int:
    """SaturatingRoundingDoublingHighMul (int32)."""
    if a == b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    v = ab + nudge
    q = abs(v) >> 31
    return q if v >= 0 else -q


def _rdbpot(x: int, exponent: int) -> int:
    """RoundingDivideByPOT (int32)."""
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def _srmbpot(x: int, exponent: int) -> int:
    """SaturatingRoundingMultiplyByPOT<exponent> (int32)."""
    if exponent == 0:
        return x
    if exponent < 0:
        return _rdbpot(x, -exponent)
    threshold = (1 << (31 - exponent)) - 1
    if x > threshold:
        return INT32_MAX
    if x < -threshold:
        return INT32_MIN
    return _wrap32(x << exponent)


def multiply_by_quantized_multiplier(x: int, multiplier: int, shift: int) -> int:
    """TFLM MultiplyByQuantizedMultiplier (int32 x, TFLite shift sign)."""
    left = shift if shift > 0 else 0
    right = -shift if shift < 0 else 0
    return _rdbpot(_srdhm(_wrap32(x * (1 << left)), multiplier), right)


# gemmlowp-style fixed point (tensorflow/lite/micro/kernels/internal/
# fixedpoint.h) on raw int32 values; a value with I integer bits has 31 - I
# fractional bits. Products of an I-bit and a J-bit value have I + J bits.

def _fp_mul(a: int, b: int) -> int:
    return _srdhm(a, b)


def _fp_one(integer_bits: int) -> int:
    return INT32_MAX if integer_bits == 0 else 1 << (31 - integer_bits)


def _rounding_half_sum(a: int, b: int) -> int:
    s = a + b
    sign = 1 if s >= 0 else -1
    v = s + sign
    q = abs(v) // 2
    return q if v >= 0 else -q


def _exp_on_interval_between_negative_one_quarter_and_0_excl(a: int) -> int:
    constant_term = 1895147668        # exp(-1/8), 0 integer bits
    constant_1_over_3 = 715827883
    x = _wrap32(a + (1 << 28))        # a + 1/8
    x2 = _fp_mul(x, x)
    x3 = _fp_mul(x2, x)
    x4 = _fp_mul(x2, x2)
    x4_over_4 = _srmbpot(x4, -2)
    poly = _srmbpot(_wrap32(_fp_mul(_wrap32(x4_over_4 + x3), constant_1_over_3) + x2), -1)
    return _wrap32(constant_term + _fp_mul(constant_term, _wrap32(x + poly)))


_EXP_BARREL = ((-2, 1672461947), (-1, 1302514674), (0, 790015084), (1, 290630308),
               (2, 39332535), (3, 720401), (4, 242))


def _exp_on_negative_values(a: int, integer_bits: int) -> int:
    """exp(a) for a <= 0 with `integer_bits` integer bits; result 0 bits."""
    frac_bits = 31 - integer_bits
    one_quarter = 1 << (frac_bits - 2)
    mask = one_quarter - 1
    a_mod = _wrap32((a & mask) - one_quarter)
    result = _exp_on_interval_between_negative_one_quarter_and_0_excl(
        _srmbpot(a_mod, integer_bits))   # Rescale<0>
    remainder = _wrap32(a_mod - a)
    for exponent, multiplier in _EXP_BARREL:
        if integer_bits > exponent:
            shift = frac_bits + exponent
            if remainder & (1 << shift):
                result = _fp_mul(result, multiplier)
    if integer_bits > 5:
        if a < -(1 << (36 - integer_bits)):
            result = 0
    if a == 0:
        result = INT32_MAX
    return result


def _newton_reciprocal_loop(half_denominator: int) -> int:
    """The three Newton-Raphson steps shared by one_over_one_plus_x and
    one_minus_x_over_one_plus_x; returns x with 2 integer bits."""
    c48_17 = 1515870810
    c_neg32_17 = -1010580540
    x = _wrap32(c48_17 + _fp_mul(half_denominator, c_neg32_17))
    one_f2 = _fp_one(2)
    for _ in range(3):
        half_denominator_times_x = _fp_mul(half_denominator, x)
        one_minus = _wrap32(one_f2 - half_denominator_times_x)
        x = _wrap32(x + _srmbpot(_fp_mul(x, one_minus), 2))   # Rescale<2> from 4 bits
    return x


def _one_over_one_plus_x_for_x_in_0_1(a: int) -> int:
    x = _newton_reciprocal_loop(_rounding_half_sum(a, INT32_MAX))
    return _srmbpot(x, 1)                 # Rescale<0>(ExactMulByPot<-1>(x)): 1 bit -> 0


def _one_minus_x_over_one_plus_x_for_x_in_0_1(a: int) -> int:
    x = _newton_reciprocal_loop(_rounding_half_sum(a, INT32_MAX))
    return _srmbpot(_wrap32(x - _fp_one(2)), 2)


def fixed_point_logistic(a: int, integer_bits: int) -> int:
    """logistic(a) for a raw value with `integer_bits` integer bits; raw
    result with 0 integer bits."""
    if a == 0:
        return 1 << 30
    abs_input = a if a > 0 else _wrap32(-a)
    positive = _one_over_one_plus_x_for_x_in_0_1(_exp_on_negative_values(_wrap32(-abs_input), integer_bits))
    return positive if a > 0 else _wrap32(INT32_MAX - positive)


def fixed_point_tanh(a: int, integer_bits: int) -> int:
    """tanh(a), raw in and out as fixed_point_logistic."""
    if a == 0:
        return 0
    n = a if a < 0 else _wrap32(-a)
    # neg_tanh_on_negative_values: ExactMulByPot<1> keeps the raw value and
    # adds one integer bit.
    t = _one_minus_x_over_one_plus_x_for_x_in_0_1(_exp_on_negative_values(n, integer_bits + 1))
    return _wrap32(-t) if a < 0 else t


def _input_params_q4(input_scale: float) -> tuple[int, int, int]:
    """TFLM's int8 LOGISTIC / TANH input parameters: (multiplier, left
    shift, input range radius) for 4 integer bits."""
    real = float(np.float32(input_scale)) * float(1 << (31 - 4))
    frac, left = math.frexp(real)
    multiplier = tflite_round(frac * float(1 << 31))
    max_rescaled = 1.0 * ((1 << 4) - 1) * float(1 << (31 - 4)) / float(1 << left) if left >= 0 \
        else 1.0 * ((1 << 4) - 1) * float(1 << (31 - 4)) * float(1 << -left)
    return multiplier, left, int(math.floor(max_rescaled))


def logistic_lut(input_scale: float, input_zero_point: int) -> list:
    """TFLM's int8 LOGISTIC (output scale 1/256, zero point -128) for every
    int8 input; entry k is the output for input k - 128."""
    multiplier, left, radius = _input_params_q4(input_scale)
    lut = []
    for q in range(-128, 128):
        x = q - input_zero_point
        if x <= -radius:
            lut.append(-128)
        elif x >= radius:
            lut.append(127)
        else:
            q4 = multiply_by_quantized_multiplier(x, multiplier, left)
            out = _rdbpot(fixed_point_logistic(q4, 4), 31 - 8) - 128
            lut.append(max(-128, min(127, out)))
    return lut


def tanh_lut(input_scale: float, input_zero_point: int) -> list:
    """TFLM's int8 TANH (output scale 1/128, zero point 0), as logistic_lut."""
    multiplier, left, radius = _input_params_q4(input_scale)
    lut = []
    for q in range(-128, 128):
        x = q - input_zero_point
        if x <= -radius:
            lut.append(-128)
        elif x >= radius:
            lut.append(127)
        else:
            q4 = multiply_by_quantized_multiplier(x, multiplier, left)
            out = _rdbpot(fixed_point_tanh(q4, 4), 31 - 7)
            lut.append(max(-128, min(127, out)))
    return lut


def activation_range(act: int, scale: float, zero_point: int, label: str) -> tuple[int, int]:
    """TFLite CalculateActivationRangeQuantized for an int8 output."""
    qmin, qmax = -128, 127

    def q(v: float) -> int:
        return zero_point + tflite_round(v / float(np.float32(scale)))

    if act == fb.ACT_NONE:
        return qmin, qmax
    if act == fb.ACT_RELU:
        return max(qmin, q(0.0)), qmax
    if act == fb.ACT_RELU6:
        return max(qmin, q(0.0)), min(qmax, q(6.0))
    if act == fb.ACT_RELU_N1_TO_1:
        return max(qmin, q(-1.0)), min(qmax, q(1.0))
    raise ValueError(f"{label}: fused activation {fb.ACT_NAMES.get(act, act)} is not supported")


def mean_requantizer(input_scale: float, output_scale: float, count: int) -> tuple[int, int]:
    """TFLM's int8 MEAN: the input / output multiplier with 1 / count folded
    in (reference_ops::QuantizedMeanOrSum). Returns (multiplier, shift) in
    tinymind's sign (shift > 0 shifts right)."""
    multiplier, shift = quantize_multiplier_tflite(float(np.float32(input_scale)) / float(np.float32(output_scale)))
    k = count.bit_length() - 1
    k = min(k, 32)
    k = min(k, 31 + shift)
    multiplier = (multiplier << k) // count
    return int(multiplier), -(shift - k)


# ----------------------------------------------------------------------------
# Model -> layer records.
# ----------------------------------------------------------------------------

_SHAPE_OPS = ("SHAPE", "STRIDED_SLICE", "PACK", "CONCATENATION")
_NAME_STEM = {"CONV_2D": "conv", "DEPTHWISE_CONV_2D": "dwconv", "FULLY_CONNECTED": "fc", "ADD": "add",
              "MAX_POOL_2D": "maxpool", "AVERAGE_POOL_2D": "avgpool", "MEAN": "mean", "SOFTMAX": "softmax",
              "LOGISTIC": "sigmoid", "TANH": "tanh", "RELU": "relu"}


def _same_pads(in_size: int, kernel: int, stride: int) -> tuple[int, int, int]:
    out = (in_size + stride - 1) // stride
    total = max((out - 1) * stride + kernel - in_size, 0)
    return total // 2, total - total // 2, out


def parse_tflite_model(model) -> list:
    """Parse a .tflite model (path, bytes or a decoded TfliteModel) into
    tinymind layer records (tinymind_import_onnx.OnnxQuantizedLayer)."""
    m = model if isinstance(model, fb.TfliteModel) else fb.read_tflite(model)
    if m.subgraph_count != 1:
        raise ValueError(f"the model has {m.subgraph_count} subgraphs; only single-subgraph models are supported")
    if len(m.inputs) != 1 or len(m.outputs) != 1:
        raise ValueError(f"the model has {len(m.inputs)} inputs and {len(m.outputs)} outputs; "
                         "only one of each is supported")
    T = m.tensors
    ops = m.operators

    consumers: dict[int, list] = {}
    for op in ops:
        for t in op.inputs:
            if t >= 0:
                consumers.setdefault(t, []).append(op)

    model_in, model_out = m.inputs[0], m.outputs[0]
    skip: set[int] = set()

    # A float boundary: QUANTIZE right after the input, DEQUANTIZE right
    # before the output. The int8 tensor behind each one is the boundary.
    if T[model_in].type == fb.FLOAT32:
        users = consumers.get(model_in, [])
        if len(users) != 1 or users[0].op != "QUANTIZE" or T[users[0].outputs[0]].type != fb.INT8:
            raise ValueError("the model's input is float and is not quantized to int8 by a single QUANTIZE")
        skip.add(users[0].index)
        model_in = users[0].outputs[0]
    if T[model_out].type == fb.FLOAT32:
        producer = [op for op in ops if model_out in op.outputs]
        if len(producer) != 1 or producer[0].op != "DEQUANTIZE":
            raise ValueError("the model's output is float and is not produced by a DEQUANTIZE")
        skip.add(producer[0].index)
        model_out = producer[0].inputs[0]

    # Run-time shape computations (Keras' Flatten): int32 tensors derived
    # from SHAPE that feed only a RESHAPE's shape operand.
    shape_tensors: set[int] = set()
    for op in ops:
        if op.index in skip:
            continue
        if op.op == "SHAPE" or (op.op in _SHAPE_OPS and all(
                t < 0 or t in shape_tensors or T[t].data is not None for t in op.inputs)
                and all(T[t].type == fb.INT32 for t in op.outputs)):
            if op.op == "SHAPE" or any(t in shape_tensors for t in op.inputs):
                shape_tensors.update(op.outputs)
                skip.add(op.index)

    layers: list = []
    origin: dict[int, int] = {model_in: -1}   # tensor -> producing record (-1: model input)

    def qparams(t: int, label: str) -> tuple[float, int]:
        tensor = T[t]
        if tensor.type == fb.INT16:
            raise ValueError(f"{label}: int16 activations (TFLite's 16x8 models) are not supported yet")
        if tensor.type != fb.INT8:
            raise ValueError(f"{label}: tensor {tensor.name!r} is {tensor.type_name}; only int8 activations "
                             "are supported (convert with full-integer int8 quantization)")
        if len(tensor.scales) != 1 or len(tensor.zero_points) != 1:
            raise ValueError(f"{label}: tensor {tensor.name!r} needs one scale and zero point, "
                             f"has {len(tensor.scales)}")
        return float(tensor.scales[0]), int(tensor.zero_points[0])

    def source(t: int, label: str) -> int:
        if t not in origin:
            if T[t].data is not None:
                raise ValueError(f"{label}: a constant activation operand ({T[t].name!r}) is not supported")
            raise ValueError(f"{label}: reads tensor {T[t].name!r}, which no supported layer produced")
        return origin[t]

    def nhwc(t: int, label: str) -> tuple[int, int, int]:
        shape = T[t].shape
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"{label}: expected a [1, H, W, C] tensor, {T[t].name!r} has shape {shape}")
        return shape[1], shape[2], shape[3]

    def weights(t: int, label: str):
        tensor = T[t]
        if tensor.data is None:
            raise ValueError(f"{label}: weights {tensor.name!r} are not a constant")
        if tensor.type != fb.INT8:
            raise ValueError(f"{label}: weights {tensor.name!r} are {tensor.type_name}, not int8")
        if any(z != 0 for z in tensor.zero_points):
            raise ValueError(f"{label}: weights {tensor.name!r} have a nonzero zero point; tinymind weights "
                             "are symmetric")
        if not tensor.scales:
            raise ValueError(f"{label}: weights {tensor.name!r} are not quantized")
        return tensor.data, [float(s) for s in tensor.scales], tensor.quantized_dimension

    def bias(op, label: str, count: int):
        if len(op.inputs) < 3 or op.inputs[2] < 0:
            return None
        tensor = T[op.inputs[2]]
        if tensor.data is None or tensor.type != fb.INT32:
            raise ValueError(f"{label}: bias {tensor.name!r} is not an int32 constant")
        b = tensor.data.reshape(-1).astype(np.int32)
        if b.size != count:
            raise ValueError(f"{label}: bias has {b.size} values for {count} outputs")
        return b

    def requantizers(in_scale: float, w_scales: list, out_scale: float) -> tuple[list, list]:
        mults, shifts = [], []
        for s in w_scales:
            q, sh = quantize_multiplier_tflite(in_scale * s / out_scale)
            mults.append(q)
            shifts.append(-sh)
        return mults, shifts

    def add_record(op, rec) -> None:
        origin[op.outputs[0]] = len(layers)
        layers.append(rec)

    for op in ops:
        if op.index in skip:
            continue
        name = f"{_NAME_STEM.get(op.op, op.op.lower())}{op.index}"
        label = f"operator {op.index} ({op.op})"
        o = op.options

        if op.op in ("RESHAPE", "SQUEEZE", "EXPAND_DIMS"):
            src = op.inputs[0]
            if src not in origin:
                raise ValueError(f"{label}: reshapes a tensor no supported layer produced")
            if T[op.outputs[0]].type != T[src].type:
                raise ValueError(f"{label}: changes the tensor type")
            if len(op.inputs) > 1 and op.inputs[1] >= 0:
                st = op.inputs[1]
                if st not in shape_tensors and T[st].data is None:
                    raise ValueError(f"{label}: its shape operand is computed by an unsupported op")
            in_q, out_q = qparams(src, label), qparams(op.outputs[0], label)
            if in_q != out_q:
                raise ValueError(f"{label}: requantizes ({in_q} -> {out_q})")
            origin[op.outputs[0]] = origin[src]
            continue

        if op.op in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            x, wt, y = op.inputs[0], op.inputs[1], op.outputs[0]
            depthwise = op.op == "DEPTHWISE_CONV_2D"
            if o["dilation_w_factor"] != 1 or o["dilation_h_factor"] != 1:
                raise ValueError(f"{label}: dilated convolutions are not supported")
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            h, w, c = nhwc(x, label)
            oh_t, ow_t, f = nhwc(y, label)
            wq, ws, qdim = weights(wt, label, )
            if depthwise:
                if wq.shape[0] != 1 or wq.shape[3] != c:
                    raise ValueError(f"{label}: depth multiplier {wq.shape[3] // max(c, 1)} is not supported; "
                                     "only 1")
                kh, kw = wq.shape[1], wq.shape[2]
                if len(ws) > 1 and qdim != 3:
                    raise ValueError(f"{label}: per-channel scales on axis {qdim}, expected 3")
                tm_w = np.ascontiguousarray(np.transpose(wq[0], (2, 0, 1)))
                n_out = c
            else:
                f_w, kh, kw, wc = wq.shape
                if wc != c or f_w != f:
                    raise ValueError(f"{label}: weights {list(wq.shape)} do not match input channels {c} "
                                     f"and output channels {f}")
                if len(ws) > 1 and qdim != 0:
                    raise ValueError(f"{label}: per-channel scales on axis {qdim}, expected 0")
                tm_w = np.ascontiguousarray(wq)
                n_out = f
            if len(ws) not in (1, n_out):
                raise ValueError(f"{label}: {len(ws)} weight scales for {n_out} output channels")
            sh, sw = o["stride_h"], o["stride_w"]
            if o["padding"] == fb.PADDING_SAME:
                pt, pb, oh = _same_pads(h, kh, sh)
                pl, pr, ow = _same_pads(w, kw, sw)
            else:
                pt = pb = pl = pr = 0
                oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
            if (oh, ow) != (oh_t, ow_t):
                raise ValueError(f"{label}: computed output {oh}x{ow}, the model says {oh_t}x{ow_t}")
            act_min, act_max = activation_range(o["fused_activation_function"], s_out, z_out, label)
            mults, shifts = requantizers(s_in, ws, s_out)
            geometry = {"InputHeight": h, "InputWidth": w, "InputChannels": c, "KernelHeight": kh,
                        "KernelWidth": kw, "StrideH": sh, "StrideW": sw, "NumFilters": n_out,
                        "PadTop": pt, "PadLeft": pl, "PadBottom": pb, "PadRight": pr,
                        "OutputHeight": oh, "OutputWidth": ow}
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="DepthwiseConv" if depthwise else "Conv", name=name, input_name=T[x].name,
                input_scale=s_in, input_zp=z_in, output_scale=s_out, output_zp=z_out, output_name=T[y].name,
                weight=tm_w.astype(np.int8), weight_scale=ws[0], weight_scales=ws, per_channel=len(ws) > 1,
                bias=bias(op, label, n_out), bias_scale=s_in * ws[0], multipliers=mults, shifts=shifts,
                geometry=geometry, sources=[source(x, label)], act_min=act_min, act_max=act_max))
            continue

        if op.op == "FULLY_CONNECTED":
            x, wt, y = op.inputs[0], op.inputs[1], op.outputs[0]
            if o["weights_format"] != 0:
                raise ValueError(f"{label}: shuffled weights are not supported")
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            wq, ws, qdim = weights(wt, label)
            if wq.ndim != 2:
                raise ValueError(f"{label}: weights have shape {list(wq.shape)}, expected [out, in]")
            n_out, n_in = wq.shape
            if int(np.prod(T[x].shape)) != n_in:
                raise ValueError(f"{label}: input {T[x].shape} is not one row of {n_in} values (batch 1)")
            if len(ws) not in (1, n_out) or (len(ws) > 1 and qdim != 0):
                raise ValueError(f"{label}: {len(ws)} weight scales on axis {qdim} for {n_out} outputs")
            act_min, act_max = activation_range(o["fused_activation_function"], s_out, z_out, label)
            mults, shifts = requantizers(s_in, ws, s_out)
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="Dense", name=name, input_name=T[x].name, input_scale=s_in, input_zp=z_in,
                output_scale=s_out, output_zp=z_out, output_name=T[y].name,
                weight=np.ascontiguousarray(wq).astype(np.int8), weight_scale=ws[0], weight_scales=ws,
                per_channel=len(ws) > 1, bias=bias(op, label, n_out), bias_scale=s_in * ws[0],
                multipliers=mults, shifts=shifts, geometry={"NumInputs": n_in, "NumOutputs": n_out},
                sources=[source(x, label)], act_min=act_min, act_max=act_max))
            continue

        if op.op == "ADD":
            a, b, y = op.inputs[0], op.inputs[1], op.outputs[0]
            if not (T[a].shape == T[b].shape == T[y].shape):
                raise ValueError(f"{label}: broadcasting ({T[a].shape} + {T[b].shape}) is not supported")
            sa, za = qparams(a, label)
            sb, zb = qparams(b, label)
            so, zo = qparams(y, label)
            act_min, act_max = activation_range(o["fused_activation_function"], so, zo, label)
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="Add", name=name, input_name=T[a].name, input_scale=sa, input_zp=za,
                output_scale=so, output_zp=zo, output_name=T[y].name, input_b_name=T[b].name,
                input_b_scale=sb, input_b_zp=zb, sources=[source(a, label), source(b, label)],
                act_min=act_min, act_max=act_max))
            continue

        if op.op in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            x, y = op.inputs[0], op.outputs[0]
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            if np.float32(s_in) != np.float32(s_out) or z_in != z_out:
                raise ValueError(f"{label}: pooling that changes the grid is not supported")
            h, w, c = nhwc(x, label)
            oh_t, ow_t, _ = nhwc(y, label)
            kh, kw, sh, sw = o["filter_height"], o["filter_width"], o["stride_h"], o["stride_w"]
            if o["padding"] == fb.PADDING_SAME:
                pads = _same_pads(h, kh, sh)[:2] + _same_pads(w, kw, sw)[:2]
                if any(pads):
                    raise ValueError(f"{label}: padded pooling is not supported")
            oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
            if (oh, ow) != (oh_t, ow_t):
                raise ValueError(f"{label}: computed output {oh}x{ow}, the model says {oh_t}x{ow_t}")
            act_min, act_max = activation_range(o["fused_activation_function"], s_out, z_out, label)
            geometry = {"InputHeight": h, "InputWidth": w, "InputChannels": c, "KernelHeight": kh,
                        "KernelWidth": kw, "StrideH": sh, "StrideW": sw, "OutputHeight": oh, "OutputWidth": ow}
            if op.op == "MAX_POOL_2D" and (act_min, act_max) != (-128, 127):
                # Max pooling commutes with the clamp; keep it exact by refusing
                # what QMaxPool2D cannot clamp.
                raise ValueError(f"{label}: a fused activation that clamps a max pool is not supported")
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="MaxPool" if op.op == "MAX_POOL_2D" else "AvgPool2D", name=name, input_name=T[x].name,
                input_scale=s_in, input_zp=z_in, output_scale=s_out, output_zp=z_out, output_name=T[y].name,
                geometry=geometry, sources=[source(x, label)], act_min=act_min, act_max=act_max))
            continue

        if op.op == "MEAN":
            x, ax, y = op.inputs[0], op.inputs[1], op.outputs[0]
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            h, w, c = nhwc(x, label)
            if T[ax].data is None:
                raise ValueError(f"{label}: the axes are not a constant")
            axes = sorted(int(v) % 4 for v in T[ax].data.reshape(-1))
            if axes != [1, 2]:
                raise ValueError(f"{label}: MEAN over axes {axes} is not supported; only over H and W (1, 2)")
            mult, shift = mean_requantizer(s_in, s_out, h * w)
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="Mean", name=name, input_name=T[x].name, input_scale=s_in, input_zp=z_in,
                output_scale=s_out, output_zp=z_out, output_name=T[y].name,
                multipliers=[mult], shifts=[shift],
                geometry={"InputHeight": h, "InputWidth": w, "InputChannels": c},
                sources=[source(x, label)]))
            continue

        if op.op == "SOFTMAX":
            x, y = op.inputs[0], op.outputs[0]
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            if np.float32(s_out) != np.float32(1.0 / 256) or z_out != -128:
                raise ValueError(f"{label}: an int8 softmax output must have scale 1/256 and zero point -128")
            shape = T[x].shape
            if int(np.prod(shape[:-1])) != 1:
                raise ValueError(f"{label}: softmax over more than one row ({shape}) is not supported")
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="Softmax", name=name, input_name=T[x].name, input_scale=s_in, input_zp=z_in,
                output_scale=s_out, output_zp=z_out, output_name=T[y].name, sources=[source(x, label)],
                beta=float(o["beta"])))
            continue

        if op.op in ("LOGISTIC", "TANH"):
            x, y = op.inputs[0], op.outputs[0]
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            if op.op == "LOGISTIC":
                if np.float32(s_out) != np.float32(1.0 / 256) or z_out != -128:
                    raise ValueError(f"{label}: an int8 LOGISTIC output must have scale 1/256 and zero point -128")
                lut = logistic_lut(s_in, z_in)
            else:
                if np.float32(s_out) != np.float32(1.0 / 128) or z_out != 0:
                    raise ValueError(f"{label}: an int8 TANH output must have scale 1/128 and zero point 0")
                lut = tanh_lut(s_in, z_in)
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="Sigmoid" if op.op == "LOGISTIC" else "Tanh", name=name, input_name=T[x].name,
                input_scale=s_in, input_zp=z_in, output_scale=s_out, output_zp=z_out, output_name=T[y].name,
                sources=[source(x, label)], lut=lut))
            continue

        if op.op == "RELU":
            x, y = op.inputs[0], op.outputs[0]
            s_in, z_in = qparams(x, label)
            s_out, z_out = qparams(y, label)
            if np.float32(s_in) != np.float32(s_out) or z_in != z_out:
                raise ValueError(f"{label}: a RELU that changes the grid is not supported")
            add_record(op, ti.OnnxQuantizedLayer(
                op_type="Relu", name=name, input_name=T[x].name, input_scale=s_in, input_zp=z_in,
                output_scale=s_out, output_zp=z_out, output_name=T[y].name, sources=[source(x, label)]))
            continue

        if op.op in ("QUANTIZE", "DEQUANTIZE"):
            raise ValueError(f"{label}: a {op.op} inside the graph (a change of grid between layers) is not "
                             "supported; only at a float input or output")
        raise ValueError(f"{label}: unsupported operator {op.op}")

    if not layers:
        raise ValueError("the model has no supported layers")
    if model_out not in origin or origin[model_out] != len(layers) - 1:
        raise ValueError("the model's output is not written by its last layer")
    return layers


def model_io(model) -> dict:
    """The tensors at the model's boundary: whether the input / output are
    float (behind QUANTIZE / DEQUANTIZE) and their int8 grids."""
    m = model if isinstance(model, fb.TfliteModel) else fb.read_tflite(model)
    return {"float_input": m.tensors[m.inputs[0]].type == fb.FLOAT32,
            "float_output": m.tensors[m.outputs[0]].type == fb.FLOAT32}


def import_tflite_model(model_path: str, weights_path: str, namespace: str,
                        meta: Optional[dict] = None) -> list:
    """Parse a .tflite model and write its weights.hpp. Returns the records."""
    layers = parse_tflite_model(model_path)
    ti.emit_weights_header(weights_path, namespace, layers, meta, generator=GENERATOR)
    return layers


def import_tflite_model_with_code(model_path: str, weights_path: str, model_header_path: str,
                                  namespace: str, alignment: int = 4, in_place: bool = True,
                                  int4_dense: bool = False, int4_conv: bool = False,
                                  nested_namespace: Optional[str] = None, streaming: bool = False):
    """Parse a .tflite model and write weights.hpp and model.hpp. Returns
    (layers, GeneratedModel). int4_dense / int4_conv requantize those layers'
    weights to packed int4 as for an ONNX model (the result then no longer
    matches TFLM bit for bit). streaming: model.hpp runs the model one input
    frame per step() call (see the code generator's plan_stream)."""
    layers = import_tflite_model(model_path, weights_path, namespace)
    generated = cg.emit_model_header(model_header_path, namespace, layers, os.path.basename(weights_path),
                                     alignment=alignment, in_place=in_place, int4_dense=int4_dense,
                                     int4_conv=int4_conv, nested_namespace=nested_namespace,
                                     generator=GENERATOR, streaming=streaming)
    return layers, generated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import an int8 .tflite model into tinymind: write weights.hpp "
                                             "and model.hpp (layers, arena plan, forward()).")
    ap.add_argument("model", help="the .tflite file")
    ap.add_argument("--namespace", required=True, help="C++ namespace for the generated model")
    ap.add_argument("--out-dir", default=".", help="where to write the headers (default: .)")
    ap.add_argument("--weights", default="weights.hpp", help="weights header file name")
    ap.add_argument("--model-header", default="model.hpp", help="model header file name")
    ap.add_argument("--alignment", type=int, default=4, help="arena alignment in bytes")
    ap.add_argument("--int4-dense", action="store_true", help="requantize dense layers to packed int4")
    ap.add_argument("--int4-conv", action="store_true", help="requantize convolutions to packed int4")
    ap.add_argument("--streaming", action="store_true",
                    help="write step(), which takes one input frame per call, instead of forward()")
    ap.add_argument("--nested-namespace", default=None,
                    help="put the model in NAMESPACE::NESTED (to share one weights header between two models)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    layers, gen = import_tflite_model_with_code(
        args.model, os.path.join(args.out_dir, args.weights), os.path.join(args.out_dir, args.model_header),
        args.namespace, alignment=args.alignment, int4_dense=args.int4_dense, int4_conv=args.int4_conv,
        nested_namespace=args.nested_namespace, streaming=args.streaming)
    print(f"{len(layers)} layers: {[l.op_type for l in layers]}")
    print(f"input {gen.input_size} values, output {gen.output_size}, arena {gen.plan.arena_size} B "
          f"(unshared {gen.plan.unshared_size} B), weights {gen.weight_bytes} B")
    if gen.stream is not None:
        print(f"streaming: one {gen.input_size}-value frame per step(), window {gen.stream.frames[-1]} frames, "
              f"state {gen.stream.state_size} B, output valid from frame {gen.stream.receptive_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
