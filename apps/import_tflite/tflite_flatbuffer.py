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

A reader for the part of the TensorFlow Lite flatbuffer schema
(tensorflow/compiler/mlir/lite/schema/schema.fbs) that an int8 model needs:
the model's subgraph, its tensors with their quantization and constant data,
its operators with their builtin options, and the operator codes.

It needs only the Python standard library and numpy, so importing a .tflite
model does not need TensorFlow. Field numbers below are the schema's field
ids; a field is found through its table's vtable, so a model written by any
schema version that keeps these ids reads the same. The tests check every
value read here against TensorFlow's own generated schema reader.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Schema constants (the subset this importer uses).
# ----------------------------------------------------------------------------

# BuiltinOperator (every code in TensorFlow 2.21's schema, so an error names the op)
BUILTIN_OPS = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE", 6: "DEQUANTIZE", 7: "EMBEDDING_LOOKUP", 8: "FLOOR", 9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP", 11: "L2_NORMALIZATION", 12: "L2_POOL_2D", 13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC", 15: "LSH_PROJECTION", 16: "LSTM", 17: "MAX_POOL_2D", 18: "MUL", 19: "RELU",
    20: "RELU_N1_TO_1", 21: "RELU6", 22: "RESHAPE", 23: "RESIZE_BILINEAR", 24: "RNN", 25: "SOFTMAX",
    26: "SPACE_TO_DEPTH", 27: "SVDF", 28: "TANH", 29: "CONCAT_EMBEDDINGS", 30: "SKIP_GRAM", 31: "CALL",
    32: "CUSTOM", 33: "EMBEDDING_LOOKUP_SPARSE", 34: "PAD", 35: "UNIDIRECTIONAL_SEQUENCE_RNN", 36: "GATHER",
    37: "BATCH_TO_SPACE_ND", 38: "SPACE_TO_BATCH_ND", 39: "TRANSPOSE", 40: "MEAN", 41: "SUB", 42: "DIV",
    43: "SQUEEZE", 44: "UNIDIRECTIONAL_SEQUENCE_LSTM", 45: "STRIDED_SLICE", 46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP", 48: "TOPK_V2", 49: "SPLIT", 50: "LOG_SOFTMAX", 51: "DELEGATE",
    52: "BIDIRECTIONAL_SEQUENCE_LSTM", 53: "CAST", 54: "PRELU", 55: "MAXIMUM", 56: "ARG_MAX", 57: "MINIMUM",
    58: "LESS", 59: "NEG", 60: "PADV2", 61: "GREATER", 62: "GREATER_EQUAL", 63: "LESS_EQUAL", 64: "SELECT",
    65: "SLICE", 66: "SIN", 67: "TRANSPOSE_CONV", 68: "SPARSE_TO_DENSE", 69: "TILE", 70: "EXPAND_DIMS",
    71: "EQUAL", 72: "NOT_EQUAL", 73: "LOG", 74: "SUM", 75: "SQRT", 76: "RSQRT", 77: "SHAPE", 78: "POW",
    79: "ARG_MIN", 80: "FAKE_QUANT", 81: "REDUCE_PROD", 82: "REDUCE_MAX", 83: "PACK", 84: "LOGICAL_OR",
    85: "ONE_HOT", 86: "LOGICAL_AND", 87: "LOGICAL_NOT", 88: "UNPACK", 89: "REDUCE_MIN", 90: "FLOOR_DIV",
    91: "REDUCE_ANY", 92: "SQUARE", 93: "ZEROS_LIKE", 94: "FILL", 95: "FLOOR_MOD", 96: "RANGE",
    97: "RESIZE_NEAREST_NEIGHBOR", 98: "LEAKY_RELU", 99: "SQUARED_DIFFERENCE", 100: "MIRROR_PAD", 101: "ABS",
    102: "SPLIT_V", 103: "UNIQUE", 104: "CEIL", 105: "REVERSE_V2", 106: "ADD_N", 107: "GATHER_ND",
    108: "COS", 109: "WHERE", 110: "RANK", 111: "ELU", 112: "REVERSE_SEQUENCE", 113: "MATRIX_DIAG",
    114: "QUANTIZE", 115: "MATRIX_SET_DIAG", 116: "ROUND", 117: "HARD_SWISH", 118: "IF", 119: "WHILE",
    120: "NON_MAX_SUPPRESSION_V4", 121: "NON_MAX_SUPPRESSION_V5", 122: "SCATTER_ND", 123: "SELECT_V2",
    124: "DENSIFY", 125: "SEGMENT_SUM", 126: "BATCH_MATMUL", 127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
    128: "CUMSUM", 129: "CALL_ONCE", 130: "BROADCAST_TO", 131: "RFFT2D", 132: "CONV_3D", 133: "IMAG",
    134: "REAL", 135: "COMPLEX_ABS", 136: "HASHTABLE", 137: "HASHTABLE_FIND", 138: "HASHTABLE_IMPORT",
    139: "HASHTABLE_SIZE", 140: "REDUCE_ALL", 141: "CONV_3D_TRANSPOSE", 142: "VAR_HANDLE",
    143: "READ_VARIABLE", 144: "ASSIGN_VARIABLE", 145: "BROADCAST_ARGS", 146: "RANDOM_STANDARD_NORMAL",
    147: "BUCKETIZE", 148: "RANDOM_UNIFORM", 149: "MULTINOMIAL", 150: "GELU", 151: "DYNAMIC_UPDATE_SLICE",
    152: "RELU_0_TO_1", 153: "UNSORTED_SEGMENT_PROD", 154: "UNSORTED_SEGMENT_MAX",
    155: "UNSORTED_SEGMENT_SUM", 156: "ATAN2", 157: "UNSORTED_SEGMENT_MIN", 158: "SIGN", 159: "BITCAST",
    160: "BITWISE_XOR", 161: "RIGHT_SHIFT", 162: "STABLEHLO_LOGISTIC", 163: "STABLEHLO_ADD",
    164: "STABLEHLO_DIVIDE", 165: "STABLEHLO_MULTIPLY", 166: "STABLEHLO_MAXIMUM", 167: "STABLEHLO_RESHAPE",
    168: "STABLEHLO_CLAMP", 169: "STABLEHLO_CONCATENATE", 170: "STABLEHLO_BROADCAST_IN_DIM",
    171: "STABLEHLO_CONVOLUTION", 172: "STABLEHLO_SLICE", 173: "STABLEHLO_CUSTOM_CALL",
    174: "STABLEHLO_REDUCE", 175: "STABLEHLO_ABS", 176: "STABLEHLO_AND", 177: "STABLEHLO_COSINE",
    178: "STABLEHLO_EXPONENTIAL", 179: "STABLEHLO_FLOOR", 180: "STABLEHLO_LOG", 181: "STABLEHLO_MINIMUM",
    182: "STABLEHLO_NEGATE", 183: "STABLEHLO_OR", 184: "STABLEHLO_POWER", 185: "STABLEHLO_REMAINDER",
    186: "STABLEHLO_RSQRT", 187: "STABLEHLO_SELECT", 188: "STABLEHLO_SUBTRACT", 189: "STABLEHLO_TANH",
    190: "STABLEHLO_SCATTER", 191: "STABLEHLO_COMPARE", 192: "STABLEHLO_CONVERT",
    193: "STABLEHLO_DYNAMIC_SLICE", 194: "STABLEHLO_DYNAMIC_UPDATE_SLICE", 195: "STABLEHLO_PAD",
    196: "STABLEHLO_IOTA", 197: "STABLEHLO_DOT_GENERAL", 198: "STABLEHLO_REDUCE_WINDOW",
    199: "STABLEHLO_SORT", 200: "STABLEHLO_WHILE", 201: "STABLEHLO_GATHER", 202: "STABLEHLO_TRANSPOSE",
    203: "DILATE", 204: "STABLEHLO_RNG_BIT_GENERATOR", 205: "REDUCE_WINDOW", 206: "STABLEHLO_COMPOSITE",
    207: "STABLEHLO_SHIFT_LEFT", 208: "STABLEHLO_CBRT", 209: "STABLEHLO_CASE",
}

# TensorType
FLOAT32, FLOAT16, INT32, UINT8, INT64, STRING, BOOL, INT16, COMPLEX64, INT8 = range(10)
TENSOR_TYPE_NAMES = {FLOAT32: "float32", FLOAT16: "float16", INT32: "int32", UINT8: "uint8", INT64: "int64",
                     STRING: "string", BOOL: "bool", INT16: "int16", COMPLEX64: "complex64", INT8: "int8"}
_NUMPY_TYPES = {FLOAT32: np.float32, FLOAT16: np.float16, INT32: np.int32, UINT8: np.uint8,
                INT64: np.int64, BOOL: np.bool_, INT16: np.int16, INT8: np.int8}

# Padding
PADDING_SAME, PADDING_VALID = 0, 1

# ActivationFunctionType
ACT_NONE, ACT_RELU, ACT_RELU_N1_TO_1, ACT_RELU6, ACT_TANH, ACT_SIGN_BIT = range(6)
ACT_NAMES = {ACT_NONE: "NONE", ACT_RELU: "RELU", ACT_RELU_N1_TO_1: "RELU_N1_TO_1", ACT_RELU6: "RELU6",
             ACT_TANH: "TANH", ACT_SIGN_BIT: "SIGN_BIT"}

# The builtin options each operator's options table holds, as
# (field id, name, struct format, default). Options for an op not listed
# here are not read.
_OPTION_FIELDS = {
    "CONV_2D": [(0, "padding", "b", 0), (1, "stride_w", "i", 0), (2, "stride_h", "i", 0),
                (3, "fused_activation_function", "b", 0), (4, "dilation_w_factor", "i", 1),
                (5, "dilation_h_factor", "i", 1)],
    "DEPTHWISE_CONV_2D": [(0, "padding", "b", 0), (1, "stride_w", "i", 0), (2, "stride_h", "i", 0),
                          (3, "depth_multiplier", "i", 0), (4, "fused_activation_function", "b", 0),
                          (5, "dilation_w_factor", "i", 1), (6, "dilation_h_factor", "i", 1)],
    "AVERAGE_POOL_2D": [(0, "padding", "b", 0), (1, "stride_w", "i", 0), (2, "stride_h", "i", 0),
                        (3, "filter_width", "i", 0), (4, "filter_height", "i", 0),
                        (5, "fused_activation_function", "b", 0)],
    "FULLY_CONNECTED": [(0, "fused_activation_function", "b", 0), (1, "weights_format", "b", 0),
                        (2, "keep_num_dims", "?", False), (3, "asymmetric_quantize_inputs", "?", False)],
    "SOFTMAX": [(0, "beta", "f", 0.0)],
    "ADD": [(0, "fused_activation_function", "b", 0), (1, "pot_scale_int16", "?", True)],
    "MEAN": [(0, "keep_dims", "?", False)],
}
_OPTION_FIELDS["MAX_POOL_2D"] = _OPTION_FIELDS["AVERAGE_POOL_2D"]


# ----------------------------------------------------------------------------
# Flatbuffer tables.
# ----------------------------------------------------------------------------


class _Table:
    """One flatbuffer table: a position whose vtable gives each field's
    offset (0 for a field the writer left at its default)."""

    __slots__ = ("buf", "pos", "vt", "vt_size")

    def __init__(self, buf: bytes, pos: int):
        self.buf = buf
        self.pos = pos
        self.vt = pos - struct.unpack_from("<i", buf, pos)[0]
        self.vt_size = struct.unpack_from("<H", buf, self.vt)[0]

    def _offset(self, fid: int) -> int:
        o = 4 + 2 * fid
        if o >= self.vt_size:
            return 0
        return struct.unpack_from("<H", self.buf, self.vt + o)[0]

    def has(self, fid: int) -> bool:
        return self._offset(fid) != 0

    def scalar(self, fid: int, fmt: str, default):
        o = self._offset(fid)
        if not o:
            return default
        return struct.unpack_from("<" + fmt, self.buf, self.pos + o)[0]

    def _indirect(self, fid: int):
        o = self._offset(fid)
        if not o:
            return None
        p = self.pos + o
        return p + struct.unpack_from("<I", self.buf, p)[0]

    def table(self, fid: int):
        p = self._indirect(fid)
        return None if p is None else _Table(self.buf, p)

    def _vector(self, fid: int):
        p = self._indirect(fid)
        if p is None:
            return None, 0
        return p + 4, struct.unpack_from("<I", self.buf, p)[0]

    def numbers(self, fid: int, dtype) -> np.ndarray:
        start, n = self._vector(fid)
        dt = np.dtype(dtype).newbyteorder("<")
        if start is None:
            return np.zeros(0, dtype=dt)
        return np.frombuffer(self.buf, dtype=dt, count=n, offset=start).copy()

    def tables(self, fid: int) -> list:
        start, n = self._vector(fid)
        if start is None:
            return []
        out = []
        for k in range(n):
            p = start + 4 * k
            out.append(_Table(self.buf, p + struct.unpack_from("<I", self.buf, p)[0]))
        return out

    def string(self, fid: int) -> str:
        start, n = self._vector(fid)
        if start is None:
            return ""
        return self.buf[start:start + n].decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------
# Decoded model.
# ----------------------------------------------------------------------------


@dataclass
class Tensor:
    index: int
    name: str
    shape: list
    type: int
    data: object = None            # numpy array for a constant tensor, else None
    scales: list = field(default_factory=list)
    zero_points: list = field(default_factory=list)
    quantized_dimension: int = 0

    @property
    def type_name(self) -> str:
        return TENSOR_TYPE_NAMES.get(self.type, str(self.type))


@dataclass
class Operator:
    index: int
    op: str                        # builtin name ("CONV_2D"), or "CUSTOM:<code>"
    inputs: list
    outputs: list
    options: dict = field(default_factory=dict)


@dataclass
class TfliteModel:
    version: int
    tensors: list
    operators: list
    inputs: list
    outputs: list
    subgraph_count: int
    description: str = ""


def _builtin_name(code: int) -> str:
    return BUILTIN_OPS.get(code, f"BUILTIN_{code}")


def read_tflite(data) -> TfliteModel:
    """Decode a .tflite flatbuffer (bytes, or a path) into its first
    subgraph's tensors and operators."""
    if isinstance(data, (str, os.PathLike)):
        with open(data, "rb") as f:
            data = f.read()
    buf = bytes(data)
    if len(buf) < 8:
        raise ValueError("not a TFLite model: file too short")
    if buf[4:8] != b"TFL3":
        raise ValueError(f"not a TFLite model: file identifier {buf[4:8]!r}, expected b'TFL3'")
    model = _Table(buf, struct.unpack_from("<I", buf, 0)[0])
    version = model.scalar(0, "I", 0)

    codes = []
    for oc in model.tables(1):
        deprecated = oc.scalar(0, "b", 0)
        builtin = oc.scalar(3, "i", 0)
        code = max(deprecated, builtin)
        if code == 32:  # CUSTOM
            codes.append("CUSTOM:" + oc.string(1))
        else:
            codes.append(_builtin_name(code))

    buffers = []
    for b in model.tables(4):
        offset = b.scalar(1, "Q", 0)
        size = b.scalar(2, "Q", 0)
        if offset > 1:
            buffers.append(buf[offset:offset + size])
        else:
            start, n = b._vector(0)
            buffers.append(b"" if start is None else buf[start:start + n])

    subgraphs = model.tables(2)
    if not subgraphs:
        raise ValueError("TFLite model has no subgraph")
    sg = subgraphs[0]

    tensors = []
    for i, t in enumerate(sg.tables(0)):
        shape = [int(v) for v in t.numbers(0, np.int32)]
        ttype = t.scalar(1, "b", 0)
        bidx = t.scalar(2, "I", 0)
        name = t.string(3)
        tensor = Tensor(index=i, name=name, shape=shape, type=ttype)
        raw = buffers[bidx] if bidx < len(buffers) else b""
        if raw:
            dtype = _NUMPY_TYPES.get(ttype)
            if dtype is None:
                raise ValueError(f"tensor {name!r}: constant of unsupported type {tensor.type_name}")
            arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).copy()
            count = int(np.prod(shape)) if shape else 1
            if arr.size != count:
                raise ValueError(f"tensor {name!r}: {arr.size} values for shape {shape}")
            tensor.data = arr.reshape(shape) if shape else arr.reshape(())
        q = t.table(4)
        if q is not None:
            tensor.scales = [float(v) for v in q.numbers(2, np.float32)]
            tensor.zero_points = [int(v) for v in q.numbers(3, np.int64)]
            tensor.quantized_dimension = q.scalar(6, "i", 0)
        tensors.append(tensor)

    operators = []
    for k, o in enumerate(sg.tables(3)):
        idx = o.scalar(0, "I", 0)
        if idx >= len(codes):
            raise ValueError(f"operator {k}: opcode index {idx} out of range")
        name = codes[idx]
        opts = {}
        spec = _OPTION_FIELDS.get(name)
        table = o.table(4)
        if spec is not None:
            for fid, key, fmt, default in spec:
                opts[key] = default if table is None else table.scalar(fid, fmt, default)
        operators.append(Operator(index=k, op=name, inputs=[int(v) for v in o.numbers(1, np.int32)],
                                  outputs=[int(v) for v in o.numbers(2, np.int32)], options=opts))

    return TfliteModel(version=version, tensors=tensors, operators=operators,
                       inputs=[int(v) for v in sg.numbers(1, np.int32)],
                       outputs=[int(v) for v in sg.numbers(2, np.int32)],
                       subgraph_count=len(subgraphs), description=model.string(3))
