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

Tests for the ONNX importer: real ONNX Runtime quantized models, imported,
compiled against tinymind's int8 layers and run, compared with ONNX Runtime.

    make test PY=<python with torch, onnx, onnxruntime, numpy>

Each end-to-end case exports a small PyTorch model to ONNX, quantizes it with
onnxruntime.quantization.quantize_static (QDQ and QOperator, per-tensor and
per-channel), imports it, generates a C++ program that builds the matching
QConv2D / QConv2DPerChannel / QDense chain from the emitted weights.hpp,
compiles it with g++, and feeds both runtimes the same int8 inputs. The int8
outputs must agree to within 1 LSB (ONNX Runtime and tinymind round the
requantization differently), and almost all must agree exactly.

The models are chosen so a wrong mapping cannot pass: a non-square kernel
with unequal strides and padding (catches OIHW vs OHWI and pad order), a
second convolution (catches a wrong NHWC order between convolutions), a
Flatten into a dense layer (catches the NCHW-to-NHWC column order) and a
MatMul without bias (catches an untransposed [in][out] weight).

The graph cases (GraphModels) run a small residual network -- convolution,
max pooling, a depthwise convolution, a pointwise one, a residual Add, average
pooling, global average pooling, a dense layer and optionally a softmax --
through a C++ program that wires every record by its `sources` onto
QConv2D, QMaxPool2D, QDepthwiseConv2D, QAdd (buildQAddParams), QDense and
QSoftmaxFixedPoint1D.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import tinymind_import_onnx as ti  # noqa: E402
import tinymind_model_codegen as cg  # noqa: E402

try:
    import onnx  # noqa: F401
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    HAVE_STACK = True
except ImportError:  # pragma: no cover - exercised only without the deps
    HAVE_STACK = False

GXX = shutil.which("g++")


# ---------------------------------------------------------------------------
# Model building and quantization.
# ---------------------------------------------------------------------------

if HAVE_STACK:
    class ConvNet(nn.Module):
        """Conv(3x2, stride 1x2, pad 1x0) + ReLU -> Conv(3x3, stride 2) + ReLU
        -> Flatten -> Linear + ReLU -> Linear without bias (exports MatMul)."""

        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(2, 3, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
            self.c2 = nn.Conv2d(3, 4, kernel_size=3, stride=2)
            self.f1 = nn.Linear(4 * 3 * 2, 6)
            self.f2 = nn.Linear(6, 5, bias=False)

        def forward(self, x):
            h = torch.relu(self.c1(x))
            h = torch.relu(self.c2(h))
            h = torch.relu(self.f1(h.flatten(1)))
            return self.f2(h)

    INPUT_SHAPE = (1, 2, 8, 10)

    class Reader(CalibrationDataReader):
        def __init__(self, name, shape, seed):
            rng = np.random.default_rng(seed)
            self._it = iter([{name: rng.standard_normal(shape).astype(np.float32)} for _ in range(32)])

        def get_next(self):
            return next(self._it, None)


def export_and_quantize(model, shape, workdir, fmt, per_channel, tag,
                        activation_type=None, weight_type=None):
    torch.manual_seed(0)
    fp32 = os.path.join(workdir, f"{tag}_fp32.onnx")
    torch.onnx.export(model.eval(), torch.zeros(shape), fp32, input_names=["input"],
                      output_names=["out"], opset_version=13, dynamo=False)
    out = os.path.join(workdir, f"{tag}_{fmt}_{int(per_channel)}.onnx")
    quantize_static(fp32, out, Reader("input", shape, 1),
                    quant_format=getattr(QuantFormat, fmt),
                    weight_type=weight_type or QuantType.QInt8,
                    activation_type=activation_type or QuantType.QInt8,
                    per_channel=per_channel)
    return out


# ---------------------------------------------------------------------------
# C++ harness generation.
# ---------------------------------------------------------------------------

def harness_source(layers, namespace, header):
    """C++ program that runs the imported weight layers in order on int8
    NHWC input read from stdin (one integer per value) and prints the int8
    output."""
    lines = [
        "#include <cstdint>",
        "#include <cstdio>",
        '#include "qaffine.hpp"',
        '#include "qconv2d.hpp"',
        '#include "qdense.hpp"',
        f'#include "{header}"',
        "",
        "typedef tinymind::Requantizer<int32_t, int8_t> Rq;",
        "static Rq makeRq(int32_t m, int32_t s, int32_t zp)",
        "{",
        "    Rq r;",
        "    r.multiplier = m;",
        "    r.shift = s;",
        "    r.zero_point = static_cast<int8_t>(zp);",
        "    r.qmin = -128;",
        "    r.qmax = 127;",
        "    return r;",
        "}",
        "",
        "int main()",
        "{",
    ]
    weight_layers = [l for l in layers if l.weight is not None]
    first = weight_layers[0]
    g = first.geometry
    in_size = (g["InputHeight"] * g["InputWidth"] * g["InputChannels"]) if first.op_type == "Conv" else g["NumInputs"]
    lines.append(f"    static int8_t buf0[{in_size}];")
    lines.append(f"    for (int i = 0; i < {in_size}; ++i)")
    lines.append("    {")
    lines.append("        int v;")
    lines.append("        if (std::scanf(\"%d\", &v) != 1) { return 2; }")
    lines.append("        buf0[i] = static_cast<int8_t>(v);")
    lines.append("    }")
    out_size = in_size
    for i, layer in enumerate(weight_layers):
        k = f"{namespace}::k{ti.sanitize(layer.name)}"
        g = layer.geometry
        if layer.op_type == "Conv":
            dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                    f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}, {g['NumFilters']}, "
                    f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}")
            out_size = g["OutputHeight"] * g["OutputWidth"] * g["NumFilters"]
            n_out = g["NumFilters"]
        else:
            dims = f"1, 1, {g['NumInputs']}, 1, 1, 1, 1, {g['NumOutputs']}"
            out_size = g["NumOutputs"]
            n_out = g["NumOutputs"]
        lines.append(f"    static int8_t buf{i + 1}[{out_size}];")
        bias = f"{k}_Biases" if layer.bias is not None else "nullptr"
        if layer.per_channel:
            lines.append(f"    static Rq rq{i}[{n_out}];")
            lines.append(f"    for (int c = 0; c < {n_out}; ++c)")
            lines.append("    {")
            lines.append(f"        rq{i}[c] = makeRq({k}_Multipliers[c], {k}_Shifts[c], {k}_OutputZeroPoint);")
            lines.append("    }")
            lines.append(f"    tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, {dims}> layer{i};")
            lines.append(f"    layer{i}.requantizers = rq{i};")
        else:
            if layer.op_type == "Conv":
                lines.append(f"    tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, {dims}> layer{i};")
            else:
                lines.append(f"    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, "
                             f"{g['NumInputs']}, {g['NumOutputs']}> layer{i};")
            lines.append(f"    layer{i}.requantizer = makeRq({k}_Multiplier, {k}_Shift, {k}_OutputZeroPoint);")
        lines.append(f"    layer{i}.weights = {k}_Weights;")
        lines.append(f"    layer{i}.biases = {bias};")
        lines.append(f"    layer{i}.input_zero_point = static_cast<int8_t>({k}_InputZeroPoint);")
        lines.append(f"    layer{i}.forward(buf{i}, buf{i + 1});")
    last = len(weight_layers)
    lines.append(f"    for (int i = 0; i < {out_size}; ++i)")
    lines.append("    {")
    lines.append(f"        std::printf(\"%d\\n\", static_cast<int>(buf{last}[i]));")
    lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compile_harness(layers, workdir, tag):
    header = os.path.join(workdir, f"{tag}_weights.hpp")
    ti.emit_weights_header(header, tag, layers)
    src = os.path.join(workdir, f"{tag}_harness.cpp")
    with open(src, "w") as f:
        f.write(harness_source(layers, tag, os.path.basename(header)))
    exe = os.path.join(workdir, f"{tag}_harness")
    subprocess.run([GXX, "-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                    "-DTINYMIND_ENABLE_QUANTIZATION=1",
                    "-I" + os.path.join(REPO, "cpp"), "-I" + os.path.join(REPO, "cpp", "include"),
                    "-I" + os.path.join(REPO, "include"), "-I" + workdir,
                    src, "-o", exe], check=True)
    return exe


def run_both(model_path, layers, exe, n_samples=24, seed=5, calibration_like=False):
    """Feed identical int8 inputs to ONNX Runtime and the tinymind build;
    return the int8 outputs of both as [n_samples, outputs] arrays.

    Inputs are uniform over the whole int8 range, or, with calibration_like,
    half-scale standard-normal values, which stay inside the ranges the
    standard-normal calibration data measured."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    first = [l for l in layers if l.weight is not None][0]
    s_in, zp_in = first.input_scale, first.input_zp
    last = [l for l in layers if l.weight is not None][-1]
    s_out, zp_out = last.output_scale, last.output_zp
    in_meta = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in in_meta.shape]
    rng = np.random.default_rng(seed)
    ref, got = [], []
    for _ in range(n_samples):
        if calibration_like:
            q_in = np.clip(np.rint(0.5 * rng.standard_normal(shape) / s_in) + zp_in, -128, 127).astype(np.int64)
        else:
            q_in = rng.integers(-128, 128, size=shape).astype(np.int64)
        x = ((q_in - zp_in) * s_in).astype(np.float32)
        y = sess.run(None, {in_meta.name: x})[0].reshape(-1)
        ref.append(np.rint(y / s_out).astype(np.int64) + zp_out)
        q_nhwc = np.transpose(q_in, (0, 2, 3, 1)) if q_in.ndim == 4 else q_in
        stdin = "\n".join(str(v) for v in q_nhwc.reshape(-1).tolist()) + "\n"
        out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
        got.append(np.array([int(v) for v in out.split()], dtype=np.int64))
    return np.array(ref), np.array(got)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAVE_STACK, "needs torch, onnx and onnxruntime")
@unittest.skipIf(GXX is None, "needs g++")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workdir = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_model(self, fmt, per_channel):
        torch.manual_seed(0)
        path = export_and_quantize(ConvNet(), INPUT_SHAPE, self.workdir, fmt, per_channel, "convnet")
        layers = ti.parse_onnx_model(path)
        kinds = [l.op_type for l in layers]
        self.assertEqual(kinds, ["Conv", "Conv", "Dense", "Dense"],
                         f"{fmt}: every compute layer must be imported, got {kinds}")
        self.assertTrue(all(l.per_channel == per_channel for l in layers))
        self.assertIsNotNone(layers[2].bias, "the Linear's bias must be carried")
        self.assertIsNone(layers[3].bias)
        self.assertEqual(layers[0].geometry["KernelHeight"], 3)
        self.assertEqual(layers[0].geometry["KernelWidth"], 2)
        self.assertEqual(layers[0].weight.shape, (3, 3, 2, 2))   # OHWI
        self.assertEqual(layers[3].weight.shape, (5, 6))         # [out][in]
        tag = f"m_{fmt}_{int(per_channel)}"
        exe = compile_harness(layers, self.workdir, tag)
        ref, got = run_both(path, layers, exe)
        diff = np.abs(ref - got)
        exact = float(np.mean(diff == 0))
        print(f"\n  {fmt:9s} per_channel={per_channel!s:5s}: max |ort - tinymind| = {diff.max()} LSB, "
              f"{100 * exact:.1f}% exact over {ref.size} outputs")
        self.assertLessEqual(int(diff.max()), 1)
        self.assertGreater(exact, 0.9)
        self.assertGreater(len(np.unique(ref)), 4, "outputs must vary, or the comparison is empty")

    def test_qdq_per_tensor(self):
        self.check_model("QDQ", False)

    def test_qdq_per_channel(self):
        self.check_model("QDQ", True)

    def test_qoperator_per_tensor(self):
        self.check_model("QOperator", False)

    def test_qoperator_per_channel(self):
        self.check_model("QOperator", True)

    def test_matmul_add_bias_is_folded(self):
        """A MatMul followed by a bias Add (QLinearMatMul + QLinearAdd in
        QOperator form) becomes one dense layer with that bias."""
        from onnx import TensorProto, helper, numpy_helper
        rng = np.random.default_rng(3)
        w = rng.standard_normal((7, 4)).astype(np.float32)
        b = (rng.standard_normal(4) * 2.0).astype(np.float32)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["input", "W"], ["mm"], name="dense"),
             helper.make_node("Add", ["mm", "B"], ["out"], name="bias")],
            "mm_add",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 7])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])],
            [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        fp32 = os.path.join(self.workdir, "mm_add_fp32.onnx")
        onnx.save(model, fp32)
        for fmt in ("QDQ", "QOperator"):
            out = os.path.join(self.workdir, f"mm_add_{fmt}.onnx")
            quantize_static(fp32, out, Reader("input", (1, 7), 2),
                            quant_format=getattr(QuantFormat, fmt),
                            weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
            layers = ti.parse_onnx_model(out)
            self.assertEqual([l.op_type for l in layers], ["Dense"], fmt)
            self.assertIsNotNone(layers[0].bias, f"{fmt}: the Add's bias must be folded in")
            real_bias = layers[0].bias.astype(np.float64) * layers[0].input_scale * layers[0].weight_scale
            np.testing.assert_allclose(real_bias, b, atol=0.05)
            exe = compile_harness(layers, self.workdir, f"mm_add_{fmt}")
            # Calibration-like inputs: ONNX Runtime clamps the MatMul result
            # to its own calibrated range before adding the bias, and the
            # folded layer has no such intermediate clamp, so inputs far
            # outside the calibration distribution legitimately differ.
            ref, got = run_both(out, layers, exe, calibration_like=True)
            diff = np.abs(ref - got)
            print(f"\n  MatMul+Add {fmt:9s}: max |ort - tinymind| = {diff.max()} LSB")
            # QOperator requantizes twice (after the MatMul, then after the
            # Add); the folded layer rounds once, so allow one more LSB.
            self.assertLessEqual(int(diff.max()), 1 if fmt == "QDQ" else 2)

    def test_activation_records_carry_graph_quantization(self):
        class SigNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Linear(5, 3)

            def forward(self, x):
                return torch.sigmoid(self.f(x))
        for fmt in ("QDQ", "QOperator"):
            path = export_and_quantize(SigNet(), (1, 5), self.workdir, fmt, False, "signet")
            layers = ti.parse_onnx_model(path)
            self.assertEqual([l.op_type for l in layers], ["Dense", "Sigmoid"], fmt)
            dense, sig = layers
            self.assertEqual(sig.input_scale, dense.output_scale)
            self.assertEqual(sig.input_zp, dense.output_zp)
            self.assertGreater(sig.output_scale, 0.0)

    def test_unsupported_ops_are_refused(self):
        class PadPoolNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(2, 3, 3)

            def forward(self, x):
                return torch.nn.functional.max_pool2d(torch.relu(self.c(x)), 2, padding=1)

        class CeilPoolNet(PadPoolNet):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(self.c(x), 2, ceil_mode=True)

        class ConcatNet(PadPoolNet):
            def forward(self, x):
                h = self.c(x)
                return torch.cat([h, h], dim=1)
        for net, needle in ((PadPoolNet, "padded pooling"), (CeilPoolNet, "ceil_mode"),
                            (ConcatNet, "unsupported ONNX op")):
            for fmt in ("QDQ", "QOperator"):
                path = export_and_quantize(net(), INPUT_SHAPE, self.workdir, fmt, False, net.__name__)
                with self.assertRaisesRegex(ValueError, needle, msg=f"{net.__name__} {fmt}"):
                    ti.parse_onnx_model(path)

    def test_grouped_conv_is_refused(self):
        """A depthwise convolution (group = channels = filters) is imported;
        other groupings, and a depthwise one with two filters per channel,
        are not."""
        for c_in, c_out, groups in ((4, 4, 2), (2, 4, 2)):
            class GroupNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.c = nn.Conv2d(c_in, c_out, 3, groups=groups)

                def forward(self, x):
                    return self.c(x)
            shape = (1, c_in, 8, 10)
            path = export_and_quantize(GroupNet(), shape, self.workdir, "QDQ", False, f"g{c_in}{c_out}")
            with self.assertRaisesRegex(ValueError, "group"):
                ti.parse_onnx_model(path)

    def test_uint8_is_refused(self):
        path = export_and_quantize(ConvNet(), INPUT_SHAPE, self.workdir, "QDQ", False, "u8",
                                   activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)
        with self.assertRaisesRegex(ValueError, "uint8"):
            ti.parse_onnx_model(path)


if HAVE_STACK:
    class ResNetish(nn.Module):
        """Conv 3x3 (padding 1) + ReLU -> MaxPool 2x2 -> a residual block
        (depthwise Conv 3x3 + ReLU -> pointwise Conv, added to the block's
        input) -> AvgPool 2x2 -> global average pool -> Linear [-> Softmax].
        Unequal channel counts and a 2x6 input after pooling catch a wrong
        depthwise layout, swapped Add operands and a wrong pooling window."""

        def __init__(self, softmax):
            super().__init__()
            self.softmax = softmax
            self.c1 = nn.Conv2d(2, 5, 3, padding=1)
            self.dw = nn.Conv2d(5, 5, 3, padding=1, groups=5)
            self.pw = nn.Conv2d(5, 5, 1)
            self.f = nn.Linear(5, 4)
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(2.0)

        def forward(self, x):
            h = torch.nn.functional.max_pool2d(torch.relu(self.c1(x)), 2)
            h = h + self.pw(torch.relu(self.dw(h)))
            h = torch.nn.functional.avg_pool2d(h, 2)
            h = torch.nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
            y = self.f(h)
            return torch.softmax(y, dim=1) if self.softmax else y

    GRAPH_INPUT_SHAPE = (1, 2, 8, 24)


def _out_size(layer, sizes):
    g = layer.geometry
    if layer.op_type in ("Conv", "DepthwiseConv", "AveragePool"):
        return g["OutputHeight"] * g["OutputWidth"] * g["NumFilters"]
    if layer.op_type == "MaxPool":
        return g["OutputHeight"] * g["OutputWidth"] * g["InputChannels"]
    if layer.op_type == "Dense":
        return g["NumOutputs"]
    return sizes[layer.sources[0]]


def graph_harness_source(layers, namespace, header):
    """C++ program that runs every imported record, each reading the
    buffers its `sources` name, on int8 NHWC input from stdin, and prints the
    last layer's int8 output."""
    first = layers[0]
    g = first.geometry
    in_size = g["NumInputs"] if first.op_type == "Dense" else \
        g["InputHeight"] * g["InputWidth"] * g["InputChannels"]
    sizes = {-1: in_size}
    lines = [
        "#include <cstdint>",
        "#include <cstdio>",
        '#include "qaffine.hpp"',
        '#include "qactivations.hpp"',
        '#include "qadd.hpp"',
        '#include "qcalibration.hpp"',
        '#include "qconv2d.hpp"',
        '#include "qdense.hpp"',
        '#include "qdepthwiseconv2d.hpp"',
        '#include "qpool2d.hpp"',
        '#include "qsoftmax_fixedpoint.hpp"',
        f'#include "{header}"',
        "",
        "typedef tinymind::Requantizer<int32_t, int8_t> Rq;",
        "static Rq makeRq(int32_t m, int32_t s, int32_t zp)",
        "{",
        "    Rq r;",
        "    r.multiplier = m;",
        "    r.shift = s;",
        "    r.zero_point = static_cast<int8_t>(zp);",
        "    r.qmin = -128;",
        "    r.qmax = 127;",
        "    return r;",
        "}",
        "",
        "int main()",
        "{",
        f"    static int8_t bufin[{in_size}];",
        f"    for (int i = 0; i < {in_size}; ++i)",
        "    {",
        "        int v;",
        "        if (std::scanf(\"%d\", &v) != 1) { return 2; }",
        "        bufin[i] = static_cast<int8_t>(v);",
        "    }",
    ]

    def buf(j):
        return "bufin" if j < 0 else f"buf{j}"

    for i, layer in enumerate(layers):
        k = f"{namespace}::k{ti.sanitize(layer.name)}"
        g = layer.geometry
        n = _out_size(layer, sizes)
        sizes[i] = n
        x = buf(layer.sources[0])
        lines.append(f"    static int8_t buf{i}[{n}];")
        if layer.op_type in ("Conv", "DepthwiseConv", "AveragePool"):
            dims = (f"{g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, {g['KernelHeight']}, "
                    f"{g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}")
            pads = f"{g['PadTop']}, {g['PadBottom']}, {g['PadLeft']}, {g['PadRight']}"
            bias = f"{k}_Biases" if layer.bias is not None else "nullptr"
            if layer.op_type == "Conv":
                if layer.per_channel:
                    lines += [f"    static Rq rq{i}[{g['NumFilters']}];",
                              f"    for (int c = 0; c < {g['NumFilters']}; ++c) {{ rq{i}[c] = makeRq({k}_Multipliers[c], {k}_Shifts[c], {k}_OutputZeroPoint); }}",
                              f"    tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, {dims}, {g['NumFilters']}, {pads}> layer{i};",
                              f"    layer{i}.requantizers = rq{i};"]
                else:
                    lines += [f"    tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t, {dims}, {g['NumFilters']}, {pads}> layer{i};",
                              f"    layer{i}.requantizer = makeRq({k}_Multiplier, {k}_Shift, {k}_OutputZeroPoint);"]
            else:
                m = f"{k}_Multipliers[c], {k}_Shifts[c]" if layer.per_channel else f"{k}_Multiplier, {k}_Shift"
                lines += [f"    static Rq rq{i}[{g['NumFilters']}];",
                          f"    for (int c = 0; c < {g['NumFilters']}; ++c) {{ rq{i}[c] = makeRq({m}, {k}_OutputZeroPoint); }}",
                          f"    tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, {dims}, {pads}> layer{i};",
                          f"    layer{i}.requantizers = rq{i};"]
            lines += [f"    layer{i}.weights = {k}_Weights;",
                      f"    layer{i}.biases = {bias};",
                      f"    layer{i}.input_zero_point = static_cast<int8_t>({k}_InputZeroPoint);",
                      f"    layer{i}.forward({x}, buf{i});"]
        elif layer.op_type == "Dense":
            if layer.per_channel:
                lines += [f"    static Rq rq{i}[{g['NumOutputs']}];",
                          f"    for (int c = 0; c < {g['NumOutputs']}; ++c) {{ rq{i}[c] = makeRq({k}_Multipliers[c], {k}_Shifts[c], {k}_OutputZeroPoint); }}",
                          f"    tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 1, 1, {g['NumInputs']}, 1, 1, 1, 1, {g['NumOutputs']}> layer{i};",
                          f"    layer{i}.requantizers = rq{i};"]
            else:
                lines += [f"    tinymind::QDense<int8_t, int8_t, int32_t, int8_t, {g['NumInputs']}, {g['NumOutputs']}> layer{i};",
                          f"    layer{i}.requantizer = makeRq({k}_Multiplier, {k}_Shift, {k}_OutputZeroPoint);"]
            bias = f"{k}_Biases" if layer.bias is not None else "nullptr"
            lines += [f"    layer{i}.weights = {k}_Weights;",
                      f"    layer{i}.biases = {bias};",
                      f"    layer{i}.input_zero_point = static_cast<int8_t>({k}_InputZeroPoint);",
                      f"    layer{i}.forward({x}, buf{i});"]
        elif layer.op_type == "MaxPool":
            lines += [f"    tinymind::QMaxPool2D<int8_t, {g['InputHeight']}, {g['InputWidth']}, {g['InputChannels']}, "
                      f"{g['KernelHeight']}, {g['KernelWidth']}, {g['StrideH']}, {g['StrideW']}> layer{i};",
                      f"    layer{i}.forward({x}, buf{i});"]
        elif layer.op_type == "Add":
            lines += [f"    const tinymind::QAddParams p{i} = tinymind::buildQAddParams({k}_InputScale, {k}_InputBScale, {k}_OutputScale);",
                      f"    tinymind::QAdd<int8_t, int8_t, int8_t, {n}> layer{i};",
                      f"    layer{i}.input_a_zero_point = static_cast<int8_t>({k}_InputZeroPoint);",
                      f"    layer{i}.input_b_zero_point = static_cast<int8_t>({k}_InputBZeroPoint);",
                      f"    layer{i}.left_shift = p{i}.left_shift;",
                      f"    layer{i}.input_a_multiplier = p{i}.input_a_multiplier;",
                      f"    layer{i}.input_a_shift = p{i}.input_a_shift;",
                      f"    layer{i}.input_b_multiplier = p{i}.input_b_multiplier;",
                      f"    layer{i}.input_b_shift = p{i}.input_b_shift;",
                      f"    layer{i}.output_requantizer = makeRq(p{i}.output_multiplier, p{i}.output_shift, {k}_OutputZeroPoint);",
                      f"    layer{i}.forward({x}, {buf(layer.sources[1])}, buf{i});"]
        elif layer.op_type == "Softmax":
            lines += [f"    int32_t mult{i} = 0, left{i} = 0, diff{i} = 0;",
                      f"    tinymind::buildQSoftmaxFixedPointParams({k}_InputScale, 1.0, mult{i}, left{i}, diff{i});",
                      f"    const tinymind::QSoftmaxFixedPoint1D<1, {n}> layer{i} = {{mult{i}, left{i}, diff{i}}};",
                      f"    layer{i}.forward({x}, buf{i});"]
        elif layer.op_type in ("Sigmoid", "Tanh"):
            lines += [f"    static int8_t lut{i}[256];",
                      f"    tinymind::buildQ{layer.op_type}LUT({k}_InputScale, {k}_InputZeroPoint, {k}_OutputScale, {k}_OutputZeroPoint, lut{i});",
                      f"    for (int j = 0; j < {n}; ++j) {{ buf{i}[j] = tinymind::qApplyLUT({x}[j], lut{i}); }}"]
        else:
            raise ValueError(f"no harness wiring for {layer.op_type}")
    last = len(layers) - 1
    lines += [f"    for (int i = 0; i < {sizes[last]}; ++i)",
              "    {",
              f"        std::printf(\"%d\\n\", static_cast<int>(buf{last}[i]));",
              "    }",
              "    return 0;",
              "}"]
    return "\n".join(lines) + "\n"


def compile_graph_harness(layers, workdir, tag):
    header = os.path.join(workdir, f"{tag}_weights.hpp")
    ti.emit_weights_header(header, tag, layers)
    src = os.path.join(workdir, f"{tag}_graph.cpp")
    with open(src, "w") as f:
        f.write(graph_harness_source(layers, tag, os.path.basename(header)))
    exe = os.path.join(workdir, f"{tag}_graph")
    # The calibration helpers (buildQAddParams, the LUT and softmax builders)
    # are host-side: they need the float and std switches.
    subprocess.run([GXX, "-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                    "-DTINYMIND_ENABLE_QUANTIZATION=1", "-DTINYMIND_ENABLE_FLOAT=1", "-DTINYMIND_ENABLE_STD=1",
                    "-I" + os.path.join(REPO, "cpp"), "-I" + os.path.join(REPO, "cpp", "include"),
                    "-I" + os.path.join(REPO, "include"), "-I" + workdir,
                    src, "-o", exe], check=True)
    return exe


def run_graph(model_path, layers, exe, n_samples=24, seed=5):
    """ONNX Runtime's float output and tinymind's int8 output on the same
    int8 inputs (in the first layer's input grid), half of them uniform
    over int8, half calibration-like."""
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    s_in, zp_in = layers[0].input_scale, layers[0].input_zp
    in_meta = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in in_meta.shape]
    rng = np.random.default_rng(seed)
    ref, got = [], []
    for i in range(n_samples):
        if i % 2:
            q_in = np.clip(np.rint(0.5 * rng.standard_normal(shape) / s_in) + zp_in, -128, 127).astype(np.int64)
        else:
            q_in = rng.integers(-128, 128, size=shape).astype(np.int64)
        x = ((q_in - zp_in) * s_in).astype(np.float32)
        ref.append(sess.run(None, {in_meta.name: x})[0].reshape(-1))
        stdin = "\n".join(str(v) for v in np.transpose(q_in, (0, 2, 3, 1)).reshape(-1).tolist()) + "\n"
        out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
        got.append(np.array([int(v) for v in out.split()], dtype=np.int64))
    return np.array(ref), np.array(got)


@unittest.skipUnless(HAVE_STACK, "needs torch, onnx and onnxruntime")
@unittest.skipIf(GXX is None, "needs g++")
class GraphModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workdir = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, fmt, per_channel, softmax):
        torch.manual_seed(1)
        tag = f"res_{fmt}_{int(per_channel)}_{int(softmax)}"
        path = export_and_quantize(ResNetish(softmax), GRAPH_INPUT_SHAPE, self.workdir, fmt, per_channel, tag)
        layers = ti.parse_onnx_model(path)
        kinds = [l.op_type for l in layers]
        expected = ["Conv", "MaxPool", "DepthwiseConv", "Conv", "Add", "AveragePool", "AveragePool", "Dense"]
        self.assertEqual(kinds, expected + (["Softmax"] if softmax else []), f"{fmt}: {kinds}")
        self.assertEqual(layers[2].weight.shape, (5, 3, 3))       # [C][KH][KW]
        self.assertEqual(sorted(layers[4].sources), [1, 3])        # the residual
        self.assertEqual(layers[6].geometry["KernelHeight"], 2)    # global: the whole 2x6 map
        self.assertEqual(layers[6].geometry["KernelWidth"], 6)
        exe = compile_graph_harness(layers, self.workdir, tag)
        ref, got = run_graph(path, layers, exe)
        if softmax:
            # tinymind's softmax gives probabilities on the 1/256 grid (as
            # TFLite does); ONNX Runtime's QDQ output is on 1/255.
            diff = np.abs((got + 128) / 256.0 - ref)
            print(f"\n  residual {fmt:9s} per_channel={per_channel!s:5s} softmax: "
                  f"max |ort - tinymind| = {diff.max() * 256:.2f}/256")
            self.assertLessEqual(diff.max(), 3 / 256)
        else:
            last = layers[-1]
            q_ref = np.rint(ref / last.output_scale).astype(np.int64) + last.output_zp
            diff = np.abs(q_ref - got)
            exact = float(np.mean(diff == 0))
            print(f"\n  residual {fmt:9s} per_channel={per_channel!s:5s}: max |ort - tinymind| = "
                  f"{diff.max()} LSB, {100 * exact:.1f}% exact over {got.size} outputs")
            # QOperator form runs QLinearAveragePool and QLinearAdd, which
            # match tinymind's integer layers to within a step. In QDQ form
            # ONNX Runtime averages in float and rounds ties to even, where
            # the integer layer rounds them away from zero: one output in
            # eight of a 2x2 pool, which the global pool and the dense layer
            # after it carry on.
            limit, share = (1, 0.9) if fmt == "QOperator" else (3, 0.5)
            self.assertLessEqual(int(diff.max()), limit)
            self.assertGreater(exact, share)
            self.assertGreater(len(np.unique(q_ref)), 8, "outputs must vary, or the comparison is empty")

    def test_qdq_per_tensor(self):
        self.check("QDQ", False, False)

    def test_qdq_per_channel(self):
        self.check("QDQ", True, False)

    def test_qoperator_per_tensor(self):
        self.check("QOperator", False, False)

    def test_qoperator_per_channel(self):
        self.check("QOperator", True, False)

    def test_softmax_qdq(self):
        self.check("QDQ", False, True)

    def test_softmax_qoperator(self):
        """QOperator's QLinearSoftmax."""
        self.check("QOperator", False, True)

    def test_requantization_between_layers_is_refused(self):
        """A Q -> DQ pair at another scale between two layers changes the
        tensor's grid, which no record carries."""
        from onnx import helper, numpy_helper
        path = export_and_quantize(ResNetish(False), GRAPH_INPUT_SHAPE, self.workdir, "QDQ", False, "requant")
        model = onnx.load(path)
        g = model.graph
        gemm = next(n for n in g.node if n.op_type == "Gemm")
        dq = next(n for n in g.node if n.output[0] == gemm.input[0])
        scale = numpy_helper.to_array(next(i for i in g.initializer if i.name == dq.input[1]))
        g.initializer.append(numpy_helper.from_array(np.array(scale * 1.5, dtype=np.float32), "rq_scale"))
        nodes = list(g.node)
        at = nodes.index(gemm)
        nodes[at:at] = [helper.make_node("QuantizeLinear", [dq.output[0], "rq_scale", dq.input[2]], ["rq_q"]),
                        helper.make_node("DequantizeLinear", ["rq_q", "rq_scale", dq.input[2]], ["rq_dq"])]
        gemm.input[0] = "rq_dq"
        del g.node[:]
        g.node.extend(nodes)
        edited = os.path.join(self.workdir, "requant_edited.onnx")
        onnx.save(model, edited)
        with self.assertRaisesRegex(ValueError, "another grid"):
            ti.parse_onnx_model(edited)

    def test_softmax_over_another_axis_is_refused(self):
        class ColSoftmax(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Linear(5, 3)

            def forward(self, x):
                return torch.softmax(self.f(x), dim=0)
        for fmt in ("QDQ", "QOperator"):
            path = export_and_quantize(ColSoftmax(), (1, 5), self.workdir, fmt, False, "colsoftmax")
            with self.assertRaisesRegex(ValueError, "axis", msg=fmt):
                ti.parse_onnx_model(path)


class Emitter(unittest.TestCase):
    """Runs without onnx: the header layout for a hand-built layer."""

    def test_header_fields(self):
        w = np.arange(-6, 6, dtype=np.int8).reshape(2, 6)
        layer = ti._finish_weight_layer("Dense", "fc", "x", 0.05, 3, "y", 0.1, -5, w,
                                        [0.02], np.array([0.5, -0.25]),
                                        {"NumInputs": 6, "NumOutputs": 2})
        self.assertEqual(layer.bias.tolist(), [lround_ref(0.5 / 0.001), lround_ref(-0.25 / 0.001)])
        m, s = ti.quantize_multiplier(0.05 * 0.02 / 0.1)
        self.assertEqual((layer.multipliers[0], layer.shifts[0]), (m, s))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.hpp")
            ti.emit_weights_header(path, "net", [layer])
            text = open(path).read()
        for needle in ("kfc_Weights[12]", "kfc_Biases[2]", "kfc_Multiplier", "kfc_Shift",
                       "kfc_NumInputs = 6", "kfc_InputZeroPoint  = 3", "kfc_OutputZeroPoint = -5",
                       "kfc_HasBias = true", "kfc_PerChannel = false"):
            self.assertIn(needle, text)

    def test_add_and_maxpool_fields(self):
        add = ti.OnnxQuantizedLayer(op_type="Add", name="res", input_name="a", input_scale=0.05,
                                    input_zp=3, output_name="y", output_scale=0.1, output_zp=-5,
                                    input_b_name="b", input_b_scale=0.02, input_b_zp=-7, sources=[0, -1])
        pool = ti.OnnxQuantizedLayer(op_type="MaxPool", name="mp", input_name="y", input_scale=0.1,
                                     input_zp=-5, output_name="z", output_scale=0.1, output_zp=-5,
                                     geometry={"InputHeight": 4, "KernelHeight": 2}, sources=[1])
        stand_in = ti.OnnxQuantizedLayer(op_type="Relu", name="first", input_name="x", input_scale=0.05,
                                         input_zp=3, output_name="a", output_scale=0.05, output_zp=3,
                                         sources=[-1])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.hpp")
            ti.emit_weights_header(path, "net", [stand_in, add, pool])
            text = open(path).read()
        for needle in ("kres_InputBScale     = 0.02f", "kres_InputBZeroPoint = -7",
                       "// Add res (reads first, the model's input)", "kmp_InputHeight = 4",
                       "kmp_KernelHeight = 2", "kb_ZeroPoint = -7"):
            self.assertIn(needle, text)


# ---------------------------------------------------------------------------
# Whole-model code generation (tinymind_model_codegen).
# ---------------------------------------------------------------------------

def simulate_plan(layers, sizes, plan):
    """Run the plan on a byte-tagged arena, independently of the planner's
    own bookkeeping: each layer checks that every arena input still holds the
    bytes its producer wrote, then stamps its output. An element-wise layer may
    write over the input it reads. Returns the first violation, or None."""
    last = len(layers) - 1
    arena = [None] * max(plan.arena_size, 1)
    for i, layer in enumerate(layers):
        for j in layer.sources:
            if j < 0:
                continue
            o = plan.offsets[j]
            if any(arena[o + k] != j for k in range(sizes[j])):
                return f"layer {i} reads layer {j}'s output after it was overwritten"
        if i == last:
            break
        o = plan.offsets[i]
        for j in layer.sources:
            if j < 0 or layer.op_type in cg.IN_PLACE_OPS and plan.offsets[j] == o:
                continue
            if o < plan.offsets[j] + sizes[j] and plan.offsets[j] < o + sizes[i]:
                return f"layer {i} writes over its input {j}"
        for k in range(sizes[i]):
            arena[o + k] = i
    return None


def _rec(op, sources, name=None):
    return ti.OnnxQuantizedLayer(op_type=op, name=name or f"{op}{len(sources)}", input_name="x",
                                 input_scale=0.1, input_zp=0, output_scale=0.1, output_zp=0,
                                 sources=list(sources))


class Planner(unittest.TestCase):
    """Runs without onnx: the memory planner on hand-built graphs."""

    def plan(self, layers, sizes, **kw):
        plan = cg.plan_memory(layers, sizes, **kw)
        cg.check_plan(plan)
        self.assertIsNone(simulate_plan(layers, sizes, plan))
        self.assertLessEqual(plan.lower_bound, plan.arena_size)
        self.assertLessEqual(plan.arena_size, plan.unshared_size)
        return plan

    def test_chain_reuses_two_buffers(self):
        layers = [_rec("Conv", [-1]), _rec("Conv", [0]), _rec("Conv", [1]), _rec("Conv", [2]), _rec("Dense", [3])]
        sizes = {0: 100, 1: 52, 2: 100, 3: 12, 4: 4}
        plan = self.plan(layers, sizes)
        self.assertEqual(plan.arena_size, 152)
        self.assertEqual(plan.unshared_size, 264)

    def test_element_wise_layer_overwrites_its_dying_input(self):
        layers = [_rec("Conv", [-1]), _rec("Sigmoid", [0]), _rec("Dense", [1])]
        plan = self.plan(layers, {0: 40, 1: 40, 2: 4})
        self.assertEqual(plan.offsets[0], plan.offsets[1])
        self.assertEqual(plan.arena_size, 40)
        off = self.plan(layers, {0: 40, 1: 40, 2: 4}, in_place=False)
        self.assertEqual(off.arena_size, 80)

    def test_element_wise_layer_keeps_an_input_read_later(self):
        # conv -> relu -> add(relu, conv): the conv output is still needed.
        layers = [_rec("Conv", [-1]), _rec("Relu", [0]), _rec("Add", [1, 0]), _rec("Dense", [2])]
        plan = self.plan(layers, {0: 40, 1: 40, 2: 40, 3: 4})
        self.assertNotEqual(plan.offsets[0], plan.offsets[1])
        # The add may overwrite either input, both die there.
        self.assertIn(plan.offsets[2], (plan.offsets[0], plan.offsets[1]))

    def test_alignment(self):
        layers = [_rec("Conv", [-1]), _rec("Conv", [0]), _rec("Conv", [1]), _rec("Dense", [2])]
        plan = self.plan(layers, {0: 5, 1: 7, 2: 3, 3: 1}, alignment=8)
        self.assertTrue(all(o % 8 == 0 for o in plan.offsets.values()))
        self.assertEqual(plan.arena_size, 16)

    def test_random_graphs(self):
        rng = np.random.default_rng(3)
        for trial in range(400):
            n = int(rng.integers(2, 14))
            layers, sizes = [], {}
            for i in range(n):
                op = str(rng.choice(["Conv", "Relu", "Sigmoid", "Add", "MaxPool"]))
                if op == "Add" and i >= 1:
                    srcs = [int(rng.integers(-1, i)), int(rng.integers(-1, i))]
                    sizes[i] = sizes.get(srcs[0], 0) or int(rng.integers(1, 200))
                    # an Add needs equal input sizes; make the second match the first
                    if srcs[1] >= 0 and sizes.get(srcs[1]) != sizes[i]:
                        srcs[1] = srcs[0]
                    if srcs[0] < 0:
                        sizes[i] = sizes.setdefault(-1, int(rng.integers(1, 200)))
                        if srcs[1] >= 0 and sizes[srcs[1]] != sizes[i]:
                            srcs[1] = -1
                elif op in ("Relu", "Sigmoid") and i >= 1:
                    srcs = [int(rng.integers(-1, i))]
                    sizes[i] = sizes.setdefault(srcs[0], int(rng.integers(1, 200)))
                else:
                    op = "Conv"
                    srcs = [int(rng.integers(-1, i)) if i else -1]
                    sizes[i] = int(rng.integers(1, 200))
                layers.append(_rec(op, srcs, name=f"l{i}"))
            for in_place in (True, False):
                with self.subTest(trial=trial, in_place=in_place):
                    self.plan(layers, sizes, in_place=in_place)

    def test_simulator_catches_a_broken_plan(self):
        """The checks above must be able to fail: put every buffer at offset 0."""
        layers = [_rec("Conv", [-1]), _rec("Conv", [0]), _rec("Add", [1, 0]), _rec("Dense", [2])]
        sizes = {0: 16, 1: 16, 2: 16, 3: 4}
        plan = cg.plan_memory(layers, sizes)
        for b in plan.buffers:
            b.offset = 0
        plan.offsets = {i: 0 for i in plan.offsets}
        self.assertIsNotNone(simulate_plan(layers, sizes, plan))
        with self.assertRaises(AssertionError):
            cg.check_plan(plan)


BUILDER_SOURCE = r"""
#include <cstdint>
#include <cstdio>
#include "qaffine.hpp"
#include "qactivations.hpp"
#include "qcalibration.hpp"

int main()
{
    float s_in, s_out, s_b;
    int zp_in, zp_out;
    while (std::scanf("%f %d %f %d %f", &s_in, &zp_in, &s_out, &zp_out, &s_b) == 5)
    {
        int8_t lut[256];
        tinymind::buildQSigmoidLUT(s_in, zp_in, s_out, zp_out, lut);
        for (int i = 0; i < 256; ++i) { std::printf("%d ", lut[i]); }
        tinymind::buildQTanhLUT(s_in, zp_in, s_out, zp_out, lut);
        for (int i = 0; i < 256; ++i) { std::printf("%d ", lut[i]); }
        const tinymind::QAddParams a = tinymind::buildQAddParams(s_in, s_b, s_out);
        std::printf("%d %d %d %d %d %d %d ", a.left_shift, a.input_a_multiplier, a.input_a_shift,
                    a.input_b_multiplier, a.input_b_shift, a.output_multiplier, a.output_shift);
        int32_t m = 0, l = 0, d = 0;
        tinymind::buildQSoftmaxFixedPointParams(s_in, 1.0, m, l, d);
        std::printf("%d %d %d\n", m, l, d);
    }
    return 0;
}
"""


def _compile(src_text, workdir, tag, defines):
    src = os.path.join(workdir, f"{tag}.cpp")
    with open(src, "w") as f:
        f.write(src_text)
    exe = os.path.join(workdir, tag)
    subprocess.run([GXX, "-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror", "-Wpedantic"] + defines +
                   ["-I" + os.path.join(REPO, "cpp"), "-I" + os.path.join(REPO, "cpp", "include"),
                    "-I" + os.path.join(REPO, "include"), "-I" + workdir, src, "-o", exe], check=True)
    return exe


@unittest.skipIf(GXX is None, "needs g++")
class BuilderMirrors(unittest.TestCase):
    """Runs without onnx: the constants model.hpp bakes in equal what the
    host-only C++ builders compute for the same scales."""

    def test_luts_and_parameters_match_the_cpp_builders(self):
        rng = np.random.default_rng(11)
        cases = [(0.05, 3, 1 / 256, -128, 0.02), (1 / 16, 0, 1 / 128, 0, 1 / 16), (0.2, -128, 1 / 256, -128, 0.7)]
        for _ in range(300):
            cases.append((float(np.float32(10 ** rng.uniform(-3.5, 0))), int(rng.integers(-128, 128)),
                          float(np.float32(10 ** rng.uniform(-3, -1.5))), int(rng.integers(-128, 128)),
                          float(np.float32(10 ** rng.uniform(-3.5, 0)))))
        with tempfile.TemporaryDirectory() as d:
            exe = _compile(BUILDER_SOURCE, d, "builders",
                           ["-DTINYMIND_ENABLE_QUANTIZATION=1", "-DTINYMIND_ENABLE_FLOAT=1", "-DTINYMIND_ENABLE_STD=1"])
            stdin = "".join(f"{a!r} {b} {c!r} {e} {f!r}\n" for a, b, c, e, f in cases)
            out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
        rows = out.strip().split("\n")
        self.assertEqual(len(rows), len(cases))
        for (s_in, zp_in, s_out, zp_out, s_b), row in zip(cases, rows):
            v = [int(x) for x in row.split()]
            self.assertEqual(cg.activation_lut("Sigmoid", s_in, zp_in, s_out, zp_out), v[:256], (s_in, zp_in))
            self.assertEqual(cg.activation_lut("Tanh", s_in, zp_in, s_out, zp_out), v[256:512], (s_in, zp_in))
            a = cg.qadd_params(s_in, s_b, s_out)
            self.assertEqual([a["left_shift"], a["input_a_multiplier"], a["input_a_shift"], a["input_b_multiplier"],
                              a["input_b_shift"], a["output_multiplier"], a["output_shift"]], v[512:519])
            self.assertEqual(list(cg.softmax_params(s_in)), v[519:522])


MODEL_DRIVER = r"""
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "@HEADER@"

int main()
{
    static int8_t input[@NS@::kModelInputSize];
    static int8_t output[@NS@::kModelOutputSize];
    alignas(@NS@::kModelArenaAlignment) static int8_t arena[@NS@::kModelArenaSize > 0 ? @NS@::kModelArenaSize : 1];
    // Garbage in the arena: the model must not read a byte it did not write.
    std::memset(arena, 0x5A, sizeof(arena));
    for (std::size_t i = 0; i < @NS@::kModelInputSize; ++i)
    {
        int v;
        if (std::scanf("%d", &v) != 1) { return 2; }
        input[i] = static_cast<int8_t>(v);
    }
    @NS@::forward(input, output, arena);
    for (std::size_t i = 0; i < @NS@::kModelOutputSize; ++i)
    {
        std::printf("%d\n", static_cast<int>(output[i]));
    }
    return 0;
}
"""


def compile_model(layers, workdir, tag, in_place=True, extra=()):
    """Generate weights.hpp + model.hpp and build a driver around forward().
    Built freestanding (FLOAT=0, STD=0): the generated model must not need
    the host-only builders."""
    weights = os.path.join(workdir, f"{tag}_weights.hpp")
    model = os.path.join(workdir, f"{tag}_model.hpp")
    ti.emit_weights_header(weights, tag, layers)
    generated = cg.emit_model_header(model, tag, layers, os.path.basename(weights), in_place=in_place)
    src = MODEL_DRIVER.replace("@HEADER@", os.path.basename(model)).replace("@NS@", tag)
    exe = _compile(src, workdir, f"{tag}_driver",
                   ["-DTINYMIND_ENABLE_QUANTIZATION=1", "-DTINYMIND_ENABLE_FLOAT=0", "-DTINYMIND_ENABLE_STD=0"]
                   + list(extra))
    return exe, generated


def run_exes(exes, q_inputs):
    outs = [[] for _ in exes]
    for q in q_inputs:
        stdin = "\n".join(str(v) for v in q.reshape(-1).tolist()) + "\n"
        for k, exe in enumerate(exes):
            out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
            outs[k].append([int(v) for v in out.split()])
    return [np.array(o) for o in outs]


def nhwc_inputs(shape, layer, n=24, seed=7):
    rng = np.random.default_rng(seed)
    s_in, zp_in = layer.input_scale, layer.input_zp
    qs = []
    for i in range(n):
        if i % 2:
            q = np.clip(np.rint(0.5 * rng.standard_normal(shape) / s_in) + zp_in, -128, 127).astype(np.int64)
        else:
            q = rng.integers(-128, 128, size=shape).astype(np.int64)
        qs.append(np.transpose(q, (0, 2, 3, 1)) if q.ndim == 4 else q)
    return qs


if HAVE_STACK:
    class ActNet(nn.Module):
        """Linear -> Sigmoid -> Linear -> Tanh -> Linear: standalone LUT
        activations, which no compute layer absorbs."""

        def __init__(self):
            super().__init__()
            self.a = nn.Linear(6, 8)
            self.b = nn.Linear(8, 8)
            self.c = nn.Linear(8, 4)

        def forward(self, x):
            return self.c(torch.tanh(self.b(torch.sigmoid(self.a(x)))))


@unittest.skipUnless(HAVE_STACK, "needs torch, onnx and onnxruntime")
@unittest.skipIf(GXX is None, "needs g++")
class ModelCodegen(unittest.TestCase):
    """model.hpp must compute exactly what a hand-wired program computes from
    the same records, with every activation in one planned arena."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workdir = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_graph(self, fmt, per_channel, softmax):
        torch.manual_seed(1)
        tag = f"cg_res_{fmt}_{int(per_channel)}_{int(softmax)}"
        path = export_and_quantize(ResNetish(softmax), GRAPH_INPUT_SHAPE, self.workdir, fmt, per_channel, tag)
        layers = ti.parse_onnx_model(path)
        hand = compile_graph_harness(layers, self.workdir, tag + "_hand")
        exes = {}
        plans = {}
        for in_place in (True, False):
            exes[in_place], gen = compile_model(layers, self.workdir, f"{tag}_{int(in_place)}", in_place)
            plans[in_place] = gen.plan
            self.assertIsNone(simulate_plan(layers, gen.sizes, gen.plan))
        want, got, got_no_in_place = run_exes([hand, exes[True], exes[False]],
                                              nhwc_inputs(GRAPH_INPUT_SHAPE, layers[0]))
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got_no_in_place, want)
        self.assertGreater(len(np.unique(want)), 4, "outputs must vary, or the comparison is empty")
        p = plans[True]
        print(f"\n  model.hpp residual {fmt:9s} per_channel={per_channel!s:5s} softmax={softmax!s:5s}: "
              f"arena {p.arena_size} B (unshared {p.unshared_size} B, busiest layer {p.lower_bound} B, "
              f"no in-place {plans[False].arena_size} B), bit-exact with hand wiring")
        self.assertLess(p.arena_size, p.unshared_size)
        # The first convolution's output and the pooled copy of it are live
        # together and no plan can do better; the Add writes over its input.
        self.assertEqual(p.arena_size, p.lower_bound)
        self.assertEqual(len(p.buffers), len(plans[False].buffers) - 1, "the residual Add should run in place")

    def test_residual_qdq_per_tensor(self):
        self.check_graph("QDQ", False, False)

    def test_residual_qdq_per_channel(self):
        self.check_graph("QDQ", True, False)

    def test_residual_qoperator_per_tensor(self):
        self.check_graph("QOperator", False, False)

    def test_residual_qoperator_per_channel(self):
        self.check_graph("QOperator", True, True)

    def test_residual_softmax_qdq(self):
        self.check_graph("QDQ", False, True)

    def test_chain_matches_hand_wiring(self):
        for fmt, per_channel in (("QDQ", False), ("QOperator", True)):
            tag = f"cg_chain_{fmt}_{int(per_channel)}"
            torch.manual_seed(0)
            path = export_and_quantize(ConvNet(), INPUT_SHAPE, self.workdir, fmt, per_channel, tag)
            layers = ti.parse_onnx_model(path)
            hand = compile_harness(layers, self.workdir, tag + "_hand")
            exe, gen = compile_model(layers, self.workdir, tag)
            want, got = run_exes([hand, exe], nhwc_inputs(INPUT_SHAPE, layers[0]))
            np.testing.assert_array_equal(got, want, err_msg=fmt)
            self.assertEqual(gen.plan.arena_size, gen.sizes[0] + gen.sizes[1])

    def test_lut_activations_match_onnx_runtime(self):
        for fmt in ("QDQ", "QOperator"):
            tag = f"cg_act_{fmt}"
            torch.manual_seed(2)
            path = export_and_quantize(ActNet(), (1, 6), self.workdir, fmt, False, tag)
            layers = ti.parse_onnx_model(path)
            kinds = [l.op_type for l in layers]
            self.assertIn("Sigmoid", kinds, f"{fmt}: {kinds}")
            self.assertIn("Tanh", kinds, f"{fmt}: {kinds}")
            exe, gen = compile_model(layers, self.workdir, tag)
            # Both activations overwrite the dense output they read.
            self.assertEqual(len(gen.plan.buffers), 2, f"{fmt}: {gen.plan.buffers}")
            ref, got = run_both(path, layers, exe, calibration_like=True)
            diff = np.abs(ref - got)
            print(f"\n  model.hpp sigmoid/tanh {fmt:9s}: max |ort - tinymind| = {diff.max()} LSB, "
                  f"{100 * np.mean(diff == 0):.1f}% exact, arena {gen.plan.arena_size} B")
            self.assertLessEqual(int(diff.max()), 2)
            self.assertGreater(len(np.unique(ref)), 4)

    def test_sanitizers_and_a_short_arena(self):
        """Under ASan and UBSan the model runs clean in exactly
        kModelArenaSize bytes, and ASan reports an arena one alignment step
        short (the arena in the driver is a static array of that size)."""
        torch.manual_seed(1)
        tag = "cg_asan"
        path = export_and_quantize(ResNetish(True), GRAPH_INPUT_SHAPE, self.workdir, "QDQ", True, tag)
        layers = ti.parse_onnx_model(path)
        flags = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
        exe, gen = compile_model(layers, self.workdir, tag, extra=flags)
        hand = compile_graph_harness(layers, self.workdir, tag + "_hand")
        q = nhwc_inputs(GRAPH_INPUT_SHAPE, layers[0], n=4)
        want, got = run_exes([hand, exe], q)
        np.testing.assert_array_equal(got, want)
        model = os.path.join(self.workdir, f"{tag}_model.hpp")
        text = open(model).read()
        size = f"kModelArenaSize      = {gen.plan.arena_size};"
        self.assertIn(size, text)
        with open(model, "w") as f:
            f.write(text.replace(size, f"kModelArenaSize      = {gen.plan.arena_size - 4};"))
        src = MODEL_DRIVER.replace("@HEADER@", os.path.basename(model)).replace("@NS@", tag)
        short = _compile(src, self.workdir, f"{tag}_short",
                         ["-DTINYMIND_ENABLE_QUANTIZATION=1"] + flags)
        stdin = "\n".join(str(v) for v in q[0].reshape(-1).tolist()) + "\n"
        res = subprocess.run([short], input=stdin, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("AddressSanitizer", res.stderr)

    def test_softmax_feeding_a_layer_is_refused(self):
        layers = [_rec("Softmax", [-1], "sm"), _rec("Relu", [0], "r")]
        layers[0].geometry = {}
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, "input|softmax"):
            cg.emit_model_header(os.path.join(d, "m.hpp"), "net", layers, "w.hpp")


def lround_ref(x):
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


if __name__ == "__main__":
    unittest.main()
