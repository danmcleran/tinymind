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
        class PoolNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(2, 3, 3)

            def forward(self, x):
                return torch.nn.functional.max_pool2d(torch.relu(self.c(x)), 2)
        for fmt in ("QDQ", "QOperator"):
            path = export_and_quantize(PoolNet(), INPUT_SHAPE, self.workdir, fmt, False, "poolnet")
            with self.assertRaises(ValueError, msg=fmt):
                ti.parse_onnx_model(path)

    def test_grouped_conv_is_refused(self):
        class DwNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(2, 2, 3, groups=2)

            def forward(self, x):
                return self.c(x)
        path = export_and_quantize(DwNet(), INPUT_SHAPE, self.workdir, "QDQ", False, "dwnet")
        with self.assertRaisesRegex(ValueError, "group"):
            ti.parse_onnx_model(path)

    def test_uint8_is_refused(self):
        path = export_and_quantize(ConvNet(), INPUT_SHAPE, self.workdir, "QDQ", False, "u8",
                                   activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8)
        with self.assertRaisesRegex(ValueError, "uint8"):
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


def lround_ref(x):
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


if __name__ == "__main__":
    unittest.main()
