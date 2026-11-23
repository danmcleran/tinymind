#
# Copyright (c) 2026 Dan McLeran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Train a small residual CNN on the UCI optical digits, quantize it with ONNX
Runtime, and generate the tinymind model the C++ example runs.

    make regenerate PY=<python with torch, onnx, onnxruntime, numpy>

Writes, next to this script:
  weights.hpp    int8 weights, biases and requantizers (tinymind_import_onnx)
  model.hpp      layer objects, memory plan and forward() (tinymind_model_codegen)
  model_int4.hpp the same model with its dense layer requantized to packed,
                 per-channel int4 weights, in namespace digits::int4
  reference.hpp  the input quantization table and, for every test digit, the
                 class the float model and ONNX Runtime's int8 model predict

The generated headers are committed, so building and running the example needs
no Python.
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "optical_digits")
sys.path.insert(0, os.path.join(HERE, "..", "..", "apps", "import_onnx"))
import tinymind_model_codegen as cg  # noqa: E402

SEED = 1
EPOCHS = 60
NAMESPACE = "digits"


def load(name):
    raw = np.loadtxt(os.path.join(DATA, name), delimiter=",", dtype=np.int64)
    pixels, labels = raw[:, :64], raw[:, 64]
    return pixels, labels


def to_float(pixels):
    return (pixels.astype(np.float32) / 16.0).reshape(-1, 1, 8, 8)


class DigitNet(nn.Module):
    """Conv 3x3 -> residual block (conv 3x3, added to its input) -> max pool
    -> depthwise-separable block -> flatten -> dense."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw = nn.Conv2d(8, 16, 1)
        self.fc = nn.Linear(4 * 4 * 16, 10)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = h + torch.relu(self.c2(h))
        h = torch.nn.functional.max_pool2d(h, 2)
        h = torch.relu(self.pw(torch.relu(self.dw(h))))
        h = h.flatten(1)
        return self.fc(h)


class Reader(CalibrationDataReader):
    def __init__(self, x):
        self._it = iter([{"input": x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self._it, None)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    train_px, train_y = load("optdigits.tra")
    test_px, test_y = load("optdigits.tes")
    x_train, x_test = to_float(train_px), to_float(test_px)

    model = DigitNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(train_y)
    for epoch in range(EPOCHS):
        model.train()
        order = torch.from_numpy(rng.permutation(len(xt)))
        for start in range(0, len(xt), 64):
            idx = order[start:start + 64]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        float_pred = model(torch.from_numpy(x_test)).argmax(1).numpy()
    print(f"float test accuracy: {np.mean(float_pred == test_y):.4f}")

    fp32 = os.path.join(HERE, "output", "digits_fp32.onnx")
    int8 = os.path.join(HERE, "output", "digits_int8.onnx")
    os.makedirs(os.path.dirname(fp32), exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 1, 8, 8), fp32, input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamo=False)
    calib = x_train[rng.permutation(len(x_train))[:512]]
    quantize_static(fp32, int8, Reader(calib), quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8, per_channel=True)

    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    ort_pred = np.array([int(np.argmax(sess.run(None, {"input": x_test[i:i + 1]})[0]))
                         for i in range(len(x_test))])
    print(f"ONNX Runtime int8 test accuracy: {np.mean(ort_pred == test_y):.4f}")

    layers, generated = cg.import_onnx_model_with_code(
        int8, os.path.join(HERE, "weights.hpp"), os.path.join(HERE, "model.hpp"), NAMESPACE)
    plan = generated.plan
    print(f"{len(layers)} layers: {[l.op_type for l in layers]}")
    print(f"arena {plan.arena_size} B (unshared {plan.unshared_size} B, busiest layer {plan.lower_bound} B)")
    int4 = cg.emit_model_header(os.path.join(HERE, "model_int4.hpp"), NAMESPACE, layers, "weights.hpp",
                                int4_dense=True, nested_namespace="int4")
    print(f"weights: {generated.weight_bytes} B int8, {int4.weight_bytes} B with the dense layer in int4")

    # The input grid: pixel p in 0..16 is the float p / 16, which ONNX
    # Runtime's QuantizeLinear rounds half to even onto the int8 grid.
    first = next(l for l in layers if -1 in l.sources)
    table = [int(np.clip(np.rint(np.float32(p / 16.0) / np.float32(first.input_scale)) + first.input_zp,
                         -128, 127)) for p in range(17)]

    def arr(name, values, ctype, per_line=32):
        rows = [", ".join(str(v) for v in values[i:i + per_line]) for i in range(0, len(values), per_line)]
        return f"    constexpr {ctype} {name}[{len(values)}] = {{\n        " + ",\n        ".join(rows) + "\n    };\n"

    with open(os.path.join(HERE, "reference.hpp"), "w") as f:
        f.write(cg.ti.COPYRIGHT_BLOCK.replace("tinymind_import_onnx.", "examples/onnx_digits_int8/train.py.")
                + "\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n"
                f"namespace {NAMESPACE} {{\n"
                "    // Pixel value 0..16 -> the model's int8 input grid.\n"
                + arr("kPixelToInput", table, "int8_t")
                + f"    constexpr std::size_t kTestCount = {len(test_y)};\n"
                "    // The class the float PyTorch model predicts for each test digit.\n"
                + arr("kFloatPrediction", float_pred.tolist(), "uint8_t")
                + "    // The class ONNX Runtime's int8 model predicts for each test digit.\n"
                + arr("kOrtPrediction", ort_pred.tolist(), "uint8_t")
                + f"}} // namespace {NAMESPACE}\n")


if __name__ == "__main__":
    main()
