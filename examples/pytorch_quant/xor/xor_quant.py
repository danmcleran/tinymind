"""
PyTorch -> TinyMind int8 quantization demo (XOR).

Trains a tiny 2-4-1 MLP (Linear -> ReLU -> Linear -> Sigmoid) in float32,
then runs post-training int8 quantization in the TinyMind / TFLite style:

  * Symmetric per-tensor weights (zero_point = 0, qmax = 127)
  * Asymmetric activations (zero_point in [-128, 127])
  * int32 biases held at scale = input_scale * weight_scale

Calibration ranges are observed by sweeping a float forward pass over a
9 x 9 grid covering [0, 1]^2 plus the four exact XOR corners. The script
emits weights.hpp containing all int8 weights, int32 biases, and per-tensor
(scale, zero_point) values; the C++ side rebuilds the Requantizers and the
qsigmoid LUT from those constants at startup.

The resulting C++ inference pipeline is pure-integer at runtime aside from
the host-side LUT / requantizer construction (which lives behind
TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD).
"""

from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 4
SEED = 42
EPOCHS = 4000
LR = 0.1


class XORNet(nn.Module):
    """Two-layer MLP for XOR (float32)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(h))

    def trace(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return all intermediate tensors for calibration."""
        z1 = self.fc1(x)
        h = self.relu(z1)
        z2 = self.fc2(h)
        y = self.sigmoid(z2)
        return {"input": x, "fc1_pre": z1, "hidden": h, "logit": z2, "output": y}


def xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, y


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    crit = nn.BCELoss()
    opt = optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1:>5d}  loss={loss.item():.6f}")


def calibration_grid() -> torch.Tensor:
    """Return a 9x9 grid over [0,1]^2 plus the four corners (deduped)."""
    pts = np.linspace(0.0, 1.0, 9)
    grid = np.array([[x, y] for x in pts for y in pts], dtype=np.float32)
    return torch.from_numpy(grid)


def lround(x: float) -> int:
    """C++ std::lround semantics: ties round away from zero.

    Python's built-in round() uses banker's rounding (ties to even); the
    TinyMind C++ side uses std::lround everywhere quantization rounding
    matters. Matching here keeps the calibration metadata produced by this
    script bit-identical to what the C++ Requantizer would expect.
    """
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def affine_asymmetric(fmin: float, fmax: float,
                      qmin: int = -128, qmax: int = 127) -> tuple[float, int]:
    """Match cpp/include/qcalibration.hpp::computeAffineParamsAsymmetric."""
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


def fmt_int_array(name: str, values: Iterable[int], cpp_type: str,
                  per_line: int = 12) -> str:
    vs = list(values)
    lines: list[str] = []
    for i in range(0, len(vs), per_line):
        chunk = ", ".join(f"{v:>5d}" for v in vs[i:i + per_line])
        lines.append("        " + chunk + ("," if i + per_line < len(vs) else ""))
    body = "\n".join(lines)
    return (f"    constexpr {cpp_type} {name}[{len(vs)}] = {{\n{body}\n    }};\n")


def emit_weights_header(path: str, model: XORNet,
                        ranges: dict[str, tuple[float, float]]) -> None:
    """Write weights.hpp consumed by xor_quant.cpp."""
    fc1_w = model.fc1.weight.detach().numpy()  # [hidden, 2]
    fc1_b = model.fc1.bias.detach().numpy()    # [hidden]
    fc2_w = model.fc2.weight.detach().numpy()  # [1, hidden]
    fc2_b = model.fc2.bias.detach().numpy()    # [1]

    input_scale, input_zp = affine_asymmetric(*ranges["input"])
    hidden_scale, hidden_zp = affine_asymmetric(*ranges["hidden"])
    logit_scale, logit_zp = affine_asymmetric(*ranges["logit"])
    sigmoid_scale = 1.0 / 256.0
    sigmoid_zp = -128

    fc1_w_scale = symmetric_weight_scale(fc1_w)
    fc2_w_scale = symmetric_weight_scale(fc2_w)

    q_fc1_w = quantize_symmetric_weights(fc1_w, fc1_w_scale).reshape(-1)
    q_fc2_w = quantize_symmetric_weights(fc2_w, fc2_w_scale).reshape(-1)
    q_fc1_b = quantize_bias_int32(fc1_b, input_scale * fc1_w_scale)
    q_fc2_b = quantize_bias_int32(fc2_b, hidden_scale * fc2_w_scale)

    fc1_w_lines = fmt_int_array("kFc1Weights", q_fc1_w.tolist(), "int8_t")
    fc1_b_lines = fmt_int_array("kFc1Biases", q_fc1_b.tolist(), "int32_t")
    fc2_w_lines = fmt_int_array("kFc2Weights", q_fc2_w.tolist(), "int8_t")
    fc2_b_lines = fmt_int_array("kFc2Biases", q_fc2_b.tolist(), "int32_t")

    body = f"""/**
* Copyright (c) 2026 Dan McLeran
*
* GENERATED FILE - do not edit by hand. Regenerate via xor_quant.py.
*
* Per-tensor int8 affine quantization metadata + weights for the XOR
* pytorch_quant example. Format mirrors TFLite / CMSIS-NN: symmetric
* per-tensor weights (zero_point = 0), asymmetric activations, int32
* biases at scale = input_scale * weight_scale.
*/

#pragma once

#include <cstddef>
#include <cstdint>

namespace pytorch_quant_xor {{

    constexpr std::size_t NumInputs   = 2;
    constexpr std::size_t HiddenSize  = {HIDDEN_SIZE};
    constexpr std::size_t NumOutputs  = 1;

    // Activation calibration (per-tensor, asymmetric int8).
    constexpr float   kInputScale         = {input_scale!r}f;
    constexpr int32_t kInputZeroPoint     = {input_zp};
    constexpr float   kHiddenScale        = {hidden_scale!r}f;
    constexpr int32_t kHiddenZeroPoint    = {hidden_zp};
    constexpr float   kLogitScale         = {logit_scale!r}f;
    constexpr int32_t kLogitZeroPoint     = {logit_zp};
    constexpr float   kSigmoidOutScale    = {sigmoid_scale!r}f;
    constexpr int32_t kSigmoidOutZeroPoint= {sigmoid_zp};

    // Weight calibration (per-tensor, symmetric int8, zero_point = 0).
    constexpr float kFc1WeightScale = {fc1_w_scale!r}f;
    constexpr float kFc2WeightScale = {fc2_w_scale!r}f;

{fc1_w_lines}
{fc1_b_lines}
{fc2_w_lines}
{fc2_b_lines}
}} // namespace pytorch_quant_xor
"""

    with open(path, "w") as f:
        f.write(body)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = XORNet()
    X, y = xor_data()
    train(model, X, y)

    with torch.no_grad():
        cal = calibration_grid()
        # Append the exact corners so the calibrated range is at least as
        # wide as the actual training distribution.
        cal_full = torch.cat([cal, X], dim=0)
        traced = model.trace(cal_full)
        ranges = {
            name: (float(t.min()), float(t.max()))
            for name, t in traced.items()
        }

    print("calibration ranges:")
    for name, (lo, hi) in ranges.items():
        print(f"  {name:<10s}  [{lo: .6f}, {hi: .6f}]")

    here = os.path.dirname(os.path.abspath(__file__))
    emit_weights_header(os.path.join(here, "weights.hpp"), model, ranges)
    print("\nwrote weights.hpp")

    # Print float reference predictions for sanity.
    with torch.no_grad():
        preds = model(X).squeeze(1).tolist()
    print("\nfloat XOR predictions:")
    for inp, p in zip(X.tolist(), preds):
        print(f"  {inp} -> {p:.4f}")


if __name__ == "__main__":
    main()
