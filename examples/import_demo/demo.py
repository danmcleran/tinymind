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
Phase 15 import_demo: full PyTorch -> TinyMind importer round-trip.

Trains a tiny 3-8-4-2 MLP with sigmoid output on a deterministic synthetic
regression task, then drives the apps/import_pytorch/tinymind_import
module to emit weights.hpp + float_reference.hpp. The C++ side
(import_demo.cpp) consumes weights.hpp, runs a pure-integer forward
through the q*.hpp layer family, and checks parity against the float
reference stored in float_reference.hpp.

This script is the regenerate-weights entry point only; the C++ binary
shipped in this directory builds against the checked-in weights.hpp and
float_reference.hpp without needing PyTorch installed.

Dependencies: torch, numpy. Use pyenv to isolate the install (the system
Python is PEP 668-managed).
"""

from __future__ import annotations

import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
APPS = os.path.normpath(os.path.join(HERE, "..", "..", "apps", "import_pytorch"))
sys.path.insert(0, APPS)
from tinymind_import import (  # noqa: E402
    Dense, ReLU, Sigmoid, MinMaxObserver, PercentileObserver,
    import_pytorch_model,
)


SEED = 1337
N_IN = 3
N_H1 = 8
N_H2 = 4
N_OUT = 2
EPOCHS = 3000
LR = 0.05


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_H1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(N_H1, N_H2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(N_H2, N_OUT)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        return self.sigmoid(self.fc3(h2))


def synthetic_dataset(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic regression-style task: f(x) = sigmoid(linear)."""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1.0, 1.0, size=(n, N_IN)).astype(np.float32)
    Wgt = np.array([[1.2, -0.5, 0.3, 0.7],
                    [-0.4, 1.1, 0.6, -0.9],
                    [0.5,  0.2, -0.8, 1.0]], dtype=np.float32)[:, :N_OUT]
    yf = 1.0 / (1.0 + np.exp(-(X @ Wgt)))
    return torch.from_numpy(X), torch.from_numpy(yf.astype(np.float32))


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    crit = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1:>5d}  loss={loss.item():.6f}")


def write_float_reference(path: str, model: MLP, X: torch.Tensor) -> None:
    """Emit a C++ header with float reference inputs and outputs."""
    with torch.no_grad():
        preds = model(X).numpy()
    X_np = X.numpy()
    lines = ["/**",
             "* Copyright (c) 2026 Dan McLeran",
             "*",
             "* GENERATED FILE - regenerate via demo.py.",
             "*/",
             "",
             "#pragma once",
             "",
             "#include <cstddef>",
             "",
             "namespace import_demo {",
             "",
             f"    constexpr std::size_t NumSamples = {X_np.shape[0]};",
             f"    constexpr std::size_t NumIn      = {N_IN};",
             f"    constexpr std::size_t NumOut     = {N_OUT};",
             ""]
    lines.append(f"    constexpr float kInputs[NumSamples][NumIn] = {{")
    for row in X_np:
        cells = ", ".join(f"{v: .6f}f" for v in row.tolist())
        lines.append(f"        {{ {cells} }},")
    lines.append("    };")
    lines.append("")
    lines.append(f"    constexpr float kFloatRef[NumSamples][NumOut] = {{")
    for row in preds:
        cells = ", ".join(f"{v: .6f}f" for v in row.tolist())
        lines.append(f"        {{ {cells} }},")
    lines.append("    };")
    lines.append("")
    lines.append("} // namespace import_demo")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = MLP()
    Xtr, ytr = synthetic_dataset(256)
    train(model, Xtr, ytr)

    # Pull numpy weights out of the state dict. The importer module
    # takes plain numpy and does NOT depend on torch.
    fc1_w = model.fc1.weight.detach().numpy()
    fc1_b = model.fc1.bias.detach().numpy()
    fc2_w = model.fc2.weight.detach().numpy()
    fc2_b = model.fc2.bias.detach().numpy()
    fc3_w = model.fc3.weight.detach().numpy()
    fc3_b = model.fc3.bias.detach().numpy()

    def fc1_fwd(x): return x @ fc1_w.T + fc1_b
    def fc2_fwd(x): return x @ fc2_w.T + fc2_b
    def fc3_fwd(x): return x @ fc3_w.T + fc3_b

    layers = [
        Dense(name="fc1", weight=fc1_w, bias=fc1_b, input_name="input",
              forward=fc1_fwd, observer=MinMaxObserver()),
        ReLU(name="h1", input_name="fc1", observer=MinMaxObserver()),
        Dense(name="fc2", weight=fc2_w, bias=fc2_b, input_name="h1",
              forward=fc2_fwd, observer=MinMaxObserver()),
        ReLU(name="h2", input_name="fc2", observer=MinMaxObserver()),
        Dense(name="fc3", weight=fc3_w, bias=fc3_b, input_name="h2",
              forward=fc3_fwd,
              observer=PercentileObserver(0.05, 99.95)),
        Sigmoid(name="output", input_name="fc3"),
    ]

    cal = Xtr.numpy()
    dataset = [row for row in cal]
    ranges = import_pytorch_model(
        layers, input_name="input",
        input_observer=MinMaxObserver(),
        dataset=dataset,
        output_path=os.path.join(HERE, "weights.hpp"),
        namespace="import_demo",
        meta={"NumInputs": N_IN, "Hidden1": N_H1, "Hidden2": N_H2,
              "NumOutputs": N_OUT},
    )
    print("wrote weights.hpp")
    print("ranges:")
    for k, v in ranges.items():
        print(f"  {k:<8s}  scale={v[0]: .6g}  zp={v[1]}")

    # Write reference inputs + float predictions for the C++ parity check.
    test_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    X_test = Xtr.index_select(0, test_idx)
    write_float_reference(os.path.join(HERE, "float_reference.hpp"),
                          model, X_test)
    print("wrote float_reference.hpp")


if __name__ == "__main__":
    main()
