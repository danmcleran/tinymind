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
LSTM importer round-trip.

Builds a deterministic float sequence model -- one nn.LSTM-shaped layer
(3 inputs, 8 hidden units, 24 steps) followed by a linear head (8 -> 2) --
calibrates it with the apps/import_pytorch importer, and emits:

  weights_cell16.hpp  the model with an int16 cell state (the default)
  weights_cell8.hpp   the same model with an int8 cell state
  lstm_reference.hpp  held-out test sequences, the float model's outputs and
                      the float hidden-state trace of the first sequence

The C++ side (import_lstm_demo.cpp) rebuilds cpp/qlstm.hpp::QLSTMCell and
cpp/qdense.hpp::QDense from the emitted constants and checks parity against
the float reference.

No torch dependency: the float weights are laid out exactly as nn.LSTM's
state_dict (weight_ih_l0 [4H, IN], weight_hh_l0 [4H, H], bias_ih_l0,
bias_hh_l0, gates i, f, g, o) and go through LSTM.from_state_dict, the path a
real model takes. The checked-in headers let the C++ binary build without
numpy.

Dependencies: numpy. Use the repo .venv-plot (or any numpy env); the system
Python is PEP 668-managed.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APPS = os.path.normpath(os.path.join(HERE, "..", "..", "apps", "import_pytorch"))
sys.path.insert(0, APPS)
from tinymind_import import (  # noqa: E402
    LSTM, Dense, MinMaxObserver, import_pytorch_model,
)


SEED = 11
N_IN = 3
N_HIDDEN = 8
N_OUT = 2
SEQ_LEN = 24
N_CAL = 128
N_TEST = 16


def float_state_dict() -> dict:
    """nn.LSTM(3, 8)-shaped parameters. Weights large enough that the gates
    move through their nonlinear range; a positive forget-gate bias (the
    usual initialization trick) so the cell state carries across steps."""
    rng = np.random.default_rng(SEED)
    k = 1.6 / np.sqrt(N_HIDDEN)
    w_ih = rng.uniform(-k, k, size=(4 * N_HIDDEN, N_IN)) * 2.0
    w_hh = rng.uniform(-k, k, size=(4 * N_HIDDEN, N_HIDDEN))
    b_ih = rng.uniform(-0.2, 0.2, size=4 * N_HIDDEN)
    b_hh = rng.uniform(-0.2, 0.2, size=4 * N_HIDDEN)
    b_hh[N_HIDDEN:2 * N_HIDDEN] += 1.0
    return {
        "weight_ih_l0": w_ih.astype(np.float32),
        "weight_hh_l0": w_hh.astype(np.float32),
        "bias_ih_l0": b_ih.astype(np.float32),
        "bias_hh_l0": b_hh.astype(np.float32),
    }


def float_head() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    w = rng.uniform(-0.8, 0.8, size=(N_OUT, N_HIDDEN)).astype(np.float32)
    b = rng.uniform(-0.1, 0.1, size=N_OUT).astype(np.float32)
    return w, b


def synthetic_sequences(n: int, seed: int) -> np.ndarray:
    """Noisy sinusoids, one frequency and phase per feature and sequence,
    in roughly [-1.2, 1.2]."""
    rng = np.random.default_rng(seed)
    t = np.arange(SEQ_LEN, dtype=np.float64)
    out = np.empty((n, SEQ_LEN, N_IN), dtype=np.float32)
    for s in range(n):
        for f in range(N_IN):
            freq = rng.uniform(0.05, 0.4)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rng.uniform(0.4, 1.0)
            out[s, :, f] = (amp * np.sin(2.0 * np.pi * freq * t + phase)
                            + rng.normal(0.0, 0.05, size=SEQ_LEN))
    return out


def head_forward(w: np.ndarray, b: np.ndarray, h: np.ndarray) -> np.ndarray:
    y = np.einsum("i,oi->o", h.astype(np.float64), w.astype(np.float64),
                  optimize=False) + b.astype(np.float64)
    return y.astype(np.float32)


def reference_lstm(state: dict, seq: np.ndarray) -> list[np.ndarray]:
    """The float model's hidden states, written straight from nn.LSTM's
    equations and state dict in float64 -- deliberately not through the
    importer's own float forward, so the parity check cannot pass by the
    importer agreeing with itself (e.g. both sides dropping bias_hh)."""
    w_ih = state["weight_ih_l0"].astype(np.float64)
    w_hh = state["weight_hh_l0"].astype(np.float64)
    b = state["bias_ih_l0"].astype(np.float64) + state["bias_hh_l0"].astype(np.float64)
    h = np.zeros(N_HIDDEN)
    c = np.zeros(N_HIDDEN)
    hs = []
    for x in seq.astype(np.float64):
        z = w_ih @ x + w_hh @ h + b
        i, f, g, o = np.split(z, 4)
        i, f, o = (1.0 / (1.0 + np.exp(-v)) for v in (i, f, o))
        c = f * c + i * np.tanh(g)
        h = o * np.tanh(c)
        hs.append(h.copy())
    return hs


def build_layers(cell_bits: int) -> list:
    """Fresh descriptors: observers are stateful, so each import gets its own."""
    w, b = float_head()
    lstm = LSTM.from_state_dict("lstm", float_state_dict(), input_name="input",
                                sequence_length=SEQ_LEN, cell_bits=cell_bits)
    head = Dense(name="head", weight=w, bias=b, input_name="lstm",
                 forward=lambda h: head_forward(w, b, h),
                 observer=MinMaxObserver())
    return [lstm, head]


def _f(v: float) -> str:
    return f"{float(np.float32(v))!r}f"


def write_reference(path: str, X_test: np.ndarray) -> None:
    state = float_state_dict()
    w, b = float_head()

    lines = [
        "/**",
        "* Copyright (c) 2026 Dan McLeran",
        "*",
        "* GENERATED FILE - regenerate via demo.py.",
        "*/",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "namespace import_lstm_demo {",
        "",
        f"    constexpr std::size_t NumInputs = {N_IN};",
        f"    constexpr std::size_t NumHidden = {N_HIDDEN};",
        f"    constexpr std::size_t NumOutputs = {N_OUT};",
        f"    constexpr std::size_t SequenceLength = {SEQ_LEN};",
        f"    constexpr std::size_t NumTest = {X_test.shape[0]};",
        "",
        "    // [sample][step][feature]",
        "    constexpr float kTestSequences[NumTest][SequenceLength][NumInputs] = {",
    ]
    for seq in X_test:
        lines.append("        {")
        for step in seq:
            lines.append("            { " + ", ".join(_f(v) for v in step) + " },")
        lines.append("        },")
    lines.append("    };")
    lines.append("")

    outputs = []
    trace = None
    for s, seq in enumerate(X_test):
        hs = reference_lstm(state, seq)
        outputs.append(head_forward(w, b, hs[-1]))
        if s == 0:
            trace = hs

    lines.append("    // Float model output for each test sequence.")
    lines.append("    constexpr float kFloatOutputs[NumTest][NumOutputs] = {")
    for row in outputs:
        lines.append("        { " + ", ".join(_f(v) for v in row) + " },")
    lines.append("    };")
    lines.append("")
    lines.append("    // Float hidden state after every step of test sequence 0.")
    lines.append("    constexpr float kFloatHiddenTrace[SequenceLength][NumHidden] = {")
    for h in trace:
        lines.append("        { " + ", ".join(_f(v) for v in h) + " },")
    lines.append("    };")
    lines.append("")
    lines.append("} // namespace import_lstm_demo")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    X_cal = synthetic_sequences(N_CAL, seed=SEED)
    for cell_bits in (16, 8):
        layers = build_layers(cell_bits)
        name = f"weights_cell{cell_bits}.hpp"
        ranges = import_pytorch_model(
            layers, input_name="input",
            input_observer=MinMaxObserver(),
            dataset=list(X_cal),
            output_path=os.path.join(HERE, name),
            namespace=f"import_lstm_demo_cell{cell_bits}",
            meta={"NumInputs": N_IN, "NumHidden": N_HIDDEN,
                  "NumOutputs": N_OUT, "SequenceLength": SEQ_LEN},
        )
        lstm = layers[0]
        print(f"wrote {name}")
        for k, v in ranges.items():
            print(f"  {k:<6s} scale={v[0]: .6g}  zp={v[1]}")
        print(f"  cell   scale={lstm.cell_scale: .6g}  "
              f"({cell_bits}-bit, range +-{lstm.cell_scale * lstm.cell_qmax:.4f})")

    X_test = synthetic_sequences(N_TEST, seed=SEED + 100)
    write_reference(os.path.join(HERE, "lstm_reference.hpp"), X_test)
    print("wrote lstm_reference.hpp")


if __name__ == "__main__":
    main()
