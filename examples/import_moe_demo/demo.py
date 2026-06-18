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
Mixture-of-Experts importer round-trip (Phase 3).

Builds a deterministic top-1 MoE in float (4 inputs -> 3 experts -> 3 outputs),
calibrates it with the apps/import_pytorch importer, and emits weights.hpp +
moe_reference.hpp. The C++ side (import_moe_demo.cpp) CONSUMES the emitted
weights.hpp -- rebuilding cpp/qmoe.hpp::QMixtureOfExperts from the generated
router/expert constants -- and checks parity against the float reference.

No torch dependency: the float weights are generated with numpy directly, so
the model and the importer fully define the round-trip. The checked-in
weights.hpp / moe_reference.hpp let the C++ binary build without numpy.

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
    MoE, MinMaxObserver, PercentileObserver, moe_float_forward,
    affine_asymmetric, quantize_symmetric_weights, symmetric_weight_scale,
    import_pytorch_model,
)


SEED = 7
N_IN = 4
N_OUT = 3
N_EXPERTS = 3
N_CAL = 256
N_TEST = 12


def build_float_moe() -> MoE:
    """Deterministic float MoE. Router columns are biased so each expert owns
    a distinct slice of the input space; experts are distinct linear maps."""
    rng = np.random.default_rng(SEED)

    # Router: emphasize a different input dimension per expert so routing is
    # input-dependent and all three experts fire across the dataset.
    router_w = np.zeros((N_EXPERTS, N_IN), dtype=np.float32)
    router_w[0] = np.array([2.0, 0.0, 0.0, 0.5], dtype=np.float32)
    router_w[1] = np.array([0.0, 2.0, 0.0, -0.5], dtype=np.float32)
    router_w[2] = np.array([0.0, 0.0, 2.0, 0.5], dtype=np.float32)
    router_b = np.array([0.1, 0.0, -0.1], dtype=np.float32)

    expert_w = [
        (rng.uniform(-1.0, 1.0, size=(N_OUT, N_IN)) * scale).astype(np.float32)
        for scale in (0.8, 1.0, 0.5)   # distinct dynamic ranges -> per-expert scale
    ]
    expert_b = [
        rng.uniform(-0.3, 0.3, size=(N_OUT,)).astype(np.float32)
        for _ in range(N_EXPERTS)
    ]

    return MoE(
        name="moe",
        input_name="input",
        router_weight=router_w,
        router_bias=router_b,
        expert_weights=expert_w,
        expert_biases=expert_b,
        observer=PercentileObserver(0.05, 99.95),
    )


def synthetic_dataset(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, N_IN)).astype(np.float32)


def write_reference(path: str, moe: MoE, X_test: np.ndarray) -> None:
    """Emit held-out test inputs, expected selected expert, and the top-1
    float output for the C++ parity check."""
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
        "namespace import_moe_demo {",
        "",
        f"    constexpr std::size_t NumTest = {X_test.shape[0]};",
        "",
        f"    constexpr float kTestInputs[NumTest][NumInputs] = {{",
    ]
    for row in X_test:
        cells = ", ".join(f"{v: .6f}f" for v in row.tolist())
        lines.append(f"        {{ {cells} }},")
    lines.append("    };")
    lines.append("")

    selected = []
    outputs = []
    for row in X_test:
        logits = row @ moe.router_weight.T + moe.router_bias
        e = int(np.argmax(logits))
        selected.append(e)
        outputs.append(moe_float_forward(moe, row))

    sel_cells = ", ".join(str(e) for e in selected)
    lines.append(f"    constexpr std::size_t kExpectedExpert[NumTest] = {{ {sel_cells} }};")
    lines.append("")
    lines.append(f"    constexpr float kFloatRef[NumTest][NumOutputs] = {{")
    for row in outputs:
        cells = ", ".join(f"{v: .6f}f" for v in np.asarray(row).tolist())
        lines.append(f"        {{ {cells} }},")
    lines.append("    };")
    lines.append("")
    lines.append("} // namespace import_moe_demo")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    moe = build_float_moe()

    X_cal = synthetic_dataset(N_CAL, seed=SEED)
    dataset = [row for row in X_cal]

    ranges = import_pytorch_model(
        [moe], input_name="input",
        input_observer=MinMaxObserver(),
        dataset=dataset,
        output_path=os.path.join(HERE, "weights.hpp"),
        namespace="import_moe_demo",
        meta={"NumInputs": N_IN, "NumOutputs": N_OUT, "NumExperts": N_EXPERTS},
    )
    print("wrote weights.hpp")
    print("ranges:")
    for k, v in ranges.items():
        print(f"  {k:<8s}  scale={v[0]: .6g}  zp={v[1]}")

    # Report expert routing distribution over the calibration set.
    counts = [0] * N_EXPERTS
    for row in X_cal:
        logits = row @ moe.router_weight.T + moe.router_bias
        counts[int(np.argmax(logits))] += 1
    print(f"calibration routing counts: {counts}")
    print(f"per-expert weight scales: {moe.expert_weight_scales}")

    X_test = synthetic_dataset(N_TEST, seed=SEED + 100)
    write_reference(os.path.join(HERE, "moe_reference.hpp"), moe, X_test)
    print("wrote moe_reference.hpp")


if __name__ == "__main__":
    main()
