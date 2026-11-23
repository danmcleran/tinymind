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

"""Generated ONNX model: the activation memory plan and the test confusion."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

RESULTS = os.path.join(HERE, "output", "onnx_digits_int8.csv")
PLAN = os.path.join(HERE, "output", "onnx_digits_int8_plan.csv")
SUMMARY = os.path.join(HERE, "output", "onnx_digits_int8_summary.csv")


def main():
    with open(PLAN, newline="") as f:
        plan = [dict(r) for r in csv.DictReader(f)]
    for r in plan:
        for k in ("offset", "size", "first_layer", "last_layer"):
            r[k] = int(r[k])
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))
    mat = [[0] * 10 for _ in range(10)]
    for r in rows:
        mat[int(r["label"])][int(r["tinymind"])] += 1
    acc = sum(mat[i][i] for i in range(10)) / len(rows)
    agree = sum(r["tinymind"] == r["onnxruntime"] for r in rows)

    with open(SUMMARY, newline="") as f:
        summary = {r["metric"]: int(r["value"]) for r in csv.DictReader(f)}
    arena, unshared, layers = summary["arena_bytes"], summary["unshared_bytes"], summary["layers"]

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("Generated int8 CNN from ONNX: one planned arena, 97% on optical digits",
                 fontsize=14, fontweight="bold")

    # 1) memory plan: each buffer is a box spanning the layers it is live for,
    #    at the arena bytes it occupies.
    for i, r in enumerate(sorted(plan, key=lambda r: r["first_layer"])):
        x0 = r["first_layer"] - 0.42
        width = r["last_layer"] - r["first_layer"] + 0.84
        color = tp.PALETTE[i % len(tp.PALETTE)]
        ax1.add_patch(tp.plt.Rectangle((x0, r["offset"]), width, r["size"], facecolor=color,
                                       edgecolor=tp.BG, linewidth=1.2, alpha=0.9))
        ax1.text(x0 + width / 2, r["offset"] + r["size"] / 2,
                 "%s\n%d B" % (r["outputs"].replace(" + ", " then "), r["size"]),
                 ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax1.axhline(arena, color=tp.FG, linewidth=1.0, linestyle="--")
    ax1.text(layers - 0.45, arena, "arena %d B; one buffer per output would need %d B" % (arena, unshared),
             ha="right", va="bottom", fontsize=9, color=tp.FG)
    ax1.set_xlim(-0.6, layers - 0.4)
    ax1.set_ylim(0, arena * 1.12)
    ax1.set_xticks(range(layers))
    ax1.set_xlabel("layer (graph order); a box spans the layers its output is live for")
    ax1.set_ylabel("arena offset (bytes)")
    ax1.set_title("activation memory plan", fontsize=10, color=tp.MUTED)
    ax1.grid(axis="x", visible=False)

    # 2) confusion matrix of the generated int8 model
    im = ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xlabel("predicted digit (tinymind int8)")
    ax2.set_ylabel("actual digit")
    vmax = max(max(r) for r in mat)
    for r in range(10):
        for c in range(10):
            if mat[r][c]:
                ax2.text(c, r, str(mat[r][c]), ha="center", va="center", fontsize=8, fontweight="bold",
                         color="white" if mat[r][c] > vmax / 2 else "#222")
    ax2.set_title("test confusion (acc %.3f; same class as ONNX Runtime on %d of %d)"
                  % (acc, agree, len(rows)), fontsize=10, color=tp.MUTED)
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    tp.finish(fig, tp.png_for(RESULTS))


if __name__ == "__main__":
    main()
