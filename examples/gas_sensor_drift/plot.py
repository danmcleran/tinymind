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

"""Gas sensor drift: training loss, the drift curve, batch-1 confusion matrix."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "gas_loss.csv")
DRIFT = os.path.join(HERE, "output", "gas_drift.csv")
CM = os.path.join(HERE, "output", "gas_confusion.csv")
NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
N = len(NAMES)


def main():
    loss, _ = tp.read_csv(LOSS)
    drift, _ = tp.read_csv(DRIFT)
    with open(CM, newline="") as f:
        rows = list(csv.reader(f))
    mat = [[int(rows[r + 1][c + 1]) for c in range(N)] for r in range(N)]
    total = sum(sum(r) for r in mat)
    acc = sum(mat[i][i] for i in range(N)) / total if total else 0.0

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Gas sensor array drift (Q16.16 MLP 128-32-6)",
                 fontsize=14, fontweight="bold")

    # 1) training loss (batch 1)
    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss (batch 1)", fontsize=10, color=tp.MUTED)

    # 2) the drift curve -- the headline plot
    tp.line(ax2, drift["batch"], {"accuracy": drift["accuracy"]},
            xlabel="batch (month)", ylabel="classification accuracy",
            markers=True)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xticks(drift["batch"])
    ax2.axhline(drift["accuracy"][0], color=tp.MUTED, linestyle="--",
                linewidth=1.0, alpha=0.7)
    ax2.set_title("DRIFT CURVE: accuracy decays on later batches",
                  fontsize=10, color=tp.MUTED)

    # 3) batch-1 validation confusion matrix
    im = ax3.imshow(mat, cmap="Blues")
    ax3.set_xticks(range(N)); ax3.set_xticklabels(NAMES, rotation=35, ha="right")
    ax3.set_yticks(range(N)); ax3.set_yticklabels(NAMES)
    ax3.set_xlabel("predicted"); ax3.set_ylabel("actual")
    vmax = max(max(r) for r in mat)
    for r in range(N):
        for c in range(N):
            ax3.text(c, r, str(mat[r][c]), ha="center", va="center",
                     color="white" if mat[r][c] > vmax / 2 else "#222",
                     fontsize=9, fontweight="bold")
    ax3.set_title("batch-1 confusion  (acc=%.3f)" % acc, fontsize=10,
                  color=tp.MUTED)
    ax3.grid(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = tp.png_for(DRIFT, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
