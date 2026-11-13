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

"""Gas sensor drift: training loss, the drift curve, batch-1 confusion matrix,
and the on-device adaptation comparison (gas_adapt.csv)."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "gas_loss.csv")
DRIFT = os.path.join(HERE, "output", "gas_drift.csv")
CM = os.path.join(HERE, "output", "gas_confusion.csv")
ADAPT = os.path.join(HERE, "output", "gas_adapt.csv")
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

    plot_adaptation()

    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


# Each method keeps its colour and marker in every panel. The unadapted model
# is the gray baseline; renorm_all is renorm's dashed reference. biases +
# DenseHead lands almost on DenseHead, so it is drawn first and dashed, with
# DenseHead's markers on top.
METHODS = [
    ("none", "no adaptation", tp.PALETTE[7], "-", "x"),
    ("renorm", "re-normalize (shots)", tp.PALETTE[0], "-", "s"),
    ("renorm_all", "re-normalize (whole batch)", tp.PALETTE[0], "--", "s"),
    ("bias", "backbone biases only", tp.PALETTE[4], "-", "v"),
    ("bias_head", "biases + DenseHead", tp.PALETTE[3], "--", "D"),
    ("head", "DenseHead (head only)", tp.PALETTE[1], "-", "^"),
    ("prototype", "PrototypeHead", tp.PALETTE[2], "-", "o"),
]


def plot_adaptation():
    if not os.path.exists(ADAPT):
        return
    rows = {}
    with open(ADAPT, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["method"], int(r["batch"]), int(r["shots"]))] = (
                float(r["accuracy_mean"]), float(r["accuracy_std"]),
                float(r["batch1_accuracy_mean"]))
    batches = sorted({k[1] for k in rows})
    shots = sorted({k[2] for k in rows})
    last = batches[-1]
    k_mid = 5 if 5 in shots else shots[len(shots) // 2]

    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(16, 4.9))
    fig.suptitle("Adapting the drifted classifier on the device "
                 "(mean of 10 draws of the labelled shots)",
                 fontsize=14, fontweight="bold")

    for key, label, color, style, marker in METHODS:
        ys = [rows[(key, b, k_mid)][0] for b in batches]
        sd = [rows[(key, b, k_mid)][1] for b in batches]
        ax1.plot(batches, ys, color=color, linestyle=style, marker=marker,
                 markersize=6, label=label)
        ax1.fill_between(batches, [y - d for y, d in zip(ys, sd)],
                         [y + d for y, d in zip(ys, sd)], color=color,
                         alpha=0.12, linewidth=0)
        ax2.plot(shots, [rows[(key, last, k)][0] for k in shots], color=color,
                 linestyle=style, marker=marker, markersize=6, label=label)
        ax3.plot(shots, [rows[(key, last, k)][2] for k in shots], color=color,
                 linestyle=style, marker=marker, markersize=6, label=label)

    ax1.set_xlabel("batch (month)")
    ax1.set_ylabel("accuracy on the rest of the batch")
    ax1.set_xticks(batches)
    ax1.set_ylim(0.75, 1.01)
    ax1.set_title("%d labelled samples per class" % k_mid, fontsize=10,
                  color=tp.MUTED)

    for ax in (ax2, ax3):
        ax.set_xscale("log")
        ax.set_xticks(shots)
        ax.set_xticklabels([str(k) for k in shots])
        ax.minorticks_off()
        ax.set_xlabel("labelled samples per class")
    ax2.set_ylim(0.75, 1.01)
    ax3.set_ylim(0.0, 1.03)
    ax2.set_ylabel("batch %d accuracy" % last)
    ax2.set_title("batch %d: accuracy against labelled samples" % last,
                  fontsize=10, color=tp.MUTED)
    ax3.set_ylabel("batch 1 accuracy")
    ax3.set_title("batch 1 accuracy after adapting to batch %d" % last,
                  fontsize=10, color=tp.MUTED)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS),
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out = tp.png_for(ADAPT, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
