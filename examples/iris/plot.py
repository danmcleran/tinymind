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

"""Iris classifier behavior: training loss, confusion matrix, petal scatter."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "iris_loss.csv")
CM = os.path.join(HERE, "output", "iris_confusion.csv")
SCAT = os.path.join(HERE, "output", "iris_scatter.csv")
NAMES = ["setosa", "versicolor", "virginica"]


def main():
    loss, _ = tp.read_csv(LOSS)
    with open(CM, newline="") as f:
        rows = list(csv.reader(f))
    mat = [[int(rows[r + 1][c + 1]) for c in range(3)] for r in range(3)]
    total = sum(sum(r) for r in mat)
    acc = sum(mat[i][i] for i in range(3)) / total if total else 0.0
    scat, _ = tp.read_csv(SCAT)

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Iris species classifier (Q16.16 MLP 4-8-3)",
                 fontsize=14, fontweight="bold")

    # 1) training loss
    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)

    # 2) confusion matrix
    im = ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(NAMES, rotation=20, ha="right")
    ax2.set_yticks(range(3)); ax2.set_yticklabels(NAMES)
    ax2.set_xlabel("predicted"); ax2.set_ylabel("actual")
    vmax = max(max(r) for r in mat)
    for r in range(3):
        for c in range(3):
            ax2.text(c, r, str(mat[r][c]), ha="center", va="center",
                     color="white" if mat[r][c] > vmax / 2 else "#222",
                     fontsize=12, fontweight="bold")
    ax2.set_title("test confusion  (acc=%.3f)" % acc, fontsize=10, color=tp.MUTED)
    ax2.grid(False)

    # 3) petal scatter colored by predicted class; X = misclassified
    pl, pw = scat["petal_len"], scat["petal_width"]
    tcl, pcl = scat["true_class"], scat["pred_class"]
    for k in range(3):
        xs = [pl[i] for i in range(len(pl)) if int(pcl[i]) == k]
        ys = [pw[i] for i in range(len(pw)) if int(pcl[i]) == k]
        ax3.scatter(xs, ys, s=40, color=tp.PALETTE[k], label=NAMES[k],
                    edgecolor="white", linewidth=0.5)
    miss_x = [pl[i] for i in range(len(pl)) if int(pcl[i]) != int(tcl[i])]
    miss_y = [pw[i] for i in range(len(pw)) if int(pcl[i]) != int(tcl[i])]
    if miss_x:
        ax3.scatter(miss_x, miss_y, s=120, marker="x", color="#C44E52",
                    label="misclassified", linewidth=2)
    ax3.set_xlabel("petal length (cm)"); ax3.set_ylabel("petal width (cm)")
    ax3.set_title("test predictions", fontsize=10, color=tp.MUTED)
    ax3.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = tp.png_for(CM, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
