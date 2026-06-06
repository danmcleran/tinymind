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

"""Predictive-maintenance behavior: training loss + test confusion matrix."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "predictive_maintenance_loss.csv")
CM = os.path.join(HERE, "output", "predictive_maintenance_confusion.csv")


def main():
    loss, _ = tp.read_csv(LOSS)
    with open(CM, newline="") as f:
        rows = list(csv.reader(f))
    tn, fp = int(rows[1][1]), int(rows[1][2])
    fn, tp_ = int(rows[2][1]), int(rows[2][2])
    mat = [[tn, fp], [fn, tp_]]
    total = tn + fp + fn + tp_
    acc = (tn + tp_) / total if total else 0.0
    prec = tp_ / (tp_ + fp) if (tp_ + fp) else 0.0
    rec = tp_ / (tp_ + fn) if (tp_ + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Predictive maintenance (machine-failure classifier)",
                 fontsize=14, fontweight="bold")

    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color="#555")

    im = ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["pred no-fail", "pred fail"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["actual no-fail", "actual fail"])
    vmax = max(max(r) for r in mat)
    for r in range(2):
        for c in range(2):
            ax2.text(c, r, str(mat[r][c]), ha="center", va="center",
                     color="white" if mat[r][c] > vmax / 2 else "#222",
                     fontsize=12, fontweight="bold")
    ax2.set_title("test confusion  (acc=%.3f  P=%.3f  R=%.3f  F1=%.3f)"
                  % (acc, prec, rec, f1), fontsize=9, color="#555")
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CM, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
