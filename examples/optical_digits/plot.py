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

"""Digit classifier behavior: training loss, confusion matrix, image grid."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "digits_loss.csv")
CM = os.path.join(HERE, "output", "digits_confusion.csv")
SAMPLES = os.path.join(HERE, "output", "digits_samples.csv")
GRID_PNG = os.path.join(HERE, "output", "digits_grid.png")
DIGITS = [str(d) for d in range(10)]


def main():
    loss, _ = tp.read_csv(LOSS)
    with open(CM, newline="") as f:
        rows = list(csv.reader(f))
    mat = [[int(rows[r + 1][c + 1]) for c in range(10)] for r in range(10)]
    total = sum(sum(r) for r in mat)
    acc = sum(mat[i][i] for i in range(10)) / total if total else 0.0

    tp.apply_style()

    # ---- main behavior figure: training loss + confusion matrix ----
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Optical digit classifier (Q16.16 MLP 64-32-10)",
                 fontsize=14, fontweight="bold")

    # 1) training loss
    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)

    # 2) confusion matrix
    im = ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks(range(10)); ax2.set_xticklabels(DIGITS)
    ax2.set_yticks(range(10)); ax2.set_yticklabels(DIGITS)
    ax2.set_xlabel("predicted digit"); ax2.set_ylabel("actual digit")
    vmax = max(max(r) for r in mat)
    for r in range(10):
        for c in range(10):
            v = mat[r][c]
            if v == 0:
                continue
            ax2.text(c, r, str(v), ha="center", va="center",
                     color="white" if v > vmax / 2 else "#222",
                     fontsize=8, fontweight="bold")
    ax2.set_title("test confusion  (acc=%.3f)" % acc, fontsize=10, color=tp.MUTED)
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = tp.png_for(CM, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)

    # ---- second figure: grid of sampled 8x8 digit images ----
    with open(SAMPLES, newline="") as f:
        srows = list(csv.reader(f))
    samples = []
    for row in srows[1:]:
        if len(row) < 66:
            continue
        px = [int(row[p]) for p in range(64)]
        true = int(row[64])
        pred = int(row[65])
        samples.append((px, true, pred))

    nrows, ncols = 5, 8
    gfig, axes = tp.plt.subplots(nrows, ncols, figsize=(11, 7))
    gfig.suptitle("Sampled test digits  (red = misclassified)",
                  fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if i >= len(samples):
            ax.axis("off")
            continue
        px, true, pred = samples[i]
        img = [px[r * 8:(r + 1) * 8] for r in range(8)]
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=16)
        wrong = (true != pred)
        ax.set_title("t=%d p=%d" % (true, pred), fontsize=9,
                     color="#C44E52" if wrong else tp.MUTED,
                     fontweight="bold" if wrong else "normal")

    gfig.tight_layout(rect=(0, 0, 1, 0.95))
    gfig.savefig(GRID_PNG)
    print("wrote %s" % GRID_PNG)

    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
