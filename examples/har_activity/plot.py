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

"""HAR LSTM behavior: training loss, confusion matrix, accelerometer traces."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "har_loss.csv")
CM = os.path.join(HERE, "output", "har_confusion.csv")
SAMPLES = os.path.join(HERE, "output", "har_samples.csv")
NAMES = ["walking", "upstairs", "sitting", "standing"]


def main():
    loss, _ = tp.read_csv(LOSS)
    with open(CM, newline="") as f:
        rows = list(csv.reader(f))
    n = len(NAMES)
    mat = [[int(rows[r + 1][c + 1]) for c in range(n)] for r in range(n)]
    total = sum(sum(r) for r in mat)
    acc = sum(mat[i][i] for i in range(n)) / total if total else 0.0
    samp, _ = tp.read_csv(SAMPLES)

    # Group sample rows by class -> {ax, ay, az traces over t}.
    classes = {}
    sclass = samp.get("class", [])
    with open(SAMPLES, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = row["class"]
            classes.setdefault(c, {"t": [], "ax": [], "ay": [], "az": []})
            classes[c]["t"].append(float(row["t"]))
            classes[c]["ax"].append(float(row["ax"]))
            classes[c]["ay"].append(float(row["ay"]))
            classes[c]["az"].append(float(row["az"]))
    _ = sclass

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("HAR activity classifier (Q16.16 LSTM 3-16-4)",
                 fontsize=14, fontweight="bold")

    # 1) training loss
    tp.line(ax1, loss["epoch"], {"avg |error|": loss["avg_err"]},
            xlabel="training epoch", ylabel="average |error| (last step)")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)

    # 2) confusion matrix
    im = ax2.imshow(mat, cmap="Blues")
    _ = im
    ax2.set_xticks(range(n)); ax2.set_xticklabels(NAMES, rotation=20, ha="right")
    ax2.set_yticks(range(n)); ax2.set_yticklabels(NAMES)
    ax2.set_xlabel("predicted"); ax2.set_ylabel("actual")
    vmax = max(max(r) for r in mat) if total else 1
    for r in range(n):
        for c in range(n):
            ax2.text(c, r, str(mat[r][c]), ha="center", va="center",
                     color="white" if mat[r][c] > vmax / 2 else "#222",
                     fontsize=12, fontweight="bold")
    ax2.set_title("test confusion  (acc=%.3f)" % acc, fontsize=10, color=tp.MUTED)
    ax2.grid(False)

    # 3) accelerometer traces (ax/ay/az) for one window of each activity.
    axis_style = {"ax": "-", "ay": "--", "az": ":"}
    for k, name in enumerate(NAMES):
        if name not in classes:
            continue
        tr = classes[name]
        color = tp.PALETTE[k]
        for axis, ls in axis_style.items():
            ax3.plot(tr["t"], tr[axis], ls, color=color, linewidth=1.4,
                     alpha=0.9, label=(name if axis == "ax" else None))
    ax3.set_xlabel("timestep"); ax3.set_ylabel("acceleration (norm.)")
    ax3.set_title("example traces  (solid=ax dash=ay dot=az)",
                  fontsize=10, color=tp.MUTED)
    ax3.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = tp.png_for(CM, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
