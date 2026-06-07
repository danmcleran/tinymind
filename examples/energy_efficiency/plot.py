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

"""Energy Efficiency regression behavior: training loss + predicted-vs-actual."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "energy_loss.csv")
PRED = os.path.join(HERE, "output", "energy_pred.csv")


def _metrics(true, pred):
    n = len(true)
    mae = sum(abs(pred[i] - true[i]) for i in range(n)) / n
    mean_true = sum(true) / n
    ss_res = sum((pred[i] - true[i]) ** 2 for i in range(n))
    ss_tot = sum((true[i] - mean_true) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return mae, r2


def _scatter(ax, true, pred, name, color):
    mae, r2 = _metrics(true, pred)
    lo = min(min(true), min(pred))
    hi = max(max(true), max(pred))
    ax.plot([lo, hi], [lo, hi], color=tp.MUTED, linestyle="--",
            linewidth=1.2, label="y = x")
    ax.scatter(true, pred, s=30, color=color, edgecolor="white",
               linewidth=0.5, label="test samples")
    ax.set_xlabel("actual %s" % name)
    ax.set_ylabel("predicted %s" % name)
    ax.set_title("%s  (MAE=%.2f, R²=%.3f)" % (name, mae, r2),
                 fontsize=10, color=tp.MUTED)
    ax.legend()


def main():
    loss, _ = tp.read_csv(LOSS)
    pred, _ = tp.read_csv(PRED)

    tp.apply_style()
    fig, (ax1, ax2, ax3) = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Energy Efficiency regression (Q16.16 MLP 8-16-2)",
                 fontsize=14, fontweight="bold")

    # 1) training loss
    tp.line(ax1, loss["iter"], {"avg |error|": loss["avg_err"]},
            xlabel="training iteration", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)

    # 2) heating-load predicted-vs-actual
    _scatter(ax2, pred["y1_true"], pred["y1_pred"],
             "heating load", tp.PALETTE[3])

    # 3) cooling-load predicted-vs-actual
    _scatter(ax3, pred["y2_true"], pred["y2_pred"],
             "cooling load", tp.PALETTE[0])

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = tp.png_for(PRED, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
