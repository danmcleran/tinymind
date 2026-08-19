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

"""int8 ANFIS: tier parity, per-sample error, and the footprint it costs."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

PARITY = os.path.join(HERE, "output", "anfis_int8_parity.csv")
FOOTPRINT = os.path.join(HERE, "output", "anfis_int8_footprint.csv")

WINDOW = 200


def main():
    p, _ = tp.read_csv(PARITY)
    f, _ = tp.read_csv(FOOTPRINT)

    tp.apply_style()
    fig, axes = tp.plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("int8 ANFIS: accuracy holds, memory does not",
                 fontsize=14, fontweight="bold")

    # 1. Tier overlay. Q16.16 sits on the float reference; int8 is visibly
    # coarser, which is the honest picture.
    ax = axes[0]
    w = slice(0, WINDOW)
    tp.line(ax, p["t"][w], {"target": p["target"][w]},
            xlabel="held-out sample", ylabel="x(t+6)")
    ax.plot(p["t"][w][::4], p["int8"][w][::4], linestyle="none", marker="o",
            markersize=4.0, markerfacecolor="none", markeredgewidth=1.2,
            color=tp.PALETTE[1], label="int8")
    ax.legend()
    ax.set_title("held-out fit, first %d samples" % WINDOW,
                 fontsize=10, color=tp.MUTED)

    # 2. Per-sample deviation from the float reference, both integer tiers.
    ax = axes[1]
    dq = [abs(a - b) for a, b in zip(p["q16_16"], p["float"])]
    di = [abs(a - b) for a, b in zip(p["int8"], p["float"])]
    tp.line(ax, p["t"], {"|Q16.16 - float|": dq, "|int8 - float|": di},
            xlabel="held-out sample", ylabel="absolute deviation", logy=True)
    ax.set_title("deviation from the float reference", fontsize=10, color=tp.MUTED)

    # 3. The point of the exemplar.
    ax = axes[2]
    tp.bars(ax, list(f["tier"]), f["bytes"], xlabel="tier",
            ylabel="model bytes", value_fmt="{:.0f}")
    ax.set_title("int8 is %.1fx LARGER (grade lookup tables)"
                 % (max(f["bytes"]) / min(f["bytes"])),
                 fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = tp.png_for(PARITY, "_tiers")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
