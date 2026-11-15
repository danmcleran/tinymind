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

"""import_lstm_demo: hidden-state trace of one sequence (float vs int16 and
int8 cell builds) and the head output, int8 against float, over every test
sequence."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "import_lstm_demo.csv")
TRACE = os.path.join(HERE, "output", "import_lstm_demo_trace.csv")
UNITS = (0, 3, 6)


def main():
    out, _ = tp.read_csv(CSV)
    tr, _ = tp.read_csv(TRACE)

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle("Imported nn.LSTM on QLSTMCell: int8 vs float  "
                 "(3-8 LSTM, 24 steps, 8-2 head)", fontsize=14, fontweight="bold")

    for k, unit in enumerate(UNITS):
        color = tp.PALETTE[k % len(tp.PALETTE)]
        rows = [i for i, (b, u) in enumerate(zip(tr["cell_bits"], tr["unit"]))
                if b == 16 and u == unit]
        steps = [tr["step"][i] for i in rows]
        ax1.plot(steps, [tr["float"][i] for i in rows], color=color,
                 linewidth=2.0, label="unit %d float" % unit)
        ax1.plot(steps, [tr["int8"][i] for i in rows], color=color,
                 linestyle="none", marker="o", markersize=4,
                 label="unit %d int8 (int16 cell)" % unit)
        rows8 = [i for i, (b, u) in enumerate(zip(tr["cell_bits"], tr["unit"]))
                 if b == 8 and u == unit]
        ax1.plot([tr["step"][i] for i in rows8], [tr["int8"][i] for i in rows8],
                 color=color, linestyle="none", marker="x", markersize=5,
                 label="unit %d int8 (int8 cell)" % unit)
    ax1.set_xlabel("step")
    ax1.set_ylabel("hidden state")
    ax1.set_title("hidden state, test sequence 0", fontsize=10, color=tp.MUTED)
    ax1.legend(fontsize=7, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.13))

    lo = min(min(out["float"]), min(out["int8"]))
    hi = max(max(out["float"]), max(out["int8"]))
    ax2.plot([lo, hi], [lo, hi], color=tp.MUTED, linestyle="--", linewidth=1.2,
             label="y = x")
    for k, bits in enumerate((16, 8)):
        idx = [i for i, b in enumerate(out["cell_bits"]) if b == bits]
        fl = [out["float"][i] for i in idx]
        q = [out["int8"][i] for i in idx]
        err = max(abs(a - b) for a, b in zip(fl, q))
        ax2.scatter(fl, q, s=30 if bits == 16 else 22,
                    color=tp.PALETTE[3 + k], edgecolor="white", linewidth=0.5,
                    marker="o" if bits == 16 else "s",
                    label="int%d cell (max err %.3f)" % (bits, err))
    ax2.set_xlabel("float model output")
    ax2.set_ylabel("int8 output, dequantized")
    ax2.set_title("head output, 16 test sequences x 2 outputs",
                  fontsize=10, color=tp.MUTED)
    ax2.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = tp.png_for(CSV)
    fig.savefig(png)
    print("wrote %s" % png)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
