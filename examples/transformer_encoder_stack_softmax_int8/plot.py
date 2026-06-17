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

"""int8 vs float parity overlay + per-element error (index,float,int8 CSV)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

NAME = os.path.basename(HERE)
CSV = os.path.join(HERE, "output", NAME + ".csv")


def main():
    cols, _ = tp.read_csv(CSV)
    idx, fl, q = cols["index"], cols["float"], cols["int8"]
    err = [abs(a - b) for a, b in zip(fl, q)]

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("int8 vs float parity  (%s)" % NAME,
                 fontsize=14, fontweight="bold")

    tp.line(ax1, idx, {"float reference": fl, "int8 dequantized": q},
            xlabel="output element", ylabel="value", markers=len(idx) <= 32)
    ax1.set_title("output overlay", fontsize=10, color=tp.MUTED)

    ax2.bar(idx, err, color=tp.PALETTE[3], width=0.9)
    ax2.set_xlabel("output element")
    ax2.set_ylabel("|int8 - float|")
    ax2.set_title("per-element quantization error", fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV)
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
