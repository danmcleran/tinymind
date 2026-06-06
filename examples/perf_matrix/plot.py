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

"""SIMD backend speed comparison from output/perf_report.csv (make report)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "perf_report.csv")


def main():
    cols, _ = tp.read_csv(CSV)
    backends = cols["active_backend"]
    conv = cols["conv_us_per_call"]
    dense = cols["dense_us_per_call"]

    tp.apply_style()
    fig, ax = tp.new_fig(
        "SIMD backend performance (int8 QConv2D + QDense)",
        "lower is faster; output_checksum is identical across backends (bit-exact)")

    xs = list(range(len(backends)))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], conv, width=w, label="QConv2D us/call",
           color=tp.PALETTE[0], edgecolor="white", linewidth=0.6)
    ax.bar([x + w / 2 for x in xs], dense, width=w, label="QDense us/call",
           color=tp.PALETTE[2], edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(backends, rotation=15, ha="right")
    ax.set_ylabel("microseconds / call")
    ax.legend()
    ax.margins(y=0.15)

    fig.tight_layout()
    out = tp.png_for(CSV)
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
