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

    # QConv2D (~100s of us) and QDense (~0.05 us) differ by ~4 orders of
    # magnitude, so they get separate panels with their own y-scales instead of
    # one shared axis where the QDense bars vanish.
    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(13, 5.0))
    fig.suptitle("SIMD backend performance (int8, bit-exact across backends)",
                 fontsize=14, fontweight="bold")

    tp.bars(ax1, backends, conv, ylabel="microseconds / call",
            value_fmt="{:.0f}")
    ax1.set_title("QConv2D 3x3  (us/call, lower is faster)",
                  fontsize=10, color=tp.MUTED)

    tp.bars(ax2, backends, dense, ylabel="microseconds / call",
            value_fmt="{:.3f}")
    ax2.set_title("QDense  (us/call, lower is faster)",
                  fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV)
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
