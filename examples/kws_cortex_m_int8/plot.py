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

"""Per-layer cost report for a KWS-style pipeline (bench CSV bar charts)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

# CSV file name defaults to the example dir name; override with argv[1].
NAME = os.path.basename(HERE)
CSV = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else os.path.join(HERE, "output", NAME + ".csv")


def main():
    cols, _ = tp.read_csv(CSV)
    names = cols["name"]
    cycles = cols["cycles"]
    wbytes = cols["weight_bytes"]
    abytes = cols["activation_bytes"]

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("KWS pipeline per-layer cost  (%s)" % NAME,
                 fontsize=14, fontweight="bold")

    tp.bars(ax1, names, cycles, ylabel="cycles / call (host: ns)")
    ax1.set_title("compute cost per layer", fontsize=10, color="#555")

    xs = list(range(len(names)))
    ax2.bar(xs, wbytes, width=0.66, label="weight bytes",
            color=tp.PALETTE[0], edgecolor="white", linewidth=0.6)
    ax2.bar(xs, abytes, width=0.66, bottom=wbytes, label="activation bytes",
            color=tp.PALETTE[1], edgecolor="white", linewidth=0.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("bytes")
    ax2.set_title("memory footprint per layer", fontsize=10, color="#555")
    ax2.legend()
    ax2.margins(y=0.15)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV)
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
