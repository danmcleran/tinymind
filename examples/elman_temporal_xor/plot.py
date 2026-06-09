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

"""Elman vs feed-forward MLP on temporal XOR: recurrent memory vs none."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "elman_temporal_xor.csv")


def main():
    cols, _ = tp.read_csv(CSV)
    fig, ax = tp.new_fig(
        "Temporal XOR: Elman (recurrent) vs feed-forward MLP",
        "target[t] = x[t] XOR x[t-1]; the net sees only x[t]. "
        "Elman recovers it from memory (~98%); the memoryless MLP sits at ~0.5 (~51%)")
    tp.line(ax, cols["step"],
            {"target (x[t] XOR x[t-1])": cols["target"],
             "Elman (recurrent)": cols["elman"],
             "MLP (no memory)": cols["mlp"]},
            xlabel="timestep", ylabel="output / target", markers=True)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
    tp.finish(fig, tp.png_for(CSV))


if __name__ == "__main__":
    main()
