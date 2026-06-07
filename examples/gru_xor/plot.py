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

"""GRU XOR learning curve from output/gru_xor_results.csv.

Plots the deterministic 4-case XOR error (state reset before each case) sampled
through training -- a clean convergence signal, not the noisy per-sample error.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "gru_xor_results.csv")


def main():
    cols, _ = tp.read_csv(CSV)
    fig, ax = tp.new_fig("GRU XOR learning curve",
                         "GRU recurrent net (2->8->1), Q16.16, state reset per sample")
    tp.line(ax, cols["iteration"], {"4-case XOR error": cols["xor_error"]},
            xlabel="training iteration", ylabel="average |error|")
    tp.finish(fig, tp.png_for(CSV))


if __name__ == "__main__":
    main()
