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

"""GRU XOR learning curve from output/gru_xor_results.csv (per-iteration error).

The raw per-iteration error is noisy (4 XOR patterns cycle); a moving average
makes the convergence trend legible alongside the raw signal.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "gru_xor_results.csv")


def moving_avg(xs, w):
    out, acc = [], 0.0
    from collections import deque
    win = deque()
    for x in xs:
        win.append(abs(x)); acc += abs(x)
        if len(win) > w:
            acc -= win.popleft()
        out.append(acc / len(win))
    return out


def main():
    cols, _ = tp.read_csv(CSV)
    it, err = cols["iteration"], cols["error"]
    fig, ax = tp.new_fig("GRU XOR learning curve",
                         "GRU recurrent net (2->3->1), Q-format")
    tp.line(ax, it, {"|error| (raw)": [abs(e) for e in err],
                     "moving avg (200)": moving_avg(err, 200)},
            xlabel="training iteration", ylabel="|error|")
    ax.lines[0].set_alpha(0.35)
    tp.finish(fig, tp.png_for(CSV))


if __name__ == "__main__":
    main()
