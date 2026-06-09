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

"""LSTM sinusoid: floating-point vs Q16.16, one-step-ahead and free-run."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CSV = os.path.join(HERE, "output", "lstm_sinusoid_float.csv")


def main():
    cols, _ = tp.read_csv(CSV)
    tp.apply_style()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle("LSTM sinusoid: float (double) vs Q16.16 fixed-point",
                 fontsize=14, fontweight="bold", y=0.98)

    ax_top.set_title("one-step-ahead (teacher forced) -- both track; float is tighter",
                     fontsize=10, color="#666666")
    tp.line(ax_top, cols["step"],
            {"ground truth": cols["true"],
             "float": cols["float_one_step"],
             "Q16.16": cols["q_one_step"]},
            ylabel="value", markers=False)

    ax_bot.set_title("auto-regressive free-run -- float holds the oscillation; "
                     "Q16.16 collapses",
                     fontsize=10, color="#666666")
    tp.line(ax_bot, cols["step"],
            {"ground truth": cols["true"],
             "float": cols["float_free_run"],
             "Q16.16": cols["q_free_run"]},
            xlabel="prediction step", ylabel="value", markers=False)

    tp.finish(fig, tp.png_for(CSV))


if __name__ == "__main__":
    main()
