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

"""CfC cell: training loss + held-out irregularly-sampled fit, with and without ts."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "cfc_loss.csv")
FIT = os.path.join(HERE, "output", "cfc_fit.csv")


def main():
    loss, _ = tp.read_csv(LOSS)
    fit, _ = tp.read_csv(FIT)

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Closed-form Continuous-time (CfC) cell: irregular sampling",
                 fontsize=14, fontweight="bold")

    tp.line(ax1, loss["epoch"], {"training loss": loss["loss"]},
            xlabel="epoch", ylabel="MSE loss", logy=True)
    ax1.set_title("reverse-mode autodiff training", fontsize=10, color=tp.MUTED)

    tp.line(ax2, fit["t"],
            {"target": fit["target"],
             "CfC (ts into time-gate)": fit["predicted"],
             "CfC, ts-blind": fit["predicted_ts_blind"]},
            xlabel="time (irregular samples)", ylabel="output", markers=True)
    ax2.set_title("held-out sequence: first-order response, ts varies 0.1 to 2.5",
                  fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(FIT, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
