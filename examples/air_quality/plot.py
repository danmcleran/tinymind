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

"""Air-quality LSTM forecaster behavior: training loss + one-step forecast."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "airq_loss.csv")
FC = os.path.join(HERE, "output", "airq_forecast.csv")


def main():
    loss, _ = tp.read_csv(LOSS)
    fc, _ = tp.read_csv(FC)

    true = fc["true"]
    pred = fc["predicted"]
    mae = sum(abs(t - p) for t, p in zip(true, pred)) / len(true) if true else 0.0

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Air-quality hourly LSTM forecaster (Q16.16, 1-16-1)",
                 fontsize=14, fontweight="bold")

    # 1) training loss over epochs
    tp.line(ax1, loss["epoch"], {"avg |error|": loss["avg_err"]},
            xlabel="epoch", ylabel="average |error|")
    ax1.set_title("training loss", fontsize=10, color=tp.MUTED)

    # 2) one-step-ahead forecast vs actual over the held-out hours
    tp.line(ax2, fc["hour"], {"true": true, "predicted": pred},
            xlabel="hour", ylabel="CO concentration (real units)")
    ax2.set_title("one-step-ahead forecast over held-out tail  (MAE=%.3f)" % mae,
                  fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = tp.png_for(FC, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
