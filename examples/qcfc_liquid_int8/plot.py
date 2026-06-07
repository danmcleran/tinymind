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

"""int8 QCfC behavior: int8-vs-float hidden trajectory + per-step error."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "qcfc_parity.csv")


def _infer_lsb(vals):
    """Int8 hidden codes dequantize to integer multiples of one scale (the LSB),
    so the smallest gap between distinct observed values recovers it."""
    uniq = sorted(set(round(v, 9) for v in vals))
    gaps = [b - a for a, b in zip(uniq, uniq[1:]) if (b - a) > 1e-9]
    return min(gaps) if gaps else 0.0


def main():
    cols, _ = tp.read_csv(CSV)
    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("int8 Closed-form Continuous-time (QCfC) cell vs float reference",
                 fontsize=14, fontweight="bold")

    step = cols["step"]
    err = cols["max_abs_err"]
    lsb = _infer_lsb(cols["h0_int8"])

    # Left: the int8 hidden trajectory rendered as a quantization staircase
    # riding the smooth float curve -- the two track because the signal now
    # spans a real dynamic range, not because the plot hides the int8 grid.
    ax1.plot(step, cols["h0_float"], color=tp.PALETTE[0], marker="o",
             label="float h[0]")
    ax1.plot(step, cols["h0_int8"], color=tp.PALETTE[1], drawstyle="steps-mid",
             marker="s", markersize=4, label="int8 h[0] (quantized)")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("hidden state")
    ax1.set_title("recurrent state tracking", fontsize=10, color=tp.MUTED)
    ax1.legend()

    # Right: residual zoomed against the int8 LSB. The error sits at a few LSB,
    # so it is grid-resolution limited, not a modelling/convergence failure.
    ax2.plot(step, err, color=tp.PALETTE[2], marker="o",
             label="max |int8 - float|")
    if lsb > 0.0:
        for k in (1, 2, 4):
            ax2.axhline(k * lsb, color=tp.MUTED, lw=0.8, ls="--")
            ax2.text(step[-1], k * lsb, " %d LSB" % k, va="center",
                     ha="left", fontsize=8, color=tp.MUTED)
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("abs error")
    ax2.set_title("residual vs int8 LSB", fontsize=10, color=tp.MUTED)
    ax2.legend(loc="upper left")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
