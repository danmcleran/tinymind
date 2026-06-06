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


def main():
    cols, _ = tp.read_csv(CSV)
    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("int8 Closed-form Continuous-time (QCfC) cell vs float reference",
                 fontsize=14, fontweight="bold")

    tp.line(ax1, cols["step"],
            {"float h[0]": cols["h0_float"], "int8 h[0]": cols["h0_int8"]},
            xlabel="time step", ylabel="hidden state", markers=True)
    ax1.set_title("recurrent state tracking", fontsize=10, color="#555")

    tp.line(ax2, cols["step"], {"max |int8 - float|": cols["max_abs_err"]},
            xlabel="time step", ylabel="abs error", markers=True)
    ax2.set_title("per-step quantization error", fontsize=10, color="#555")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
