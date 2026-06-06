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

"""PINN heat-equation behavior: training loss / solution error + field fit.

Run the example first with training enabled:  make train
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

LOSS = os.path.join(HERE, "output", "pinn_loss.csv")
SOL = os.path.join(HERE, "output", "pinn_solution.csv")


def main():
    loss, _ = tp.read_csv(LOSS)
    sol, _ = tp.read_csv(SOL)

    tp.apply_style()
    fig, (ax1, ax2) = tp.plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Physics-Informed NN -- 1-D heat equation  u_t = nu * u_xx",
                 fontsize=14, fontweight="bold")

    tp.line(ax1, loss["epoch"],
            {"PINN residual loss": loss["loss"], "solution L2 error": loss["sol_l2"]},
            xlabel="epoch", ylabel="value", logy=True)
    ax1.set_title("training (exact-autodiff residual)", fontsize=10, color=tp.MUTED)

    tp.line(ax2, sol["x"], {
        "learned u(x, T/2)": sol["u_pred_mid"],
        "analytic u(x, T/2)": sol["u_analytic_mid"],
        "learned u(x, T)": sol["u_pred_end"],
        "analytic u(x, T)": sol["u_analytic_end"],
    }, xlabel="x", ylabel="u(x, t)")
    ax2.set_title("learned field vs analytic solution", fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(SOL, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
