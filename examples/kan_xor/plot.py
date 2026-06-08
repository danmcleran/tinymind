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

"""KAN XOR learning curve and decision surface."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CURVE_CSV = os.path.join(HERE, "output", "kan_xor_training.csv")
SURF_CSV = os.path.join(HERE, "output", "kan_xor_decision_surface.csv")
DENSE_SURF_CSV = os.path.join(HERE, "output", "kan_xor_dense_decision_surface.csv")


def plot_learning_curve():
    cols, _ = tp.read_csv(CURVE_CSV)
    fig, ax = tp.new_fig(
        "KAN XOR learning curve",
        "Kolmogorov-Arnold Network (2->5->1), learnable B-spline edges")
    tp.line(ax, cols["iteration"], {"average error": cols["avg_error"]},
            xlabel="training iteration", ylabel="average |error|", logy=True)
    tp.finish(fig, tp.png_for(CURVE_CSV))


def plot_decision_surface(csv, title, subtitle):
    cols, _ = tp.read_csv(csv)
    x0, prob = cols["x0"], cols["prob"]
    g = int(round(len(x0) ** 0.5))
    grid = [[0.0] * g for _ in range(g)]
    for k in range(len(prob)):
        grid[k // g][k % g] = prob[k]

    fig, ax = tp.new_fig(title, subtitle)
    im = ax.imshow(grid, origin="lower", extent=(0, 1, 0, 1),
                   cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.contour(
        [i / (g - 1) for i in range(g)], [i / (g - 1) for i in range(g)],
        grid, levels=[0.5], colors=tp.FG, linewidths=1.5)
    for (cx, cy, lbl) in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        ax.scatter([cx], [cy], s=160, edgecolor="white", linewidth=1.5,
                   color=("#C44E52" if lbl else "#4C72B0"), zorder=5)
    ax.set_xlabel("x0"); ax.set_ylabel("x1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(XOR=1)")
    tp.finish(fig, tp.png_for(csv))


def main():
    plot_learning_curve()
    if os.path.exists(SURF_CSV):
        plot_decision_surface(
            SURF_CSV, "KAN XOR decision surface (Q16.16)",
            "Trained on the 4 XOR corners (2->5->1)")
    if os.path.exists(DENSE_SURF_CSV):
        plot_decision_surface(
            DENSE_SURF_CSV, "KAN XOR decision surface -- dense training (Q16.16)",
            "Trained on dense [0,1]^2 samples (2->5->1)")


if __name__ == "__main__":
    main()
