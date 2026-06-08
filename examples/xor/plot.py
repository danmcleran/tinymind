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

"""Plot the XOR network learning curve and decision surface."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CURVE_CSV = os.path.join(HERE, "output", "xor_training.csv")
SURF_CSV = os.path.join(HERE, "output", "xor_decision_surface.csv")


def plot_learning_curve():
    cols, _ = tp.read_csv(CURVE_CSV)
    fig, ax = tp.new_fig(
        "XOR network learning curve",
        "Q16.16 fixed-point MLP (2->4->1), tanh hidden + sigmoid output")
    tp.line(ax, cols["iteration"], {"average error": cols["avg_error"]},
            xlabel="training iteration", ylabel="average |error|")
    tp.finish(fig, tp.png_for(CURVE_CSV))


def plot_decision_surface():
    cols, _ = tp.read_csv(SURF_CSV)
    x0, prob = cols["x0"], cols["prob"]
    g = int(round(len(x0) ** 0.5))
    grid = [[0.0] * g for _ in range(g)]
    for k in range(len(prob)):
        grid[k // g][k % g] = prob[k]

    fig, ax = tp.new_fig("XOR decision surface (Q16.16)",
                         "Fixed-point MLP (2->4->1) trained in TinyMind")
    im = ax.imshow(grid, origin="lower", extent=(0, 1, 0, 1),
                   cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.contour(
        [i / (g - 1) for i in range(g)], [i / (g - 1) for i in range(g)],
        grid, levels=[0.5], colors=tp.FG, linewidths=1.5)
    # XOR ground-truth corners (red = 1, blue = 0).
    for (cx, cy, lbl) in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        ax.scatter([cx], [cy], s=160, edgecolor="white", linewidth=1.5,
                   color=("#C44E52" if lbl else "#4C72B0"), zorder=5)
    ax.set_xlabel("x0"); ax.set_ylabel("x1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(XOR=1)")
    tp.finish(fig, tp.png_for(SURF_CSV))


def main():
    plot_learning_curve()
    if os.path.exists(SURF_CSV):
        plot_decision_surface()


if __name__ == "__main__":
    main()
