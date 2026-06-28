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

"""int8 GBDT decision-region map over the 2D feature grid (f0, f1, class)."""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

NAME = os.path.basename(HERE)
CSV = os.path.join(HERE, "output", NAME + ".csv")


def main():
    cols, _ = tp.read_csv(CSV)
    f0, f1, cls = cols["f0"], cols["f1"], cols["class"]
    n = len(cls)
    g = int(round(math.sqrt(n)))

    # CSV iterates iy (f1) outer, ix (f0) inner.
    grid = [[0] * g for _ in range(g)]
    for k in range(n):
        iy, ix = divmod(k, g)
        grid[iy][ix] = int(cls[k])

    tp.apply_style()
    fig, ax = tp.plt.subplots(figsize=(7.2, 6.4))
    fig.suptitle("int8 GBDT decision regions  (%s)" % NAME,
                 fontsize=14, fontweight="bold")

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([tp.PALETTE[0], tp.PALETTE[1], tp.PALETTE[3]])
    im = ax.imshow(grid, origin="lower", extent=(-1, 1, -1, 1),
                   cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto")
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    ax.set_title("argmax class over the feature grid (5 trees, 3 classes)",
                 fontsize=10, color=tp.MUTED)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["class 0", "class 1", "class 2"])

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV)
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
