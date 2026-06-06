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

"""Generic per-column viewer for a TinyMind NetworkPropertiesFileManager dump.

Reads the header-row file written by ``writeHeader`` / ``storeNetworkProperties``
(every tracked weight, bias, target, and output, one column per series, one row
per training iteration) and renders each numeric column as its own small panel
so the whole network's training trajectory is visible at a glance.

Usage:
    python3 nn_plot.py <data-file> [--no-show]

Robust to ragged rows (e.g. a trailing un-named error column): extra fields are
auto-named ``col<N>``. Saves a PNG next to the data file.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "examples", "plotting"))
import tinymind_plot as tp  # noqa: E402


def _read(path):
    with open(path) as f:
        lines = [ln.strip(" \r\n") for ln in f if ln.strip(" \r\n")]
    if not lines:
        raise SystemExit("empty data file: %s" % path)
    header = [h.strip() for h in lines[0].split(",") if h.strip() != ""]
    rows = []
    width = len(header)
    for ln in lines[1:]:
        parts = [p for p in ln.split(",") if p != ""]
        width = max(width, len(parts))
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(None)
        rows.append(vals)
    # Pad header for any extra unnamed columns (e.g. trailing error value).
    while len(header) < width:
        header.append("col%d" % len(header))
    return header, rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: python3 nn_plot.py <data-file> [--no-show]")
    path = args[0]
    if not os.path.exists(path):
        raise SystemExit("data file not found: %s" % path)

    header, rows = _read(path)
    series = []
    for c in range(len(header)):
        col = [r[c] for r in rows if c < len(r) and r[c] is not None]
        if col:
            series.append((header[c], col))

    tp.apply_style()
    n = len(series)
    cols = min(4, n) if n else 1
    grid_rows = int(math.ceil(n / float(cols))) if n else 1
    fig, axes = tp.plt.subplots(grid_rows, cols,
                                figsize=(3.2 * cols, 2.2 * grid_rows),
                                squeeze=False, sharex=True)
    fig.suptitle("TinyMind network training trajectory\n%s" %
                 os.path.basename(path), fontsize=13, fontweight="bold")
    for idx in range(grid_rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n:
            name, col = series[idx]
            ax.plot(range(len(col)), col, color=tp.PALETTE[idx % len(tp.PALETTE)])
            ax.set_title(name, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True)
        else:
            ax.axis("off")

    out = tp.png_for(path)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
