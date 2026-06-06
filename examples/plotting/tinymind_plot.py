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

"""Shared plotting helpers for the TinyMind examples.

Every example writes a header-row CSV to its ``output/`` directory and ships a
small ``plot.py`` that turns that CSV into a graph. This module gives them a
single, consistent, good-looking style plus a few high-level chart helpers, so
the example scripts stay tiny and the graphs look uniform across the repo.

The CSV-first contract: the C++ side owns the numbers (header + rows), the
Python side only visualizes -- so users can drop the CSV into pandas / Excel /
their own tooling and ignore these scripts entirely.

Dependencies: matplotlib only (numpy optional). Headless-safe: if no display is
available the Agg backend is selected automatically, so ``make plot`` works over
SSH / CI and just writes the PNG.
"""

import csv
import os
import sys


# ---------------------------------------------------------------------------
# Backend selection: pick Agg when there is no display so saving still works.
# ---------------------------------------------------------------------------
def _select_backend():
    import matplotlib
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    return matplotlib


_select_backend()
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)


# A calm, high-contrast categorical palette (color-blind friendly-ish).
PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
    "#CCB974",  # gold
    "#64B5CD",  # cyan
]

_FG = "#222222"
_GRID = "#D9D9D9"


def apply_style():
    """Apply the shared TinyMind matplotlib style. Idempotent."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": _FG,
        "axes.labelcolor": _FG,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "grid.color": _GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "xtick.color": _FG,
        "ytick.color": _FG,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": _FG,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    """Read a header-row CSV. Returns (columns_dict, header_list).

    ``columns_dict`` maps each header name to a list of values (floats where
    every cell in the column parses as a number, else the raw strings).
    """
    if not os.path.exists(path):
        raise SystemExit("CSV not found: %s (run the example first)" % path)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    rows = [r for r in rows if r and not (len(r) == 1 and r[0].strip() == "")]
    if not rows:
        raise SystemExit("CSV is empty: %s" % path)
    header = [h.strip() for h in rows[0]]
    raw_cols = {h: [] for h in header}
    for r in rows[1:]:
        for i, h in enumerate(header):
            raw_cols[h].append(r[i].strip() if i < len(r) else "")
    cols = {}
    for h in header:
        floats = [_to_float(v) for v in raw_cols[h]]
        cols[h] = floats if all(v is not None for v in floats) else raw_cols[h]
    return cols, header


def new_fig(title, subtitle=None, figsize=(9, 5.2)):
    """Create a styled figure + axes with a title (and optional subtitle)."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    if subtitle:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        ax.set_title(subtitle, fontsize=10, fontweight="normal", color="#555555")
    else:
        ax.set_title(title, fontsize=14)
    return fig, ax


def line(ax, x, series, xlabel=None, ylabel=None, logy=False, markers=False):
    """Plot one or more y-series against x.

    ``series`` is a dict of {label: y_values}. ``x`` may be None (uses index).
    """
    for i, (label, ys) in enumerate(series.items()):
        xs = x if x is not None else list(range(len(ys)))
        ax.plot(xs, ys, label=label,
                marker="o" if markers else None,
                color=PALETTE[i % len(PALETTE)])
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if len(series) > 1 or next(iter(series)):
        ax.legend()


def bars(ax, labels, values, xlabel=None, ylabel=None, value_fmt="{:.0f}"):
    """Vertical bar chart with value annotations."""
    xs = list(range(len(labels)))
    rects = ax.bar(xs, values, color=[PALETTE[i % len(PALETTE)] for i in xs],
                   width=0.66, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    top = max(values) if values else 1.0
    for r, v in zip(rects, values):
        ax.text(r.get_x() + r.get_width() / 2.0, r.get_height() + top * 0.01,
                value_fmt.format(v), ha="center", va="bottom", fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.margins(y=0.15)


def finish(fig, out_path, show=None):
    """Tight-layout, save a PNG next to the data, and optionally show.

    ``show`` defaults to True only when a display is available and the user did
    not pass ``--no-show``; pass an explicit bool to override.
    """
    fig.tight_layout()
    fig.savefig(out_path)
    print("wrote %s" % out_path)
    if show is None:
        show = bool(os.environ.get("DISPLAY")) and ("--no-show" not in sys.argv)
    if show:
        plt.show()
    plt.close(fig)


def png_for(csv_path, suffix=""):
    """Default PNG path: alongside the CSV, same stem (+ optional suffix)."""
    base, _ = os.path.splitext(csv_path)
    return base + suffix + ".png"
