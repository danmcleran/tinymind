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

"""Plot Q-learning maze state trajectories from a ragged path-trace file.

Each line is one episode: ``start_state, next_state, next_state, ...`` (comment
lines starting with '#' are ignored). One panel per distinct start state; each
episode's state progression is drawn vs. step number. Saves a PNG next to the
data file.

Usage:  python3 mazeplot.py <data-file> [--no-show]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "plotting"))
import tinymind_plot as tp  # noqa: E402


def load(path):
    by_start = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.split(",") if p.strip() != ""]
            try:
                states = [int(p, 10) for p in parts]
            except ValueError:
                continue
            if states:
                by_start.setdefault(states[0], []).append(states)
    return by_start


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: python3 mazeplot.py <data-file> [--no-show]")
    path = args[0]
    if not os.path.exists(path):
        raise SystemExit("data file not found: %s" % path)

    by_start = load(path)
    if not by_start:
        raise SystemExit("no episodes parsed from %s" % path)

    tp.apply_style()
    starts = sorted(by_start)
    n = len(starts)
    cols = min(3, n)
    rows = int(math.ceil(n / float(cols)))
    fig, axes = tp.plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows),
                                squeeze=False)
    fig.suptitle("Q-learning maze trajectories\n%s" % os.path.basename(path),
                 fontsize=13, fontweight="bold")

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            s = starts[i]
            for j, episode in enumerate(by_start[s]):
                ax.plot(range(len(episode)), episode, "-o",
                        color=tp.PALETTE[j % len(tp.PALETTE)],
                        markersize=4, alpha=0.85)
            ax.set_title("start state %d  (%d run%s)" %
                         (s, len(by_start[s]), "" if len(by_start[s]) == 1 else "s"),
                         fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel("state")
            ax.grid(True)
        else:
            ax.axis("off")

    out = tp.png_for(path)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
