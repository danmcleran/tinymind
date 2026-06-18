"""
Copyright (c) 2026 Dan McLeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""int8 MoE regime routing: nonlinear target vs the top-1 piecewise-linear
prediction, colored by which expert the router selected at each x."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

CSV = os.path.join(HERE, "output", "moe_regimes_int8.csv")


def main():
    cols, _ = tp.read_csv(CSV)
    x = cols["x"]
    target = cols["target"]
    pred = cols["prediction"]
    expert = [int(round(e)) for e in cols["expert"]]
    num_experts = max(expert) + 1

    fig, ax = tp.new_fig(
        "int8 Mixture-of-Experts: regime routing",
        subtitle="target f(x)=0.6 sin(1.5x)+0.25x  -  top-1 of 3 linear "
                 "experts (one runs per inference)")

    # Smooth target curve.
    ax.plot(x, target, color=tp.MUTED, lw=2.0, label="target f(x)", zorder=1)

    # Prediction colored by selected expert: each contiguous regime is the
    # output of a single expert, so the color blocks ARE the routing map.
    for e in range(num_experts):
        xs = [xi for xi, ei in zip(x, expert) if ei == e]
        ys = [yi for yi, ei in zip(pred, expert) if ei == e]
        ax.scatter(xs, ys, s=14, color=tp.PALETTE[e % len(tp.PALETTE)],
                   label="expert %d" % e, zorder=3)

    # Mark the routing boundaries (envelope crossings of the router lines).
    for boundary in (-1.0, 1.0):
        ax.axvline(boundary, color=tp.MUTED, lw=0.8, ls="--", zorder=2)

    ax.set_xlabel("input x")
    ax.set_ylabel("output")
    ax.legend(loc="upper left", ncol=2)

    tp.finish(fig, tp.png_for(CSV))


if __name__ == "__main__":
    main()
