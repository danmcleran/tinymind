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

"""Generated token sequences: float reference vs int8 greedy decode.

The CSV is the flattened (index, float, int8) token stream for NUM_PROMPTS
sequences of GEN tokens each. Each sequence is the autoregressive "+1 counter"
the nano-LM is wired to emit; the int8 decode is overlaid as markers on the
float line to show it reproduces the same tokens.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

NAME = os.path.basename(HERE)
CSV = os.path.join(HERE, "output", NAME + ".csv")
GEN = 12  # tokens per prompt (matches the source)


def main():
    cols, _ = tp.read_csv(CSV)
    fl, q = cols["float"], cols["int8"]
    n = len(fl)
    nseq = max(1, n // GEN)

    tp.apply_style()
    matches = sum(1 for a, b in zip(fl, q) if a == b)

    fig, axes = tp.plt.subplots(nseq, 1, figsize=(11, 7.2), sharex=True)
    if nseq == 1:
        axes = [axes]
    fig.suptitle("int8 greedy decode vs float  (%s)  --  %d/%d tokens match"
                 % (NAME, matches, n), fontsize=14, fontweight="bold")

    for s in range(nseq):
        ax = axes[s]
        seq_f = fl[s * GEN:(s + 1) * GEN]
        seq_q = q[s * GEN:(s + 1) * GEN]
        steps = list(range(1, len(seq_f) + 1))
        ax.step(steps, seq_f, where="mid", color=tp.PALETTE[0],
                linewidth=2.2, label="float reference" if s == 0 else None)
        ax.scatter(steps, seq_q, color=tp.PALETTE[1], s=44, zorder=3,
                   marker="x", label="int8 decode" if s == 0 else None)
        ax.set_ylabel("prompt %d\ntoken" % s, fontsize=9, color=tp.MUTED)
        ax.set_ylim(-0.7, 7.7)
    axes[-1].set_xlabel("generation step")
    axes[0].legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(CSV)
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
