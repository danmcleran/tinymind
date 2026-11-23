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

"""Fixed-point vs float log-mel spectrogram of the synthetic vibration run."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

BANDS = os.path.join(HERE, "output", "logmel_vibration.csv")
ERRORS = os.path.join(HERE, "output", "logmel_vibration_error.csv")


def main():
    with open(BANDS, newline="") as f:
        rows = list(csv.DictReader(f))
    nb = sum(1 for k in rows[0] if k.startswith("fixed_"))
    times = [float(r["time_s"]) for r in rows]
    fixed = [[float(r["fixed_%d" % m]) for r in rows] for m in range(nb)]
    flt = [[float(r["float_%d" % m]) for r in rows] for m in range(nb)]
    diff = [[a - b for a, b in zip(fa, fb)] for fa, fb in zip(fixed, flt)]
    fault_start = next(t for t, r in zip(times, rows) if r["fault"] == "1")
    with open(ERRORS, newline="") as f:
        err = list(csv.DictReader(f))

    tp.apply_style()
    fig, axes = tp.plt.subplots(2, 2, figsize=(13, 8.4))
    fig.suptitle("Fixed-point log-mel front end vs float, synthetic vibration (-50 to -6 dBFS)",
                 fontsize=14, fontweight="bold")
    extent = (times[0], times[-1], -0.5, nb - 0.5)
    lo = min(min(r) for r in flt)
    hi = max(max(r) for r in flt)
    for ax, data, title in ((axes[0][0], flt, "float reference: ln(mel energy)"),
                            (axes[0][1], fixed, "tinymind QLogMelSpectrogram<256, 32> (int16, Q8)")):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap="magma", vmin=lo, vmax=hi)
        ax.axvline(fault_start, color="white", linestyle="--", linewidth=1.0)
        ax.text(fault_start + 0.1, nb - 2, "bearing fault starts", color="white", fontsize=8, va="top")
        ax.set_xlabel("time (s); level rises 44 dB across the run")
        ax.set_ylabel("mel band (5 to 1,600 Hz)")
        ax.set_title(title, fontsize=10, color=tp.MUTED)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="nats")

    # Signed error, clipped to +-0.1 nats so the small values show.
    ax = axes[1][0]
    im = ax.imshow(diff, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel band")
    ax.set_title("fixed - float (nats, clipped to +-0.1)", fontsize=10, color=tp.MUTED)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="nats")

    ax = axes[1][1]
    labels = [r["db_below_peak"] + " dB" for r in err]
    xs = list(range(len(labels)))
    width = 0.38
    p99 = [float(r["p99_abs_error"]) for r in err]
    mx = [float(r["max_abs_error"]) for r in err]
    ax.bar([x - width / 2 for x in xs], p99, width, color=tp.PALETTE[0], label="99th percentile")
    ax.bar([x + width / 2 for x in xs], mx, width, color=tp.PALETTE[1], label="max")
    for x, v in zip(xs, mx):
        ax.text(x + width / 2, v, "%.3f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("band level below the frame's loudest band")
    ax.set_ylabel("|fixed - float| (nats)")
    ax.set_title("error grows only where the int16 FFT's floor is reached", fontsize=10, color=tp.MUTED)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tp.finish(fig, tp.png_for(BANDS))


if __name__ == "__main__":
    main()
