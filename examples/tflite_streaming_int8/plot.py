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

"""Streaming int8 CNN from a .tflite file: the test recording's band
energies, the fault probability step() gives frame by frame, and the memory
step() needs against forward() over the whole window."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

RESULTS = os.path.join(HERE, "output", "tflite_streaming_int8.csv")
SUMMARY = os.path.join(HERE, "output", "tflite_streaming_int8_summary.csv")


def main():
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(SUMMARY, newline="") as f:
        summary = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
    bands = sorted((k for k in rows[0] if k.startswith("band")), key=lambda k: int(k[4:]))
    t = [float(r["seconds"]) for r in rows]
    energy = [[int(r[b]) for r in rows] for b in bands]
    valid = [r for r in rows if r["valid"] == "1"]
    tv = [float(r["seconds"]) for r in valid]
    p = [int(r["fault_per256"]) / 256.0 for r in valid]
    sev = [int(r["severity_percent"]) / 100.0 for r in rows]
    differing = int(summary["step_outputs_differing"])

    tp.apply_style()
    fig = tp.plt.figure(figsize=(14, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 1.1])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(grid[:, 1])
    fig.suptitle("A streaming int8 CNN from a .tflite file: one 20 ms frame per step(), every output equal to "
                 "TFLM's (%d of %d differ)" % (differing, 2 * int(summary["windows"])),
                 fontsize=14, fontweight="bold")

    # 1) the model's input: 16 int8 log band energies per frame
    im = ax1.imshow(energy, aspect="auto", origin="lower", cmap="magma",
                    extent=(t[0], t[-1], -0.5, len(bands) - 0.5))
    ax1.set_ylabel("band (0-25 Hz ... 1575-1600 Hz)")
    ax1.set_title("test recording: int8 log band energies, 50 frames per second", fontsize=10, color=tp.MUTED)
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.01, label="int8")

    # 2) fault probability per step, against the fault's true severity
    ax2.plot(t, sev, color=tp.PALETTE[2], linewidth=1.6, label="fault severity (truth)")
    ax2.plot(tv, p, color=tp.PALETTE[0], linewidth=1.2, label="fault probability from step()")
    ax2.axhline(0.5, color=tp.FG, linewidth=0.8, linestyle="--")
    held = summary["sustained_alarm_seconds"]
    if held >= 0:
        ax2.axvline(held, color=tp.PALETTE[3], linewidth=1.0, linestyle=":")
        ax2.text(held + 0.1, 0.06, "alarm held 0.5 s from %.2f s\n(severity %d%%)"
                 % (held, int(summary["sustained_alarm_severity"])), fontsize=9, color=tp.FG)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("probability / severity")
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_title("false alarms on %d of %d healthy windows; %d of %d faulty windows flagged"
                  % (summary["false_alarms"], summary["healthy_windows"], summary["detected"],
                     summary["faulty_windows"]), fontsize=10, color=tp.MUTED)
    ax2.legend(loc="upper left", fontsize=9)

    # 3) RAM: forward() over the window vs step() over one frame
    labels = ["forward()\nwhole window", "step()\none frame"]
    window, whole_arena = summary["window_bytes"], summary["whole_arena_bytes"]
    frame, stream_arena, state = summary["frame_bytes"], summary["stream_arena_bytes"], summary["state_bytes"]
    ax3.bar(labels, [window, frame], color=tp.PALETTE[1], label="input")
    ax3.bar(labels, [whole_arena, stream_arena], bottom=[window, frame], color=tp.PALETTE[0], label="arena (scratch)")
    ax3.bar(labels, [0, state], bottom=[window + whole_arena, frame + stream_arena], color=tp.PALETTE[4],
            label="state (kept between steps)")
    for x, total in enumerate([window + whole_arena, frame + stream_arena + state]):
        ax3.text(x, total, "%d B" % total, ha="center", va="bottom", fontsize=10, fontweight="bold", color=tp.FG)
    ax3.set_ylim(0, 1.3 * (window + whole_arena))
    ax3.set_ylabel("RAM (bytes)")
    ax3.set_title("RAM; both read the same %d bytes of weights" % summary["weight_bytes"],
                  fontsize=10, color=tp.MUTED)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(axis="x", visible=False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    tp.finish(fig, tp.png_for(RESULTS))


if __name__ == "__main__":
    main()
