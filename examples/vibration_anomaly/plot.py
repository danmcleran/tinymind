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

"""Vibration anomaly scores over the test run, and alarm rates per fault."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

FRAMES = os.path.join(HERE, "output", "vibration_anomaly.csv")
SUMMARY = os.path.join(HERE, "output", "vibration_anomaly_summary.csv")
DETECTORS = [("kmeans", "on-device k-means, weighted"),
             ("euclidean", "on-device k-means, Euclidean"),
             ("gaussian", "host-fitted diagonal Gaussian")]
FAULTS = ["bearing", "imbalance", "looseness"]


def main():
    with open(FRAMES, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(SUMMARY, newline="") as f:
        summary = list(csv.DictReader(f))
    t0 = float(rows[0]["time_s"])
    t = [float(r["time_s"]) - t0 for r in rows]

    tp.apply_style()
    fig = tp.plt.figure(figsize=(13, 8.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    fig.suptitle("On-device vibration anomaly detection: log-mel features, k-means learned frame by frame",
                 fontsize=14, fontweight="bold")

    # 1) score / threshold over the test run, with the fault segments shaded.
    ax = fig.add_subplot(grid[0, :])
    spans = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["condition"] != rows[start]["condition"]:
            spans.append((rows[start]["condition"], t[start], t[i - 1]))
            start = i
    for name, a, b in spans:
        if name != "normal":
            ax.axvspan(a, b, color=tp.PALETTE[FAULTS.index(name) + 3], alpha=0.18, linewidth=0)
            ax.text((a + b) / 2, 30, "%s\n(severity 0 to 1)" % name, ha="center", va="top", fontsize=9, color=tp.FG)
    # The Gaussian first, thin, so the on-device k-means draws on top.
    for i, (key, label) in ((2, DETECTORS[2]), (0, DETECTORS[0])):
        ratio = [max(float(r[key + "_score"]) / float(r[key + "_threshold"]), 1e-3) for r in rows]
        if key == "gaussian":
            # A log-likelihood: measure it from the bias, as the threshold was.
            bias = min(float(r["gaussian_score"]) for r in rows)
            ratio = [max((float(r["gaussian_score"]) - bias) / (float(r["gaussian_threshold"]) - bias), 1e-3)
                     for r in rows]
        ax.plot(t, ratio, color=tp.PALETTE[i], linewidth=0.6 if i == 2 else 0.9, alpha=0.6 if i == 2 else 0.9,
                label=label)
    ax.axhline(1.0, color=tp.FG, linestyle="--", linewidth=1.0)
    ax.text(t[-1], 1.05, "alarm threshold", ha="right", va="bottom", fontsize=9, color=tp.FG)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 40)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("test time (s)")
    ax.set_ylabel("score / threshold")
    ax.set_title("normal running, then three faults growing from nothing to full severity", fontsize=10,
                 color=tp.MUTED)
    ax.legend(loc="upper left")

    # 2) alarm rate per fault and detector.
    ax = fig.add_subplot(grid[1, 0])
    width = 0.26
    for i, (key, label) in enumerate(DETECTORS):
        rates = []
        for fault in FAULTS:
            row = next(r for r in summary if r["detector"] == key and r["condition"] == fault)
            rates.append(100.0 * float(row["alarm_rate"]))
        xs = [x + (i - 1) * width for x in range(len(FAULTS))]
        ax.bar(xs, rates, width, color=tp.PALETTE[i], label=label)
        for x, v in zip(xs, rates):
            ax.text(x, v + 1, "%.0f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(FAULTS)))
    ax.set_xticklabels(FAULTS)
    ax.set_ylim(0, 110)
    ax.set_ylabel("frames in alarm (%)")
    false_alarm = {r["detector"]: 100.0 * float(r["alarm_rate"]) for r in summary if r["condition"] == "normal"}
    ax.set_title("alarm rate over each fault segment (false alarms in normal running: %.2f / %.2f / %.2f%%)"
                 % (false_alarm["kmeans"], false_alarm["euclidean"], false_alarm["gaussian"]),
                 fontsize=9, color=tp.MUTED)
    ax.legend(fontsize=8, loc="upper right")

    # 3) severity at the first sustained alarm.
    ax = fig.add_subplot(grid[1, 1])
    for i, (key, label) in enumerate(DETECTORS):
        sev = []
        for fault in FAULTS:
            row = next(r for r in summary if r["detector"] == key and r["condition"] == fault)
            v = float(row["first_sustained_alarm_severity"])
            sev.append(v if v >= 0 else 1.05)
        xs = [x + (i - 1) * width for x in range(len(FAULTS))]
        ax.bar(xs, sev, width, color=tp.PALETTE[i], label=label)
        for x, v in zip(xs, sev):
            ax.text(x, v + 0.02, "never" if v > 1.0 else "%.2f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(FAULTS)))
    ax.set_xticklabels(FAULTS)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("fault severity")
    ax.set_title("severity at the first sustained alarm (lower is earlier)", fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tp.finish(fig, tp.png_for(FRAMES))


if __name__ == "__main__":
    main()
