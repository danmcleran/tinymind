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

"""Elman Japanese Vowels: accuracy-vs-format, training loss, Q8.8 confusion."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

ACCURACY = os.path.join(HERE, "output", "vowels_accuracy.csv")
LOSS = os.path.join(HERE, "output", "vowels_loss.csv")
CONFUSION = os.path.join(HERE, "output", "vowels_confusion.csv")

NUM_CLASSES = 9


def plot_accuracy():
    cols, _ = tp.read_csv(ACCURACY)
    labels = [
        "%s\n(%d B weights)" % (fmt, int(b))
        for fmt, b in zip(cols["format"], cols["weight_bytes"])
    ]
    fig, ax = tp.new_fig(
        "Speaker recognition accuracy vs. numeric format",
        "UCI Japanese Vowels, 370 test utterances -- offline float training, fixed-point inference")
    tp.bars(ax, labels, cols["accuracy"], ylabel="test accuracy (%)", value_fmt="{:.2f}%")
    ax.axhline(cols["accuracy"][0], linestyle="--", linewidth=1.0, color=tp.MUTED)
    tp.finish(fig, tp.png_for(ACCURACY))


def plot_loss():
    cols, _ = tp.read_csv(LOSS)
    fig, ax = tp.new_fig(
        "Offline training loss (double precision)",
        "per-frame MSE against the one-hot speaker target")
    tp.line(ax, cols["epoch"], {"train MSE": cols["mse"]}, xlabel="epoch", ylabel="MSE", logy=True)
    tp.finish(fig, tp.png_for(LOSS))


def plot_confusion():
    cols, _ = tp.read_csv(CONFUSION)
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for actual, predicted, count in zip(cols["true_speaker"], cols["predicted_speaker"], cols["count"]):
        matrix[int(actual)][int(predicted)] = int(count)

    fig, ax = tp.new_fig(
        "Q8.8 confusion matrix",
        "row = true speaker, column = predicted (test split)")
    image = ax.imshow(matrix, cmap="Blues")
    speakers = [str(s + 1) for s in range(NUM_CLASSES)]
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(speakers)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("predicted speaker")
    ax.set_ylabel("true speaker")
    peak = max(max(row) for row in matrix)
    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            if matrix[r][c]:
                ax.text(c, r, str(matrix[r][c]), ha="center", va="center", fontsize=8,
                        color="white" if matrix[r][c] > (peak / 2) else "#333333")
    fig.colorbar(image, ax=ax, shrink=0.8)
    tp.finish(fig, tp.png_for(CONFUSION))


def main():
    plot_accuracy()
    plot_loss()
    plot_confusion()


if __name__ == "__main__":
    main()
