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

"""Mackey-Glass ANFIS: series fit, hybrid training, rule base, rule pruning."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plotting"))
import tinymind_plot as tp  # noqa: E402

OUT = os.path.join(HERE, "output")
PREDICTION = os.path.join(OUT, "anfis_prediction.csv")
TRAINING = os.path.join(OUT, "anfis_training.csv")
RULES = os.path.join(OUT, "anfis_rules.csv")
PRUNING = os.path.join(OUT, "anfis_pruning.csv")

# Only the first slice of the held-out set is drawn; 500 overlaid points
# turn the fit into a solid band and hide whether it actually tracks.
FIT_WINDOW = 200


def main():
    pred, _ = tp.read_csv(PREDICTION)
    train, _ = tp.read_csv(TRAINING)
    rules, _ = tp.read_csv(RULES)
    prune, _ = tp.read_csv(PRUNING)

    tp.apply_style()
    fig, axes = tp.plt.subplots(2, 2, figsize=(13, 8.4))
    fig.suptitle("Mackey-Glass prediction with a 16-rule Takagi-Sugeno ANFIS",
                 fontsize=14, fontweight="bold")

    # 1. Series fit on the held-out set. The double and Q16.16 predictions are
    # indistinguishable from the target at this scale -- which is the result --
    # so the fixed-point series is drawn as sparse markers rather than a third
    # line that would simply hide under the other two.
    ax = axes[0][0]
    window = slice(0, FIT_WINDOW)
    tp.line(ax, pred["t"][window], {"target": pred["target"][window]},
            xlabel="held-out sample", ylabel="x(t+6)")
    ax.plot(pred["t"][window][::4], pred["predicted_q16_16"][window][::4],
            linestyle="none", marker="o", markersize=4.0,
            markerfacecolor="none", markeredgewidth=1.2,
            color=tp.PALETTE[1], label="ANFIS (Q16.16)")
    ax.legend()

    gap = max(abs(q - d) for q, d
              in zip(pred["predicted_q16_16"], pred["predicted_double"]))
    ax.set_title("held-out fit, first %d samples (max |Q16.16 - double| %.1e)"
                 % (FIT_WINDOW, gap), fontsize=10, color=tp.MUTED)

    # 2. Hybrid training: exact consequents, descended premises.
    ax = axes[0][1]
    tp.line(ax, train["epoch"],
            {"train RMSE": train["train_rmse"], "test RMSE": train["test_rmse"]},
            xlabel="epoch", ylabel="RMSE")
    ax.set_title("hybrid training (least squares + premise descent)",
                 fontsize=10, color=tp.MUTED)

    # 3. Which rules actually carry the prediction.
    ax = axes[1][0]
    labels = ["%d" % int(r) for r in rules["rule"]]
    tp.bars(ax, labels, rules["mean_firing_strength"],
            xlabel="rule", ylabel="mean normalized firing strength",
            value_fmt="{:.2f}")
    ax.set_title("rule base read back over the held-out set",
                 fontsize=10, color=tp.MUTED)

    # 4. Why the rule table is explicit: the full grid overfits, pruning fixes it.
    ax = axes[1][1]
    kept = ["%d/%d" % (int(k), int(t))
            for k, t in zip(prune["rules_kept"], prune["rules_total"])]
    tp.line(ax, list(range(len(kept))),
            {"train RMSE": prune["train_rmse"], "test RMSE": prune["test_rmse"]},
            xlabel="rules kept after pruning", ylabel="RMSE", logy=True,
            markers=True)
    ax.set_xticks(list(range(len(kept))))
    ax.set_xticklabels(kept)
    ax.set_title("pruning an oversized 81-rule grid", fontsize=10, color=tp.MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = tp.png_for(PREDICTION, "_behavior")
    fig.savefig(out)
    print("wrote %s" % out)
    if os.environ.get("DISPLAY") and "--no-show" not in sys.argv:
        tp.plt.show()


if __name__ == "__main__":
    main()
