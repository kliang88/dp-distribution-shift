"""Plot the AUPRC gain from public pretraining (no distribution shift) against epsilon.

Gain = unshifted-source pretrained minus scratch, for SGD and for DP-SGD, averaged over
every shift type and severity in the sweep. Reads <input>/data/runs.csv and writes a PNG
next to the figures made by `dp-shift plot` (default: <input>/plots/), in the same style.

Usage: python analysis/pretraining_gain_by_epsilon.py --input results/amy_full2
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Style constants copied from src/dp_shift/evaluation/plots.py so the figures match.
SGD_COLOR, DP_COLOR = "#bc6630", "#28658c"
TITLE_COLOR, MUTED_COLOR, GRID_COLOR = "#172332", "#505963", "#e6e9f0"
NAME = "pretraining_gain_by_epsilon"
KEYS = ["shift_type", "shift_fraction", "comparison_epsilon", "repeat"]


def pretraining_gains(runs):
    """Per-run gains in percentage points: baseline-pretrained minus scratch."""
    wide = runs.pivot_table(index=KEYS, columns=["method", "pretraining"], values="auprc")
    gains = pd.DataFrame({
        "sgd": 100 * (wide[("sgd", "baseline")] - wide[("sgd", "none")]),
        "dp": 100 * (wide[("dp_sgd", "baseline")] - wide[("dp_sgd", "none")]),
    })
    return gains.reset_index()


def summarize(gains):
    """Mean and 95% t half-width across repeats, after averaging each repeat over all cells."""
    rows = {}
    for epsilon, group in gains.groupby("comparison_epsilon"):
        per_repeat = group.groupby("repeat")[["sgd", "dp"]].mean()
        n = len(per_repeat)
        t = stats.t.ppf(0.975, n - 1)
        rows[epsilon] = {name: (per_repeat[name].mean(), t * per_repeat[name].std(ddof=1) / np.sqrt(n))
                         for name in ("sgd", "dp")}
    return rows


def draw(summary, output, repeats):
    epsilons = sorted(summary)
    x = np.arange(len(epsilons))
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    dp = [summary[e]["dp"] for e in epsilons]
    sgd = [summary[e]["sgd"] for e in epsilons]
    ax.errorbar(x, [m for m, _ in dp], yerr=[h for _, h in dp], fmt="o-", color=DP_COLOR,
                linewidth=1.5, markersize=4, capsize=3, label="DP-SGD")
    ax.errorbar(x, [m for m, _ in sgd], yerr=[h for _, h in sgd], fmt="o-", color=SGD_COLOR,
                linewidth=1.5, markersize=4, capsize=3, label="SGD (no privacy)")
    ax.axhline(0, color="#3d4a5c", ls="--", lw=1)
    for xi, (mean, half) in zip(x, dp):
        ax.annotate(f"+{mean:.1f}", (xi, mean + half), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color=DP_COLOR)
    ax.annotate(f"+{sgd[-1][0]:.1f}", (x[-1], sgd[-1][0]), textcoords="offset points",
                xytext=(10, -3), ha="left", fontsize=8, color=SGD_COLOR)
    ax.set_xticks(x, [f"{e:g}" for e in epsilons])
    ax.set_xlim(-0.4, len(epsilons) - 0.4)
    ax.set_ylim(-0.3, 3.2)
    ax.set_xlabel("Privacy budget ε (smaller = stronger privacy)")
    ax.set_ylabel("Pretraining gain (AUPRC percentage points)")
    ax.set_title("Public Pretraining Helps DP-SGD More Than Standard SGD", loc="left", pad=22,
                 fontweight="bold", color=TITLE_COLOR)
    ax.text(0, 1.02, "No distribution shift: unshifted source minus scratch", transform=ax.transAxes,
            va="bottom", fontsize=9, color=MUTED_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    caption = ("Gain = unshifted-source pretrained minus scratch, averaged over all sweep cells (replicates of the unshifted comparison).\n"
               f"SGD has no ε, so its gain is constant. Error bars: 95% confidence intervals across {repeats} repeats; fixed target test set.")
    fig.text(0.5, 0.015, caption, ha="center", fontsize=8, color=MUTED_COLOR)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Results directory containing data/runs.csv")
    parser.add_argument("--output", help="Figure directory (default: <input>/plots)")
    args = parser.parse_args()
    source = Path(args.input)
    runs = pd.read_csv(source / "data" / "runs.csv")
    gains = pretraining_gains(runs)
    summary = summarize(gains)
    for epsilon, values in summary.items():
        print(f"epsilon {epsilon:g}: " + ", ".join(f"{k} {m:+.2f} ± {h:.2f} pp" for k, (m, h) in values.items()))
    directory = Path(args.output) if args.output else source / "plots"
    draw(summary, directory / f"{NAME}.png", gains.repeat.nunique())
    print(f"Saved {directory / (NAME + '.png')}")


if __name__ == "__main__":
    main()
