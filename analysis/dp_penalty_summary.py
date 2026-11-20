"""Forest plot: does DP-SGD make the cost of a shifted source larger than SGD does?

For every shift type and severity:
    penalty(method) = AUPRC(unshifted-source pretrained) - AUPRC(shifted-source pretrained)
    extra DP penalty = penalty(DP-SGD) - penalty(SGD), in AUPRC percentage points.
Positive means DP amplifies the damage from the shifted source; negative means DP is hurt less.
Each repeat's value is averaged over all privacy budgets, then summarized across repeats.
Reads <input>/data/runs.csv and writes a PNG next to the `dp-shift plot` figures (<input>/plots/).

Default figure: one row per shift type plus all shifts combined. Add --detail for every severity.

Usage: python analysis/dp_penalty_summary.py --input results/amy_full2 [--detail]
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
DP_COLOR = "#28658c"
TITLE_COLOR, MUTED_COLOR, GRID_COLOR, ZERO_COLOR = "#172332", "#505963", "#e6e9f0", "#3d4a5c"
NAME = "dp_penalty_summary"
KEYS = ["shift_type", "shift_fraction", "comparison_epsilon", "repeat"]
SHIFT_LABELS = {
    "readmission": ("Readmission prevalence", "{:g}% source readmission rate"),
    "age_70_plus": ("Age distribution", "{:g}% aged ≥70"),
    "primary_diagnosis_circulatory": ("Primary diagnosis", "{:g}% replaced (circulatory)"),
    "clinical_complexity_2plus": ("Clinical complexity", "{:g}% replaced (≥2 diagnosis groups)"),
}
PERCENT = {"readmission", "age_70_plus", "primary_diagnosis_circulatory", "clinical_complexity_2plus"}


def extra_dp_penalty(runs):
    """Per-run extra DP penalty (percentage points) = DP penalty - SGD penalty."""
    wide = runs.pivot_table(index=KEYS, columns=["method", "pretraining"], values="auprc")
    penalty_sgd = wide[("sgd", "baseline")] - wide[("sgd", "shifted")]
    penalty_dp = wide[("dp_sgd", "baseline")] - wide[("dp_sgd", "shifted")]
    return (100 * (penalty_dp - penalty_sgd)).rename("extra").reset_index()


def estimate(per_repeat):
    """Mean, 95% t half-width and two-sided p-value for a per-repeat series."""
    n = len(per_repeat)
    mean, sd = per_repeat.mean(), per_repeat.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, half, stats.ttest_1samp(per_repeat, 0).pvalue


def summarize(extra):
    """Rows for every severity, each shift type pooled, and all shifts pooled."""
    rows = []
    for kind in SHIFT_LABELS:
        group = extra[extra.shift_type == kind]
        if group.empty:
            continue
        rows.append(("header", kind, SHIFT_LABELS[kind][0], None))
        for fraction, cell in group.groupby("shift_fraction"):
            rows.append(("cell", kind, SHIFT_LABELS[kind][1].format(100 * fraction),
                         estimate(cell.groupby("repeat").extra.mean())))
        rows.append(("pooled", kind, "All severities", estimate(group.groupby("repeat").extra.mean())))
    rows.append(("overall", "all", "All shifts and severities", estimate(extra.groupby("repeat").extra.mean())))
    return rows


def draw(rows, output, repeats):
    n_cells = sum(1 for row in rows if row[0] == "cell")
    bonferroni = 0.05 / n_cells
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(rows) + 2.6))
    y = {}
    for position, row in enumerate(rows):
        y[position] = len(rows) - position
    lim = max(0.6, max(abs(r[3][0]) + r[3][1] for r in rows if r[3]) * 1.15)
    for position, (kind, shift, label, stat) in enumerate(rows):
        yy = y[position]
        if kind == "header":
            ax.text(-lim * 1.02, yy, label, ha="right", va="center", fontweight="bold", color=TITLE_COLOR, fontsize=10)
            continue
        mean, half, p = stat
        bold = kind in ("pooled", "overall")
        ax.errorbar(mean, yy, xerr=half, fmt="D" if bold else "o", color=DP_COLOR if not bold else TITLE_COLOR,
                    ecolor=DP_COLOR if not bold else TITLE_COLOR, capsize=3, linewidth=1.5,
                    markersize=6 if bold else 4.5)
        star = "*" if p < 0.05 else ""
        ax.text(-lim * 1.02, yy, label, ha="right", va="center", fontsize=9,
                fontweight="bold" if bold else "normal", color=TITLE_COLOR if bold else MUTED_COLOR)
        ax.text(lim * 1.02, yy, f"{mean:+.2f}{star}", ha="left", va="center", fontsize=8.5,
                color=TITLE_COLOR if bold else MUTED_COLOR)
    ax.axvline(0, color=ZERO_COLOR, ls="--", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0.3, len(rows) + 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Extra AUPRC loss under DP-SGD relative to SGD (percentage points)")
    ax.text(0.02, 1.01, "← DP less affected by the shift", transform=ax.transAxes, fontsize=8, color=MUTED_COLOR, va="bottom")
    ax.text(0.98, 1.01, "DP amplifies the shift penalty →", transform=ax.transAxes, fontsize=8, color=MUTED_COLOR,
            va="bottom", ha="right")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR)
    ax.set_axisbelow(True)
    survivors = sum(1 for row in rows if row[0] == "cell" and row[3][2] < bonferroni)
    fig.suptitle("Does Differential Privacy Make Distribution Shift More Harmful?", x=0.01, ha="left",
                 fontweight="bold", color=TITLE_COLOR, fontsize=13)
    caption = ("Extra DP penalty = (unshifted − shifted AUPRC under DP-SGD) − (same under SGD), averaged over ε = 1–10.\n"
               f"Points: mean over {repeats} repeats; bars: 95% confidence intervals across repeats; fixed target test set.\n"
               f"* p < 0.05 uncorrected; {survivors} of {n_cells} severities remain significant after Bonferroni correction (p < {bonferroni:.3f}).")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color=MUTED_COLOR)
    fig.tight_layout(rect=(0.0, 0.1, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_simple(rows, output, repeats):
    """Five rows: each shift type pooled over severities and budgets, then everything combined."""
    chosen = [row for row in rows if row[0] in ("pooled", "overall")]
    labels = [SHIFT_LABELS[row[1]][0] if row[0] == "pooled" else "All shifts combined" for row in chosen]
    n = len(chosen)
    lim = max(0.5, max(abs(r[3][0]) + r[3][1] for r in chosen) * 1.3)
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * n + 2.7))
    for i, ((kind, shift, label, (mean, half, p)), name) in enumerate(zip(chosen, labels)):
        yy = n - i
        overall = kind == "overall"
        color = TITLE_COLOR if overall else DP_COLOR
        ax.errorbar(mean, yy, xerr=half, fmt="D" if overall else "o", color=color, ecolor=color, capsize=4,
                    linewidth=2, markersize=8 if overall else 6.5)
        ax.text(-lim * 1.03, yy, name, ha="right", va="center", fontsize=10,
                fontweight="bold" if overall else "normal", color=TITLE_COLOR)
        ax.text(lim * 1.03, yy, f"{mean:+.2f} pp", ha="left", va="center", fontsize=9, color=MUTED_COLOR)
    ax.axhline(1.5, color=GRID_COLOR, lw=1.2)
    ax.axvline(0, color=ZERO_COLOR, ls="--", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0.4, n + 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Extra AUPRC loss under DP-SGD vs. SGD when the source is shifted (percentage points)")
    ax.text(0.02, 1.015, "← DP less affected", transform=ax.transAxes, fontsize=8.5, color=MUTED_COLOR, va="bottom")
    ax.text(0.98, 1.015, "DP amplifies the penalty →", transform=ax.transAxes, fontsize=8.5, color=MUTED_COLOR,
            va="bottom", ha="right")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR)
    ax.set_axisbelow(True)
    all_zero = all(r[3][0] - r[3][1] <= 0 <= r[3][0] + r[3][1] for r in chosen)
    verdict = ("Every interval includes zero: no evidence that DP amplifies the shift penalty"
               if all_zero else "Some intervals exclude zero; see the detailed figure (--detail)")
    fig.suptitle("Does Differential Privacy Make Distribution Shift More Harmful?", x=0.01, ha="left",
                 fontweight="bold", color=TITLE_COLOR, fontsize=13)
    fig.text(0.01, 0.905, verdict, ha="left", fontsize=9.5, color=MUTED_COLOR)
    caption = ("Extra DP penalty = (unshifted − shifted AUPRC under DP-SGD) − (same under SGD), averaged over all\n"
               f"severities and ε = 1–10. Points: mean over {repeats} repeats; bars: 95% confidence intervals across repeats; "
               "fixed target test set.")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color=MUTED_COLOR)
    fig.tight_layout(rect=(0.0, 0.14, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Results directory containing data/runs.csv")
    parser.add_argument("--output", help="Figure directory (default: <input>/plots)")
    parser.add_argument("--detail", action="store_true", help="Show every severity (writes dp_penalty_detail.png)")
    args = parser.parse_args()
    source = Path(args.input)
    runs = pd.read_csv(source / "data" / "runs.csv")
    extra = extra_dp_penalty(runs)
    rows = summarize(extra)
    for kind, shift, label, stat in rows:
        if stat:
            print(f"{shift:32s} {label:40s} {stat[0]:+.2f} ± {stat[1]:.2f} pp (p={stat[2]:.3f})")
    directory = Path(args.output) if args.output else source / "plots"
    repeats = extra.repeat.nunique()
    if args.detail:
        path = directory / "dp_penalty_detail.png"
        draw(rows, path, repeats)
    else:
        path = directory / f"{NAME}.png"
        draw_simple(rows, path, repeats)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
