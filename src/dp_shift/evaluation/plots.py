"""Five research figures, faceted automatically by registered shift type."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dp_shift_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from dp_shift.data.shifts import SHIFTS
from dp_shift.stages.common import atomic_path
from .summarize import interval


FIGURES = ("overall_performance", "pretraining_benefit", "dp_amplification", "privacy_utility", "source_target_prevalence")
INITIALIZATIONS = ("none", "baseline", "shifted")
LABELS = ("Scratch", "Baseline-source\npretraining", "Shifted-source\npretraining")
COLORS = ("#777777", "#28658c", "#bc6630")
CI_NOTE = "95% t intervals across repeats on the fixed target test set."


def save(figure, output, name, caption):
    try:
        figure.text(.5, .015, caption, ha="center", fontsize=8, color="#505963")
        figure.tight_layout(rect=(0, .12, 1, 1))
        with atomic_path(Path(output) / f"{name}.png") as temporary:
            figure.savefig(temporary, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def panels(kinds, *, sharey=True):
    figure, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), sharey=sharey, squeeze=False)
    return figure, axes[0]


def estimates(groups, scale=1):
    rows = [interval(np.asarray(values) * scale) for values in groups]
    means = np.array([row["mean"] for row in rows])
    errors = np.array([[row["mean"] - row["ci_lower"] for row in rows],
                       [row["ci_upper"] - row["mean"] for row in rows]])
    return means, None if all(row["n"] == 1 for row in rows) else errors


def line(axis, points, label, color, scale=1):
    points = sorted(points, key=lambda point: point[0])
    x = np.array([value for value, _ in points], dtype=float)
    means, errors = estimates([values for _, values in points], scale)
    axis.errorbar(x, means, yerr=errors, fmt="o-", color=color,
                  label=label, capsize=3, linewidth=1.5, markersize=4)


def fraction_axis(axis, baseline, kind):
    axis.set_xlabel(f"Source share: {SHIFTS[kind].label}")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.axvline(baseline, color="#777777", linestyle=":", linewidth=1,
                 label=f"Baseline source: {baseline:.1%}")
    axis.grid(alpha=.15)


def overall_performance(runs, output, epsilon, fractions, note=CI_NOTE):
    figure, axes = panels(fractions)
    x = np.arange(3)
    for axis, (kind, fraction) in zip(axes, fractions.items()):
        selected = runs.loc[runs.shift_type.eq(kind) & runs.comparison_epsilon.eq(epsilon) & runs.shift_fraction.eq(fraction)]
        for offset, method, color, label in ((-.18, "sgd", COLORS[1], "SGD"), (.18, "dp_sgd", COLORS[2], f"DP-SGD, ε={epsilon:g}")):
            means, errors = estimates([selected.loc[selected.method.eq(method) & selected.pretraining.eq(init), "auprc"]
                                       for init in INITIALIZATIONS])
            axis.bar(x + offset, means, width=.36, yerr=errors, capsize=4, color=color, label=label, alpha=.9)
        axis.set(xticks=x, xticklabels=LABELS, ylabel="Target-test AUPRC",
                 title=f"{SHIFTS[kind].label} · source share {fraction:.1%}")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=.15)
    figure.suptitle("Overall model performance")
    save(figure, output, "overall_performance", note)


def pretraining_benefit(effects, output, budgets, colors, baselines, fractions, note=CI_NOTE):
    figure, axes = panels(fractions)
    for axis, (kind, fraction) in zip(axes, fractions.items()):
        series = [(budgets[0], "gain_sgd_shifted", "gain_sgd_baseline", "SGD", COLORS[1])]
        series.extend((budget, "gain_dp_shifted", "gain_dp_baseline", f"DP-SGD, ε={budget:g}", colors[budget]) for budget in budgets)
        for budget, field, baseline_field, label, color in series:
            data = effects.loc[effects.shift_type.eq(kind) & effects.comparison_epsilon.eq(budget)]
            points = [(value, group[field]) for value, group in data.groupby("shift_fraction")]
            reference = data.loc[data.shift_fraction.eq(fraction), baseline_field]
            points.append((baselines[kind], reference))
            line(axis, points, label, color, scale=100)
            axis.plot(baselines[kind], reference.mean() * 100, "s", color=color, markersize=6)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        fraction_axis(axis, baselines[kind], kind)
        axis.set(title=SHIFTS[kind].label, ylabel="AUPRC gain (percentage points)")
        axis.legend(frameon=False)
    figure.suptitle("Pretraining benefit versus source composition")
    save(figure, output, "pretraining_benefit",
         "Pretrained − scratch; positive means pretraining helps. Squares: baseline-source benefit.\n" + note)


def dp_amplification(effects, output, budgets, colors, baselines, note=CI_NOTE):
    figure, axes = panels(baselines)
    for axis, (kind, baseline) in zip(axes, baselines.items()):
        for budget in budgets:
            data = effects.loc[effects.shift_type.eq(kind) & effects.comparison_epsilon.eq(budget)]
            points = [(fraction, group.dp_amplification) for fraction, group in data.groupby("shift_fraction")]
            points.append((baseline, np.zeros(data.repeat.nunique())))
            line(axis, points, f"ε={budget:g}", colors[budget], scale=100)
            axis.plot(baseline, 0, "s", color=colors[budget], markersize=6)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        fraction_axis(axis, baseline, kind)
        axis.set(title=SHIFTS[kind].label, ylabel="Additional DP shift penalty\n(AUPRC percentage points)")
        axis.legend(frameon=False)
    figure.suptitle("DP amplification of the source-shift penalty")
    save(figure, output, "dp_amplification",
         "Positive = DP amplifies the penalty. Baseline self-comparison is zero by definition.\n" + note)


def privacy_utility(runs, output, budgets, fractions, note=CI_NOTE):
    figure, axes = panels(fractions)
    for axis, (kind, fraction) in zip(axes, fractions.items()):
        data = runs.loc[runs.shift_type.eq(kind) & runs.method.eq("dp_sgd") & runs.shift_fraction.eq(fraction)]
        for initialization, label, color in zip(INITIALIZATIONS, LABELS, COLORS):
            group = data.loc[data.pretraining.eq(initialization)]
            line(axis, [(budget, values.auprc) for budget, values in group.groupby("comparison_epsilon")],
                 label.replace("\n", " "), color)
        axis.set(xticks=budgets, xlabel="Privacy budget ε (smaller = stronger privacy)", ylabel="Target-test AUPRC",
                 title=f"{SHIFTS[kind].label} · source share {fraction:.1%}")
        axis.grid(alpha=.15)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Privacy–utility tradeoff")
    save(figure, output, "privacy_utility", "DP-SGD only; nonprivate SGD is not assigned an epsilon value.\n" + note)


def size_label(values):
    low, high = int(values.min()), int(values.max())
    return f"n={low:,}" if low == high else f"n={low:,}–{high:,}"


def source_target_prevalence(runs, output, target_size, fractions, note=""):
    figure, axes = panels(fractions, sharey=False)
    for axis, (kind, fraction) in zip(axes, fractions.items()):
        data = runs.loc[runs.shift_type.eq(kind) & runs.method.eq("sgd") & runs.comparison_epsilon.eq(runs.comparison_epsilon.min())]
        reference = data.loc[data.shift_fraction.eq(fraction) & data.pretraining.eq("baseline")]
        target = data.test_fraction.iloc[0]
        labels = [f"Target test\nn={target_size:,}", f"Baseline\n{size_label(reference.source_size)}"]
        values = [[target], reference.source_fraction.to_numpy()]
        for value, group in data.loc[data.pretraining.eq("shifted")].groupby("shift_fraction"):
            labels.append(f"{value:.1%}\n{size_label(group.source_size)}")
            values.append(group.source_fraction.to_numpy())
        positions = np.arange(len(labels))
        axis.bar(positions, [np.mean(value) for value in values], color=[COLORS[0], COLORS[1], *[COLORS[2]] * (len(labels) - 2)], alpha=.8)
        for x, observed in zip(positions[1:], values[1:]):
            axis.scatter(x + np.linspace(-.07, .07, len(observed)), observed, s=15, color="#333333", zorder=3)
        axis.axhline(target, color="#555555", linestyle="--", linewidth=1, label=f"Target test: {target:.1%}")
        axis.set(xticks=positions, xticklabels=labels, ylabel="Patients with characteristic", title=SHIFTS[kind].label)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=.15)
        axis.legend(frameon=False)
    figure.suptitle("Observed source–target composition")
    save(figure, output, "source_target_prevalence",
         "Bars: observed percentages averaged across repeats. Dots: source repeats.\nThe target test population is fixed; source sample sizes are shown below each bar." + ("\n" + note if note else ""))


def plot(runs, effects, target_size, output, epsilon, fractions):
    single = runs["repeat"].nunique() == 1
    note = "Single repeat; uncertainty not estimated." if single else CI_NOTE
    budgets = sorted(runs.comparison_epsilon.unique())
    colors = dict(zip(budgets, plt.get_cmap("viridis")(np.linspace(.15, .8, len(budgets)))))
    baselines = {kind: runs.loc[runs.shift_type.eq(kind) & runs.method.eq("sgd") & runs.pretraining.eq("baseline")
                               & runs.comparison_epsilon.eq(budgets[0]) & runs.shift_fraction.eq(fraction), "source_fraction"].mean()
                 for kind, fraction in fractions.items()}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        overall_performance(runs, output, epsilon, fractions, note=note)
        pretraining_benefit(effects, output, budgets, colors, baselines, fractions, note=note)
        dp_amplification(effects, output, budgets, colors, baselines, note=note)
        privacy_utility(runs, output, budgets, fractions, note=note)
        source_target_prevalence(runs, output, target_size, fractions, note=note if single else "")
    return len(FIGURES)
