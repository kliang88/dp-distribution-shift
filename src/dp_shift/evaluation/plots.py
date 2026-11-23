"""Plot the signed study's AUPRC and AUROC research figures from results.csv."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dp_shift_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dp_shift.data.utils import atomic_output
from . import tables
from .summarize import interaction, interval, signed_effects


LABELS = {"readmission": "30-day readmission", "age": "Age 70+",
          "circulatory": "Circulatory primary diagnosis", "complexity": "Clinical complexity"}
DP_COLORS = {1: "#2166ac", 2.5: "#7967a9", 5: "#33876e", 7.5: "#b07a25", 10: "#b34b34"}
SHIFT_COLORS = {"readmission": "#b34b34", "age": "#2166ac",
                "circulatory": "#33876e", "complexity": "#7967a9"}
SHIFT_AXIS_LABEL = "Change in share of source patients with this characteristic (percentage points)"


def save(figure, output, name, caption, *, layout_rect=(0, .07, 1, 1)):
    try:
        figure.text(.5, .015, caption, ha="center", fontsize=8, color="#505963")
        if not figure.get_constrained_layout():
            figure.tight_layout(rect=layout_rect)
        with atomic_output(Path(output) / f"{name}.png") as temporary:
            figure.savefig(temporary, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def estimates(groups, scale=1):
    means, widths = np.asarray([interval(np.asarray(values) * scale) for values in groups]).T
    return means, None if np.isnan(widths).all() else widths


def line(axis, points, label, color, scale=1, *, linewidth=1.5, markersize=4):
    levels, groups = zip(*sorted(points, key=lambda point: point[0]))
    means, errors = estimates(groups, scale)
    axis.errorbar(np.asarray(levels, dtype=float), means, yerr=errors, fmt="o-", label=label, color=color,
                  capsize=3, linewidth=linewidth, markersize=markersize)


def shift_tick_labels(levels):
    return [f"{level:+g}".replace("-", "−") if level else "0" for level in levels]


def signed_curves(effects, series, output, name, *, title, ylabel, include_origin=False):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    repeats = effects.repeat.nunique()
    origin = [(0, np.zeros(repeats))] if include_origin else []
    levels = sorted(set(effects.shift_pp) | ({0} if include_origin else set()))
    for axis, (kind, label) in zip(axes.flat, LABELS.items()):
        axis.axhline(0, color="gray", linestyle="--", linewidth=.8)
        for legend, mask, color in series:
            selected = effects.loc[mask & effects.shift_type.eq(kind)]
            points = [(level, block.gain) for level, block in selected.groupby("shift_pp")]
            line(axis, [*points, *origin], legend, color, scale=100)
        axis.set_title(label)
        axis.set_xticks(levels, shift_tick_labels(levels))
    figure.suptitle(title, y=.97, fontsize=16, fontweight="semibold")
    figure.supxlabel(SHIFT_AXIS_LABEL, y=.07)
    figure.supylabel(ylabel, x=.02)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .92),
                  ncol=len(series), frameon=False)
    caption = f"n = {repeats} paired repeats"
    if repeats > 1:
        caption += ", pointwise 95% CI"
    save(figure, output, name, caption, layout_rect=(.025, .06, 1, .90))


def private_series(effects, budgets):
    available = set(effects.epsilon.dropna())
    return [(f"DP-SGD ε = {epsilon:g}", effects.epsilon.eq(epsilon), DP_COLORS[epsilon])
            for epsilon in budgets if epsilon in available]


def response(effects, output, metric="auprc"):
    series = [("SGD", effects.epsilon.isna(), "#b07a25"), *private_series(effects, (1, 10))]
    signed_curves(effects, series, output, f"source_shift_{metric}",
                  title=f"How Source Shifts Affect Target {metric.upper()}",
                  ylabel=f"{metric.upper()} difference (percentage points)\nShifted pretraining − unshifted pretraining",
                  include_origin=True)


def source_shift_effects(results, output, epsilon=10):
    """Compare all four source shifts across metrics at one privacy budget."""
    selected = results.loc[results.method.eq("DP-SGD") & results.epsilon.eq(epsilon)]
    if selected.empty:
        return 0
    repeats = selected.repeat.nunique()
    levels = sorted(set(selected.shift_pp))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, label in zip(axes, tables.METRICS, ("AUPRC", "AUROC", "Brier score")):
        effects = signed_effects(selected, metric)
        axis.axhline(0, color="gray", linestyle="--", linewidth=.8)
        for kind, legend in LABELS.items():
            points = [(level, block.gain) for level, block in
                      effects.loc[effects.shift_type.eq(kind)].groupby("shift_pp")]
            line(axis, [*points, (0, np.zeros(repeats))], legend, SHIFT_COLORS[kind], scale=100)
        axis.set_title(label + (" (lower is better)" if metric == "brier_score" else ""))
        axis.set_ylabel(f"{label} difference (percentage points)")
        axis.set_xticks(levels, shift_tick_labels(levels))
    figure.suptitle("How Source Shifts Affect Target Performance", y=.98, fontsize=16, fontweight="semibold")
    figure.supxlabel(SHIFT_AXIS_LABEL, y=.13)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .90),
                  ncol=4, frameon=False)
    caption = f"DP-SGD ε = {epsilon:g}, n = {repeats} paired repeats"
    if repeats > 1:
        caption += ", pointwise 95% CI"
    caption += ". Shifted pretraining − unshifted pretraining."
    save(figure, output, "source_shift_effects", caption, layout_rect=(0, .13, 1, .80))
    return 1


def negative_transfer(effects, output, metric="auprc"):
    signed_curves(effects, private_series(effects, (1, 5, 10)), output, f"pretraining_vs_scratch_{metric}",
                  title=f"Does Shifted Pretraining Improve Target {metric.upper()}?",
                  ylabel=f"{metric.upper()} difference (percentage points)\nShifted pretraining − training from scratch")


def heatmaps(effects, output, metric="auprc"):
    values = interaction(effects).groupby(["shift_type", "epsilon", "shift_pp"]).interaction.mean() * 100
    bound = max(float(values.abs().max()), 1e-6)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    figure.get_layout_engine().set(rect=(0, .10, 1, .90))
    for axis, (kind, label) in zip(axes.flat, LABELS.items()):
        matrix = values.loc[kind].unstack("shift_pp")
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_title(label)
        axis.set_xticks(range(len(matrix.columns)), shift_tick_labels(matrix.columns))
        axis.set_yticks(range(len(matrix.index)), [f"{epsilon:g}" for epsilon in matrix.index])
    figure.suptitle(f"How Privacy Changes the Response to Source Shifts ({metric.upper()})", fontsize=16, fontweight="semibold")
    figure.supxlabel(SHIFT_AXIS_LABEL, y=.065)
    figure.supylabel("DP-SGD privacy budget (ε)")
    figure.colorbar(image, ax=axes, label=f"{metric.upper()} change under SGD − {metric.upper()} change under DP-SGD (percentage points)")
    save(figure, output, f"privacy_shift_interaction_{metric}",
         f"n = {effects.repeat.nunique()} paired repeats. "
         f"Positive values indicate a larger {metric.upper()} drop or a smaller gain under DP-SGD than under SGD.")


def average_dp_penalty(effects, output, metric="auprc", *, axis=None):
    """Average paired interactions within repeats before estimating uncertainty."""
    contrasts = interaction(effects)
    averaged = contrasts.groupby(["shift_type", "repeat"]).interaction.mean()
    means, errors = estimates([averaged.loc[kind] for kind in LABELS], scale=100)
    standalone = axis is None
    if standalone:
        figure, axis = plt.subplots(figsize=(7, 3.7))
    else:
        figure = axis.figure
    axis.use_sticky_edges = False
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#e8e8e8", linewidth=.5)
    axis.axhline(0, color="#555555", linestyle="--", linewidth=.8, zorder=3)
    for position, (kind, mean) in enumerate(zip(LABELS, means)):
        axis.errorbar(position, mean, yerr=None if errors is None else errors[position],
                      fmt="o", color=SHIFT_COLORS[kind], markersize=6,
                      elinewidth=1.3, capsize=4, capthick=1.3, zorder=4)
    axis.set(xticks=range(len(LABELS)),
             xticklabels=["Readmission", "Age ≥70", "Circulatory\ndiagnosis", "Clinical\ncomplexity"])
    axis.set_ylabel(f"Additional {metric.upper()} loss under DP-SGD\n(percentage points)", fontsize=10, labelpad=8)
    axis.tick_params(axis="both", labelsize=9, width=.7, length=3)
    axis.tick_params(axis="x", length=0, pad=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_linewidth(.7)
    axis.margins(x=.10, y=.15)
    budgets = ", ".join(f"{value:g}" for value in sorted(contrasts.epsilon.unique()))
    caption = f"n = {contrasts.repeat.nunique()} paired repeats"
    if contrasts.repeat.nunique() > 1:
        caption += ", pointwise 95% CI"
    caption += f"; averaged over all signed shifts\nand ε = {budgets}."
    if standalone:
        save(figure, output, f"average_dp_penalty_{metric}", caption, layout_rect=(0, .13, 1, 1))
    return caption


def paired_metrics(output, name, title, draw, *, xlabel=None, legend=False):
    """Lay out complementary metrics with independent scales and shared context."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("auprc", "auroc")):
        caption = draw(axis, metric)
        axis.set_title(metric.upper())
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title, y=.98, fontsize=16, fontweight="semibold")
    if xlabel:
        figure.supxlabel(xlabel, y=.10)
    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .90),
                      ncol=len(labels), frameon=False)
    save(figure, output, name, caption, layout_rect=(0, .10, 1, .80 if legend else .90))


def average_dp_penalty_pair(results, output):
    paired_metrics(output, "average_dp_penalty", "Additional Loss under DP-SGD by Source Shift Type",
                   lambda axis, metric: average_dp_penalty(signed_effects(results, metric), output, metric, axis=axis))


def run(input, output=None, progress=print):
    path = tables.results_path(input)
    output = Path(output) if output is not None else path.parent / "plots"
    results = tables.read(path)
    output.mkdir(parents=True, exist_ok=True)
    progress("Plotting AUPRC and AUROC source shifts, privacy interactions, and pretraining gains…")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        for metric in ("auprc", "auroc"):
            effects = signed_effects(results, metric)
            response(effects, output, metric)
            heatmaps(effects, output, metric)
            negative_transfer(signed_effects(results, metric, reference="scratch"), output, metric)
        extra = source_shift_effects(results, output)
        average_dp_penalty_pair(results, output)
    return {"figures": 7 + extra, "output": str(output)}
