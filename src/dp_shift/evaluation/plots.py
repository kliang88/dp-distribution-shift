"""Plot the signed study's three research figures from results.csv."""

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


def response(effects, output):
    series = [("SGD", effects.epsilon.isna(), "#303942"), *private_series(effects, (1, 5, 10))]
    signed_curves(effects, series, output, "signed_shift_response",
                  title="Effect of source shifts on target AUPRC",
                  ylabel="AUPRC difference (percentage points)\nShifted pretraining − unshifted pretraining",
                  include_origin=True)


def negative_transfer(effects, output):
    signed_curves(effects, private_series(effects, DP_COLORS), output, "negative_transfer",
                  title="Shifted pretraining vs. training from scratch",
                  ylabel="AUPRC difference (percentage points)\nShifted pretraining − training from scratch")


def heatmaps(effects, output):
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
    figure.suptitle("How privacy changes the effect of source shifts", fontsize=16, fontweight="semibold")
    figure.supxlabel(SHIFT_AXIS_LABEL, y=.065)
    figure.supylabel("DP-SGD privacy budget (ε)")
    figure.colorbar(image, ax=axes, label="AUPRC change under SGD − AUPRC change under DP-SGD (percentage points)")
    save(figure, output, "privacy_shift_interaction",
         f"n = {effects.repeat.nunique()} paired repeats. "
         "Positive values indicate a larger AUPRC drop or a smaller gain under DP-SGD than under SGD.")


def run(input, output=None, progress=print):
    path = tables.results_path(input)
    output = Path(output) if output is not None else path.parent / "plots"
    results = tables.read(path)
    effects = signed_effects(results)
    output.mkdir(parents=True, exist_ok=True)
    progress("Plotting signed response, privacy–shift interaction, and negative transfer…")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        response(effects, output)
        heatmaps(effects, output)
        negative_transfer(signed_effects(results, reference="scratch"), output)
    return {"figures": 3, "output": str(output)}
