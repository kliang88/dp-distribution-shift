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


def signed_curves(effects, series, output, name, ylabel, *, include_origin=False):
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
        axis.set(title=label, xticks=levels)
    figure.supxlabel("Change in source share (percentage points)", y=.08)
    figure.supylabel(ylabel, x=.02)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(series), frameon=False)
    caption = f"{repeats} paired repeats on the fixed target test set."
    if repeats > 1:
        caption += " Bars show pointwise 95% t intervals."
    save(figure, output, name, caption, layout_rect=(.03, .09, 1, .94))


def private_series(effects, budgets):
    available = set(effects.epsilon.dropna())
    return [(f"DP-SGD ε={epsilon:g}", effects.epsilon.eq(epsilon), DP_COLORS[epsilon])
            for epsilon in budgets if epsilon in available]


def response(effects, output):
    series = [("SGD", effects.epsilon.isna(), "#303942"), *private_series(effects, (1, 5, 10))]
    signed_curves(effects, series, output, "signed_shift_response", "S1 − S0 AUPRC (percentage points)",
                  include_origin=True)


def negative_transfer(effects, output):
    signed_curves(effects, private_series(effects, DP_COLORS), output, "negative_transfer",
                  "Shifted pretraining − scratch AUPRC (percentage points)")


def heatmaps(effects, output):
    values = interaction(effects).groupby(["shift_type", "epsilon", "shift_pp"]).interaction.mean() * 100
    bound = max(float(values.abs().max()), 1e-6)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for axis, (kind, label) in zip(axes.flat, LABELS.items()):
        matrix = values.loc[kind].unstack("shift_pp")
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set(title=label, xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
                 yticks=range(len(matrix.index)), yticklabels=[f"{epsilon:g}" for epsilon in matrix.index],
                 xlabel="Signed shift (percentage points)", ylabel="ε")
    figure.colorbar(image, ax=axes, label="Extra DP-SGD AUPRC penalty (percentage points)")
    save(figure, output, "privacy_shift_interaction", "Positive values indicate a larger shift penalty under DP-SGD.")


def run(input, output=None, progress=print):
    path = tables.results_path(input)
    output = Path(output) if output is not None else path.parent / "plots"
    results = tables.read(path)
    effects = signed_effects(results)
    output.mkdir(parents=True, exist_ok=True)
    progress("Plotting signed response, privacy–shift interaction, and negative transfer…")
    response(effects, output)
    heatmaps(effects, output)
    negative_transfer(signed_effects(results, reference="scratch"), output)
    return {"figures": 3, "output": str(output)}
