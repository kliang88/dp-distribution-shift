"""Four research figures built entirely from retained run scores and curve summaries."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dp_shift_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import CONDITIONS, curve_key, model_label, validate_curves
from .provenance import atomic_path
from .summarize import interval, validate_runs


FIGURES = ("performance", "paired_effects", "precision_recall", "calibration")
COLORS = {"sgd": "#28658c", "dp_sgd": "#bc6630"}
INTERVAL_NOTE = "Dots: individual repeats. Bars: mean and 95% t interval across repeats on the fixed test set."


def save_figure(figure, output, name):
    try:
        with atomic_path(Path(output) / f"figures/{name}.png") as temporary:
            figure.savefig(temporary, format="png", dpi=300, bbox_inches="tight",
                           metadata={"Software": "dp-shift-evaluation"})
    finally:
        plt.close(figure)


def plot_performance(runs, reference, output):
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
    specs = [("average_precision", "Average precision ↑", reference["constant_ap"]),
             ("auroc", "AUROC ↑", .5), ("brier", "Brier score ↓", reference["constant_brier"])]
    for axis, (metric, title, baseline) in zip(axes, specs):
        for y, (path, (method, _)) in enumerate(CONDITIONS.items()):
            values = runs.loc[runs.path_id.eq(path)].sort_values("repeat_id")[metric].to_numpy()
            row = interval(values)
            axis.scatter(values, y + np.linspace(-.08, .08, len(values)), color=COLORS[method],
                         alpha=.3, s=13, zorder=3)
            axis.errorbar(row["mean"], y, xerr=[[row["mean"] - row["ci_lower"]],
                          [row["ci_upper"] - row["mean"]]], fmt="o", color=COLORS[method],
                          capsize=3, markersize=5, zorder=4)
        axis.axvline(baseline, color="gray", linestyle="--", linewidth=1)
        axis.axhspan(2.5, 5.5, color="#f1f4f5", zorder=0)
        axis.set_title(title, fontsize=11, pad=12)
        axis.grid(axis="x", alpha=.15)
        axis.tick_params(axis="y", length=0)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].set_yticks(range(6), [model_label(path) for path in CONDITIONS])
    axes[0].invert_yaxis()
    figure.suptitle("Model performance", fontsize=14, fontweight="bold", x=.02, ha="left")
    figure.text(.02, .025, f"{INTERVAL_NOTE}\nDashed: constant-score AP, chance AUROC, and observed-prevalence Brier reference "
                f"(test prevalence {reference['test_prevalence']:.2%}).", fontsize=8, color="#505963")
    figure.tight_layout(rect=[0, .13, 1, .94], w_pad=2)
    save_figure(figure, output, "performance")


def plot_effects(effects, threshold, output):
    """Separate three questions, each with its own baseline and explicit sign."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
    panels = [
        ("1. Does pretraining help?", "Pretrained − scratch",
         [("ap_gain_sgd_s0", "SGD · S0", "sgd"),
          ("ap_gain_sgd_s1", "SGD · S1", "sgd"),
          ("ap_gain_dp_s0", "DP-SGD · S0", "dp_sgd"),
          ("ap_gain_dp_s1", "DP-SGD · S1", "dp_sgd")],
         "Positive = pretraining helps"),
        ("2. Does the source shift hurt?", "S1-pretrained − S0-pretrained",
         [("ap_shift_sgd", "SGD", "sgd"), ("ap_shift_dp", "DP-SGD", "dp_sgd")],
         "Negative = S1 performs worse"),
        ("3. Is DP affected differently?", "DP shift − SGD shift",
         [("ap_dp_interaction", "DP − SGD", None)],
         "Negative = DP responds less favorably"),
    ]
    indexed = effects.set_index("effect")
    for axis, (title, formula, comparisons, direction) in zip(axes, panels):
        for y, (name, label, method) in enumerate(comparisons):
            row = indexed.loc[name]
            color = COLORS[method] if method else "#343d46"
            axis.errorbar(row["mean"] * 100, y,
                          xerr=[[(row["mean"] - row.ci_lower) * 100], [(row.ci_upper - row["mean"]) * 100]],
                          fmt="o" if method else "D", color=color, capsize=4,
                          markersize=7, linewidth=1.8, zorder=3)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(comparisons)), [label for _, label, _ in comparisons])
        axis.set_ylim(len(comparisons) - .45, -.55)
        axis.set_xlabel("AP difference (percentage points)", labelpad=10)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=38)
        axis.text(0, 1.05, formula, transform=axis.transAxes, color="#505963", fontsize=9)
        axis.text(.5, -.52, direction, transform=axis.transAxes, ha="center", fontsize=8.5)
        axis.grid(axis="x", alpha=.15)
        axis.tick_params(axis="y", length=0, pad=8)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        for spine in ("left", "top", "right"):
            axis.spines[spine].set_visible(False)
    band = threshold * 100
    axes[2].axvspan(-band, band, color="gray", alpha=.12, zorder=0)
    axes[2].set_xlim(min(axes[2].get_xlim()[0], -band * 1.2), max(axes[2].get_xlim()[1], band * 1.2))
    figure.suptitle("Three questions about pretraining and source shift", x=.02, ha="left",
                    fontsize=14, fontweight="bold")
    figure.text(.02, .025, "Points: mean paired differences. Bars: 95% t intervals across repeats on the fixed test set. Each panel has its own horizontal scale.\n"
                f"S0: natural source prevalence; S1: shifted source prevalence. Gray band: interaction within ±{band:g} AP percentage points.",
                fontsize=8, color="#505963")
    figure.subplots_adjust(left=.10, right=.99, top=.72, bottom=.34, wspace=.70)
    save_figure(figure, output, "paired_effects")


def precision_on_grid(precision, recall, grid):
    """Step interpolation for display only; scalar AP uses the original curve."""
    indices = np.searchsorted(recall[::-1], grid, side="left")
    return precision[::-1][np.minimum(indices, len(precision) - 1)]


def plot_precision_recall(runs, curves, reference, source_rates, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    grid = np.linspace(0, 1, 501)
    for axis, source in zip(axes, ("s0", "s1")):
        for path in ("A", f"B-{source.upper()}", "C", f"D-{source.upper()}"):
            method, initialization = CONDITIONS[path]
            style = "--" if initialization == "scratch" else "-"
            aligned = []
            group = runs.loc[runs.path_id.eq(path)].sort_values("repeat_id")
            for run in group.itertuples():
                precision = curves[curve_key(run.repeat_id, path, "precision")]
                recall = curves[curve_key(run.repeat_id, path, "recall")]
                axis.step(recall, precision, where="post", color=COLORS[method], linestyle=style,
                          alpha=.12, linewidth=.6)
                aligned.append(precision_on_grid(precision, recall, grid))
            axis.step(grid, np.mean(aligned, axis=0), where="pre", color=COLORS[method], linestyle=style,
                      linewidth=1.8, label=f"{model_label(path)} · AP {group.average_precision.mean():.3f}")
        axis.axhline(reference["test_prevalence"], color="gray", linestyle=":", linewidth=1)
        axis.set(title=f"{source.upper()} source · {source_rates[source]:.1%} positive",
                 xlim=(0, 1), ylim=(0, 1), xlabel="Recall")
        axis.grid(alpha=.12)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Precision")
    figure.suptitle("Precision–recall trade-offs", fontsize=14, fontweight="bold", x=.02, ha="left")
    figure.text(.02, .025, "Bold: mean step precision on a common recall grid. Faint: individual repeats; no prediction ensembling.\n"
                "AP is averaged from exact per-run scores. Dotted: test prevalence. Scratch curves are reused in both panels.",
                fontsize=8, color="#505963")
    figure.tight_layout(rect=[0, .13, 1, .94])
    save_figure(figure, output, "precision_recall")


def plot_calibration(runs, curves, reference, output):
    figure = plt.figure(figsize=(10, 7))
    layout = figure.add_gridspec(2, 3, hspace=.5, wspace=.22)
    values = [curves[curve_key(row.repeat_id, row.path_id, field)].max()
              for row in runs.itertuples()
              for field in ("calibration_mean_prediction", "calibration_observed_rate")]
    # Round to a histogram edge so cropped tail mass can be reported exactly.
    limit = min(1., max(.1, np.ceil(max(values) * 1.1 * 50) / 50))
    for index, path in enumerate(CONDITIONS):
        method, initialization = CONDITIONS[path]
        cell = layout[index // 3, index % 3].subgridspec(2, 1, height_ratios=[4, 1], hspace=.08)
        axis = figure.add_subplot(cell[0])
        histogram_axis = figure.add_subplot(cell[1], sharex=axis)
        histograms = []
        for run in runs.loc[runs.path_id.eq(path)].sort_values("repeat_id").itertuples():
            data = {field: curves[curve_key(run.repeat_id, path, field)] for field in
                    ("calibration_mean_prediction", "calibration_observed_rate", "histogram_edges", "histogram_n")}
            axis.plot(data["calibration_mean_prediction"], data["calibration_observed_rate"],
                      marker="o", markersize=3, color=COLORS[method], linewidth=.9, alpha=.5)
            histograms.append(data["histogram_n"] / reference["n"])
        edges = data["histogram_edges"]
        frequencies = np.mean(histograms, axis=0)
        histogram_axis.bar(edges[:-1], frequencies * 100, width=np.diff(edges), align="edge",
                           color=COLORS[method], alpha=.55)
        tail = frequencies[edges[:-1] >= limit - 1e-12].sum()
        if tail > 0:
            histogram_axis.text(.98, .85, f"{tail:.1%} above range", transform=histogram_axis.transAxes,
                                ha="right", va="top", fontsize=7)
        axis.plot([0, limit], [0, limit], color="gray", linestyle="--", linewidth=.8)
        axis.set(xlim=(0, limit), ylim=(0, limit),
                 title=f"{'SGD' if method == 'sgd' else 'DP-SGD'} · {initialization.upper() if initialization != 'scratch' else 'Scratch'}")
        axis.tick_params(labelbottom=False)
        axis.grid(alpha=.12)
        histogram_axis.grid(axis="y", alpha=.12)
        if index % 3 == 0:
            axis.set_ylabel("Observed positive fraction")
            histogram_axis.set_ylabel("Patients\n(%)", fontsize=8)
        else:
            axis.tick_params(labelleft=False)
        if index >= 3:
            histogram_axis.set_xlabel("Predicted probability")
    figure.suptitle("Calibration and predicted-risk distributions", x=.07, y=.99, ha="left",
                    fontsize=14, fontweight="bold")
    figure.text(.07, .02, "Each curve is one repeat, using up to 10 quantile bins; tied predictions stay together. Dashed: perfect calibration.\n"
                f"Histograms: mean patient fraction across repeats. Shared axes; test prevalence {reference['test_prevalence']:.2%}.",
                fontsize=8, color="#505963")
    figure.subplots_adjust(top=.90, bottom=.14)
    save_figure(figure, output, "calibration")


def plot(output, runs, curves, effects, reference, source_rates):
    """Render from retained inputs, including after temporary models are deleted."""
    validate_runs(runs, runs.repeat_id.nunique())
    validate_curves(curves, runs, reference)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        plot_performance(runs, reference, output)
        plot_effects(effects, .005, output)
        plot_precision_recall(runs, curves, reference, source_rates, output)
        plot_calibration(runs, curves, reference, output)
    return {"figures": len(FIGURES)}
