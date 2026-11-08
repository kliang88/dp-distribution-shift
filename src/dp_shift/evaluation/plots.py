"""Four research figures built entirely from retained run scores and curve summaries."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dp_shift_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 5.2), sharey=True)
    specs = [("average_precision", "Average precision ↑", reference["constant_ap"]),
             ("auroc", "AUROC ↑", .5), ("brier", "Brier score ↓", None)]
    for axis, (metric, title, baseline) in zip(axes, specs):
        for y, (path, (method, _)) in enumerate(CONDITIONS.items()):
            values = runs.loc[runs.path_id.eq(path)].sort_values("repeat_id")[metric].to_numpy()
            row = interval(values)
            axis.scatter(values, y + np.linspace(-.08, .08, len(values)), color=COLORS[method],
                         alpha=.3, s=13, zorder=3)
            axis.errorbar(row["mean"], y, xerr=[[row["mean"] - row["ci_lower"]],
                          [row["ci_upper"] - row["mean"]]], fmt="o", color=COLORS[method],
                          capsize=3, markersize=5, zorder=4)
        if baseline is not None:
            axis.axvline(baseline, color="gray", linestyle="--", linewidth=1)
        axis.axhspan(2.5, 5.5, color="#f1f4f5", zorder=0)
        axis.set_title(title, fontsize=11, pad=12)
        axis.grid(axis="x", alpha=.15)
        axis.tick_params(axis="y", length=0)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].set_yticks(range(6), [model_label(path).replace(" · ", "\n") for path in CONDITIONS])
    axes[0].invert_yaxis()
    figure.suptitle("Model performance", fontsize=14, fontweight="bold", x=.02, ha="left")
    figure.text(.02, .025, f"{INTERVAL_NOTE}\n"
                f"Dashed: constant-score AP and chance AUROC. Test prevalence: {reference['test_prevalence']:.2%}.",
                fontsize=8, color="#505963")
    figure.tight_layout(rect=[0, .16, 1, .94], w_pad=2)
    save_figure(figure, output, "performance")


def plot_effects(effects, output):
    """Separate three questions, each with its own baseline and explicit sign."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
    panels = [
        ("A. Pretraining benefit", "Pretraining − no pretraining",
         [("ap_gain_sgd_s0", model_label("B-S0").replace(" · ", "\n"), "sgd"),
          ("ap_gain_sgd_s1", model_label("B-S1").replace(" · ", "\n"), "sgd"),
          ("ap_gain_dp_s0", model_label("D-S0").replace(" · ", "\n"), "dp_sgd"),
          ("ap_gain_dp_s1", model_label("D-S1").replace(" · ", "\n"), "dp_sgd")],
         "Positive = pretraining helps"),
        ("B. Source distribution-shift effect", "Shifted-source − baseline-source pretraining",
         [("ap_shift_sgd", "SGD", "sgd"), ("ap_shift_dp", "DP-SGD", "dp_sgd")],
         "Negative = source shift hurts"),
        ("C. Additional source-shift effect under DP", "DP-SGD shift − SGD shift",
         [("ap_dp_interaction", "DP-SGD\n− SGD", None)],
         "Negative = DP-SGD responds less favorably"),
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
            axis.annotate(f"{row['mean'] * 100:+.2f}", (row["mean"] * 100, y),
                          xytext=(0, 10), textcoords="offset points", ha="center",
                          fontsize=8, color=color)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(comparisons)), [label for _, label, _ in comparisons])
        axis.set_ylim(len(comparisons) - .45, -.55)
        axis.set_xlabel("AP difference (percentage points)", labelpad=10)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=38)
        axis.text(0, 1.05, formula, transform=axis.transAxes, color="#505963", fontsize=9)
        axis.text(.5, -.42, direction, transform=axis.transAxes, ha="center", fontsize=8.5)
        axis.grid(axis="x", alpha=.15)
        axis.tick_params(axis="y", length=0, pad=8)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        for spine in ("left", "top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("Paired effects on average precision", x=.02, ha="left",
                    fontsize=14, fontweight="bold")
    figure.text(.02, .025, "Points: paired differences. Bars: 95% t intervals across repeats on the fixed test set.",
                fontsize=8, color="#505963")
    figure.subplots_adjust(left=.18, right=.99, top=.72, bottom=.34, wspace=.75)
    save_figure(figure, output, "paired_effects")


def precision_on_grid(precision, recall, grid):
    """Step interpolation for display only; scalar AP uses the original curve."""
    indices = np.searchsorted(recall[::-1], grid, side="left")
    return precision[::-1][np.minimum(indices, len(precision) - 1)]


def plot_precision_recall(runs, curves, reference, source_rates, output):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True, sharey="row",
                                gridspec_kw={"height_ratios": [1, 1.5]})
    grid = np.linspace(0, 1, 501)
    for column, source in enumerate(("s0", "s1")):
        overview, detail = axes[:, column]
        for path in ("A", f"B-{source.upper()}", "C", f"D-{source.upper()}"):
            method, initialization = CONDITIONS[path]
            style = "--" if initialization == "scratch" else "-"
            aligned = []
            group = runs.loc[runs.path_id.eq(path)].sort_values("repeat_id")
            for run in group.itertuples():
                precision = curves[curve_key(run.repeat_id, path, "precision")]
                recall = curves[curve_key(run.repeat_id, path, "recall")]
                for axis in (overview, detail):
                    axis.step(recall, precision, where="post", color=COLORS[method], linestyle=style,
                              alpha=.09, linewidth=.6)
                aligned.append(precision_on_grid(precision, recall, grid))
            for axis in (overview, detail):
                axis.step(grid, np.mean(aligned, axis=0), where="pre", color=COLORS[method], linestyle=style,
                          linewidth=1.8, label=f"{model_label(path)} · AP {group.average_precision.mean():.3f}")
        for axis in (overview, detail):
            axis.axhline(reference["test_prevalence"], color="gray", linestyle=":", linewidth=1)
            axis.set_xlim(0, 1)
            axis.grid(alpha=.12)
            axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        overview.set(title=f"{'Baseline' if source == 's0' else 'Shifted'} source ({source_rates[source]:.1%} positive)",
                     ylim=(0, 1))
        overview.legend(frameon=False, fontsize=7.5, loc="upper right")
        detail.set(ylim=(0, .35), xlabel="Recall", title="Detail: precision 0–35% (same curves)")
    axes[0, 0].set_ylabel("Precision · full range")
    axes[1, 0].set_ylabel("Precision · detail")
    figure.suptitle("Precision–recall trade-offs", fontsize=14, fontweight="bold", x=.02, ha="left")
    figure.text(.02, .025, "Bold: mean step precision on a common recall grid. Faint: individual repeats; no prediction ensembling.\n"
                "AP is averaged from exact per-run scores. Dotted: test prevalence. Models without pretraining are reused across source columns; lower panels crop high precision.",
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
    histogram_limit = min(100., max(10., np.ceil(max(
        curves[curve_key(row.repeat_id, row.path_id, "histogram_n")].max()
        / reference["n"] * 100 for row in runs.itertuples()) / 10) * 10))
    for index, path in enumerate(CONDITIONS):
        method, _ = CONDITIONS[path]
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
        histogram_axis.set_ylim(0, histogram_limit)
        histogram_axis.axvline(reference["test_prevalence"], color="#555555", linestyle=":", linewidth=1)
        if tail >= .0005:
            histogram_axis.text(.98, .85, f"{tail:.1%} above range", transform=histogram_axis.transAxes,
                                ha="right", va="top", fontsize=7)
        axis.plot([0, limit], [0, limit], color="gray", linestyle="--", linewidth=.8)
        bias = runs.loc[runs.path_id.eq(path), "calibration_bias"].mean()
        axis.text(.04, .95, f"Mean risk bias: {bias * 100:+.2f} pp", transform=axis.transAxes,
                  va="top", fontsize=8, color=COLORS[method],
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
        axis.set(xlim=(0, limit), ylim=(0, limit))
        axis.set_title(model_label(path).replace(" · ", "\n"), fontsize=10)
        axis.tick_params(labelbottom=False)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        histogram_axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(alpha=.12)
        histogram_axis.grid(axis="y", alpha=.12)
        if index % 3 == 0:
            axis.set_ylabel("Observed readmission rate")
            histogram_axis.set_ylabel("Patients\n(%)", fontsize=8)
        else:
            axis.tick_params(labelleft=False)
        if index >= 3:
            histogram_axis.set_xlabel("Predicted probability")
    figure.suptitle("Calibration and predicted-risk distributions", x=.07, y=.99, ha="left",
                    fontsize=14, fontweight="bold")
    figure.text(.07, .02, "Curves: repeats, up to 10 quantile bins; ties stay together. Above diagonal: underprediction; below: overprediction.\n"
                f"Histograms: mean patient fraction; dotted vertical line: test prevalence ({reference['test_prevalence']:.2%}). Shared scales. Bias in percentage points (pp).",
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
        plot_effects(effects, output)
        plot_precision_recall(runs, curves, reference, source_rates, output)
        plot_calibration(runs, curves, reference, output)
    return {"figures": len(FIGURES)}
