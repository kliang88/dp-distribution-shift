"""Research figures, faceted automatically by registered shift type."""

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


FIGURES = ("overall_performance", "pretraining_benefit", "dp_amplification", "privacy_utility",
           "source_target_prevalence", "source_target_gap")
INITIALIZATIONS = ("none", "baseline", "shifted")
COLORS = ("#777777", "#28658c", "#bc6630")
CI_NOTE = "95% t intervals across repeats on the fixed target test set."
SINGLE_REPEAT_CI_NOTE = "Single repeat. Confidence intervals require multiple repeats."
PERFORMANCE_PANELS = {
    "readmission": ("Readmission Prevalence", "{level} source readmission rate"),
    "age_70_plus": ("Age Distribution", "{level} of source patients aged ≥70"),
    "primary_diagnosis_circulatory": ("Primary Diagnosis", "{level} source replacement · Circulatory primary diagnosis"),
    "clinical_complexity_2plus": ("Clinical Complexity", "{level} source replacement · ≥2 diagnosis categories"),
}
SOURCE_SHIFT_PANELS = {
    "readmission": ("Readmission Prevalence Shift", "Share readmitted within 30 days", "Source readmission rate (%)"),
    "age_70_plus": ("Patient Age Distribution Shift", "Share of patients aged ≥70", "Source patients aged ≥70 (%)"),
    "primary_diagnosis_circulatory": ("Primary Diagnosis Shift", "Oversample circulatory primary diagnoses",
                                      "Source patients replaced (%)"),
    "clinical_complexity_2plus": ("Clinical Complexity Shift", "Oversample patients with ≥2 distinct diagnosis groups",
                                  "Source patients replaced (%)"),
}


def save(figure, output, name, caption, *, layout_rect=(0, .12, 1, 1)):
    try:
        figure.text(.5, .015, caption, ha="center", fontsize=8, color="#505963")
        figure.tight_layout(rect=layout_rect)
        with atomic_path(Path(output) / f"{name}.png") as temporary:
            figure.savefig(temporary, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def panel_grid(count, *, sharey=True):
    columns = min(2, count)
    rows = (count + columns - 1) // columns
    figure, grid = plt.subplots(rows, columns, figsize=(10.8 if columns == 2 else 7.4, 2.9 * rows + 2),
                                sharey=sharey, squeeze=False)
    axes = grid.ravel()
    for axis in axes[count:]:
        figure.delaxes(axis)
    return figure, axes[:count]


def estimates(groups, scale=1):
    rows = [interval(np.asarray(values) * scale) for values in groups]
    means = np.array([row["mean"] for row in rows])
    errors = np.array([[row["mean"] - row["ci_lower"] for row in rows],
                       [row["ci_upper"] - row["mean"] for row in rows]])
    return means, None if all(row["n"] == 1 for row in rows) else errors


def line(axis, points, label, color, scale=1, *, fmt="o-", linewidth=1.5, markersize=4):
    points = sorted(points, key=lambda point: point[0])
    x = np.array([value for value, _ in points], dtype=float)
    means, errors = estimates([values for _, values in points], scale)
    axis.errorbar(x, means, yerr=errors, fmt=fmt, color=color,
                  label=label, capsize=3, linewidth=linewidth, markersize=markersize)


def performance_panel_title(axis, kind, data):
    shift = SHIFTS[kind]
    default_subtitle = "{level} source replacement" if shift.replacement_rate else "{level} source subgroup share"
    title, subtitle = PERFORMANCE_PANELS.get(kind, (shift.label, default_subtitle))
    shares = data.pivot(index=["comparison_epsilon", "repeat", "method"],
                        columns="pretraining", values="source_fraction")
    achieved = shares["shifted"]
    if shift.replacement_rate:
        # Every replacement adds one subgroup member while keeping the source size fixed.
        achieved = achieved - shares["baseline"]
    axis.set_title(title, loc="left", pad=20, fontweight="bold", color="#172332")
    axis.text(0, 1.02, subtitle.format(level=f"{100 * achieved.mean():g}%"), transform=axis.transAxes,
              va="bottom", fontsize=9, color="#505963")


def overall_performance(runs, output, epsilon, fractions, note=CI_NOTE):
    figure, axes = panel_grid(len(fractions))
    height = figure.get_figheight()
    x = np.arange(len(INITIALIZATIONS))
    selected = runs.loc[runs.comparison_epsilon.eq(epsilon) & runs.shift_fraction.eq(runs.shift_type.map(fractions))]
    methods = (("sgd", "SGD", COLORS[1], -.18),
               ("dp_sgd", f"DP-SGD (ε = {epsilon:g})", COLORS[2], .18))
    for axis, kind in zip(axes, fractions):
        data = selected.loc[selected.shift_type.eq(kind)]
        for method, label, color, offset in methods:
            scores = data.loc[data.method.eq(method)]
            means, errors = estimates([scores.loc[scores.pretraining.eq(init), "auprc"] for init in INITIALIZATIONS])
            axis.bar(x + offset, means, width=.34, yerr=errors, color=color, label=label, capsize=3)
        axis.set(xticks=x, xticklabels=("Scratch", "Unshifted", "Shifted"), xlim=(-.5, 2.5))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#dbe3ec", linewidth=.7, alpha=.75)
        performance_panel_title(axis, kind, data)
    highest = max(axis.dataLim.ymax for axis in axes)
    axes[0].set_ylim(0, max(axes[0].get_ylim()[1], 1.08 * highest))
    figure.suptitle("Overall Target-Test Performance Across Training Conditions",
                    fontsize=18 if len(axes) > 1 else 15, fontweight="bold", color="#172332", y=1 - .20 / height)
    figure.supylabel("Target-test AUPRC", x=.02, fontsize=11)
    figure.legend(*axes[0].get_legend_handles_labels(), ncols=2, loc="upper center",
                  bbox_to_anchor=(.5, 1 - .57 / height), frameon=False)

    caption = ["Unshifted and Shifted use source pretraining before target training."]
    references = selected.loc[selected.method.eq("dp_sgd") & selected.pretraining.ne("shifted")]
    if references.groupby(["repeat", "pretraining"]).auprc.nunique().gt(1).any():
        caption.append("DP-SGD scratch and unshifted scores vary across panels because the models are trained separately.")
    repeats = selected.repeat.nunique()
    caption.append(f"{repeats} repeats. {note}" if repeats > 1 else note)
    if any(axis.dataLim.ymin < 0 for axis in axes):
        caption[-1] += " Intervals extending below zero are clipped at zero."
    save(figure, output, "overall_performance", "\n".join(caption),
         layout_rect=(.025, .55 / height, .985, 1 - .76 / height))


def pretraining_benefit(effects, output, budgets, colors, baselines, fractions, note=CI_NOTE):
    figure, axes = panel_grid(len(fractions))
    height = figure.get_figheight()
    series = [(budgets[0], "gain_sgd", "SGD", COLORS[1])]
    series.extend((budget, "gain_dp", f"DP-SGD ε={budget:g}", colors[budget]) for budget in budgets)
    for axis, (kind, fraction) in zip(axes, fractions.items()):
        baseline = baselines[kind]
        panel = effects.loc[effects.shift_type.eq(kind)]
        for budget, metric, label, color in series:
            data = panel.loc[panel.comparison_epsilon.eq(budget)]
            points = [(value, group[f"{metric}_shifted"]) for value, group in data.groupby("shift_fraction")]
            reference = data.loc[data.shift_fraction.eq(fraction), f"{metric}_baseline"]
            line(axis, points, label, color, scale=100, linewidth=1, markersize=5)
            line(axis, [(baseline, reference)], None, color, scale=100, fmt="s", linewidth=1, markersize=6)
        source_shift_axis(axis, kind, baseline, baseline_label="Unshifted source")

    figure.suptitle("How Does Source Distribution Shift Affect the Benefit of Pretraining?",
                    fontsize=17 if len(axes) > 1 else 13, fontweight="bold", color="#172332", y=1 - .20 / height)
    figure.text(.5, 1 - .62 / height, "Target-test AUPRC with source pretraining minus AUPRC from scratch.",
                ha="center", fontsize=10, color="#505963")
    figure.supylabel("Pretraining gain (AUPRC percentage points)", x=.02, fontsize=11)
    legend_columns = min(4 if len(axes) == 1 else 5, len(series))
    legend_rows = (len(series) + legend_columns - 1) // legend_columns
    extra_legend_height = .24 * (legend_rows - 1)
    figure.legend(*axes[0].get_legend_handles_labels(), ncols=legend_columns,
                  loc="upper center", bbox_to_anchor=(.5, 1 - .77 / height), frameon=False)
    figure.text(.5, 1 - (1.19 + extra_legend_height) / height, "Dashed line · No benefit from pretraining",
                ha="center", fontsize=8, color="#415268")

    selected = "the selected comparison in each panel (lowest shift level by default)"
    if len(fractions) == 1:
        selected = f"the {100 * next(iter(fractions.values())):g}% comparison"
    caption = ["Positive gains indicate better target-test AUPRC with pretraining. Negative gains indicate worse performance.",
               f"Squares show matched unshifted gains from {selected}."]
    private_references = effects.groupby(["repeat", "comparison_epsilon"]).gain_dp_baseline
    if private_references.nunique().gt(1).any():
        caption.append("DP-SGD reference gains can vary because the models were trained separately for each comparison.")
    repeats = effects.repeat.nunique()
    caption.append(f"{repeats} repeats. {note}" if repeats > 1 else note)
    bottom = .55 + .12 * (len(caption) - 3)
    save(figure, output, "pretraining_benefit", "\n".join(caption),
         layout_rect=(.02, bottom / height, .985, 1 - (1.08 + extra_legend_height) / height))


def source_shift_axis(axis, kind, baseline, *, baseline_label="Baseline"):
    shift = SHIFTS[kind]
    default_unit = "Source patients replaced (%)" if shift.replacement_rate else "Source subgroup share (%)"
    title, description, xlabel = SOURCE_SHIFT_PANELS.get(kind, (shift.label, "", default_unit))
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", pad=20 if description else 8, fontweight="bold", color="#172332")
    if description:
        axis.text(0, 1.02, description, transform=axis.transAxes, va="bottom", fontsize=9, color="#505963")

    percentage = PercentFormatter(1)
    axis.xaxis.set_major_formatter(percentage)
    if shift.replacement_rate:
        axis.xaxis.set_major_formatter(
            lambda value, position: "0%\nUnshifted source" if value == 0 else percentage(value, position))
    else:
        axis.axvline(baseline, color="#687586", linestyle=":", linewidth=1)
        left, right = axis.get_xlim()
        near_right = baseline > (left + right) / 2
        axis.annotate(f"{baseline_label} {baseline:.1%}", (baseline, 1), xycoords=axis.get_xaxis_transform(),
                      xytext=(-5 if near_right else 5, -6), textcoords="offset points",
                      ha="right" if near_right else "left", va="top", fontsize=8, color="#505963")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dbe3ec", linewidth=.7, alpha=.75)
    axis.axhline(0, color="#415268", linestyle="--", linewidth=1.15)


def dp_amplification(effects, output, budgets, colors, baselines, note=CI_NOTE):
    figure, axes = panel_grid(len(baselines))
    height = figure.get_figheight()

    for axis, (kind, baseline) in zip(axes, baselines.items()):
        shifted = effects.loc[effects.shift_type.eq(kind)]
        for budget in budgets:
            data = shifted.loc[shifted.comparison_epsilon.eq(budget)]
            points = [(fraction, group.dp_amplification) for fraction, group in data.groupby("shift_fraction")]
            points.append((baseline, np.zeros(data.repeat.nunique())))
            line(axis, points, f"ε={budget:g}", colors[budget], scale=100)
        axis.plot(baseline, 0, "s", color="#415268", markeredgecolor="white", markersize=6, zorder=4)
        source_shift_axis(axis, kind, baseline)

    figure.suptitle("Does Differential Privacy Make Distribution Shift More Harmful?",
                    fontsize=18 if len(axes) > 1 else 15, fontweight="bold", color="#172332", y=1 - .20 / height)
    figure.text(.5, 1 - .63 / height,
                "Extra loss in target AUPRC from source shift under DP-SGD versus standard SGD.",
                ha="center", fontsize=10, color="#505963")
    figure.supylabel("Extra AUPRC loss under DP-SGD (percentage points)", x=.02, fontsize=11)
    legend_columns = min(5, len(budgets))
    legend_rows = (len(budgets) + legend_columns - 1) // legend_columns
    extra_legend_height = .20 * (legend_rows - 1)
    figure.legend(*axes[0].get_legend_handles_labels(), title="DP-SGD privacy budget", ncols=legend_columns,
                  loc="upper center", bbox_to_anchor=(.5, 1 - .83 / height), frameon=False)
    figure.text(.5, 1 - (1.37 + extra_legend_height) / height, "Dashed line · No additional DP effect",
                ha="center", fontsize=8, color="#415268")

    save(figure, output, "dp_amplification",
         "Positive values indicate greater sensitivity to source shift under DP-SGD. Negative values indicate lower sensitivity.\n"
         "AUPRC differences are in percentage points. " + note,
         layout_rect=(.02, .58 / height, .985, 1 - (1.54 + extra_legend_height) / height))


def privacy_utility(runs, output, budgets, fractions):
    figure, axes = panel_grid(len(fractions))
    height = figure.get_figheight()
    selected = runs.loc[runs.method.eq("dp_sgd") & runs.shift_fraction.eq(runs.shift_type.map(fractions))]
    labels = ("Scratch", "Unshifted pretraining", "Shifted pretraining")
    for axis, kind in zip(axes, fractions):
        data = selected.loc[selected.shift_type.eq(kind)]
        for initialization, label, color, marker in zip(INITIALIZATIONS, labels, COLORS, ("o", "s", "^")):
            group = data.loc[data.pretraining.eq(initialization)]
            line(axis, [(budget, values.auprc) for budget, values in group.groupby("comparison_epsilon")],
                 label, color, fmt=f"{marker}-", linewidth=1, markersize=5)
        axis.set(xticks=budgets, xticklabels=[f"{budget:g}" for budget in budgets], xlabel="Privacy budget (ε)")
        axis.margins(y=.08)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#dbe3ec", linewidth=.7, alpha=.75)
        performance_panel_title(axis, kind, data)

    figure.suptitle("How Does Privacy Strength Affect Model Performance?",
                    fontsize=18 if len(axes) > 1 else 15, fontweight="bold", color="#172332", y=1 - .20 / height)
    figure.text(.5, 1 - .62 / height, "Target-test AUPRC under DP-SGD · Smaller ε means stronger privacy",
                ha="center", fontsize=10, color="#505963")
    figure.supylabel("Target-test AUPRC", x=.02, fontsize=11)
    figure.legend(*axes[0].get_legend_handles_labels(), title="Training condition", ncols=3, loc="upper center",
                  bbox_to_anchor=(.5, 1 - .76 / height), frameon=False)

    caption = ["Pretraining benefit = Unshifted − Scratch. Shift effect = Shifted − Unshifted."]
    references = selected.loc[selected.pretraining.ne("shifted")]
    if references.groupby(["comparison_epsilon", "repeat", "pretraining"]).auprc.nunique().gt(1).any():
        caption.append("Scratch and unshifted scores can vary across panels because DP-SGD runs use independent sampling and noise.")
    repeats = selected.repeat.nunique()
    uncertainty = f"{repeats} repeats. {CI_NOTE}" if repeats > 1 else SINGLE_REPEAT_CI_NOTE
    caption.append(f"The shared AUPRC axis is zoomed to the plotted data. {uncertainty}")
    save(figure, output, "privacy_utility", "\n".join(caption),
         layout_rect=(.02, (.22 + .12 * len(caption)) / height, .985, 1 - 1.24 / height))


def size_label(values):
    low, high = int(values.min()), int(values.max())
    return f"n={low:,}" if low == high else f"n={low:,}–{high:,}"


def composition_bars(axis, x, means, errors, *, width):
    for points, color, label in ((slice(0, 1), COLORS[1], "Unshifted source"),
                                 (slice(1, None), COLORS[2], "Shifted source")):
        axis.bar(x[points], means[points], width=width, yerr=None if errors is None else errors[:, points].tolist(),
                 color=color, label=label, capsize=3)


def source_target_prevalence(composition, output, caption):
    figure, axes = panel_grid(len(composition), sharey=False)
    height = figure.get_figheight()
    for axis, (kind, (target, levels, means, errors)) in zip(axes, composition.items()):
        shift = SHIFTS[kind]
        title, description, _ = SOURCE_SHIFT_PANELS.get(kind, (shift.label, "", ""))
        separator = "\n" if len(levels) > 3 else " "
        suffix = f"{separator}replaced" if shift.replacement_rate else ""
        x = np.arange(len(means))
        composition_bars(axis, x, means, errors, width=.58)
        axis.axhline(target, color="#415268", linestyle="--", linewidth=1)
        axis.annotate(f"Target: {target:.1%}", (1, target), xycoords=axis.get_yaxis_transform(),
                      xytext=(-5, 5), textcoords="offset points", ha="right", fontsize=8, color="#415268",
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        axis.set(xticks=x, xticklabels=["Unshifted", *[f"{level}{suffix}" for level in levels]])
        axis.margins(x=.24, y=.16)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#dbe3ec", linewidth=.7, alpha=.75)
        axis.set_title(title, loc="left", pad=20 if description else 8, fontweight="bold", color="#172332")
        if description:
            axis.text(0, 1.02, description, transform=axis.transAxes, va="bottom", fontsize=9, color="#505963")

    figure.suptitle("Source and Target Population Characteristics",
                    fontsize=18 if len(axes) > 1 else 15, fontweight="bold", color="#172332", y=1 - .20 / height)
    figure.text(.5, 1 - .62 / height, "Observed characteristic prevalence before and after source shifting",
                ha="center", fontsize=10, color="#505963")
    figure.supylabel("Patients with characteristic (%)", x=.02, fontsize=11)
    figure.legend(*axes[0].get_legend_handles_labels(), ncols=2, loc="upper center",
                  bbox_to_anchor=(.5, 1 - .77 / height), frameon=False)
    scale_note = "Y-axis scales differ by panel.\n" if len(axes) > 1 else ""
    save(figure, output, "source_target_prevalence", scale_note + caption,
         layout_rect=(.02, .46 / height, .985, 1 - 1.09 / height))


def source_target_gap(composition, output, caption):
    figure, axis = plt.subplots(figsize=(max(7.4, 2.65 * len(composition)), 4.9))
    height = figure.get_figheight()
    labels = []
    slot = .8 / max(len(means) for _, _, means, _ in composition.values())
    for index, (kind, (target, levels, means, errors)) in enumerate(composition.items()):
        shift = SHIFTS[kind]
        title = PERFORMANCE_PANELS.get(kind, (shift.label, ""))[0]
        suffix = " replaced" if shift.replacement_rate else ""
        labels.append(f"{title}\n{' / '.join(levels)}{suffix}")
        offsets = (np.arange(len(means)) - (len(means) - 1) / 2) * slot
        composition_bars(axis, index + offsets, 100 * (means - target),
                         None if errors is None else 100 * errors, width=.8 * slot)

    axis.axhline(0, color="#415268", linewidth=1.3, zorder=.9)
    axis.set(xticks=np.arange(len(labels)), xticklabels=labels, xlim=(-.5, len(labels) - .5),
             ylabel="Source minus target prevalence (percentage points)")
    axis.margins(y=.12)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dbe3ec", linewidth=.7, alpha=.75)
    figure.suptitle("How Different Is the Source Population from the Target?",
                    fontsize=17 if len(composition) > 1 else 15, fontweight="bold", color="#172332",
                    y=1 - .20 / height)
    figure.text(.5, 1 - .62 / height, "Difference in observed characteristic prevalence by source condition",
                ha="center", fontsize=10, color="#505963")
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(handles[:2], legend_labels[:2], ncols=2, loc="upper center",
                  bbox_to_anchor=(.5, 1 - .77 / height), frameon=False)
    save(figure, output, "source_target_gap",
         "Positive gaps mean higher source prevalence. Negative gaps mean higher target prevalence. Zero means equal shares.\n"
         "Shifted bars follow the listed levels from left to right.\n" + caption,
         layout_rect=(.02, .58 / height, .985, 1 - 1.09 / height))


def source_target_composition(runs, output, target_size, fractions):
    # The cohort records are repeated across methods and privacy budgets.
    cohorts = runs.loc[runs.method.eq("sgd") & runs.comparison_epsilon.eq(runs.comparison_epsilon.min())
                       & runs.shift_type.isin(fractions)]
    baseline = cohorts.pretraining.eq("baseline") & cohorts.shift_fraction.eq(cohorts.shift_type.map(fractions))
    source = cohorts.loc[baseline | cohorts.pretraining.eq("shifted")]
    composition = {}
    for kind in fractions:
        data = source.loc[source.shift_type.eq(kind)]
        unshifted = data.loc[data.pretraining.eq("baseline"), "source_fraction"]
        shifted = data.loc[data.pretraining.eq("shifted")].groupby("shift_fraction").source_fraction
        levels, values = zip(*shifted)
        means, errors = estimates([unshifted, *values])
        composition[kind] = (data.test_fraction.iloc[0], [f"{100 * level:g}%" for level in levels], means, errors)

    repeats = source.repeat.nunique()
    uncertainty = f"Means and 95% t intervals across {repeats} repeats." if repeats > 1 else SINGLE_REPEAT_CI_NOTE
    caption = f"Source {size_label(source.source_size)} per cohort. Fixed target test n={target_size:,}. {uncertainty}"
    source_target_prevalence(composition, output, caption)
    source_target_gap(composition, output, caption)


def plot(runs, effects, target_size, output, epsilon, fractions, *, overall_fractions=None):
    single = runs["repeat"].nunique() == 1
    note = "Single repeat; uncertainty not estimated." if single else CI_NOTE
    budgets = sorted(runs.comparison_epsilon.unique())
    colors = dict(zip(budgets, plt.get_cmap("viridis")(np.linspace(.15, .8, len(budgets)))))
    performance_fractions = fractions if overall_fractions is None else overall_fractions
    baselines = {kind: 0.0 if SHIFTS[kind].replacement_rate else
                 runs.loc[runs.shift_type.eq(kind) & runs.method.eq("sgd") & runs.pretraining.eq("baseline")
                               & runs.comparison_epsilon.eq(budgets[0]) & runs.shift_fraction.eq(fraction), "source_fraction"].mean()
                 for kind, fraction in fractions.items()}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        overall_performance(runs, output, epsilon, performance_fractions, note=note)
        pretraining_benefit(effects, output, budgets, colors, baselines, fractions, note=note)
        dp_amplification(effects, output, budgets, colors, baselines, note=note)
        privacy_utility(runs, output, budgets, performance_fractions)
        source_target_composition(runs, output, target_size, fractions)
    return len(FIGURES)
