"""Presentation figures for paired AUPRC and AUROC comparisons at epsilon 5."""

from dp_shift.data.utils import require
from . import plots, tables
from .report import plot_paths
from .summarize import paired_effects


EPSILON = 5
REPEATS = 20
METRICS = ("auprc", "auroc")
READMISSION_LEVELS = (.01, .04, .14, .17)
SHIFT_LABELS = {
    "readmission": "Readmission",
    "age_70_plus": "Age 70+",
    "clinical_complexity_2plus": "Clinical\ncomplexity",
    "primary_diagnosis_circulatory": "Circulatory\ndiagnosis",
}
METHODS = (("dp", "DP-SGD", "#2166ac", -.18), ("sgd", "SGD", "#d95f02", .18))
CAPTION = f"ε = {EPSILON}, n = {REPEATS} repeats, 95% CI"


def save_slide(figure, axis, output, name, metric):
    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    axis.legend(frameon=False)
    filename = name if metric == "auprc" else f"{name}_{metric}"
    plots.save(figure, output, filename, CAPTION, layout_rect=(0, .07, 1, 1))


def readmission_prevalence(effects, target, output, metric):
    figure, axis = plots.plt.subplots(figsize=(8.5, 4.7))
    for method, label, color, _ in METHODS:
        # The report stores baseline minus shifted. This slide shows shifted minus baseline.
        points = [(100 * fraction, -group[f"shift_penalty_{method}"])
                  for fraction, group in effects.groupby("shift_fraction")]
        plots.line(axis, points, label, color, scale=100, linewidth=2.2, markersize=6)
    axis.axvline(100 * target, color="gray", linestyle=":", linewidth=1,
                 label=f"Target prevalence ≈{target:.2%}")
    axis.set(xticks=[100 * level for level in READMISSION_LEVELS],
             xlabel="Source readmission prevalence (%)",
             ylabel=f"Shifted − baseline {metric.upper()}\n(percentage points)",
             title="Readmission prevalence shift changes pretraining benefit")
    save_slide(figure, axis, output, "slide6_readmission_prevalence", metric)


def pretraining_gain(effects, output, metric):
    gains = effects.groupby(["shift_type", "repeat"])[["gain_dp_shifted", "gain_sgd_shifted"]].mean()
    figure, axis = plots.plt.subplots(figsize=(9, 4.7))
    axis.use_sticky_edges = False  # Keep the zero reference visible when all intervals are above zero.
    positions = range(len(SHIFT_LABELS))
    for method, label, color, offset in METHODS:
        means, errors = plots.estimates([gains.loc[kind, f"gain_{method}_shifted"] for kind in SHIFT_LABELS], scale=100)
        axis.bar([position + offset for position in positions], means, width=.32, color=color,
                 label=label, yerr=errors, ecolor="#303942", capsize=4)
    axis.set(xticks=positions, xticklabels=list(SHIFT_LABELS.values()),
             xlim=(-.5, len(SHIFT_LABELS) - .5),
             ylabel=f"Shifted pretraining − scratch {metric.upper()}\n(percentage points)",
             title="Shifted pretraining benefits DP-SGD across shift types")
    save_slide(figure, axis, output, "slide7_pretraining_gain", metric)


def run(input, output=None, progress=print):
    path, output = plot_paths(input, output, default_directory="plots/slides")
    progress("Reading and validating model scores…")
    runs = tables.read(path)
    selected = runs.loc[runs.comparison_epsilon.eq(EPSILON)]
    require(not selected.empty, f"Slide figures require comparison_epsilon = {EPSILON}")
    require(selected.repeat.nunique() == REPEATS, f"Slide figures require {REPEATS} paired repeats")
    levels = selected.groupby("shift_type").shift_fraction.nunique()
    require(set(levels.index) == SHIFT_LABELS.keys() and levels.eq(4).all(),
            "Slide figures require the four presentation shift types with four severities each")
    readmission = selected.loc[selected.shift_type.eq("readmission")]
    require(set(readmission.shift_fraction) == set(READMISSION_LEVELS),
            "Slide 6 requires source readmission prevalence levels of 1%, 4%, 14%, and 17%")

    figure_count = 2 * len(METRICS)
    progress(f"Creating {figure_count} AUPRC and AUROC slide figures at ε={EPSILON}…")
    target = readmission.test_fraction.iloc[0]
    with plots.plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        for metric in METRICS:
            effects = paired_effects(selected, metric=metric)
            readmission_prevalence(effects.loc[effects.shift_type.eq("readmission")], target, output, metric)
            pretraining_gain(effects, output, metric)
    return {"figures": figure_count, "epsilon": EPSILON, "repeats": REPEATS, "output": str(output)}
