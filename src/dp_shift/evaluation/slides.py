"""Presentation figures for paired AUPRC and AUROC comparisons at epsilon 5."""

from pathlib import Path

from dp_shift.data.utils import require
from . import plots, tables
from .summarize import signed_effects


EPSILON = 5
METRICS = ("auprc", "auroc")
SHIFT_LABELS = {
    "readmission": "Readmission",
    "age": "Age 70+",
    "complexity": "Clinical\ncomplexity",
    "circulatory": "Circulatory\ndiagnosis",
}
METHODS = (("DP-SGD", "#2166ac", -.18), ("SGD", "#d95f02", .18))


def save_slide(figure, axis, output, name, metric, repeats, *, standalone=True):
    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    filename = f"{name}_{metric}"
    caption = f"ε = {EPSILON}, n = {repeats} paired repeats"
    if repeats > 1:
        caption += ", 95% CI"
    if standalone:
        axis.legend(frameon=False)
        plots.save(figure, output, filename, caption, layout_rect=(0, .07, 1, 1))
    return caption


def readmission_prevalence(effects, target, output, metric, repeats, *, axis=None):
    readmission = effects.loc[effects.shift_type.eq("readmission")]
    standalone = axis is None
    if standalone:
        figure, axis = plots.plt.subplots(figsize=(8.5, 4.7))
    else:
        figure = axis.figure
    for method, color, _ in METHODS:
        selected = readmission.loc[readmission.method.eq(method)]
        points = [(100 * (group.source_readmission_count / group.source_size).mean(), group.gain)
                  for _, group in selected.groupby("shift_pp")]
        plots.line(axis, points, method, color, scale=100, linewidth=2.2, markersize=6)
    axis.axvline(100 * target, color="gray", linestyle=":", linewidth=1,
                 label=f"Target prevalence {target:.2%}")
    axis.set(xlabel="Source readmission prevalence (%)",
             ylabel=f"Shifted − baseline {metric.upper()}\n(percentage points)",
             title=f"How Source Readmission Prevalence Affects Target {metric.upper()}")
    if not standalone:
        axis.set_xlabel("")
    return save_slide(figure, axis, output, "readmission_prevalence", metric, repeats, standalone=standalone)


def pretraining_gain(effects, output, metric, repeats, *, axis=None):
    gains = effects.groupby(["method", "shift_type", "repeat"]).gain.mean()
    standalone = axis is None
    if standalone:
        figure, axis = plots.plt.subplots(figsize=(9, 4.7))
    else:
        figure = axis.figure
    axis.use_sticky_edges = False  # Keep the zero reference visible when all intervals are above zero.
    positions = range(len(SHIFT_LABELS))
    for method, color, offset in METHODS:
        means, errors = plots.estimates([gains.loc[method, kind] for kind in SHIFT_LABELS], scale=100)
        axis.bar([position + offset for position in positions], means, width=.32, color=color,
                 label=method, yerr=errors, ecolor="#303942", capsize=4)
    axis.set(xticks=positions, xticklabels=list(SHIFT_LABELS.values()),
             xlim=(-.5, len(SHIFT_LABELS) - .5),
             ylabel=f"Shifted pretraining − scratch {metric.upper()}\n(percentage points)",
             title=f"Average {metric.upper()} Gain from Pretraining by Source Shift Type")
    return save_slide(figure, axis, output, "average_pretraining_gain", metric, repeats, standalone=standalone)


def run(input, output=None, progress=print):
    path = tables.results_path(input)
    output = Path(output) if output is not None else path.parent / "plots"
    results = tables.read(path)
    private = results.epsilon.eq(EPSILON)
    require(private.any(), f"Slide figures require DP-SGD at ε = {EPSILON}")
    selected = results.loc[results.epsilon.isna() | private]
    repeats = selected.repeat.nunique()
    target = (selected.target_test_readmission_count / selected.target_test_size).iloc[0]

    progress(f"Creating AUPRC and AUROC slide figures at ε={EPSILON}…")
    output.mkdir(parents=True, exist_ok=True)
    with plots.plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        plots.paired_metrics(output, "readmission_prevalence", "How Source Readmission Prevalence Affects Target Performance",
                             lambda axis, metric: readmission_prevalence(
                                 signed_effects(selected, metric), target, output, metric, repeats, axis=axis),
                             xlabel="Source readmission prevalence (%)", legend=True)
        plots.paired_metrics(output, "average_pretraining_gain", "Average Gain from Pretraining by Source Shift Type",
                             lambda axis, metric: pretraining_gain(
                                 signed_effects(selected, metric, reference="scratch"), output, metric, repeats, axis=axis),
                             legend=True)
    return {"figures": 2, "epsilon": EPSILON, "repeats": repeats, "output": str(output)}
