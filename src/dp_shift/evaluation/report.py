"""Prepare paired estimates and facet research figures by shift type."""

from pathlib import Path

from dp_shift.data.shifts import SHIFTS
from dp_shift.data.utils import require
from . import tables
from .summarize import paired_effects


OVERALL_FRACTIONS = {
    "readmission": .01,
    "age_70_plus": .80,
    "primary_diagnosis_circulatory": .05,
    "clinical_complexity_2plus": .05,
}


def selected(value, available, name, *, preferred=None):
    choices = sorted(available)
    if value is None:
        return preferred if preferred in choices else choices[0]
    require(value in choices, f"{name} must be one of {choices}")
    return value


def run(input, output=None, progress=print, *, epsilon=None, shift=None, fraction=None):
    from .plots import plot

    input = Path(input).resolve()
    if input.is_file():
        path = input
    elif (input / "runs.csv").is_file():
        path = input / "runs.csv"
    else:
        path = input / tables.RUNS_FILE
    data_directory = path.parent
    root = data_directory.parent if data_directory.name == "data" else data_directory
    output = Path(output).resolve() if output is not None else root / "plots"
    require(output != root and (data_directory == root or not output.is_relative_to(data_directory)),
            "Plot output must be separate from the data directory")
    progress("Reading and validating model scores and cohort counts…")
    runs = tables.read(path)
    if shift is not None:
        require(shift in set(runs.shift_type), f"Shift {shift} is absent from the CSV data")
        runs = runs.loc[runs.shift_type.eq(shift)]
    kinds = [kind for kind in SHIFTS if kind in set(runs.shift_type)]
    require(fraction is None or len(kinds) == 1, "--fraction requires --shift when plotting multiple shift types")
    budgets = set(runs.comparison_epsilon)
    epsilon = selected(epsilon, budgets, "epsilon", preferred=1)
    fractions, overall_fractions = {}, {}
    for kind in kinds:
        levels = runs.loc[runs.shift_type.eq(kind), "shift_fraction"].unique()
        fractions[kind] = selected(fraction, levels, "fraction")
        overall_fractions[kind] = selected(fraction, levels, "fraction", preferred=OVERALL_FRACTIONS.get(kind))
    require(all(set(group.comparison_epsilon) == budgets
                for _, group in runs.groupby(["shift_type", "shift_fraction"])),
            "Shift fractions must have the same privacy budgets")
    effects = paired_effects(runs)
    selection = ", ".join(f"{kind}={value:g}" for kind, value in fractions.items())
    overall_selection = ", ".join(f"{kind}={value:g}" for kind, value in overall_fractions.items())
    progress(f"Creating five figures with overall performance at {overall_selection} (ε={epsilon:g})…")
    count = plot(runs, effects, int(runs.test_size.iloc[0]), output, epsilon, fractions,
                 overall_fractions=overall_fractions)
    summary = {"figures": count, "epsilon": epsilon, "shifts": selection, "output": str(output)}
    if overall_fractions != fractions:
        summary["overall_shifts"] = overall_selection
    return summary
