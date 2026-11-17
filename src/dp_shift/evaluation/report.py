"""Prepare paired estimates and facet research figures by shift type."""

from pathlib import Path

import pandas as pd

from dp_shift.data.shifts import SHIFTS
from dp_shift.data.utils import require
from . import tables
from .summarize import paired_effects


def selected(value, available, name):
    choices = sorted(available)
    value = choices[0] if value is None else value
    require(value in choices, f"{name} must be one of {choices}")
    return value


def run(input, output=None, progress=print, *, epsilon=None, shift=None, fraction=None):
    from .plots import plot

    input = Path(input).resolve()
    root = input.parent if (input / "runs.csv").is_file() else input
    output = Path(output).resolve() if output is not None else root / "plots"
    data_directory = input if (input / "runs.csv").is_file() else input / "data"
    require(output != root and not output.is_relative_to(data_directory), "Plot output must be separate from the data directory")
    progress("Reading and validating CSV scores and cohort counts…")
    data = tables.read(input)
    runs = data["runs"]
    if shift is not None:
        require(shift in set(runs.shift_type), f"Shift {shift} is absent from the CSV data")
        ids = set(runs.loc[runs.shift_type.eq(shift), "run_id"])
        data = {name: table.loc[table.run_id.isin(ids)] for name, table in data.items()}
        runs = data["runs"]
    kinds = [kind for kind in SHIFTS if kind in set(runs.shift_type)]
    require(fraction is None or len(kinds) == 1, "--fraction requires --shift when plotting multiple shift types")
    epsilon = selected(epsilon, runs.comparison_epsilon.unique(), "epsilon")
    fractions = {kind: selected(fraction, runs.loc[runs.shift_type.eq(kind), "shift_fraction"].unique(), "fraction") for kind in kinds}
    budgets = set(runs.comparison_epsilon)
    require(all(set(group.comparison_epsilon) == budgets for _, group in runs.groupby("shift_type")),
            "Shift types must have the same privacy budgets")
    effects = []
    for (kind, value, budget), cell in tables.cells(data):
        scores, _, reference = tables.internal(cell)
        contrasts = paired_effects(scores, scores.repeat_id.nunique())
        contrasts["shift_type"], contrasts["shift_fraction"], contrasts["comparison_epsilon"] = kind, value, budget
        effects.append(contrasts)
    selection = ", ".join(f"{kind}={value:g}" for kind, value in fractions.items())
    progress(f"Creating five figures; {selection}; epsilon {epsilon:g}…")
    count = plot(runs, pd.concat(effects, ignore_index=True), reference["n"], output, epsilon, fractions)
    return {"figures": count, "epsilon": epsilon, "shifts": selection, "output": str(output)}
