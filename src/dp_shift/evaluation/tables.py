"""The single published CSV schema for experiments and plots."""

from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.data.shifts import SHIFTS
from dp_shift.data.utils import require
from dp_shift.stages.common import atomic_path


RUNS_FILE = "data/runs.csv"
GRID = ["shift_type", "shift_fraction", "comparison_epsilon"]
KEY = [*GRID, "repeat", "method", "pretraining"]
METRICS = ["auprc", "auroc", "brier"]
COLUMNS = [*KEY, *METRICS, "source_fraction", "source_size", "test_fraction", "test_size"]
CONDITIONS = {
    "A": ("sgd", "none"), "B-S0": ("sgd", "baseline"), "B-S1": ("sgd", "shifted"),
    "C": ("dp_sgd", "none"), "D-S0": ("dp_sgd", "baseline"), "D-S1": ("dp_sgd", "shifted"),
}
PATHS = {condition: path for path, condition in CONDITIONS.items()}


def cells(runs):
    return runs.groupby(GRID, sort=True)


def validate(runs):
    require(list(runs.columns) == COLUMNS and not runs.empty, "Invalid runs.csv schema or empty file")
    require(runs.shift_type.isin(SHIFTS).all(), "Unknown shift type")
    require(set(zip(runs.method, runs.pretraining)) <= set(PATHS), "Unknown method or pretraining")
    require(not runs.duplicated(KEY).any(), "Duplicate model rows")
    numeric = [name for name in COLUMNS if name not in {"shift_type", "method", "pretraining"}]
    for name in numeric:
        require(pd.api.types.is_numeric_dtype(runs[name]) and not pd.api.types.is_bool_dtype(runs[name]),
                f"{name} must be numeric")
    require(runs.shift_fraction.between(0, 1, inclusive="neither").all(), "Invalid shift fraction")
    require(np.isfinite(runs.comparison_epsilon).all() and runs.comparison_epsilon.gt(0).all(),
            "Invalid comparison epsilon")
    for name in METRICS:
        require(runs[name].between(0, 1).all(), f"Invalid {name.upper()}")
    require(np.isfinite(runs.repeat).all() and runs.repeat.ge(0).all() and runs.repeat.mod(1).eq(0).all(),
            "Invalid repeat numbers")

    scratch = runs.pretraining.eq("none")
    require(runs.loc[scratch, ["source_fraction", "source_size"]].isna().all().all(),
            "Scratch source fields must be empty")
    source = runs.loc[~scratch]
    require(source.source_fraction.between(0, 1).all(), "Invalid source fraction")
    for name, values in (("source_size", source.source_size), ("test_size", runs.test_size)):
        require(np.isfinite(values).all() and values.ge(2).all() and values.mod(1).eq(0).all(),
                f"Invalid {name}")
    require(runs.test_size.nunique() == 1, "Test size changed across runs")
    require(runs.test_fraction.between(0, 1).all(), "Invalid test fraction")
    require(runs.groupby("shift_type").test_fraction.nunique().eq(1).all(),
            "Test fraction changed within a shift type")
    # Uniqueness and the six allowed conditions make this a completeness check.
    repeats = runs.repeat.nunique()
    require(runs.repeat.max() == repeats - 1
            and runs.groupby(GRID + ["repeat"]).size().eq(len(CONDITIONS)).all()
            and cells(runs).size().eq(repeats * len(CONDITIONS)).all(),
            "Missing model conditions or repeats")

    replacement_kinds = {name for name, shift in SHIFTS.items() if shift.replacement_rate}
    uses_replacements = source.shift_type.isin(replacement_kinds)
    shifted = source.loc[~uses_replacements & source.pretraining.eq("shifted")]
    require(np.allclose(shifted.source_fraction, shifted.shift_fraction, rtol=0, atol=1e-12),
            "Shifted source fraction mismatch")
    if uses_replacements.any():
        pairs = source.loc[uses_replacements].pivot(
            index=[*GRID, "repeat", "method"], columns="pretraining", values=["source_fraction", "source_size"])
        size = pairs["source_size"]["baseline"]
        expected = np.rint(size * pairs.index.get_level_values("shift_fraction"))
        actual = size * (pairs["source_fraction"]["shifted"] - pairs["source_fraction"]["baseline"])
        require(size.eq(pairs["source_size"]["shifted"]).all() and ((expected > 0) & (expected < size)).all()
                and np.allclose(actual, expected, rtol=0, atol=1e-8),
                "Shifted source fraction does not match the requested replacements")


def read(path):
    runs = pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])
    require(isinstance(runs.index, pd.RangeIndex), "Unexpected fields in runs.csv")
    validate(runs)
    return runs


def write(path, runs):
    validate(runs)
    ordered = runs.sort_values(KEY, ignore_index=True)
    for name in ("repeat", "source_size", "test_size"):
        ordered[name] = ordered[name].astype("Int64")
    with atomic_path(path) as temporary:
        ordered.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        pd.testing.assert_frame_equal(read(temporary), ordered, check_dtype=False, check_exact=True)
