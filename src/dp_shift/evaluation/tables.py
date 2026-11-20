"""The single CSV published by the signed study."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.config import PRIVACY_BUDGETS, SIGNED_LEVELS
from dp_shift.data.shifts import SHIFTS
from dp_shift.data.utils import require


RESULTS_FILE = "results.csv"
MODEL_KEYS = ["repeat", "shift_type", "shift_pp", "method", "epsilon"]
METRICS = ["auprc", "auroc", "brier_score"]
SOURCE_COUNTS = [f"source_{kind}_count" for kind in SHIFTS]
SOURCE = ["source_size", *SOURCE_COUNTS, "source_lr", "source_epochs"]
TARGET = ["target_train_size", "target_test_size", "target_test_readmission_count",
          "target_lr", "target_epochs", "target_steps"]
PRIVACY = ["epsilon_achieved", "delta", "clipping_norm", "noise_multiplier"]
COLUMNS = [*MODEL_KEYS, *METRICS, *SOURCE, *TARGET, "seed", *PRIVACY]
INTEGER_COLUMNS = ["repeat", "shift_pp", "source_size", *SOURCE_COUNTS, "source_epochs",
                   "target_train_size", "target_test_size", "target_test_readmission_count", "target_epochs",
                   "target_steps", "seed"]


def validate(results, config=None):
    require(list(results.columns) == COLUMNS and not results.empty, "Invalid results.csv schema or empty table")
    for name in (name for name in COLUMNS if name not in ("shift_type", "method")):
        dtype = results[name].dtype
        require(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype), f"{name} must be numeric")
    for name in INTEGER_COLUMNS:
        values = results[name].dropna()
        require(np.isfinite(values).all() and values.mod(1).eq(0).all(), f"{name} must be integer")
    require(results[["repeat", "shift_pp", "seed"]].notna().all().all(), "Missing repeat, signed shift, or seed")
    ordinary = results.method.eq("SGD")
    private = results.method.eq("DP-SGD")
    require((ordinary | private).all(), "Unknown target method")
    require(results.loc[ordinary, "epsilon"].isna().all()
            and results.loc[private, "epsilon"].isin(PRIVACY_BUDGETS).all(), "Invalid epsilon for target method")
    require(results[METRICS].apply(lambda column: column.between(0, 1)).all().all(), "Invalid test metric")
    require(results.shift_pp.isin((0, *SIGNED_LEVELS)).all(), "Invalid signed shift level")
    levels = config.shift_pp if config else sorted(set(results.shift_pp) - {0})
    epsilons = config.epsilons if config else sorted(results.loc[private, "epsilon"].unique())
    repeat_count = config.repeats if config else int(results.repeat.max())
    require(levels and epsilons and 0 < repeat_count <= len(results), "Missing shift, privacy budget, or repeat")
    conditions = [("scratch", 0), ("baseline", 0), *product(SHIFTS, levels)]
    methods = [("SGD", 0), *(("DP-SGD", epsilon) for epsilon in epsilons)]
    expected = {(repeat, kind, level, method, epsilon) for repeat in range(1, repeat_count + 1)
                for kind, level in conditions for method, epsilon in methods}
    actual = set(results[MODEL_KEYS].fillna({"epsilon": 0}).itertuples(index=False, name=None))
    require(len(actual) == len(results) and actual == expected, "Incomplete results grid or duplicate model rows")
    validate_details(results, private)


def validate_details(results, private):
    scratch = results.shift_type.eq("scratch")
    require(results.loc[scratch, SOURCE].isna().all().all(), "Scratch rows must have blank source fields")
    source = results.loc[~scratch, SOURCE]
    source_counts = source[SOURCE_COUNTS]
    require(source_counts.ge(0).all().all() and source_counts.le(source.source_size, axis=0).all().all(),
            "Invalid source cohort counts")
    require(positive(source[["source_size", "source_lr", "source_epochs"]]), "Invalid source training settings")

    require(positive(results[TARGET]), "Invalid target cohort counts or training settings")
    require(results.target_test_readmission_count.lt(results.target_test_size).all(), "Invalid target readmission count")
    require(results[TARGET].nunique().eq(1).all(), "Target cohort counts and training settings must be fixed")

    dp = results.loc[private]
    require(results.loc[~private, PRIVACY].isna().all().all() and positive(dp[PRIVACY]), "Invalid DP accounting values")
    require(dp.epsilon_achieved.le(dp.epsilon).all() and dp.delta.lt(1).all(), "Invalid achieved privacy budget")


def positive(values):
    return (np.isfinite(values) & values.gt(0)).all().all()


def results_path(input):
    path = Path(input)
    return path / RESULTS_FILE if path.is_dir() else path


def read(path):
    results = pd.read_csv(path, float_precision="round_trip")
    validate(results)
    return results
