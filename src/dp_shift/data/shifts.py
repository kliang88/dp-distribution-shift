"""Named binary patient characteristics used by the shared source sampler."""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import pandas as pd

from .features import DIAGNOSIS_COLUMNS, diagnosis_categories
from .utils import require


def readmission(frame):
    require(frame.y.isin([0, 1]).all(), "Readmission outcomes must be binary and complete")
    return frame.y.eq(1)


def age_70_plus(frame):
    bins = {f"[{age}-{age + 10})" for age in range(0, 100, 10)}
    require(frame.age.isin(bins).all(), "Age shift requires complete, valid ten-year age bins")
    return frame.age.isin(["[70-80)", "[80-90)", "[90-100)"])


def prior_inpatient_2_plus(frame):
    values = pd.to_numeric(frame.number_inpatient, errors="coerce")
    require((values.notna() & values.ge(0) & values.lt(float("inf")) & values.eq(values.round())).all(),
            "Prior-inpatient shift requires complete, nonnegative integer counts")
    return values.ge(2)


def primary_diagnosis_circulatory(frame):
    return diagnosis_categories(frame.diag_1).eq("Circulatory")


def clinical_complexity_2plus(frame):
    categories = frame[list(DIAGNOSIS_COLUMNS)].apply(diagnosis_categories)
    eligible = categories.mask(categories.isin(["Missing", "Invalid", "Other"]))
    distinct_pairs = [
        eligible[left].notna() & eligible[right].notna() & eligible[left].ne(eligible[right])
        for left, right in combinations(DIAGNOSIS_COLUMNS, 2)
    ]
    return pd.concat(distinct_pairs, axis=1).any(axis=1)


@dataclass(frozen=True)
class Shift:
    label: str
    columns: tuple[str, ...]
    indicator: Callable
    replacement_rate: bool = False
    allow_missing: bool = False


SHIFTS = {
    "readmission": Shift("30-day readmission", (), readmission),
    "age_70_plus": Shift("Age 70+", ("age",), age_70_plus),
    "primary_diagnosis_circulatory": Shift("Circulatory primary diagnosis", ("diag_1",),
                                          primary_diagnosis_circulatory, replacement_rate=True, allow_missing=True),
    "clinical_complexity_2plus": Shift("Two or more diagnosis categories", DIAGNOSIS_COLUMNS,
                                      clinical_complexity_2plus, replacement_rate=True, allow_missing=True),
    # Keep earlier studies readable and reproducible with their saved configurations.
    "prior_inpatient_2_plus": Shift("Prior inpatient visits ≥2", ("number_inpatient",), prior_inpatient_2_plus),
}


def members(frame, name):
    require(isinstance(name, str) and name in SHIFTS, f"Unknown shift type: {name}")
    shift = SHIFTS[name]
    require(set(shift.columns) <= set(frame.columns), f"{name} requires columns {shift.columns}")
    if not shift.allow_missing:
        require(frame[list(shift.columns)].notna().all().all(), f"{name} requires complete shift fields")
    mask = shift.indicator(frame)
    require(mask.index.equals(frame.index) and mask.dtype == bool and mask.notna().all(),
            f"{name} must return one boolean per patient")
    return mask


def describe(frame):
    """Audit available traits; unavailable unselected traits are explicitly unknown."""
    result = {}
    for name in SHIFTS:
        try:
            mask = members(frame, name)
        except ValueError:
            result[name] = {"count": None, "fraction": None}
        else:
            count = int(mask.sum())
            result[name] = {"count": count, "fraction": count / len(frame) if len(frame) else None}
    return result
