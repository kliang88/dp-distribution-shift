"""Named binary patient characteristics used by the shared source sampler."""

from dataclasses import dataclass
from typing import Callable

from .utils import require


def readmission(frame):
    require(frame.y.isin([0, 1]).all(), "Readmission outcomes must be binary and complete")
    return frame.y.eq(1)


def age_70_plus(frame):
    bins = {f"[{age}-{age + 10})" for age in range(0, 100, 10)}
    require(frame.age.isin(bins).all(), "Age shift requires complete, valid ten-year age bins")
    return frame.age.isin(["[70-80)", "[80-90)", "[90-100)"])


def prior_inpatient_2_plus(frame):
    import pandas as pd

    values = pd.to_numeric(frame.number_inpatient, errors="coerce")
    require((values.notna() & values.ge(0) & values.lt(float("inf")) & values.eq(values.round())).all(),
            "Prior-inpatient shift requires complete, nonnegative integer counts")
    return values.ge(2)


@dataclass(frozen=True)
class Shift:
    label: str
    columns: tuple[str, ...]
    indicator: Callable


SHIFTS = {
    "readmission": Shift("30-day readmission", (), readmission),
    "age_70_plus": Shift("Age 70+", ("age",), age_70_plus),
    "prior_inpatient_2_plus": Shift("Prior inpatient visits ≥2", ("number_inpatient",), prior_inpatient_2_plus),
}


def members(frame, name):
    require(isinstance(name, str) and name in SHIFTS, f"Unknown shift type: {name}")
    shift = SHIFTS[name]
    require(set(shift.columns) <= set(frame.columns), f"{name} requires columns {shift.columns}")
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
