"""The four binary characteristics used in the signed experiment."""

from itertools import combinations

import pandas as pd

from .features import DIAGNOSIS_COLUMNS, diagnosis_categories
from .utils import require


def readmission(frame):
    return frame.y.eq(1)


def age_70_plus(frame):
    bins = {f"[{age}-{age + 10})" for age in range(0, 100, 10)}
    require(frame.age.isin(bins).all(), "Age shift requires complete, valid ten-year age bins")
    return frame.age.isin(["[70-80)", "[80-90)", "[90-100)"])


def primary_diagnosis_circulatory(frame):
    return diagnosis_categories(frame.diag_1).eq("Circulatory")


def clinical_complexity_2plus(frame):
    categories = frame[list(DIAGNOSIS_COLUMNS)].apply(diagnosis_categories)
    eligible = categories.mask(categories.isin(["Missing", "Invalid", "Other"]))
    return pd.concat([
        eligible[left].notna() & eligible[right].notna() & eligible[left].ne(eligible[right])
        for left, right in combinations(DIAGNOSIS_COLUMNS, 2)
    ], axis=1).any(axis=1)


SHIFTS = {
    "readmission": readmission,
    "age": age_70_plus,
    "circulatory": primary_diagnosis_circulatory,
    "complexity": clinical_complexity_2plus,
}
