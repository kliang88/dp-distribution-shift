"""Fixed categories for diagnosis and insulin predictors."""

import re

import pandas as pd


DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
DIAGNOSIS_RANGES = (
    ("Infectious", 1, 139),
    ("Neoplasm", 140, 239),
    ("Diabetes", 250, 250),
    ("Circulatory", 390, 459),
    ("Respiratory", 460, 519),
    ("Digestive", 520, 579),
    ("Genitourinary", 580, 629),
    ("Musculoskeletal", 710, 739),
    ("Symptoms", 780, 799),
    ("Injury", 800, 999),
)
DIAGNOSIS_CATEGORIES = tuple(name for name, _, _ in DIAGNOSIS_RANGES) + (
    "Other", "V_code", "E_code", "Missing", "Invalid",
)
INSULIN_CATEGORIES = ("No", "Steady", "Up", "Down", "Missing")
FIXED_CATEGORIES = dict.fromkeys(DIAGNOSIS_COLUMNS, DIAGNOSIS_CATEGORIES)
FIXED_CATEGORIES["insulin"] = INSULIN_CATEGORIES
MISSING_DIAGNOSES = {"", "?", "NA", "N/A", "NULL", "NONE"}


def diagnosis_category(value):
    if pd.isna(value):
        return "Missing"
    code = str(value).strip().upper()
    if code in MISSING_DIAGNOSES:
        return "Missing"
    if re.fullmatch(r"[0-9]{3}(?:\.[0-9]{1,2})?", code):
        prefix = int(code[:3])
        if prefix == 0:
            return "Invalid"
        for category, start, stop in DIAGNOSIS_RANGES:
            if start <= prefix <= stop:
                return category
        return "Other"
    if re.fullmatch(r"V[0-9]{2}(?:\.[0-9]{1,2})?", code) and 1 <= int(code[1:3]) <= 91:
        return "V_code"
    if re.fullmatch(r"E[0-9]{3}(?:\.[0-9])?", code):
        return "E_code"
    return "Invalid"


def normalize_fixed_categories(frame):
    values = frame[list(DIAGNOSIS_COLUMNS)].map(diagnosis_category)
    insulin = frame["insulin"]
    values["insulin"] = insulin.where(insulin.isin(INSULIN_CATEGORIES), "Missing")
    return values
