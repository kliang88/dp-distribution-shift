"""Strict parsing of the raw records and discharge-code descriptions."""

import csv
import re

import numpy as np
import pandas as pd

from .utils import file_hash, require


def integer_value(value):
    """Parse integer strings, including categorical spellings such as '1.0'."""
    text = str(value).strip()
    require(bool(re.fullmatch(r"\+?\d+(?:\.0+)?", text)),
            f"Expected a nonnegative integer, got {value!r}")
    result = int(text.split(".")[0])
    require(result <= np.iinfo(np.int64).max, f"Integer is outside int64: {value!r}")
    return result


def load_and_validate_raw(path, config):
    expected_hash = config["inputs"]["records"]["sha256"]
    require(file_hash(path) == expected_hash, "Raw records fingerprint does not match config")
    with open(path, newline="") as stream:
        header = next(csv.reader(stream), [])
    require(bool(header), "Empty records CSV")
    require(len(header) == len(set(header)), "Duplicate CSV column names")
    features = config["numeric_features"] + config["categorical_features"]
    required = ["patient_nbr", "encounter_id", "readmitted", *features]
    require(set(required).issubset(header), f"Missing required columns: {set(required) - set(header)}")
    frame = pd.read_csv(path, usecols=required, dtype=str,
                        keep_default_na=False, na_values=["?", ""], low_memory=False)
    for name in ["patient_nbr", "encounter_id"]:
        require(frame[name].notna().all(), f"Missing {name}")
        frame[name] = frame[name].map(integer_value).astype("int64")
    require(frame.encounter_id.is_unique, "Duplicate encounter IDs")
    require(frame.readmitted.isin(["<30", ">30", "NO"]).all(), "Missing or unexpected readmitted label")
    for name in config["numeric_features"]:
        try:
            values = pd.to_numeric(frame[name], errors="raise").astype(float)
        except ValueError as error:
            raise ValueError(f"Malformed numerical feature: {name}") from error
        observed = values.dropna()
        require(np.isfinite(observed).all() and observed.ge(0).all(),
                f"Negative or nonfinite numerical feature: {name}")
        frame[name] = values
    for name in config["categorical_ids"]:
        frame[name] = frame[name].map(
            lambda value: str(integer_value(value)) if pd.notna(value) else np.nan
        )
    frame["y"] = frame.readmitted.eq("<30").astype("int64")
    return frame.drop(columns="readmitted")


def discharge_descriptions(path):
    descriptions = {}
    in_discharge_section = False
    with open(path, newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].strip():
                continue
            if row[0] == "discharge_disposition_id":
                in_discharge_section = True
            elif row[0].endswith("_id"):
                in_discharge_section = False
            elif in_discharge_section:
                descriptions[str(integer_value(row[0]))] = row[1].strip()
    return descriptions
