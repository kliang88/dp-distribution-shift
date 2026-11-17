"""Strict parsing of the raw records and discharge-code descriptions."""

import csv
import re

import numpy as np
import pandas as pd

from dp_shift.config import PROTOCOL, ResolvedConfig

from .utils import file_hash, require
from .shifts import SHIFTS, members


def integer_value(value):
    """Parse integer strings, including categorical spellings such as '1.0'."""
    text = str(value).strip()
    require(bool(re.fullmatch(r"\+?\d+(?:\.0+)?", text)),
            f"Expected a nonnegative integer, got {value!r}")
    result = int(text.split(".")[0])
    require(result <= np.iinfo(np.int64).max, f"Integer is outside int64: {value!r}")
    return result


def load_and_validate_raw(path, config: ResolvedConfig, expected_hash):
    require(file_hash(path) == expected_hash, "Raw records fingerprint does not match config")
    with open(path, newline="") as stream:
        header = next(csv.reader(stream), [])
    require(bool(header), "Empty records CSV")
    require(len(header) == len(set(header)), "Duplicate CSV column names")
    features = list(PROTOCOL.numeric_features) + list(PROTOCOL.categorical_features)
    required = list(dict.fromkeys(["patient_nbr", "encounter_id", "readmitted", *features,
                                   *SHIFTS[config.requested.source_conditions.shift_type].columns]))
    require(set(required).issubset(header), f"Missing required columns: {set(required) - set(header)}")
    available_traits = {column for shift in SHIFTS.values() for column in shift.columns} & set(header)
    columns = list(dict.fromkeys([*required, *sorted(available_traits)]))
    frame = pd.read_csv(path, usecols=columns, dtype=str,
                        keep_default_na=False, na_values=["?", ""], low_memory=False)
    for name in ["patient_nbr", "encounter_id"]:
        require(frame[name].notna().all(), f"Missing {name}")
        frame[name] = frame[name].map(integer_value).astype("int64")
    require(frame.encounter_id.is_unique, "Duplicate encounter IDs")
    require(frame.readmitted.isin(["<30", ">30", "NO"]).all(), "Missing or unexpected readmitted label")
    for name in PROTOCOL.numeric_features:
        try:
            values = pd.to_numeric(frame[name], errors="raise").astype(float)
        except ValueError as error:
            raise ValueError(f"Malformed numerical feature: {name}") from error
        observed = values.dropna()
        require(np.isfinite(observed).all() and observed.ge(0).all(),
                f"Negative or nonfinite numerical feature: {name}")
        frame[name] = values
    for name in PROTOCOL.categorical_ids:
        frame[name] = frame[name].map(
            lambda value: str(integer_value(value)) if pd.notna(value) else np.nan
        )
    frame["y"] = frame.readmitted.eq("<30").astype("int64")
    frame = frame.drop(columns="readmitted")
    members(frame, config.requested.source_conditions.shift_type)
    return frame


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
