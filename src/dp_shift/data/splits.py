"""Exact-size, patient-disjoint random partitions and ordered manifests."""

import numpy as np
import pandas as pd
from dp_shift.config import PARTITIONS

from .utils import require


def create_split_manifests(cohort, config):
    sizes = config["data"]
    require(sum(sizes.values()) <= len(cohort),
            f"Requested partitions need {sum(sizes.values())} patients; cohort has {len(cohort)}")
    shuffled = cohort.sample(frac=1, random_state=config["seeds"]["partition"])
    frames, start = {"cohort": cohort}, 0
    for name in PARTITIONS:
        end = start + sizes[f"{name}_size"]
        frames[name] = shuffled.iloc[start:end].reset_index(drop=True)
        start = end
    frames["unused"] = shuffled.iloc[start:].reset_index(drop=True)
    validate_splits(frames, config)
    return frames


def validate_splits(frames, config):
    require(set(frames) == {"cohort", *PARTITIONS, "unused"}, "Unexpected partition names")
    for name, frame in frames.items():
        require(frame.patient_nbr.is_unique, f"Duplicate patients in {name}")
        if name in PARTITIONS:
            require(len(frame) == config["data"][f"{name}_size"], f"Incorrect patient count: {name}")
            require(set(frame.y) == {0, 1}, f"Partition {name} must contain both classes")
    seen = set()
    for name in (*PARTITIONS, "unused"):
        patients = set(frames[name].patient_nbr)
        require(not seen & patients, f"Patients overlap in partition {name}")
        seen.update(patients)
    require(seen == set(frames["cohort"].patient_nbr), "Partitions do not cover the cohort")


def manifest(frame):
    result = frame[["patient_nbr", "encounter_id"]].reset_index(drop=True).copy()
    result.insert(0, "position", np.arange(len(result), dtype=np.int64))
    return result


def read_manifest(path):
    result = pd.read_csv(path, dtype={"position": "int64", "patient_nbr": "int64", "encounter_id": "int64"})
    require(list(result.columns) == ["position", "patient_nbr", "encounter_id"], f"Invalid manifest: {path}")
    require(np.array_equal(result.position, np.arange(len(result))), f"Invalid manifest order: {path}")
    require(result.patient_nbr.is_unique, f"Duplicate manifest patients: {path}")
    return result


def rows_for_manifest(cohort, membership):
    """Look up IDs explicitly, preserving membership order rather than merge order."""
    indexed = cohort.set_index("patient_nbr", drop=False)
    require(membership.patient_nbr.isin(indexed.index).all(), "Manifest contains unknown patients")
    rows = indexed.loc[membership.patient_nbr].reset_index(drop=True)
    require(np.array_equal(rows.encounter_id, membership.encounter_id), "Manifest encounter IDs do not match")
    return rows
