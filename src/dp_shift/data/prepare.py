"""Prepare each split once, with the transformer fitted on the source pool."""

import numpy as np

from dp_shift.config import PARTITIONS
from .cohort import build_cohort
from .preprocessing import fit_source_transformer, transform_features
from .schema import load_and_validate_raw
from .shifts import SHIFTS
from .splits import create_split_manifests


DEFAULT_RECORDS = "data/diabetic_data.csv"


def prepare_splits(records, expected_hash, config):
    cohort, _, _ = build_cohort(load_and_validate_raw(records, expected_hash))
    frames = create_split_manifests(cohort, config)
    pool = frames["source_pool"]
    transformer = fit_source_transformer(pool)
    arrays = {name: {"X": transform_features(transformer, frames[name]),
                     "y": frames[name].y.to_numpy(dtype=np.float32)} for name in PARTITIONS}
    masks = {name: indicator(pool).to_numpy() for name, indicator in SHIFTS.items()}
    return arrays, masks
