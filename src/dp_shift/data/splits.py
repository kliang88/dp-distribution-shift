"""Exact-size, patient-disjoint random partitions."""

from dp_shift.config import PARTITIONS
from .utils import require


def create_split_manifests(cohort, config):
    require(cohort.patient_nbr.is_unique, "Cohort contains duplicate patients")
    require(sum(config.data.values()) <= len(cohort), "Requested partitions exceed the cohort size")
    shuffled = cohort.sample(frac=1, random_state=config.seed_for("partition"))
    frames, start = {}, 0
    for name in PARTITIONS:
        end = start + config.data[f"{name}_size"]
        frame = shuffled.iloc[start:end]
        require(set(frame.y) == {0, 1}, f"Partition {name} must contain both classes")
        frames[name] = frame.reset_index(drop=True)
        start = end
    return frames
