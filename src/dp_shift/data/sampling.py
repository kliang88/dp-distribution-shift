"""Paired source samples: remove positives to reach the configured prevalence."""

import pandas as pd

from .utils import require


def create_source_pair(pool, repeat_id, config):
    seeds = config["seeds"]
    require(config["pretrain_size"] <= len(pool), "Source pool is smaller than pretrain_size")
    s0 = pool.sample(n=config["pretrain_size"], replace=False, random_state=seeds["s0"] + repeat_id)
    require(set(s0.y) == {0, 1}, "S0 sample must contain both classes")
    count = int(s0.y.sum()) - config["source_positive_n"]
    require(count > 0, "S0 must have more positives than the shifted target")
    candidates = pool.loc[~pool.patient_nbr.isin(s0.patient_nbr) & pool.y.eq(0)]
    require(len(candidates) >= count, "Not enough unused negative patients for S1")
    removed = s0.loc[s0.y.eq(1)].sample(n=count, random_state=seeds["removed"] + repeat_id)
    added = candidates.sample(n=count, random_state=seeds["added"] + repeat_id)
    s1 = pd.concat([s0.loc[~s0.patient_nbr.isin(removed.patient_nbr)], added])
    pair = {name: frame.sort_values("patient_nbr").reset_index(drop=True)
            for name, frame in dict(s0=s0, s1=s1, removed=removed, added=added).items()}
    validate_source_pair(pool, pair, config)
    return pair


def validate_source_pair(pool, pair, config):
    s0, s1, removed, added = (pair[name] for name in ["s0", "s1", "removed", "added"])
    a, b, gone, new = (set(frame.patient_nbr) for frame in [s0, s1, removed, added])
    for name, frame in pair.items():
        require(frame.patient_nbr.is_unique, f"Duplicate patients in source {name}")
        require(set(frame.patient_nbr) <= set(pool.patient_nbr), f"Source {name} outside source pool")
        require(frame.patient_nbr.is_monotonic_increasing, f"Source {name} must be sorted")
    count = int(s0.y.sum()) - config["source_positive_n"]
    require(len(s0) == len(s1) == config["pretrain_size"], "Incorrect source sample size")
    require(count > 0 and int(s1.y.sum()) == config["source_positive_n"], "Incorrect S1 positive count")
    require(len(gone) == len(new) == count, "Incorrect number of source replacements")
    require(removed.y.eq(1).all() and added.y.eq(0).all(), "Source replacement labels are wrong")
    require(gone == a - b and new == b - a, "Source replacement IDs do not match")
    require(not new & a and b == (a - gone) | new, "Invalid paired source construction")
    require(len(a & b) == config["pretrain_size"] - count, "Incorrect source overlap")
    require(set(s0.loc[s0.y.eq(0), "patient_nbr"]) <= b, "S1 did not retain every S0 negative")
