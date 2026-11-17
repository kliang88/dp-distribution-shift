"""Paired source samples with a configured share of any registered characteristic."""

import pandas as pd

from .utils import require
from dp_shift.config import ResolvedConfig
from .shifts import members


def create_source_pair(pool, repeat_id, config: ResolvedConfig):
    seeds = config.seeds
    size, target = config.requested.source_conditions.pretrain_size, config.shift_member_count
    kind = config.requested.source_conditions.shift_type
    membership = members(pool, kind)
    require(size <= len(pool), "Source pool is smaller than pretrain_size")
    require(0 < target < size, "Shifted source must contain both classes")
    positives = int(membership.sum())
    negatives = len(pool) - positives
    minimum, maximum = max(1, size - negatives), min(size - 1, positives)
    for label, needed, available in (("positive", target, positives), ("negative", size - target, negatives)):
        require(needed <= available,
                f"S1 requires {needed} {label} patients; source pool has {available}. "
                f"Shift {kind}, pretrain_size={size}: feasible fraction is {minimum / size:.4%} to {maximum / size:.4%}")
    s0 = pool.sample(n=size, replace=False, random_state=seeds.s0 + repeat_id)
    require(set(s0.y) == {0, 1}, "S0 sample must contain both classes")
    s0_members = members(s0, kind)
    difference = target - int(s0_members.sum())
    count, removed_label = abs(difference), int(difference < 0)
    candidates = pool.loc[~pool.patient_nbr.isin(s0.patient_nbr) & membership.eq(1 - removed_label)]
    # Prefixes keep replacements nested within each direction; zero difference swaps nothing.
    removed = s0.loc[s0_members.eq(removed_label)].sample(frac=1, random_state=seeds.removed + repeat_id).iloc[:count]
    added = candidates.sample(frac=1, random_state=seeds.added + repeat_id).iloc[:count]
    s1 = pd.concat([s0.loc[~s0.patient_nbr.isin(removed.patient_nbr)], added])
    pair = {name: frame.sort_values("patient_nbr").reset_index(drop=True)
            for name, frame in dict(s0=s0, s1=s1, removed=removed, added=added).items()}
    validate_source_pair(pool, pair, config)
    return pair


def validate_source_pair(pool, pair, config: ResolvedConfig):
    s0, s1, removed, added = (pair[name] for name in ["s0", "s1", "removed", "added"])
    a, b, gone, new = (set(frame.patient_nbr) for frame in [s0, s1, removed, added])
    for name, frame in pair.items():
        require(frame.patient_nbr.is_unique, f"Duplicate patients in source {name}")
        require(set(frame.patient_nbr) <= set(pool.patient_nbr), f"Source {name} outside source pool")
        require(frame.patient_nbr.is_monotonic_increasing, f"Source {name} must be sorted")
    kind = config.requested.source_conditions.shift_type
    s0_members = members(s0, kind)
    difference = config.shift_member_count - int(s0_members.sum())
    count, removed_label = abs(difference), int(difference < 0)
    require(len(s0) == len(s1) == config.requested.source_conditions.pretrain_size, "Incorrect source sample size")
    require(set(s0.y) == set(s1.y) == {0, 1}, "Source samples must contain both classes")
    require(int(members(s1, kind).sum()) == config.shift_member_count, "Incorrect S1 member count")
    require(len(gone) == len(new) == count, "Incorrect number of source replacements")
    require(members(removed, kind).eq(removed_label).all() and members(added, kind).eq(1 - removed_label).all(), "Source replacement labels are wrong")
    require(gone == a - b and new == b - a, "Source replacement IDs do not match")
    require(not new & a and b == (a - gone) | new, "Invalid paired source construction")
    require(len(a & b) == config.requested.source_conditions.pretrain_size - count, "Incorrect source overlap")
    require(set(s0.loc[s0_members.ne(removed_label), "patient_nbr"]) <= b, "S1 changed patients from the retained class")
