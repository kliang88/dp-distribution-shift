"""Paired source samples with a configured share or replacement rate."""

import hashlib

import numpy as np
import pandas as pd

from .utils import require
from dp_shift.config import ResolvedConfig
from .shifts import members


class InfeasibleReplacement(ValueError):
    def __init__(self, message, baseline):
        super().__init__(message)
        self.baseline = baseline


def replacement_seed(config, repeat_id):
    source = config.requested.source_conditions
    key = f"{config.seeds.s0 + repeat_id}:{source.shift_type}:{float(source.shift_fraction).hex()}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def matched_replacements(s0, s0_members, available, count, seed):
    removable = s0.loc[~s0_members]
    removal_groups = [removable.loc[removable.y.eq(label)] for label in (0, 1)]
    addition_groups = [available.loc[available.y.eq(label)] for label in (0, 1)]
    capacities = [min(len(remove), len(add)) for remove, add in zip(removal_groups, addition_groups)]
    if sum(capacities) < count:
        raise InfeasibleReplacement(
            f"Requested {count} readmission-matched swaps, but at most {sum(capacities)} are possible "
            f"(label 0: {capacities[0]}, label 1: {capacities[1]})", s0)

    # Cap each label before sampling removals so every chosen patient can be matched.
    rng = np.random.default_rng(seed)
    eligible = pd.concat([group.sample(n=capacity, random_state=rng)
                          for group, capacity in zip(removal_groups, capacities)])
    removed = eligible.sample(n=count, random_state=rng)
    added = pd.concat([group.sample(n=int(removed.y.eq(label).sum()), random_state=rng)
                       for label, group in enumerate(addition_groups)])
    return removed, added


def create_source_pair(pool, repeat_id, config: ResolvedConfig):
    seeds = config.seeds
    source = config.requested.source_conditions
    size, kind = source.pretrain_size, source.shift_type
    membership = members(pool, kind)
    require(size <= len(pool), "Source pool is smaller than pretrain_size")
    if config.shift_member_count is not None:
        validate_target_share(membership, size, config.shift_member_count, kind)
    s0 = pool.sample(n=size, replace=False, random_state=seeds.s0 + repeat_id)
    require(set(s0.y) == {0, 1}, "S0 sample must contain both classes")
    s0_members = members(s0, kind)
    unused = ~pool.patient_nbr.isin(s0.patient_nbr)

    if config.source_replacement_count is not None:
        removed, added = matched_replacements(s0, s0_members, pool.loc[unused & membership],
                                              config.source_replacement_count, replacement_seed(config, repeat_id))
    else:
        difference = config.shift_member_count - int(s0_members.sum())
        count, removed_label = abs(difference), int(difference < 0)
        candidates = pool.loc[unused & membership.eq(1 - removed_label)]
        # Prefixes keep replacements nested within each direction.
        removed = s0.loc[s0_members.eq(removed_label)].sample(frac=1, random_state=seeds.removed + repeat_id).iloc[:count]
        added = candidates.sample(frac=1, random_state=seeds.added + repeat_id).iloc[:count]

    s1 = pd.concat([s0.loc[~s0.patient_nbr.isin(removed.patient_nbr)], added])
    pair = {name: frame.sort_values("patient_nbr").reset_index(drop=True)
            for name, frame in dict(s0=s0, s1=s1, removed=removed, added=added).items()}
    validate_source_pair(pool, pair, config)
    return pair


def validate_target_share(membership, size, target, kind):
    require(0 < target < size, "Shifted source must contain both classes")
    positives = int(membership.sum())
    negatives = len(membership) - positives
    minimum, maximum = max(1, size - negatives), min(size - 1, positives)
    for label, needed, available in (("positive", target, positives), ("negative", size - target, negatives)):
        require(needed <= available,
                f"S1 requires {needed} {label} patients; source pool has {available}. "
                f"Shift {kind}, pretrain_size={size}: feasible fraction is {minimum / size:.4%} to {maximum / size:.4%}")


def validate_source_pair(pool, pair, config: ResolvedConfig):
    s0, s1, removed, added = (pair[name] for name in ["s0", "s1", "removed", "added"])
    a, b, gone, new = (set(frame.patient_nbr) for frame in [s0, s1, removed, added])
    for name, frame in pair.items():
        require(frame.patient_nbr.is_unique, f"Duplicate patients in source {name}")
        require(set(frame.patient_nbr) <= set(pool.patient_nbr), f"Source {name} outside source pool")
        require(frame.patient_nbr.is_monotonic_increasing, f"Source {name} must be sorted")
    kind = config.requested.source_conditions.shift_type
    s0_members = members(s0, kind)
    baseline_count = int(s0_members.sum())
    if config.source_replacement_count is not None:
        count, removed_label = config.source_replacement_count, 0
        target = baseline_count + count
        require(s0.y.sum() == s1.y.sum(), "Source replacements must match readmission labels")
    else:
        target = config.shift_member_count
        difference = target - baseline_count
        count, removed_label = abs(difference), int(difference < 0)
    require(len(s0) == len(s1) == config.requested.source_conditions.pretrain_size, "Incorrect source sample size")
    require(set(s0.y) == set(s1.y) == {0, 1}, "Source samples must contain both classes")
    require(int(members(s1, kind).sum()) == target, "Incorrect S1 member count")
    require(len(gone) == len(new) == count, "Incorrect number of source replacements")
    require(members(removed, kind).eq(removed_label).all() and members(added, kind).eq(1 - removed_label).all(), "Source replacement labels are wrong")
    require(gone == a - b and new == b - a, "Source replacement IDs do not match")
    require(not new & a and b == (a - gone) | new, "Invalid paired source construction")
    require(len(a & b) == config.requested.source_conditions.pretrain_size - count, "Incorrect source overlap")
    require(set(s0.loc[s0_members.ne(removed_label), "patient_nbr"]) <= b, "S1 changed patients from the retained class")
