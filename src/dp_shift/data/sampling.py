"""Generate nested, label-matched swap orders using only patient indices."""

import numpy as np

from .utils import require


def swap_order(remove, add, labels, rng, maximum, match_labels, description):
    groups = ((remove, add),)
    if match_labels:
        groups = tuple((remove[labels[remove] == label], add[labels[add] == label]) for label in (0, 1))
    capacities = [min(len(outgoing), len(incoming)) for outgoing, incoming in groups]
    require(sum(capacities) >= maximum,
            f"{description}: requested {maximum} swaps, capacity {sum(capacities)}")
    outgoing = np.concatenate([rng.choice(rows, n, replace=False) for (rows, _), n in zip(groups, capacities)])
    incoming = np.concatenate([rng.choice(rows, n, replace=False) for (_, rows), n in zip(groups, capacities)])
    order = rng.permutation(len(outgoing))[:maximum]
    return outgoing[order].astype(np.int32), incoming[order].astype(np.int32)


def plan_swaps(labels, masks, config):
    """Check all repeats before training and retain only the largest swap prefixes."""
    plans = []
    for index in range(config.repeats):
        baseline = np.random.RandomState(config.seed_for("s0") + index).choice(
            len(labels), config.pretrain_size, replace=False).astype(np.int32)
        require(len(np.unique(labels[baseline])) == 2, f"Repeat {index + 1}: S0 must contain both labels")
        outside = np.ones(len(labels), dtype=bool)
        outside[baseline] = False
        orders = {}
        for kind, mask in masks.items():
            for direction in (-1, 1):
                levels = [abs(level) for level in config.shift_pp if direction * level > 0]
                if not levels:
                    continue
                maximum = config.pretrain_size * max(levels) // 100
                member = direction > 0
                if kind == "readmission":
                    remaining = int(mask[baseline].sum()) + direction * maximum
                    require(0 < remaining < len(baseline), f"Repeat {index + 1}: shifted readmission needs both labels")
                orders[kind, direction] = swap_order(
                    baseline[mask[baseline] != member], np.flatnonzero(outside & (mask == member)), labels,
                    np.random.default_rng(config.seed_for(f"swaps:{kind}:{direction}") + index),
                    maximum, kind != "readmission", f"Repeat {index + 1}, {kind}, direction {direction}")
        plans.append((np.sort(baseline), orders))
    return plans


def shifted_indices(baseline, order, count):
    removed, added = order
    kept = np.isin(baseline, removed[:count], invert=True)
    return np.sort(np.concatenate((baseline[kept], added[:count])))
