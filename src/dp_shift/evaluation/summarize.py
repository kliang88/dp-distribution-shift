"""Pair each shifted score with its shared control within a repeat."""

import numpy as np
from scipy.stats import t

from dp_shift.config import PROTOCOL
from dp_shift.data.shifts import SHIFTS


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    n = len(values)
    half_width = t.ppf((1 + PROTOCOL.confidence_level) / 2, n - 1) * values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return mean, float(half_width)


def signed_effects(results, metric="auprc", *, reference="baseline"):
    keys = ["repeat", "method", "epsilon"]
    control = results.loc[results.shift_type.eq(reference), [*keys, metric]].rename(columns={metric: "control_score"})
    shifted = results.loc[results.shift_type.isin(SHIFTS)]
    effects = shifted.merge(control, on=keys, validate="many_to_one")
    return effects.assign(gain=effects[metric] - effects.control_score)


def interaction(effects):
    keys = ["repeat", "shift_type", "shift_pp"]
    ordinary = effects.loc[effects.method.eq("SGD"), [*keys, "gain"]]
    private = effects.loc[effects.method.eq("DP-SGD"), [*keys, "epsilon", "gain"]]
    values = private.merge(ordinary, on=keys, suffixes=("_dp", "_sgd"), validate="many_to_one")
    values["interaction"] = values.gain_sgd - values.gain_dp
    return values[[*keys, "epsilon", "interaction"]]
