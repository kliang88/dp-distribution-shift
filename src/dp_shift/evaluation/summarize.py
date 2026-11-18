"""Paired AUPRC effects and uncertainty across repeats on a fixed test set."""

import numpy as np
import pandas as pd
from scipy.stats import t

from dp_shift.config import PROTOCOL
from dp_shift.data.utils import require
from .tables import GRID


def paired_effects(runs):
    """Calculate contrasts within each comparison and repeat of a validated table."""
    ap = runs.pivot(index=[*GRID, "repeat"], columns=["method", "pretraining"], values="auprc")
    sgd, dp = ap["sgd"], ap["dp_sgd"]
    penalty_sgd = sgd.baseline - sgd.shifted
    penalty_dp = dp.baseline - dp.shifted
    return pd.DataFrame({
        "gain_sgd_baseline": sgd.baseline - sgd["none"],
        "gain_sgd_shifted": sgd.shifted - sgd["none"],
        "gain_dp_baseline": dp.baseline - dp["none"],
        "gain_dp_shifted": dp.shifted - dp["none"],
        "shift_penalty_sgd": penalty_sgd,
        "shift_penalty_dp": penalty_dp,
        "dp_amplification": penalty_dp - penalty_sgd,
    }).reset_index()


def interval(values, confidence=PROTOCOL.confidence_level):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) >= 1 and np.isfinite(values).all(), "Invalid repeat values")
    if len(values) == 1:
        return {"n": 1, "mean": float(values[0]), "sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    margin = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "ci_lower": mean - margin, "ci_upper": mean + margin}
