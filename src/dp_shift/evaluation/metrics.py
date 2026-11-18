"""Validate aligned binary labels and model probabilities."""

import numpy as np

from dp_shift.data.utils import require


def validate_probabilities(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    require(y.ndim == probability.ndim == 1 and y.shape == probability.shape and len(y) > 0,
            "Labels and probabilities must be nonempty aligned vectors")
    require(np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2, "Evaluation requires both binary classes")
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
            "Probabilities must be finite and in [0,1]")
