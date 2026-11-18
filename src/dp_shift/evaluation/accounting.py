"""Verify retained privacy accounting against the model rows and training settings."""

from functools import lru_cache
import math

from dp_shift.config import PROTOCOL
from dp_shift.data.utils import require
from .tables import PATHS


def model_key(repeat, path_id):
    return f"repeat_{repeat:02d}/{path_id}"


@lru_cache(maxsize=32)
def accounted_epsilon(sigma, rate, steps, delta):
    from dp_shift.training.privacy import predicted_epsilon

    return predicted_epsilon(sigma, rate, steps, delta)


def verify_privacy(runs, ledger, selected, clipping_norm, epsilon):
    private = runs.loc[runs.method.eq("dp_sgd")]
    expected = {model_key(int(row.repeat), PATHS[row.method, row.pretraining]) for row in private.itertuples()}
    require(set(ledger) == expected, "Missing or extra privacy records")
    steps = selected["target_epochs"] * PROTOCOL.steps_per_epoch
    sigma, rate = selected["noise_multiplier"], 1 / PROTOCOL.steps_per_epoch
    require(math.isfinite(sigma) and sigma > 0, "Invalid noise multiplier")
    achieved = accounted_epsilon(sigma, rate, steps, PROTOCOL.delta)
    require(math.isfinite(achieved) and 0 < achieved <= epsilon, "Privacy budget exceeded")
    for record in ledger.values():
        require(record["accountant"] == "prv" and record["steps"] == steps
                and record["noise_multiplier"] == sigma and record["sampling_rate"] == rate
                and record["delta"] == PROTOCOL.delta and record["clipping_norm"] == clipping_norm
                and 0 < record["epsilon_achieved"] <= epsilon
                and abs(record["epsilon_achieved"] - achieved) <= 1e-6, "Privacy settings mismatch")
        history = record["accountant_history"]
        require(sum(count for _, _, count in history) == steps
                and all(noise == sigma and sample_rate == rate and count > 0
                        for noise, sample_rate, count in history), "Privacy history mismatch")
