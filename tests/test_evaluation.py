"""Check paired effects, uncertainty, and cache alignment."""

import numpy as np
import pandas as pd
import pytest

from dp_shift.data.utils import file_hash, write_json
from dp_shift.evaluation.metrics import validate_probabilities
from dp_shift.evaluation.score import read_prediction_cache
from dp_shift.evaluation.summarize import interval, paired_effects
from dp_shift.evaluation.tables import CONDITIONS


@pytest.mark.parametrize("labels,probability", [
    ([0, 1], [.2]), ([0, 1], [np.nan, .2]), ([0, 1], [-.1, .2]),
    ([0, 1], [.2, 1.1]), ([0, 0], [.1, .2]), ([0, 2], [.1, .2]),
])
def test_invalid_predictions_fail(labels, probability):
    with pytest.raises(ValueError):
        validate_probabilities(labels, probability)


def test_paired_effects_cancel_scratch_and_use_positive_dp_amplification():
    scores = {"A": .1, "B-S0": .13, "B-S1": .12, "C": .08, "D-S0": .12, "D-S1": .09}
    runs = pd.DataFrame([
        dict(shift_type="readmission", shift_fraction=.04, comparison_epsilon=5,
             repeat=repeat, method=method, pretraining=initialization, auprc=scores[path] + repeat * .1)
        for repeat in range(2) for path, (method, initialization) in CONDITIONS.items()
    ])
    effects = paired_effects(runs)
    np.testing.assert_allclose(effects.gain_sgd_baseline, .03)
    np.testing.assert_allclose(effects.shift_penalty_sgd, .01)
    np.testing.assert_allclose(effects.shift_penalty_dp, .03)
    np.testing.assert_allclose(effects.dp_amplification, .02)
    runs.loc[runs.pretraining.eq("none"), "auprc"] += .5
    np.testing.assert_allclose(effects.dp_amplification, paired_effects(runs).dp_amplification)


def test_t_interval_and_zero_variance():
    result = interval([1, 3])
    assert result["mean"] == 2 and result["sd"] == pytest.approx(np.sqrt(2))
    assert result["ci_lower"] == pytest.approx(2 - 12.706204736432095)
    assert result["ci_upper"] == pytest.approx(2 + 12.706204736432095)
    result = interval([.002] * 10)
    assert result["ci_lower"] == pytest.approx(.002) and result["ci_upper"] == pytest.approx(.002)


def test_single_repeat_has_a_score_without_uncertainty():
    result = interval([.02])
    assert result["mean"] == .02
    assert all(np.isnan(result[key]) for key in ("sd", "ci_lower", "ci_upper"))


def test_cache_alignment_is_checked_independently_of_file_hash(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez_compressed(path, row_id=np.array([1, 0], dtype=np.int64), probability=np.array([.2, .8], dtype=np.float32))
    expected = {"status": "complete", "n": 2}
    write_json(path.with_suffix(".json"), {**expected, "prediction_sha256": file_hash(path)})
    with pytest.raises(ValueError, match="row IDs"):
        read_prediction_cache(path, path.with_suffix(".json"), expected, np.array([0, 1]))
