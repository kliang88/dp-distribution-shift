"""Hand-calculated checks for metrics, paired effects, and cache alignment."""

import numpy as np
import pandas as pd
import pytest

from dp_shift.data.utils import file_hash, write_json
from dp_shift.evaluation.metrics import CONDITIONS, curve_key, curve_values, metric_values, validate_curves
from dp_shift.evaluation.score import read_prediction_cache
from dp_shift.evaluation.summarize import classify_interaction, interval, paired_effects, summary_tables


def test_metrics_match_hand_calculations_and_constant_reference():
    values = metric_values([0, 1, 1, 0], [.1, .4, .8, .2])
    assert values["average_precision"] == values["auroc"] == 1
    assert values["brier"] == pytest.approx(.1125)
    assert values["mean_prediction"] == pytest.approx(.375)
    assert values["calibration_bias"] == pytest.approx(-.125)
    constant = metric_values([0, 0, 0, 1], [.25] * 4)
    assert constant["average_precision"] == .25 and constant["auroc"] == .5
    assert constant["brier"] == .25 * .75


@pytest.mark.parametrize("labels,probability", [
    ([0, 1], [.2]), ([0, 1], [np.nan, .2]), ([0, 1], [-.1, .2]),
    ([0, 1], [.2, 1.1]), ([0, 0], [.1, .2]), ([0, 2], [.1, .2]),
])
def test_invalid_metrics_fail(labels, probability):
    with pytest.raises(ValueError):
        metric_values(labels, probability)


def test_quantile_bins_are_populated_and_preserve_patient_counts():
    p = np.linspace(.001, .04, 100)
    y = np.arange(100) % 2
    curves = curve_values(y, p)
    assert curves["calibration_n"].tolist() == [10] * 10
    assert curves["calibration_positive_n"].sum() == y.sum()
    assert curves["histogram_n"].sum() == len(y)
    np.testing.assert_allclose(curves["calibration_observed_rate"], .5)
    assert np.average(curves["calibration_mean_prediction"], weights=curves["calibration_n"]) == pytest.approx(p.mean())
    assert -np.sum(np.diff(curves["recall"]) * curves["precision"][:-1]) == pytest.approx(metric_values(y, p)["average_precision"])


def test_quantile_bins_preserve_ties_constants_and_probability_endpoints():
    p = np.repeat([0., .1, 1.], [20, 20, 20])
    curves = curve_values(np.arange(60) % 2, p)
    assert curves["calibration_n"].tolist() == [40, 20]
    np.testing.assert_allclose(curves["calibration_mean_prediction"], [.05, 1])
    constant = curve_values([0, 0, 0, 1], [.25] * 4)
    assert constant["calibration_n"].tolist() == [4]
    assert constant["calibration_observed_rate"].tolist() == [.25]
    np.testing.assert_allclose(constant["precision"], [.25, 1])
    np.testing.assert_allclose(constant["recall"], [1, 0])


def fake_runs():
    ap = {"A": .1, "B-S0": .13, "B-S1": .12, "C": .08, "D-S0": .12, "D-S1": .09}
    brier = {"A": .08, "B-S0": .07, "B-S1": .075, "C": .09, "D-S0": .08, "D-S1": .10}
    return pd.DataFrame([{"repeat_id": repeat, "path_id": path, "status": "complete", "average_precision": value,
                          "brier": brier[path], "auroc": .6, "mean_prediction": .2, "test_prevalence": .1,
                          "calibration_bias": .1, "method": CONDITIONS[path][0],
                          "initialization": CONDITIONS[path][1]} for repeat in range(2) for path, value in ap.items()])


def test_paired_effects_cancel_scratch_and_preserve_ap_signs():
    runs = fake_runs()
    effects = paired_effects(runs, 2)
    assert effects.ap_gain_sgd_s0.iloc[0] == pytest.approx(.03)
    assert effects.ap_shift_sgd.iloc[0] == pytest.approx(-.01)
    assert effects.ap_shift_dp.iloc[0] == pytest.approx(-.03)
    assert effects.ap_dp_interaction.iloc[0] == pytest.approx(-.02)
    runs.loc[runs.path_id.isin(["A", "C"]), "average_precision"] += .5
    other = paired_effects(runs, 2)
    np.testing.assert_allclose(effects.ap_dp_interaction, other.ap_dp_interaction)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        paired_effects(runs.iloc[:-1], 2)


@pytest.mark.parametrize("lower,upper,expected", [
    (-.02, -.006, "negative"), (.006, .02, "positive"), (-.005, .005, "small"),
    (-.006, .004, "inconclusive"), (-.007, -.005, "inconclusive"), (.005, .007, "inconclusive"),
])
def test_interval_decision_boundaries(lower, upper, expected):
    assert classify_interaction(lower, upper) == expected


def test_t_interval_and_zero_variance():
    result = interval([1, 3])
    assert result["mean"] == 2 and result["sd"] == pytest.approx(np.sqrt(2))
    assert result["ci_lower"] == pytest.approx(2 - 12.706204736432095)
    assert result["ci_upper"] == pytest.approx(2 + 12.706204736432095)
    result = interval([.002] * 10)
    assert result["ci_lower"] == pytest.approx(.002) and result["ci_upper"] == pytest.approx(.002)
    assert classify_interaction(result["ci_lower"], result["ci_upper"]) == "small"


def test_cache_alignment_is_checked_independently_of_file_hash(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez_compressed(path, row_id=np.array([1, 0], dtype=np.int64), probability=np.array([.2, .8], dtype=np.float32))
    expected = {"status": "complete", "n": 2}
    write_json(path.with_suffix(".json"), {**expected, "prediction_sha256": file_hash(path)})
    with pytest.raises(ValueError, match="row IDs"):
        read_prediction_cache(path, path.with_suffix(".json"), expected, np.array([0, 1]))


def test_paper_tables_are_six_models_and_seven_paired_ap_effects():
    runs = fake_runs()
    tables = summary_tables(runs, {"repeats": 2, "confidence_level": .95, "practical_ap_threshold": .005})
    assert set(tables) == {"metrics.csv", "effects.csv"}
    metrics, effects = tables["metrics.csv"], tables["effects.csv"]
    assert len(metrics) == 6 and len(effects) == 7
    assert metrics.n.eq(2).all() and metrics.average_precision_sd.eq(0).all()
    np.testing.assert_allclose(metrics.average_precision_mean, [.1, .13, .12, .08, .12, .09])
    interaction = effects.loc[effects.effect.eq("ap_dp_interaction")].iloc[0]
    assert interaction["mean"] == pytest.approx(-.02)
    assert interaction.ci_lower == pytest.approx(-.02)
    assert interaction.interpretation == "negative"
    # Pairing removes the shared repeat offset; unpaired error propagation would not.
    runs.loc[runs.repeat_id.eq(1), "average_precision"] += .2
    tables = summary_tables(runs, {"repeats": 2, "confidence_level": .95, "practical_ap_threshold": .005})
    assert tables["metrics.csv"].average_precision_sd.gt(.1).all()
    np.testing.assert_allclose(tables["effects.csv"].sd, 0, atol=1e-15)


def test_retained_curve_validation_detects_inconsistent_aggregate_data():
    y, p = np.array([0, 0, 0, 1]), np.array([.1, .2, .3, .9])
    rows, arrays = [], {}
    for repeat in range(2):
        for path, (method, init) in CONDITIONS.items():
            rows.append(dict(repeat_id=repeat, path_id=path, status="complete", method=method,
                             initialization=init, **metric_values(y, p)))
            for field, values in curve_values(y, p).items():
                arrays[curve_key(repeat, path, field)] = values
    reference = dict(n=4, positive_n=1)
    runs = pd.DataFrame(rows)
    validate_curves(arrays, runs, reference)
    key = curve_key(0, "A", "calibration_positive_n")
    arrays[key] = arrays[key].copy()
    arrays[key][0] += 1
    with pytest.raises(ValueError, match="Invalid calibration positive counts"):
        validate_curves(arrays, runs, reference)
