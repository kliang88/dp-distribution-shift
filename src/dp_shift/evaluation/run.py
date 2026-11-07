"""Verify scalar scores, curve summaries, and research tables against cached predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.data.artifacts import load_evaluation_data
from dp_shift.data.utils import read_json, require
from .metrics import curve_key, curve_values, metric_values, validate_curves
from .provenance import verify_scored
from .score import cache_metadata, cache_paths, read_prediction_cache
from .summarize import summary_tables, validate_runs


def verify_analysis(output):
    """Recompute analysis tables from cached probabilities, without inference."""
    output = Path(output)
    frozen = verify_scored(output)
    test = load_evaluation_data(frozen["data_root"])
    labels = test["y"]
    reference = read_json(output / "test_reference.json")
    require(reference["n"] == len(labels) and reference["positive_n"] == int(labels.sum()), "Test reference mismatch")
    runs = pd.read_csv(output / "runs.csv", float_precision="round_trip")
    validate_runs(runs, frozen["config"]["repeats"])
    expected_curves = {}
    for run in frozen["runs"]:
        path, metadata_path = cache_paths(output, run)
        probability = read_prediction_cache(path, metadata_path, cache_metadata(output, run, reference), labels)
        actual = runs.loc[runs.repeat_id.eq(run["repeat_id"]) & runs.path_id.eq(run["path_id"])].iloc[0]
        for metric, value in metric_values(labels, probability).items():
            require(np.isclose(actual[metric], value, rtol=1e-10, atol=1e-12), f"Metric mismatch: {metric}")
        for field, values in curve_values(labels, probability, frozen["config"]["calibration_bins"]).items():
            expected_curves[curve_key(run["repeat_id"], run["path_id"], field)] = values
    with np.load(output / "curves.npz", allow_pickle=False) as curves:
        validate_curves(curves, runs, reference)
        for name, values in expected_curves.items():
            np.testing.assert_allclose(curves[name], values, rtol=1e-12, atol=1e-12)
    expected = summary_tables(runs, frozen["config"])
    for name, table in expected.items():
        actual = pd.read_csv(output / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, table, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    rows = pd.read_csv(output / "test_rows.csv")
    membership = pd.read_csv(Path(frozen["data_root"]) / "manifests/target_test.csv").rename(columns={"position": "row_id"})
    pd.testing.assert_frame_equal(rows, membership)
    return {"evaluated_models": len(runs), "test_patients": len(labels), "repeats": frozen["config"]["repeats"]}

