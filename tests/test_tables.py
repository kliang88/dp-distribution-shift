"""Check exact CSV round trips, malformed input, and plotting without training."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import resolve_config
from dp_shift.sweep import load_plan
from dp_shift.data.utils import file_hash, read_json, write_json
from dp_shift.evaluation import tables
from dp_shift.evaluation.metrics import CONDITIONS, curve_key, curve_values, metric_values


@pytest.fixture
def dataset():
    config = resolve_config(load_plan(Path(__file__).resolve().parents[1] / "configs/experiment.json")["base"])
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    p = np.array([0., .1, .1, .2, .3, .7, .8, 1.])
    runs, pairs, privacy, curves = [], [], [], {}
    for repeat in range(2):
        for condition, rate in (("s0", .25), ("s1", config.requested.source_conditions.shift_fraction)):
            pairs.append(dict(repeat_id=repeat, condition=condition, positive_rate=rate))
        for path, (method, initialization) in CONDITIONS.items():
            runs.append(dict(repeat_id=repeat, path_id=path, method=method, initialization=initialization,
                             status="complete", **metric_values(y, p)))
            for field, value in curve_values(y, p).items():
                curves[curve_key(repeat, path, field)] = value
            if method == "dp_sgd":
                privacy.append(dict(repeat_id=repeat, path_id=path, accountant="prv", epsilon_achieved=.5,
                                    accountant_history=json.dumps([[1., 1 / 12, 12]])))
    data, ledger = tables.export(pd.DataFrame(runs), curves, pd.DataFrame(pairs), config, pd.DataFrame(privacy))
    return data, ledger, pd.DataFrame(runs), curves


def test_csv_round_trip_preserves_metrics_exact_curves_and_counts(dataset, tmp_path):
    data, ledger, original, curves = dataset
    tables.write(tmp_path, data)
    restored = tables.read(tmp_path)
    assert {path.name for path in (tmp_path / "data").iterdir()} == {f"{name}.csv" for name in tables.COLUMNS}
    for name in tables.COLUMNS:
        pd.testing.assert_frame_equal(data[name], restored[name], check_dtype=False, check_exact=True)
    scores, actual_curves, reference = tables.internal(restored)
    np.testing.assert_array_equal(scores.average_precision, original.average_precision)
    np.testing.assert_array_equal(scores.auroc, original.auroc)
    np.testing.assert_array_equal(scores.brier, original.brier)
    assert reference["n"] == 8 and reference["positive_n"] == 3
    for key in curves:
        np.testing.assert_array_equal(actual_curves[key], curves[key])
    assert tables.privacy_frame(restored, ledger).shape[0] == 6
    assert restored["runs"].pretraining.eq("none").sum() == 4
    assert restored["runs"].loc[restored["runs"].pretraining.eq("none"), "source_prevalence"].isna().all()
    assert "average_precision" not in restored["runs"]
    assert restored["runs"].loc[restored["runs"].pretraining.ne("none"), "source_size"].eq(20000).all()


@pytest.mark.parametrize("table,column,value,match", [
    ("runs", "auprc", .9, "AUPRC disagree"),
    ("runs", "method", "unknown", "Unknown method"),
    ("runs", "repeat", .5, "repeat numbers"),
    ("runs", "source_size", 40, "source size"),
    ("precision_recall", "run_id", "unknown", "run IDs differ"),
    ("precision_recall", "point", 1, "point order"),
    ("calibration", "patient_count", .5, "counts"),
    ("risk_distribution", "bin_upper", .03, "contiguous"),
    ("risk_distribution", "patient_count", 100, "prediction histogram"),
])
def test_invalid_plot_data_is_rejected(dataset, table, column, value, match):
    data = {name: frame.copy() for name, frame in dataset[0].items()}
    data[table][column] = data[table][column].astype(object) if isinstance(value, str) else data[table][column].astype(float)
    data[table].loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        tables.validate(data)


def test_plot_uses_only_csv_files_without_training_packages_or_protocol(dataset, tmp_path):
    tables.write(tmp_path, dataset[0])
    script = '''
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "opacus"} or name.startswith("dp_shift.training"):
        raise AssertionError("Plotting imported training code: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from dp_shift.cli import main
import sys
main(["plot", "--input", sys.argv[1], "--output", sys.argv[2]])
'''
    output = tmp_path / "custom_plots"
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(output.glob("*.png"))) == 5
    assert not (tmp_path / "protocol.json").exists()


def test_plot_rejects_a_destination_inside_data(dataset, tmp_path):
    from dp_shift.evaluation.report import run

    tables.write(tmp_path, dataset[0])
    with pytest.raises(ValueError, match="separate"):
        run(tmp_path, tmp_path / "data/plots")
    with pytest.raises(ValueError, match="separate"):
        run(tmp_path, tmp_path)


@pytest.mark.parametrize("kind", ["experiment", "sweep"])
def test_older_csvs_recover_source_size_without_rewriting_data(dataset, tmp_path, kind):
    data = {name: table.copy() for name, table in dataset[0].items()}
    tables.write(tmp_path, data)
    runs_path = tmp_path / "data/runs.csv"
    data["runs"].drop(columns="source_size").to_csv(runs_path, index=False)
    original_hash = file_hash(runs_path)
    if kind == "experiment":
        saved = {"stages": {"prepare": {"resolved": {"pretrain_size": 20000}}}}
    else:
        saved = {"plan": {"base": {"source_conditions": {"pretrain_size": 20000}}}}
    write_json(tmp_path / "protocol.json", saved)
    restored = tables.read(tmp_path)
    pd.testing.assert_frame_equal(restored["runs"], data["runs"], check_dtype=False)
    assert file_hash(runs_path) == original_hash


def test_missing_source_size_needs_metadata_for_old_exports(dataset, tmp_path):
    tables.write(tmp_path, dataset[0])
    dataset[0]["runs"].drop(columns="source_size").to_csv(tmp_path / "data/runs.csv", index=False)
    with pytest.raises(ValueError, match="source_size"):
        tables.read(tmp_path)


def test_prevalence_only_csvs_normalize_to_readmission_without_rewriting(dataset, tmp_path):
    data = dataset[0]
    tables.write(tmp_path, data)
    path = tmp_path / "data/runs.csv"
    old = data["runs"].drop(columns=["shift_type", "source_fraction", "test_fraction"])
    old.rename(columns={"shift_fraction": "shift_prevalence"}).to_csv(path, index=False)
    original_hash = file_hash(path)
    restored = tables.read(tmp_path)
    pd.testing.assert_frame_equal(restored["runs"], data["runs"], check_dtype=False)
    assert file_hash(path) == original_hash


@pytest.mark.parametrize("options", [{"epsilon": 999}, {"fraction": .999}])
def test_unknown_plot_selections_fail_before_writing_figures(dataset, tmp_path, options):
    from dp_shift.evaluation.report import run

    tables.write(tmp_path, dataset[0])
    with pytest.raises(ValueError, match="must be one of"):
        run(tmp_path, **options)
    assert not (tmp_path / "plots").exists()
