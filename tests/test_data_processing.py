"""Tests of cohort validity, leakage prevention, and saved-data integrity."""

from copy import deepcopy
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from dp_shift.config import load_config
from dp_shift.data.artifacts import (PATH_IDS, load_evaluation_data, load_training_data,
                                     validate_evaluation_manifest, verify_artifacts)
from dp_shift.data.cohort import build_cohort
from dp_shift.data.prepare import build, materialize_test
from dp_shift.data.preprocessing import (fit_source_transformer, transform_features,
                                         transformer_metadata)
from dp_shift.data.sampling import create_source_pair
from dp_shift.data.schema import load_and_validate_raw
from dp_shift.data.splits import manifest, rows_for_manifest
from dp_shift.data.utils import file_hash, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment.json"


@pytest.fixture
def config():
    config = load_config(CONFIG_PATH)
    config["inputs"] = {"records": {"sha256": file_hash(ROOT / "data/diabetic_data.csv")}}
    return config


def make_frame(n, config):
    frame = pd.DataFrame({name: np.arange(n, dtype=float) + i
                          for i, name in enumerate(config["numeric_features"])})
    for name in config["categorical_features"]:
        frame[name] = "1" if name in config["categorical_ids"] else "known"
    frame["patient_nbr"] = np.arange(n, dtype=np.int64) + 1
    frame["encounter_id"] = np.arange(n, dtype=np.int64) + 1000
    frame["y"] = (np.arange(n) % 3 == 0).astype(np.int64)
    return frame


def save_raw(frame, path):
    frame = frame.copy()
    frame["readmitted"] = np.where(frame.y.eq(1), "<30", "NO")
    frame.drop(columns="y").to_csv(path, index=False)


def test_index_selection_precedes_exclusion(config):
    frame = make_frame(4, config)
    frame["patient_nbr"] = [1, 1, 2, 2]
    frame["encounter_id"] = [10, 20, 30, 40]
    frame["discharge_disposition_id"] = ["11", "1", "1", "1"]
    cohort, excluded, _ = build_cohort(frame.sample(frac=1, random_state=10), config)
    assert cohort.patient_nbr.tolist() == [2]
    assert cohort.encounter_id.tolist() == [30]
    assert excluded.patient_nbr.tolist() == [1]


def test_raw_parser_preserves_large_integer_ids_and_literal_categories(tmp_path, config):
    frame = make_frame(2, config)
    frame["patient_nbr"] = [9007199254740993, 9007199254740994]
    frame["race"] = ["None", "?"]
    frame["admission_type_id"] = ["1.0", "2"]
    path = tmp_path / "records.csv"
    save_raw(frame, path)
    config["inputs"]["records"]["sha256"] = file_hash(path)
    result = load_and_validate_raw(path, config)
    assert result.patient_nbr.tolist() == frame.patient_nbr.tolist()
    assert result.admission_type_id.tolist() == ["1", "2"]
    assert result.race.iloc[0] == "None" and pd.isna(result.race.iloc[1])
    assert result.y.tolist() == [1, 0]
    assert "readmitted" not in result


@pytest.mark.parametrize("column,value,error", [
    ("encounter_id", "1000", "Duplicate encounter"),
    ("patient_nbr", "?", "Missing patient"),
    ("readmitted", "unexpected", "unexpected readmitted"),
    ("time_in_hospital", "bad", "Malformed numerical"),
    ("time_in_hospital", "-1", "Negative or nonfinite"),
    ("time_in_hospital", "inf", "Negative or nonfinite"),
])
def test_raw_parser_rejects_invalid_data(tmp_path, config, column, value, error):
    path = tmp_path / "records.csv"
    save_raw(make_frame(2, config), path)
    frame = pd.read_csv(path, dtype=str)
    frame.loc[1, column] = value
    frame.to_csv(path, index=False)
    config["inputs"]["records"]["sha256"] = file_hash(path)
    with pytest.raises(ValueError, match=error):
        load_and_validate_raw(path, config)


def test_raw_fingerprint_and_duplicate_headers(tmp_path, config):
    path = tmp_path / "records.csv"
    save_raw(make_frame(2, config), path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_and_validate_raw(path, config)
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + ",patient_nbr\n" + "\n".join(line + ",1" for line in lines[1:]) + "\n")
    config["inputs"]["records"]["sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="Duplicate CSV column"):
        load_and_validate_raw(path, config)


def test_source_pair_replaces_positives_without_changing_size(config):
    pool = make_frame(100, config)
    config.update(pretrain_size=20, source_positive_n=1)
    first = create_source_pair(pool, 0, config)
    second = create_source_pair(pool, 0, config)
    assert len(first["s1"]) == 20 and first["s1"].y.sum() == 1
    assert first["removed"].y.eq(1).all() and first["added"].y.eq(0).all()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    config["source_positive_n"] = 20
    with pytest.raises(ValueError, match="more positives"):
        create_source_pair(pool, 0, config)


def test_preprocessing_uses_source_statistics_and_survives_unseen_categories(tmp_path, config):
    source = make_frame(4, config)
    source["number_outpatient"] = [0., 1., 3., 7.]
    source.loc[0, "time_in_hospital"] = np.nan
    transformer = fit_source_transformer(source, config)
    before = deepcopy(transformer_metadata(transformer, source, config))
    # Compute the expected statistics independently in the prescribed order.
    values = source[config["numeric_features"]].to_numpy().copy()
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isnan(values), medians, values)
    for name in config["log_features"]:
        index = config["numeric_features"].index(name)
        values[:, index] = np.log1p(values[:, index])
    np.testing.assert_allclose(before["medians"], medians)
    np.testing.assert_allclose(before["means"], values.mean(axis=0))
    np.testing.assert_allclose(before["scales"], values.std(axis=0))
    np.testing.assert_allclose(transform_features(transformer, source)[:, :8],
                               (values - values.mean(axis=0)) / values.std(axis=0), atol=1e-6)
    target = make_frame(2, config)
    target["race"] = ["unseen", np.nan]
    target.loc[0, "time_in_hospital"] = 1e8
    target.loc[1, "number_outpatient"] = np.nan
    transformed = transform_features(transformer, target)
    assert np.isfinite(transformed).all() and transformed.shape[1] == before["input_dimension"]
    assert transformer_metadata(transformer, source, config) == before
    joblib.dump(transformer, tmp_path / "transformer.joblib")
    np.testing.assert_array_equal(transformed, transform_features(joblib.load(tmp_path / "transformer.joblib"), target))
    source["num_medications"] = np.nan
    with pytest.raises(ValueError, match="entire source numerical"):
        fit_source_transformer(source, config)


def test_manifest_lookup_aligns_rows_by_id(config):
    frame = make_frame(10, config)
    membership = manifest(frame.iloc[[8, 1, 6]])
    actual = rows_for_manifest(frame.sample(frac=1, random_state=1), membership)
    np.testing.assert_array_equal(actual.y, frame.iloc[[8, 1, 6]].y)
    np.testing.assert_array_equal(actual.patient_nbr, membership.patient_nbr)
    membership.loc[0, "encounter_id"] += 1
    with pytest.raises(ValueError, match="encounter IDs"):
        rows_for_manifest(frame, membership)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp("real_data") / "prepared"
    summary = build(CONFIG_PATH, output)
    return output, summary


def test_real_data_build_and_reproducibility(prepared, tmp_path):
    output, summary = prepared
    repeats = read_json(CONFIG_PATH)["repeats"]
    assert summary == {"patients": 69973, "input_dimension": 74, "source_pairs": repeats, "target_train_n": 3000}
    assert not (output / "evaluation").exists()
    assert not (output / "arrays/target_test.npz").exists()
    assert verify_artifacts(output) == summary
    numeric = load_training_data(output, "source_pool")["X"][:, :8]
    np.testing.assert_allclose(numeric.mean(axis=0, dtype=np.float64), 0, atol=1e-6)
    np.testing.assert_allclose(numeric.std(axis=0, dtype=np.float64), 1, atol=1e-6)
    # An independent build checks repeatability of manifests and encoded arrays.
    second = tmp_path / "rebuilt"
    assert build(CONFIG_PATH, second) == summary
    for name in ["source_pool", "proxy_train", "proxy_validation", "target_train"]:
        first_bundle = load_training_data(output, name)
        other_bundle = load_training_data(second, name)
        np.testing.assert_array_equal(first_bundle["X"], other_bundle["X"])
        np.testing.assert_array_equal(first_bundle["y"], other_bundle["y"])
    for path in (output / "manifests").glob("*.csv"):
        assert path.read_bytes() == (second / "manifests" / path.name).read_bytes()
    assert read_json(output / "preprocessing/metadata.json") == read_json(second / "preprocessing/metadata.json")
    for repeat in range(repeats):
        shifted = load_training_data(output, "s1", repeat)
        assert shifted["X"].shape == (16000, summary["input_dimension"]) and shifted["y"].sum() == 800
        for name in ["s0", "s1", "removed", "added"]:
            relative = f"source_pairs/repeat_{repeat:02d}/{name}.csv"
            assert (output / relative).read_bytes() == (second / relative).read_bytes()
    assert build(CONFIG_PATH, output) == summary  # Existing matching artifacts are reused.


def test_artifact_corruption_and_incomplete_output_are_rejected(prepared, tmp_path):
    output, _ = prepared
    with pytest.raises(ValueError, match="no COMPLETE"):
        verify_artifacts(tmp_path)
    corrupted = tmp_path / "corrupted"
    shutil.copytree(output, corrupted)
    with (corrupted / "arrays/target_train.npz").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_training_data(corrupted, "target_train")
    with pytest.raises(ValueError, match="Unsupported training partition"):
        load_training_data(output, "target_test")
    with pytest.raises(ValueError, match="repeat ID"):
        load_training_data(output, "s1", read_json(CONFIG_PATH)["repeats"])


@pytest.fixture
def synthetic_project(tmp_path, config):
    # Evaluation plumbing is tested with synthetic patients, never real test data.
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    records = tmp_path / "data/records.csv"
    save_raw(make_frame(400, config), records)
    mapping = tmp_path / "data/mapping.csv"
    mapping.write_text("discharge_disposition_id,description\n1,Home\n")
    experiment = config["experiment"]
    experiment["repeats"] = 2
    experiment["data"] = dict(source_pool_size=200, proxy_train_size=24, proxy_validation_size=40,
                              target_train_size=24, target_test_size=32)
    experiment["source_conditions"] = dict(pretrain_size=40, shifted_positive_rate=.05)
    path = tmp_path / "configs/test.json"
    write_json(path, experiment)
    output = tmp_path / "prepared"
    build(path, output, records, mapping)
    return output, tmp_path


def test_deferred_test_requires_complete_evaluation_manifest(synthetic_project, monkeypatch):
    output, directory = synthetic_project
    complete = read_json(output / "COMPLETE.json")
    checkpoint, analysis = directory / "checkpoint.bin", directory / "analysis.py"
    checkpoint.write_bytes(b"synthetic checkpoint")
    analysis.write_text("# synthetic analysis fixture\n")
    ready = {key: complete[key] for key in ["config_sha256", "artifact_manifest_sha256"]}
    ready["analysis"] = {"path": analysis.name, "sha256": file_hash(analysis)}
    ready["runs"] = []
    path = directory / "evaluation_ready.json"
    write_json(path, ready)
    with pytest.raises(ValueError, match="every completed training path"):
        materialize_test(output, path)
    assert not (output / "evaluation").exists()
    ready["runs"] = [{"repeat_id": repeat, "path_id": name, "status": "complete",
                      "checkpoint_path": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint)}
                     for repeat in range(2) for name in PATH_IDS]
    write_json(path, ready)
    validate_evaluation_manifest(path, output)

    def cannot_fit(*args, **kwargs):
        raise AssertionError("Evaluation must not fit preprocessing")

    monkeypatch.setattr("dp_shift.data.prepare.fit_source_transformer", cannot_fit)
    assert materialize_test(output, path) == {"target_test_n": 32}
    test = load_evaluation_data(output)
    assert test["X"].shape[0] == 32 and len(test["y"]) == 32
    assert materialize_test(output, path) == {"target_test_n": 32}
    verify_artifacts(output)  # Evaluation does not modify the training artifact index.
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        load_evaluation_data(output)


def test_changed_config_and_failed_build_do_not_overwrite_output(synthetic_project):
    output, directory = synthetic_project
    config_path = directory / "configs/test.json"
    config = read_json(config_path)
    original_marker = (output / "COMPLETE.json").read_bytes()
    config["source_conditions"]["shifted_positive_rate"] = .10
    write_json(config_path, config)
    with pytest.raises(ValueError, match="different configuration"):
        build(config_path, output, directory / "data/records.csv", directory / "data/mapping.csv")
    assert (output / "COMPLETE.json").read_bytes() == original_marker
    records = directory / "data/records.csv"
    raw = pd.read_csv(records, dtype=str)
    raw.loc[0, "num_medications"] = "bad"
    raw.to_csv(records, index=False)
    write_json(config_path, config)
    failed = directory / "failed_build"
    with pytest.raises(ValueError, match="Malformed numerical"):
        build(config_path, failed, records, directory / "data/mapping.csv")
    assert not failed.exists()
