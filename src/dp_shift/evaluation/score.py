"""One inference pass per checkpoint, with resumable, aligned prediction caches."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.data.artifacts import load_evaluation_data
from dp_shift.data.prepare import materialize_test
from dp_shift.data.splits import read_manifest
from dp_shift.data.utils import file_hash, read_json, require
from dp_shift.training.checkpoints import atomic_json, load_checkpoint
from dp_shift.training.model import configure_cpu
from dp_shift.training.tune import predict
from .metrics import CONDITIONS, curve_key, curve_values, metric_values, validate_probabilities
from .provenance import atomic_csv, atomic_path, load_evaluation, require_open, verify_scored


def cache_paths(output, run):
    path = Path(output) / f"predictions/repeat_{run['repeat_id']:02d}/{run['path_id']}.npz"
    return path, path.with_suffix(".json")


def cache_metadata(output, run, reference):
    return {"status": "complete", "repeat_id": run["repeat_id"], "path_id": run["path_id"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "evaluation_freeze_sha256": file_hash(Path(output) / "frozen_evaluation.json"),
            "test_manifest_sha256": reference["test_manifest_sha256"],
            "labels_sha256": reference["labels_sha256"], "n": reference["n"]}


def read_prediction_cache(path, metadata_path, expected, labels):
    metadata = read_json(metadata_path)
    require({k: metadata.get(k) for k in expected} == expected, "Prediction-cache provenance mismatch")
    require(file_hash(path) == metadata["prediction_sha256"], "Prediction-cache fingerprint mismatch")
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {"row_id", "probability"}, "Invalid prediction cache arrays")
        rows, probability = saved["row_id"], saved["probability"]
    require(rows.dtype == np.int64 and np.array_equal(rows, np.arange(len(labels))),
            "Prediction row IDs do not match the test manifest")
    require(probability.dtype == np.float32, "Expected float32 prediction cache")
    validate_probabilities(labels, probability)
    return probability


def score_models(output, resume=False):
    output = Path(output)
    if (output / "SCORED.json").exists():
        frozen = verify_scored(output)
        return {"scored_models": len(frozen["runs"])}
    require_open(output)
    frozen = load_evaluation(output)
    require(resume or not (output / "predictions").exists(), "Prediction caches exist; use score --resume")
    configure_cpu()
    data_root = Path(frozen["data_root"])
    materialize_test(data_root, output / "inputs/evaluation_ready.json")
    test = load_evaluation_data(data_root)
    labels = test["y"]
    membership = read_manifest(data_root / "manifests/target_test.csv")
    reference = {"n": len(labels), "positive_n": int(labels.sum()), "test_prevalence": float(labels.mean()),
                 "test_manifest_sha256": file_hash(data_root / "manifests/target_test.csv"),
                 "labels_sha256": hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest()}
    require(reference["n"] == frozen["expected_test_n"], "Target test size differs from configured size")
    prevalence = reference["test_prevalence"]
    reference.update(constant_ap=prevalence, constant_auroc=.5,
                     constant_brier=prevalence * (1 - prevalence))
    reference_path = output / "test_reference.json"
    if reference_path.exists():
        require(read_json(reference_path) == reference, "Test reference changed during scoring")
    else:
        atomic_json(reference_path, reference)
    atomic_csv(output / "test_rows.csv", membership.rename(columns={"position": "row_id"}))

    # Build every cache before calculating/reporting test performance.
    for run in frozen["runs"]:
        path, metadata_path = cache_paths(output, run)
        expected = cache_metadata(output, run, reference)
        if metadata_path.exists():
            read_prediction_cache(path, metadata_path, expected, labels)
            continue
        require(resume or not path.exists(), "Uncommitted cache exists; use score --resume")
        model, _ = load_checkpoint(run["checkpoint_path"], {"input_dimension": frozen["input_dimension"],
            "repeat_id": run["repeat_id"], "path_id": run["path_id"]})
        probability = predict(model, test)
        validate_probabilities(labels, probability)
        require(probability.dtype == np.float32, "Predictor changed probability precision")
        # A data file without its metadata is uncommitted and can be replaced on resume.
        with atomic_path(path) as temporary:
            np.savez_compressed(temporary, row_id=np.arange(len(labels), dtype=np.int64), probability=probability)
        atomic_json(metadata_path, {**expected, "prediction_sha256": file_hash(path)})
        read_prediction_cache(path, metadata_path, expected, labels)

    runs = pd.read_csv(Path(frozen["experiment"]) / "training_runs.csv")
    keys = list(zip(runs.repeat_id, runs.path_id))
    expected_keys = {(r["repeat_id"], r["path_id"]) for r in frozen["runs"]}
    require(len(keys) == len(set(keys)) == len(expected_keys) and set(keys) == expected_keys,
            "Training ledger contains missing or duplicate paths")
    require(runs.status.eq("complete").all(), "An incomplete training run cannot be scored")
    rows, curves, index = [], {}, {}
    for run in frozen["runs"]:
        path, metadata_path = cache_paths(output, run)
        probability = read_prediction_cache(path, metadata_path, cache_metadata(output, run, reference), labels)
        method, initialization = CONDITIONS[run["path_id"]]
        rows.append({"repeat_id": run["repeat_id"], "path_id": run["path_id"], "status": "complete",
                     "method": method, "initialization": initialization, **metric_values(labels, probability)})
        for field, values in curve_values(labels, probability, frozen["config"]["calibration_bins"]).items():
            curves[curve_key(run["repeat_id"], run["path_id"], field)] = values
        index[f"repeat_{run['repeat_id']:02d}/{run['path_id']}"] = {
            p.relative_to(output).as_posix(): file_hash(p) for p in [path, metadata_path]}
    runs = pd.DataFrame(rows)
    atomic_csv(output / "runs.csv", runs)
    with atomic_path(output / "curves.npz") as temporary:
        np.savez_compressed(temporary, **curves)
    atomic_json(output / "prediction_index.json", index)
    names = ["runs.csv", "curves.npz", "prediction_index.json", "test_reference.json", "test_rows.csv"]
    atomic_json(output / "SCORED.json", {"status": "complete", "n_runs": len(runs),
        "freeze_sha256": file_hash(output / "frozen_evaluation.json"),
        "files": {name: file_hash(output / name) for name in names}})
    verify_scored(output)
    return {"scored_models": len(runs), "test_patients": len(labels)}
