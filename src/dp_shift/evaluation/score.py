"""One inference pass per checkpoint, with resumable, aligned prediction caches."""

import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dp_shift.data.artifacts import load_evaluation_data
from dp_shift.data.prepare import materialize_test
from dp_shift.data.splits import read_manifest
from dp_shift.data.utils import file_hash, read_json, require
from dp_shift.training.checkpoints import atomic_json, load_checkpoint
from dp_shift.training.model import configure_cpu
from dp_shift.training.tune import predict
from .metrics import validate_probabilities
from .provenance import load_evaluation
from dp_shift.stages.common import atomic_path


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
    reference_path = output / "test_reference.json"
    if reference_path.exists():
        require(read_json(reference_path) == reference, "Test reference changed during scoring")
    else:
        atomic_json(reference_path, reference)
    scores = {}
    for run in frozen["runs"]:
        path, metadata_path = cache_paths(output, run)
        expected = cache_metadata(output, run, reference)
        if not metadata_path.exists():
            require(resume or not path.exists(), "Uncommitted prediction cache exists")
            model, _ = load_checkpoint(run["checkpoint_path"], {"input_dimension": frozen["input_dimension"],
                "repeat_id": run["repeat_id"], "path_id": run["path_id"]})
            probability = predict(model, test)
            validate_probabilities(labels, probability)
            require(probability.dtype == np.float32, "Predictor changed probability precision")
            with atomic_path(path) as temporary:
                np.savez_compressed(temporary, row_id=np.arange(len(labels), dtype=np.int64), probability=probability)
            atomic_json(metadata_path, {**expected, "prediction_sha256": file_hash(path)})
        probability = read_prediction_cache(path, metadata_path, expected, labels)
        scores[run["repeat_id"], run["path_id"]] = {
            "auprc": float(average_precision_score(labels, probability)),
            "auroc": float(roc_auc_score(labels, probability)),
            "brier": float(brier_score_loss(labels, probability)),
        }
    return scores, reference
