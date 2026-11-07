"""Freeze evaluation separately, preserving every original training artifact."""

from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
import os
from pathlib import Path
import tempfile

from dp_shift.config import load_config
from dp_shift.data.artifacts import validate_evaluation_manifest, verify_artifacts
from dp_shift.data.utils import file_hash, read_json, require
from dp_shift.training.checkpoints import atomic_json
from dp_shift.training.run import verify_training
from dp_shift.training.tune import load_frozen, versions as training_versions


def code_hashes():
    paths = [*Path(__file__).parent.glob("*.py"), Path(__file__).resolve().parents[1] / "config.py"]
    return {path.name: file_hash(path) for path in sorted(paths)}


def versions():
    return {**training_versions(), **{name: version(name) for name in
            ["matplotlib", "Pillow", "fonttools", "cycler", "kiwisolver", "pyparsing", "contourpy"]}}


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path, frame):
    with atomic_path(path) as temporary:
        frame.to_csv(temporary, index=False, float_format="%.17g")


def require_open(output):
    require(not (Path(output) / "COMPLETE.json").exists(), "Evaluation is complete; use verify instead of overwriting")


def freeze(data_root, experiment, output, config_path):
    data_root, experiment, output = (Path(p).resolve() for p in [data_root, experiment, output])
    config = load_config(config_path)
    if (output / "frozen_evaluation.json").exists():
        frozen = load_evaluation(output)
        require(frozen["config"] == config and frozen["data_root"] == str(data_root)
                and frozen["experiment"] == str(experiment), "Different evaluation configuration or inputs")
        return frozen
    require(not output.exists() or not any(output.iterdir()), "Unfinished evaluation directory; investigate it first")
    require(not (experiment / "runs.csv").exists(), "Legacy evaluator has already scored the test set")
    require(not (data_root / "evaluation").exists(), "Test inputs already belong to another evaluation version")
    verify_artifacts(data_root)
    verify_training(data_root, experiment)
    training = load_frozen(data_root, experiment)
    data_config = read_json(data_root / "config.json")
    require(config["experiment"] == data_config["experiment"], "Experiment configuration differs from prepared data")
    require(config["repeats"] == data_config["repeats"] and config["repeats"] >= 2, "Evaluation repeat count mismatch")
    original_ready = experiment / "evaluation_ready.json"
    validate_evaluation_manifest(original_ready, data_root)
    ready = deepcopy(read_json(original_ready))
    ready["analysis"] = {"path": str(Path(__file__).with_name("run.py").resolve()),
                         "sha256": file_hash(Path(__file__).with_name("run.py"))}
    ready["original_readiness_sha256"] = file_hash(original_ready)
    ready["evaluation_version"] = config["evaluation_version"]
    output.mkdir(parents=True, exist_ok=True)
    ready_path = output / "inputs/evaluation_ready.json"
    atomic_json(ready_path, ready)
    validate_evaluation_manifest(ready_path, data_root)
    inputs = [experiment / "frozen_protocol.json", original_ready, experiment / "training_runs.csv",
              data_root / "COMPLETE.json", data_root / "manifests/target_test.csv"]
    frozen = {"config": config, "data_root": str(data_root), "experiment": str(experiment),
              "input_hashes": {str(path): file_hash(path) for path in inputs},
              "readiness_sha256": file_hash(ready_path), "code_hashes": code_hashes(), "versions": versions(),
              "training_settings": training["settings"], "input_dimension": training["input_dimension"],
              "expected_test_n": data_config["data"]["target_test_size"],
              "runs": ready["runs"], "cpu_threads": 1, "prediction_dtype": "float32"}
    atomic_json(output / "frozen_evaluation.json", frozen)
    return frozen


def load_evaluation(output):
    output = Path(output)
    frozen = read_json(output / "frozen_evaluation.json")
    require(frozen["code_hashes"] == code_hashes(), "Evaluation source code changed after freezing")
    require(frozen["versions"] == versions(), "Evaluation dependency versions changed")
    for path, digest in frozen["input_hashes"].items():
        require(file_hash(path) == digest, f"Frozen evaluation input changed: {path}")
    require(file_hash(output / "inputs/evaluation_ready.json") == frozen["readiness_sha256"],
            "Evaluation readiness changed")
    load_frozen(frozen["data_root"], frozen["experiment"])
    validate_evaluation_manifest(output / "inputs/evaluation_ready.json", frozen["data_root"])
    return frozen


def verify_scored(output):
    output = Path(output)
    frozen = load_evaluation(output)
    scored = read_json(output / "SCORED.json")
    require(scored.get("status") == "complete", "Scoring is incomplete")
    require(scored["freeze_sha256"] == file_hash(output / "frozen_evaluation.json"), "Scoring freeze mismatch")
    expected = {"runs.csv", "curves.npz", "prediction_index.json", "test_reference.json", "test_rows.csv"}
    require(set(scored["files"]) == expected, "Incomplete scored artifact list")
    for name, digest in scored["files"].items():
        require(file_hash(output / name) == digest, f"Scoring artifact changed: {name}")
    index = read_json(output / "prediction_index.json")
    require(len(index) == len(frozen["runs"]), "Missing prediction caches")
    require(set(index) == {f"repeat_{r['repeat_id']:02d}/{r['path_id']}" for r in frozen["runs"]},
            "Incorrect prediction-cache paths")
    for entry in index.values():
        for name, digest in entry.items():
            require(file_hash(output / name) == digest, f"Prediction cache changed: {name}")
    return frozen
