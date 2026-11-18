"""Freeze evaluation separately, preserving every original training artifact."""

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from dp_shift.config import PROTOCOL, load_config
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


def freeze(data_root, experiment, output, config_path):
    data_root, experiment, output = (Path(p).resolve() for p in [data_root, experiment, output])
    config = load_config(config_path)
    if (output / "frozen_evaluation.json").exists():
        frozen = load_evaluation(output)
        require(frozen["config"] == config.to_dict() and frozen["data_root"] == str(data_root)
                and frozen["experiment"] == str(experiment), "Different evaluation configuration or inputs")
        return frozen
    require(not output.exists() or not any(output.iterdir()), "Unfinished evaluation directory; investigate it first")
    require(not (data_root / "evaluation").exists(), "Test inputs already belong to another evaluation version")
    verify_artifacts(data_root)
    verify_training(data_root, experiment)
    training = load_frozen(data_root, experiment)
    data_config = read_json(data_root / "config.json")
    require(config.requested.to_dict() == data_config["experiment"], "Experiment configuration differs from prepared data")
    require(config.requested.repeats == data_config["repeats"] and config.requested.repeats >= 1, "Evaluation repeat count mismatch")
    original_ready = experiment / "evaluation_ready.json"
    validate_evaluation_manifest(original_ready, data_root)
    ready = deepcopy(read_json(original_ready))
    ready["original_readiness_sha256"] = file_hash(original_ready)
    ready["evaluation_version"] = PROTOCOL.evaluation_version
    output.mkdir(parents=True, exist_ok=True)
    ready_path = output / "inputs/evaluation_ready.json"
    atomic_json(ready_path, ready)
    validate_evaluation_manifest(ready_path, data_root)
    inputs = [experiment / "frozen_protocol.json", original_ready, experiment / "training_runs.csv",
              data_root / "COMPLETE.json", data_root / "manifests/target_test.csv"]
    frozen = {"config": config.to_dict(), "data_root": str(data_root), "experiment": str(experiment),
              "input_hashes": {str(path): file_hash(path) for path in inputs},
              "readiness_sha256": file_hash(ready_path), "code_hashes": code_hashes(), "versions": versions(),
              "training_settings": {**training["selected"], "noise_multiplier": training["noise_multiplier"]}, "input_dimension": training["input_dimension"],
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
