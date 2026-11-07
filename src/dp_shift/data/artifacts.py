"""Saved arrays, integrity checks, and the small training/evaluation interface."""

from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .cohort import build_cohort
from .preprocessing import transform_features, transformer_metadata
from .sampling import validate_source_pair
from .schema import discharge_descriptions, load_and_validate_raw
from dp_shift.config import PARTITIONS

from .splits import read_manifest, rows_for_manifest, validate_splits
from .utils import file_hash, read_json, require, write_json


TRAIN_PARTITIONS = ("source_pool", "proxy_train", "proxy_validation", "target_train")
PATH_IDS = ("A", "B-S0", "B-S1", "C", "D-S0", "D-S1")


def save_bundle(path, frame, transformer, manifest_path, transformer_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = transform_features(transformer, frame)
    np.savez_compressed(path, X=values, y=frame.y.to_numpy(dtype=np.int64),
                        patient_id=frame.patient_nbr.to_numpy(dtype=np.int64),
                        encounter_id=frame.encounter_id.to_numpy(dtype=np.int64))
    write_json(path.with_suffix(".json"), {
        "n": len(frame), "input_dimension": values.shape[1],
        "manifest_sha256": file_hash(manifest_path),
        "transformer_sha256": file_hash(transformer_path),
    })


def read_bundle(path, dimension):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {"X", "y", "patient_id", "encounter_id"}, f"Invalid array bundle: {path}")
        bundle = {key: saved[key] for key in saved.files}
    n = len(bundle["y"])
    require(bundle["X"].shape == (n, dimension), f"Incorrect feature shape: {path}")
    require(bundle["X"].dtype == np.float32 and np.isfinite(bundle["X"]).all(),
            f"Invalid feature dtype/values: {path}")
    for key in ["y", "patient_id", "encounter_id"]:
        require(bundle[key].dtype == np.int64 and bundle[key].shape == (n,), f"Invalid {key}: {path}")
    require(np.isin(bundle["y"], [0, 1]).all(), f"Nonbinary labels: {path}")
    require(len(np.unique(bundle["patient_id"])) == n, f"Duplicate patients: {path}")
    return bundle


def write_artifact_index(root):
    root = Path(root)
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"artifact_manifest.json", "COMPLETE.json"} or relative.startswith("evaluation/"):
            continue
        entry = {"sha256": file_hash(path)}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as bundle:
                entry["arrays"] = {key: {"shape": list(bundle[key].shape), "dtype": str(bundle[key].dtype)}
                                   for key in bundle.files}
        elif path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            entry["shape"], entry["dtype"] = list(array.shape), str(array.dtype)
        elif path.suffix == ".csv":
            table = pd.read_csv(path)
            entry["columns"], entry["n"] = table.columns.tolist(), len(table)
        entries[relative] = entry
    write_json(root / "artifact_manifest.json", entries)
    provenance = read_json(root / "provenance.json")
    write_json(root / "COMPLETE.json", {
        "status": "complete", "protocol_version": read_json(root / "config.json")["protocol_version"],
        "artifact_manifest_sha256": file_hash(root / "artifact_manifest.json"),
        "config_sha256": file_hash(root / "config.json"),
        "protocol_sha256": provenance["protocol_sha256"],
        "transformer_sha256": file_hash(root / "preprocessing/transformer.joblib"),
    })


def check_artifact_files(root):
    """Validate the completion marker and every saved file before loading a pickle."""
    root = Path(root)
    require((root / "COMPLETE.json").is_file(), "Data preparation is incomplete: no COMPLETE.json")
    complete = read_json(root / "COMPLETE.json")
    require(complete.get("status") == "complete", "Data preparation did not complete")
    require(file_hash(root / "artifact_manifest.json") == complete["artifact_manifest_sha256"],
            "Artifact index fingerprint mismatch")
    entries = read_json(root / "artifact_manifest.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p.relative_to(root).parts[0] != "evaluation"
              and p.relative_to(root).as_posix() not in {"artifact_manifest.json", "COMPLETE.json"}}
    require(actual == set(entries), "Missing or unexpected data artifact files")
    for name, entry in entries.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid artifact path")
        require(file_hash(root / name) == entry["sha256"], f"Artifact fingerprint mismatch: {name}")
    require(file_hash(root / "config.json") == complete["config_sha256"], "Configuration fingerprint mismatch")
    require(file_hash(root / "preprocessing/transformer.joblib") == complete["transformer_sha256"],
            "Transformer fingerprint mismatch")
    return complete


def input_paths(provenance, config):
    project = Path(provenance["project_root"])
    return {name: project / item["path"] for name, item in config["inputs"].items()}


def load_saved_cohort(root):
    config = read_json(Path(root) / "config.json")
    provenance = read_json(Path(root) / "provenance.json")
    paths = input_paths(provenance, config)
    for name, path in paths.items():
        require(file_hash(path) == config["inputs"][name]["sha256"], f"Input fingerprint mismatch: {name}")
    protocol = Path(provenance["project_root"]) / config["protocol_path"]
    require(file_hash(protocol) == provenance["protocol_sha256"], "Protocol changed since preparation")
    raw = load_and_validate_raw(paths["records"], config)
    return build_cohort(raw, config, discharge_descriptions(paths["mapping"]))


def verify_artifacts(root):
    """Verify persisted memberships and transforms, without resampling or fitting."""
    root = Path(root)
    complete = check_artifact_files(root)
    config = read_json(root / "config.json")
    provenance = read_json(root / "provenance.json")
    require(complete["protocol_sha256"] == provenance["protocol_sha256"], "Protocol metadata mismatch")
    require(complete["protocol_version"] == config["protocol_version"], "Protocol version mismatch")
    for name in ["numpy", "pandas", "scikit-learn", "joblib"]:
        require(version(name) == provenance["versions"][name], f"Library version changed: {name}")
    cohort, exclusions, audit = load_saved_cohort(root)
    require(audit == read_json(root / "cohort_audit.json"), "Cohort audit mismatch")
    saved_exclusions = pd.read_csv(root / "exclusions.csv", keep_default_na=False,
                                   dtype={"discharge_disposition_id": str})
    pd.testing.assert_frame_equal(saved_exclusions, exclusions, check_dtype=False)
    names = {"cohort", *PARTITIONS, "unused"}
    require({p.stem for p in (root / "manifests").glob("*.csv")} == names, "Missing partition manifests")
    manifests = {name: read_manifest(root / f"manifests/{name}.csv") for name in names}
    frames = {name: rows_for_manifest(cohort, membership) for name, membership in manifests.items()}
    require(np.array_equal(frames["cohort"].patient_nbr, cohort.patient_nbr), "Cohort manifest order mismatch")
    validate_splits(frames, config)

    transformer = joblib.load(root / "preprocessing/transformer.joblib")
    metadata = read_json(root / "preprocessing/metadata.json")
    require(metadata == transformer_metadata(transformer, frames["source_pool"], config),
            "Fitted preprocessing metadata mismatch")
    for name in TRAIN_PARTITIONS:
        path = root / f"arrays/{name}.npz"
        bundle = read_bundle(path, metadata["input_dimension"])
        frame = frames[name]
        for key, column in [("patient_id", "patient_nbr"), ("encounter_id", "encounter_id"), ("y", "y")]:
            require(np.array_equal(bundle[key], frame[column]), f"Array/manifest {key} mismatch: {name}")
        require(np.array_equal(bundle["X"], transform_features(transformer, frame)),
                f"Feature transformation mismatch: {name}")
        require(read_json(path.with_suffix(".json")) == {
            "n": len(frame), "input_dimension": metadata["input_dimension"],
            "manifest_sha256": file_hash(root / f"manifests/{name}.csv"),
            "transformer_sha256": complete["transformer_sha256"],
        }, f"Array provenance mismatch: {name}")

    pool = frames["source_pool"]
    expected_repeats = {f"repeat_{r:02d}" for r in range(config["repeats"])}
    require({p.name for p in (root / "source_pairs").iterdir()} == expected_repeats, "Missing source repeats")
    for repeat in range(config["repeats"]):
        directory = root / f"source_pairs/repeat_{repeat:02d}"
        pair = {name: rows_for_manifest(cohort, read_manifest(directory / f"{name}.csv"))
                for name in ["s0", "s1", "removed", "added"]}
        validate_source_pair(pool, pair, config)
        for name in ["s0", "s1"]:
            indices = np.load(directory / f"{name}_indices.npy", allow_pickle=False)
            require(indices.dtype == np.int64 and indices.shape == (config["pretrain_size"],),
                    f"Invalid source indices: {repeat}/{name}")
            require(((indices >= 0) & (indices < len(pool))).all(), "Source index outside cache")
            require(np.array_equal(pool.patient_nbr.to_numpy()[indices], pair[name].patient_nbr),
                    f"Source cache alignment mismatch: {repeat}/{name}")
    require(not (root / "arrays/target_test.npz").exists(), "Test bundle is in the training directory")
    return {"patients": len(cohort), "input_dimension": metadata["input_dimension"],
            "source_pairs": config["repeats"], "target_train_n": len(frames["target_train"])}


def load_training_data(root, partition, repeat_id=None):
    """Return model inputs only; identifiers never enter X. Test data are excluded."""
    root = Path(root)
    check_artifact_files(root)
    config = read_json(root / "config.json")
    metadata = read_json(root / "preprocessing/metadata.json")
    require(partition in (*TRAIN_PARTITIONS, "s0", "s1"), "Unsupported training partition")
    source = partition in {"s0", "s1"}
    if source:
        require(isinstance(repeat_id, int) and 0 <= repeat_id < config["repeats"], "Invalid source repeat ID")
    else:
        require(repeat_id is None, "repeat_id is only used for source samples")
    bundle = read_bundle(root / f"arrays/{'source_pool' if source else partition}.npz", metadata["input_dimension"])
    indices = (np.load(root / f"source_pairs/repeat_{repeat_id:02d}/{partition}_indices.npy", allow_pickle=False)
               if source else None)
    return {"X": bundle["X"] if indices is None else bundle["X"][indices],
            "y": bundle["y"] if indices is None else bundle["y"][indices],
            "input_dimension": metadata["input_dimension"], "source_row_indices": indices,
            "provenance": read_json(root / "COMPLETE.json")}


def validate_evaluation_manifest(path, root):
    path, root = Path(path), Path(root)
    evaluation = read_json(path)
    complete = read_json(root / "COMPLETE.json")
    for key in ["config_sha256", "artifact_manifest_sha256"]:
        require(evaluation.get(key) == complete[key], f"Evaluation manifest has a different {key}")
    analysis = evaluation["analysis"]
    require(file_hash(path.parent / analysis["path"]) == analysis["sha256"], "Frozen analysis fingerprint mismatch")
    config = read_json(root / "config.json")
    expected = {(repeat, name) for repeat in range(config["repeats"]) for name in PATH_IDS}
    runs = evaluation["runs"]
    actual = {(run["repeat_id"], run["path_id"]) for run in runs}
    require(len(runs) == len(expected) and actual == expected, "Evaluation requires every completed training path")
    for run in runs:
        require(run["status"] == "complete", "Evaluation includes an incomplete training run")
        require(file_hash(path.parent / run["checkpoint_path"]) == run["checkpoint_sha256"],
                "Training checkpoint fingerprint mismatch")


def load_evaluation_data(root):
    root = Path(root)
    check_artifact_files(root)
    directory = root / "evaluation"
    complete = read_json(directory / "COMPLETE.json")
    require(complete.get("status") == "complete", "Test preparation is incomplete")
    expected_files = {"target_test.npz", "target_test.json", "evaluation_ready.json"}
    require(set(complete["files"]) == expected_files, "Incomplete evaluation artifact index")
    require({p.name for p in directory.iterdir()} == expected_files | {"COMPLETE.json"},
            "Missing or unexpected evaluation files")
    require(complete["data_manifest_sha256"] == file_hash(root / "artifact_manifest.json"), "Test data provenance mismatch")
    for name, digest in complete["files"].items():
        require(file_hash(directory / name) == digest, f"Evaluation artifact mismatch: {name}")
    validate_evaluation_manifest(directory / "evaluation_ready.json", root)
    metadata = read_json(root / "preprocessing/metadata.json")
    bundle = read_bundle(directory / "target_test.npz", metadata["input_dimension"])
    membership = read_manifest(root / "manifests/target_test.csv")
    require(np.array_equal(bundle["patient_id"], membership.patient_nbr), "Evaluation row alignment mismatch")
    require(np.array_equal(bundle["encounter_id"], membership.encounter_id), "Evaluation encounter alignment mismatch")
    require(read_json(directory / "target_test.json") == {
        "n": len(membership), "input_dimension": metadata["input_dimension"],
        "manifest_sha256": file_hash(root / "manifests/target_test.csv"),
        "transformer_sha256": file_hash(root / "preprocessing/transformer.joblib"),
    }, "Evaluation bundle provenance mismatch")
    return {"X": bundle["X"], "y": bundle["y"], "input_dimension": metadata["input_dimension"],
            "provenance": complete}
