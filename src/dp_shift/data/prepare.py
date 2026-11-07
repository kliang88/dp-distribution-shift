"""Build, verify, or finish evaluation-stage data artifacts."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
import pandas as pd

from dp_shift.config import DEFAULT_CONFIG, load_config

from .artifacts import (TRAIN_PARTITIONS, input_paths, load_evaluation_data, load_saved_cohort,
                        save_bundle, validate_evaluation_manifest, verify_artifacts, write_artifact_index)
from .audit import write_audits
from .cohort import build_cohort, describe
from .preprocessing import fit_source_transformer, transformer_metadata
from .sampling import create_source_pair
from .schema import discharge_descriptions, load_and_validate_raw
from .splits import create_split_manifests, manifest, read_manifest, rows_for_manifest
from .utils import file_hash, read_json, require, write_json


DEFAULT_RECORDS = "data/diabetic_data.csv"
DEFAULT_MAPPING = "data/IDS_mapping.csv"


def implementation_hash():
    digest = hashlib.sha256()
    paths = [*Path(__file__).parent.glob("*.py"), Path(__file__).resolve().parents[1] / "config.py"]
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(config_path, config):
    project = Path.cwd()
    return {
        "project_root": str(project),
        "config_source_sha256": file_hash(config_path),
        "protocol_sha256": file_hash(project / config["protocol_path"]),
        "implementation_sha256": implementation_hash(),
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]}},
        "platform": platform.platform(),
        "thread_settings": {name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def input_fingerprints(records=None, mapping=None):
    records = DEFAULT_RECORDS if records is None else records
    mapping = DEFAULT_MAPPING if mapping is None else mapping
    return {name: {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
            for name, path in {"records": records, "mapping": mapping}.items()}


def build(config_path, output, records=None, mapping=None):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = load_config(config_path)
    config["inputs"] = input_fingerprints(records, mapping)
    info = provenance(config_path, config)
    if output.exists():
        require(read_json(output / "config.json") == config, "Existing output uses a different configuration")
        require(read_json(output / "provenance.json")["implementation_sha256"] == info["implementation_sha256"],
                "Existing output uses different preparation code; use a new output directory")
        return verify_artifacts(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        paths = input_paths(info, config)
        for name, path in paths.items():
            require(file_hash(path) == config["inputs"][name]["sha256"], f"Input fingerprint mismatch: {name}")
        raw = load_and_validate_raw(paths["records"], config)
        cohort, exclusions, cohort_audit = build_cohort(raw, config, discharge_descriptions(paths["mapping"]))
        frames = create_split_manifests(cohort, config)
        write_json(staging / "config.json", config)
        write_json(staging / "provenance.json", info)
        write_json(staging / "cohort_audit.json", cohort_audit)
        exclusions.to_csv(staging / "exclusions.csv", index=False)
        (staging / "manifests").mkdir()
        for name, frame in frames.items():
            manifest(frame).to_csv(staging / f"manifests/{name}.csv", index=False)

        pool = frames["source_pool"]
        tuning_source = pool.sample(n=config["pretrain_size"], random_state=config["seeds"]["tuning_sample"])
        require(set(tuning_source.y) == {0, 1}, "Public tuning source sample must contain both classes")
        lookup = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool.patient_nbr)
        pairs = []
        for repeat in range(config["repeats"]):
            pair = create_source_pair(pool, repeat, config)
            pairs.append(pair)
            directory = staging / f"source_pairs/repeat_{repeat:02d}"
            directory.mkdir(parents=True)
            for name, frame in pair.items():
                manifest(frame).to_csv(directory / f"{name}.csv", index=False)
                if name in {"s0", "s1"}:
                    np.save(directory / f"{name}_indices.npy", lookup.loc[frame.patient_nbr].to_numpy(dtype=np.int64))

        transformer = fit_source_transformer(pool, config)
        transformer_path = staging / "preprocessing/transformer.joblib"
        transformer_path.parent.mkdir()
        joblib.dump(transformer, transformer_path)
        write_json(staging / "preprocessing/metadata.json", transformer_metadata(transformer, pool, config))
        for name in TRAIN_PARTITIONS:
            save_bundle(staging / f"arrays/{name}.npz", frames[name], transformer,
                        staging / f"manifests/{name}.csv", transformer_path)
        write_json(staging / "partition_summary.json", {name: describe(frame) for name, frame in frames.items()})
        write_audits(staging, frames, pairs, config)
        write_artifact_index(staging)
        summary = verify_artifacts(staging)
        require(not output.exists(), "Output appeared during preparation; refusing to overwrite")
        staging.rename(output)
    return summary


def materialize_test(output, evaluation_manifest):
    output, evaluation_manifest = Path(output).resolve(), Path(evaluation_manifest).resolve()
    verify_artifacts(output)
    validate_evaluation_manifest(evaluation_manifest, output)
    ready = read_json(evaluation_manifest)
    # Preserve references when copying the readiness manifest into a different directory.
    ready["analysis"]["path"] = str((evaluation_manifest.parent / ready["analysis"]["path"]).resolve())
    for run in ready["runs"]:
        run["checkpoint_path"] = str((evaluation_manifest.parent / run["checkpoint_path"]).resolve())
    final = output / "evaluation"
    if final.exists():
        require(read_json(final / "evaluation_ready.json") == ready, "Evaluation manifest changed")
        return {"target_test_n": len(load_evaluation_data(output)["y"])}
    cohort, _, _ = load_saved_cohort(output)
    frame = rows_for_manifest(cohort, read_manifest(output / "manifests/target_test.csv"))
    transformer_path = output / "preprocessing/transformer.joblib"
    transformer = joblib.load(transformer_path)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output) as temporary:
        staging = Path(temporary)
        write_json(staging / "evaluation_ready.json", ready)
        save_bundle(staging / "target_test.npz", frame, transformer,
                    output / "manifests/target_test.csv", transformer_path)
        write_json(staging / "COMPLETE.json", {
            "status": "complete",
            "data_manifest_sha256": file_hash(output / "artifact_manifest.json"),
            "files": {p.name: file_hash(p) for p in sorted(staging.iterdir())},
        })
        require(not final.exists(), "Evaluation output appeared; refusing to overwrite")
        staging.rename(final)
    return {"target_test_n": len(load_evaluation_data(output)["y"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Prepare training data without test arrays")
    builder.add_argument("--config", default=DEFAULT_CONFIG)
    builder.add_argument("--output", default="results/prevalence_shift_v3/data")
    verifier = commands.add_parser("verify", help="Verify existing artifacts without resampling or fitting")
    verifier.add_argument("--output", default="results/prevalence_shift_v3/data")
    evaluator = commands.add_parser("materialize-test", help="Transform test data after all training is complete")
    evaluator.add_argument("--output", default="results/prevalence_shift_v3/data")
    evaluator.add_argument("--evaluation-manifest", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(args.config, args.output)
        elif args.command == "verify":
            result = verify_artifacts(args.output)
        else:
            result = materialize_test(args.output, args.evaluation_manifest)
    except (ValueError, OSError, KeyError, AssertionError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
