"""Run and verify the fixed pretraining/fine-tuning experiment."""

import argparse
import math
from pathlib import Path
import json
import os

import pandas as pd

from dp_shift.data.artifacts import load_training_data, validate_evaluation_manifest, verify_artifacts
from dp_shift.data.utils import file_hash, read_json, require
from .checkpoints import atomic_json, completed_job, execute_job, load_checkpoint
from .loaders import generator
from .model import build_model, configure_cpu, model_state, state_hash
from .privacy import predicted_epsilon, private_generators
from .train import pretrain_source, train_target
from .tune import load_frozen, public_smoke, tune


PATHS = {"A": ("sgd", None), "B-S0": ("sgd", "s0"), "B-S1": ("sgd", "s1"),
         "C": ("dp_sgd", None), "D-S0": ("dp_sgd", "s0"), "D-S1": ("dp_sgd", "s1")}


def base_metadata(data_root, output, frozen):
    return {"protocol_version": frozen["protocol_version"], "input_dimension": frozen["input_dimension"],
            "frozen_protocol_sha256": file_hash(Path(output) / "frozen_protocol.json"),
            "data_manifest_sha256": frozen["data"]["artifact_manifest_sha256"],
            "data_config_sha256": frozen["data"]["config_sha256"],
            "transformer_sha256": frozen["data"]["transformer_sha256"]}


def source_path(output, repeat, condition):
    return Path(output) / f"pretraining/repeat_{repeat:02d}/{condition}.pt"


def target_path(output, repeat, path_id):
    return Path(output) / f"training/repeat_{repeat:02d}/{path_id}.pt"


def source_metadata(data_root, output, frozen, repeat, condition, n, positive_n):
    config = frozen["settings"]
    initial = build_model(frozen["input_dimension"], config["seeds"]["initialization"] + repeat)
    return {**base_metadata(data_root, output, frozen), "stage": "source", "repeat_id": repeat,
            "path_id": f"pretrain_{condition}", "source_condition": condition, "source_n": int(n),
            "source_positive_n": int(positive_n), "source_positive_rate": positive_n / n,
            "source_membership_sha256": file_hash(Path(data_root) / f"source_pairs/repeat_{repeat:02d}/{condition}.csv"),
            "initial_state_sha256": state_hash(model_state(initial)),
            "scheduled_steps": config["source_epochs"] * math.ceil(n / config["source_batch_size"]),
            "pretraining_lr": config["source_lr"]}


def pretrain(data_root, output, repeats):
    frozen = load_frozen(data_root, output)
    chosen = frozen["settings"]
    for repeat in repeats:
        for condition in ["s0", "s1"]:
            data = load_training_data(data_root, condition, repeat)
            metadata = source_metadata(data_root, output, frozen, repeat, condition,
                                       len(data["y"]), int(data["y"].sum()))
            model = build_model(frozen["input_dimension"], chosen["seeds"]["initialization"] + repeat)
            execute_job(source_path(output, repeat, condition), metadata,
                lambda: pretrain_source(model, data, chosen, generator(chosen["seeds"]["source_shuffle"] + repeat)),
                Path(output) / f"internal/pretraining/repeat_{repeat:02d}/{condition}.pt")
    return {"source_jobs_complete": 2 * len(repeats)}


def target_metadata(data_root, output, frozen, repeat, path_id, initial_state, target_n, source_info=None):
    chosen = frozen["settings"]
    method, condition = PATHS[path_id]
    return {**base_metadata(data_root, output, frozen), "stage": "target", "repeat_id": repeat,
            "path_id": path_id, "training_method": method,
            "initialization": "pretrained" if condition else "random", "source_condition": condition or "none",
            "source_n": source_info["source_n"] if condition else 0,
            "source_positive_n": source_info["source_positive_n"] if condition else None,
            "source_positive_rate": source_info["source_positive_rate"] if condition else None,
            "source_checkpoint_sha256": file_hash(source_path(output, repeat, condition)) if condition else None,
            "initial_state_sha256": state_hash(initial_state), "target_train_n": target_n,
            "target_membership_sha256": file_hash(Path(data_root) / "manifests/target_train.csv"),
            "pretraining_lr": chosen["source_lr"] if condition else None,
            "target_lr": chosen["target_lr"], "target_epochs": chosen["target_epochs"],
            "sampling_rate": 1 / chosen["steps_per_epoch"],
            "scheduled_steps": chosen["target_epochs"] * chosen["steps_per_epoch"]}


def fit(data_root, output, repeats):
    frozen = load_frozen(data_root, output)
    chosen = frozen["settings"]
    target = load_training_data(data_root, "target_train")
    private_root = Path(output) / "private_rng"
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private_root, 0o700)
    try:
        for repeat in repeats:
            initial = model_state(build_model(frozen["input_dimension"], chosen["seeds"]["initialization"] + repeat))
            sources = {}
            for condition in ["s0", "s1"]:
                path = source_path(output, repeat, condition)
                expected = {**base_metadata(data_root, output, frozen), "repeat_id": repeat,
                            "stage": "source", "source_condition": condition}
                require(completed_job(path, expected), f"Missing source checkpoint: {path}")
                sources[condition] = load_checkpoint(path, expected)
            for path_id, (method, condition) in PATHS.items():
                state = model_state(sources[condition][0]) if condition else initial
                metadata = target_metadata(data_root, output, frozen, repeat, path_id, state,
                                           len(target["y"]), sources[condition][1] if condition else None)

                def train():
                    model = build_model(frozen["input_dimension"], chosen["seeds"]["initialization"] + repeat)
                    model.load_state_dict(state, strict=True)
                    if method == "dp_sgd":
                        sampling, noise = private_generators(private_root / f"repeat_{repeat:02d}/{path_id}.pt")
                    else:
                        sampling, noise = generator(chosen["seeds"]["target_sampling"] + repeat), None
                    return train_target(model, target, chosen, method, sampling, noise)

                execute_job(target_path(output, repeat, path_id), metadata, train,
                            Path(output) / f"internal/training/repeat_{repeat:02d}/{path_id}.pt")
    finally:
        write_training_ledger(data_root, output)
    return {"target_jobs_complete": len(PATHS) * len(repeats)}


def write_training_ledger(data_root, output):
    rows = []
    config = read_json(Path(data_root) / "config.json")
    null_fields = ["epsilon_achieved", "delta", "noise_multiplier", "clipping_norm", "accountant",
                   "average_precision", "auroc", "brier", "mean_prediction", "test_prevalence"]
    for repeat in range(config["repeats"]):
        for path_id, (method, condition) in PATHS.items():
            path = target_path(output, repeat, path_id).with_suffix(".json")
            status = read_json(path) if path.exists() else {"status": "pending", "metadata": {}}
            metadata = status["metadata"]
            row = {**dict.fromkeys(null_fields), "protocol_version": config["protocol_version"],
                   "repeat_id": repeat, "path_id": path_id, "status": status["status"],
                   "source_condition": condition or "none", "training_method": method,
                   "initialization": "pretrained" if condition else "random",
                   "target_test_n": config["data"]["target_test_size"]}
            row.update({key: metadata.get(key) for key in ["source_n", "source_positive_n", "source_positive_rate",
                       "target_train_n", "pretraining_lr", "target_lr", "target_epochs", "steps", "sampling_rate"]})
            row.update(metadata.get("privacy") or {})
            row["error"] = status.get("error")
            rows.append(row)
    path = Path(output) / "training_runs.csv"
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def verify_training(data_root, output):
    frozen = load_frozen(data_root, output)
    chosen = frozen["settings"]
    config = read_json(Path(data_root) / "config.json")
    audit = pd.read_csv(Path(data_root) / "audit/source_pair_summary.csv")
    target_n = len(pd.read_csv(Path(data_root) / "manifests/target_train.csv"))
    epsilon = predicted_epsilon(chosen["noise_multiplier"], 1 / chosen["steps_per_epoch"],
                                chosen["target_epochs"] * chosen["steps_per_epoch"], chosen["delta"])
    require(epsilon <= chosen["epsilon_cap"], "Frozen schedule exceeds the privacy cap")
    for repeat in range(config["repeats"]):
        sources = {}
        for condition in ["s0", "s1"]:
            row = audit.loc[(audit.repeat_id == repeat) & (audit.condition == condition)].iloc[0]
            expected = source_metadata(data_root, output, frozen, repeat, condition, int(row.n), int(row.positive_n))
            path = source_path(output, repeat, condition)
            require(completed_job(path, expected), f"Missing source job: {path}")
            sources[condition] = load_checkpoint(path, expected)
            require(sources[condition][1]["steps"] == expected["scheduled_steps"], "Source step count mismatch")
        initial = model_state(build_model(frozen["input_dimension"], chosen["seeds"]["initialization"] + repeat))
        for path_id, (method, condition) in PATHS.items():
            state = model_state(sources[condition][0]) if condition else initial
            expected = target_metadata(data_root, output, frozen, repeat, path_id, state, target_n,
                                       sources[condition][1] if condition else None)
            path = target_path(output, repeat, path_id)
            require(completed_job(path, expected), f"Missing target job: {path}")
            _, metadata = load_checkpoint(path, expected)
            require(metadata["steps"] == expected["scheduled_steps"], "Target step count mismatch")
            privacy = metadata["privacy"]
            if method == "dp_sgd":
                require(privacy is not None and privacy["accountant"] == "prv", "Missing PRV accountant")
                for key, value in {"noise_multiplier": chosen["noise_multiplier"], "delta": chosen["delta"],
                    "clipping_norm": chosen["clipping_norm"], "sampling_rate": 1 / chosen["steps_per_epoch"],
                    "steps": expected["scheduled_steps"]}.items():
                    require(privacy[key] == value, f"Privacy {key} mismatch: {path_id}")
                require(abs(privacy["epsilon_achieved"] - epsilon) <= 1e-6, "Accountant epsilon mismatch")
            else:
                require(privacy is None, "Non-private model has DP accounting metadata")
    write_training_ledger(data_root, output)
    return {"source_jobs_complete": 2 * config["repeats"], "target_jobs_complete": 6 * config["repeats"],
            "dp_jobs_complete": 3 * config["repeats"], "epsilon_achieved": epsilon}


def finalize(data_root, output):
    summary = verify_training(data_root, output)
    frozen = load_frozen(data_root, output)
    analysis = Path(__file__).resolve().parents[1] / "evaluation/run.py"
    require(analysis.is_file(), "Frozen evaluation code is missing")
    ready = {key: frozen["data"][key] for key in ["config_sha256", "artifact_manifest_sha256"]}
    ready["analysis"] = {"path": str(analysis), "sha256": file_hash(analysis)}
    ready["runs"] = []
    repeats = read_json(Path(data_root) / "config.json")["repeats"]
    for repeat in range(repeats):
        for path_id in PATHS:
            checkpoint = target_path(output, repeat, path_id).resolve()
            ready["runs"].append({"repeat_id": repeat, "path_id": path_id, "status": "complete",
                                  "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint)})
    path = Path(output) / "evaluation_ready.json"
    if path.exists():
        require(read_json(path) == ready, "Refusing to replace a different evaluation manifest")
    else:
        atomic_json(path, ready)
    validate_evaluation_manifest(path, data_root)
    return {**summary, "evaluation_ready": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["smoke", "tune", "pretrain", "fit", "verify", "finalize"]:
        command = commands.add_parser(name)
        command.add_argument("--data", default="results/prevalence_shift_v3/data")
        command.add_argument("--output", default="results/prevalence_shift_v3")
        if name in {"smoke", "tune"}:
            command.add_argument("--config", default="configs/experiment.json")
        if name in {"pretrain", "fit"}:
            command.add_argument("--repeats", type=int, nargs="+")
    args = parser.parse_args()
    try:
        configure_cpu()
        verify_artifacts(args.data)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        if args.command in {"pretrain", "fit"}:
            count = read_json(Path(args.data) / "config.json")["repeats"]
            repeats = args.repeats if args.repeats is not None else list(range(count))
            require(len(repeats) == len(set(repeats)) and all(0 <= r < count for r in repeats), "Invalid repeat IDs")
            result = (pretrain if args.command == "pretrain" else fit)(args.data, args.output, repeats)
        elif args.command == "smoke":
            result = public_smoke(args.data, args.output, args.config)
        elif args.command == "tune":
            frozen = tune(args.data, args.output, args.config)
            result = {key: frozen["settings"][key] for key in ["source_lr", "target_lr", "target_epochs", "noise_multiplier"]}
        else:
            result = (verify_training if args.command == "verify" else finalize)(args.data, args.output)
    except (ValueError, OSError, KeyError, RuntimeError, AssertionError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
