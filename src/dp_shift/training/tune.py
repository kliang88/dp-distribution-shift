"""The fixed public-only search. Final target data never enter this module."""

from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

from dp_shift.config import load_config
from dp_shift.data.artifacts import load_training_data
from dp_shift.data.utils import file_hash, read_json, require
from .checkpoints import atomic_json, execute_job, load_checkpoint
from .loaders import generator
from .model import build_model, model_state, state_hash
from .privacy import calibrate_noise
from .train import pretrain_source, train_target


def code_hashes():
    paths = [*Path(__file__).parent.glob("*.py"), Path(__file__).resolve().parents[1] / "config.py"]
    return {path.name: file_hash(path) for path in sorted(paths)}


def versions():
    return {"python": platform.python_version(), **{name: version(name) for name in
            ["torch", "opacus", "numpy", "pandas", "scikit-learn", "scipy", "joblib"]}}


def settings(config, source_lr, target_lr=None, target_epochs=None):
    result = {**config, "source_lr": source_lr, "target_lr": target_lr, "target_epochs": target_epochs}
    if target_epochs is not None:
        result["noise_multiplier"] = calibrate_noise(target_epochs, config["steps_per_epoch"],
            config["delta"], config["epsilon_cap"], config["calibration_epsilon"])
    return result


def predict(model, data):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(data["y"]), 1024):
            x = torch.as_tensor(data["X"][start:start + 1024], dtype=torch.float32)
            probabilities.append(torch.sigmoid(model(x).squeeze(-1)).numpy())
    result = np.concatenate(probabilities)
    require(np.isfinite(result).all(), "Nonfinite probabilities")
    return result


def score(model, data):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    probability = predict(model, data)
    return {"average_precision": float(average_precision_score(data["y"], probability)),
            "brier": float(brier_score_loss(data["y"], probability)),
            "auroc": float(roc_auc_score(data["y"], probability))}


def best_candidate(rows):
    valid = [row for row in rows if row["status"] == "complete"]
    require(bool(valid), "No valid public tuning candidates")
    best = max(row["average_precision"] for row in valid)
    tied = [row for row in valid if row["average_precision"] >= best - 1e-6]
    return min(tied, key=lambda row: (row["brier"], row.get("target_epochs", 0), row["learning_rate"]))


def best_target_candidate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["learning_rate"], row["target_epochs"]), []).append(row)
    candidates = []
    required = {(method, init) for method in ["sgd", "dp_sgd"] for init in ["scratch", "pretrained"]}
    for (lr, epochs), group in grouped.items():
        if len(group) != 4 or {(r["method"], r["initialization"]) for r in group} != required:
            continue
        if all(row["status"] == "complete" for row in group):
            candidates.append({"learning_rate": lr, "target_epochs": epochs, "status": "complete",
                               "average_precision": float(np.mean([r["average_precision"] for r in group])),
                               "brier": float(np.mean([r["brier"] for r in group]))})
    return best_candidate(candidates)


def public_source(data_root, config):
    pool = load_training_data(data_root, "source_pool")
    membership = pd.read_csv(Path(data_root) / "manifests/source_pool.csv")
    count = read_json(Path(data_root) / "config.json")["pretrain_size"]
    selected = membership.sample(n=count, random_state=config["seeds"]["tuning_sample"]).sort_values("patient_nbr")
    indices = selected.position.to_numpy(dtype=np.int64)
    return {**pool, "X": pool["X"][indices], "y": pool["y"][indices]}, pool


def public_context(data_root, config_path):
    return {"data": read_json(Path(data_root) / "COMPLETE.json"),
            "training_config_sha256": file_hash(config_path), "versions": versions(), "code_hashes": code_hashes()}


def load_frozen(data_root, output):
    frozen = read_json(Path(output) / "frozen_protocol.json")
    require(frozen["data"] == read_json(Path(data_root) / "COMPLETE.json"), "Frozen data identity mismatch")
    require(frozen["versions"] == versions(), "Training dependency versions changed")
    require(frozen["code_hashes"] == code_hashes(), "Training/evaluation source code changed after freezing")
    require(file_hash(Path(output) / "tuning/candidates.csv") == frozen["tuning_candidates_sha256"],
            "Public tuning ledger changed after freezing")
    return frozen


def run_public_job(path, context, candidate_id, model, train, internal_path):
    metadata = {**context, "input_dimension": model[0].in_features, "candidate_id": candidate_id,
                "initial_state_sha256": state_hash(model_state(model)), "stage": "public_tuning"}
    execute_job(path, metadata, lambda: train(model), internal_path)
    return load_checkpoint(path, metadata)[0]


def tune(data_root, output, config_path):
    output, config_path = Path(output), Path(config_path)
    config = load_config(config_path)
    require(config["experiment"] == read_json(Path(data_root) / "config.json")["experiment"],
            "Experiment configuration differs from prepared data; use a new output directory")
    if (output / "frozen_protocol.json").exists():
        frozen = load_frozen(data_root, output)
        require(frozen["training_config"] == config, "Frozen training configuration differs")
        return frozen
    directory = output / "tuning"
    require(not directory.exists(), "Partial tuning output exists; investigate it and use a new output directory")
    directory.mkdir(parents=True)
    context = public_context(data_root, config_path)
    atomic_json(directory / "context.json", context)
    source, pool = public_source(data_root, config)
    proxy = load_training_data(data_root, "proxy_train")
    validation = load_training_data(data_root, "proxy_validation")
    dimension = source["input_dimension"]
    seeds, rows = config["seeds"], []

    for index, lr in enumerate(config["source_learning_rates"]):
        candidate = f"source_{index}"
        path = directory / "public_checkpoints" / f"{candidate}.pt"
        row = {"phase": "source", "learning_rate": lr, "candidate_id": candidate}
        try:
            model = run_public_job(path, context, candidate, build_model(dimension, seeds["tuning_initialization"]),
                lambda model: pretrain_source(model, source, settings(config, lr),
                                               generator(seeds["tuning_source_shuffle"])),
                output / "internal/tuning" / f"{candidate}.pt")
            row.update(status="complete", **score(model, validation))
        except (ValueError, RuntimeError) as error:
            row.update(status="failed", error=str(error))
        rows.append(row)
        pd.DataFrame(rows).to_csv(directory / "candidates.csv", index=False)
    source_choice = best_candidate(rows)
    source_lr = source_choice["learning_rate"]
    pretrained, _ = load_checkpoint(directory / "public_checkpoints" / f"{source_choice['candidate_id']}.pt")
    pretrained_state = model_state(pretrained)

    for lr_index, lr in enumerate(config["target_learning_rates"]):
        for epoch_index, epochs in enumerate(config["target_epoch_candidates"]):
            chosen = settings(config, source_lr, lr, epochs)
            for method_index, method in enumerate(["sgd", "dp_sgd"]):
                for init_index, init in enumerate(["scratch", "pretrained"]):
                    candidate = f"target_{lr_index}_{epoch_index}_{method}_{init}"
                    row = {"phase": "target", "learning_rate": lr, "target_epochs": epochs,
                           "method": method, "initialization": init, "candidate_id": candidate}
                    try:
                        model = build_model(dimension, seeds["tuning_initialization"])
                        if init == "pretrained":
                            model.load_state_dict(pretrained_state, strict=True)
                        sampling_seed = seeds["tuning_sampling"] + method_index
                        noise_seed = seeds["tuning_noise"] + 100 * lr_index + 10 * epoch_index + init_index
                        model = run_public_job(directory / "public_checkpoints" / f"{candidate}.pt",
                            context, candidate, model,
                            lambda model: train_target(model, proxy, chosen, method,
                                generator(sampling_seed), generator(noise_seed)),
                            output / "internal/tuning" / f"{candidate}.pt")
                        row.update(status="complete", **score(model, validation))
                    except (ValueError, RuntimeError) as error:
                        row.update(status="failed", error=str(error))
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(directory / "candidates.csv", index=False)
    target_choice = best_target_candidate([row for row in rows if row["phase"] == "target"])
    logistic = LogisticRegression(C=1., solver="lbfgs", max_iter=2000)
    logistic.fit(pool["X"], pool["y"])
    require(logistic.n_iter_[0] < 2000, "Public logistic diagnostic failed to converge")
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    probability = logistic.predict_proba(validation["X"])[:, 1]
    atomic_json(directory / "logistic_diagnostic.json", {
        "average_precision": float(average_precision_score(validation["y"], probability)),
        "auroc": float(roc_auc_score(validation["y"], probability)),
        "brier": float(brier_score_loss(validation["y"], probability)),
        "iterations": int(logistic.n_iter_[0]), "training_n": len(pool["y"])})
    selected = settings(config, source_lr, target_choice["learning_rate"], target_choice["target_epochs"])
    frozen = {**context, "protocol_version": config["protocol_version"], "training_config": config,
              "settings": selected, "input_dimension": dimension, "candidate_results": rows,
              "tuning_candidates_sha256": file_hash(directory / "candidates.csv"),
              "cpu_threads": 1, "deterministic_algorithms": True}
    atomic_json(output / "frozen_protocol.json", frozen)
    return frozen


def public_smoke(data_root, output, config_path):
    config = load_config(config_path)
    source, _ = public_source(data_root, config)
    proxy = load_training_data(data_root, "proxy_train")
    context = public_context(data_root, config_path)
    chosen = settings(config, .01, .01, 8)
    seeds = config["seeds"]
    directory = Path(output) / "public_smoke"
    model = run_public_job(directory / "source.pt", context, "smoke_source",
        build_model(source["input_dimension"], seeds["tuning_initialization"]),
        lambda model: pretrain_source(model, source, chosen, generator(seeds["tuning_source_shuffle"])),
        Path(output) / "internal/public_smoke/source.pt")
    state = model_state(model)
    for index, method in enumerate(["sgd", "dp_sgd"]):
        model = build_model(source["input_dimension"], seeds["tuning_initialization"])
        model.load_state_dict(state, strict=True)
        run_public_job(directory / f"{method}.pt", context, f"smoke_{method}", model,
            lambda model: train_target(model, proxy, chosen, method,
                generator(seeds["tuning_sampling"] + index), generator(seeds["tuning_noise"])),
            Path(output) / f"internal/public_smoke/{method}.pt")
    return {"status": "complete", "source_steps": config["source_epochs"] *
            int(np.ceil(len(source["y"]) / config["source_batch_size"])),
            "proxy_steps_per_model": 8 * config["steps_per_epoch"]}
