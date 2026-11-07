"""One experiment schema, fixed protocol defaults, and deterministic seed derivation."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .data.utils import read_json, require


DEFAULT_CONFIG = "configs/experiment.json"
PROTOCOL_VERSION = "prevalence_shift_v3"
PARTITIONS = ("source_pool", "proxy_train", "proxy_validation", "target_train", "target_test")
DEFAULTS = {
    "numeric_features": [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    ],
    "categorical_features": [
        "age", "gender", "race", "admission_type_id", "admission_source_id",
        "discharge_disposition_id", "diabetesMed", "change",
    ],
    "categorical_ids": ["admission_type_id", "admission_source_id", "discharge_disposition_id"],
    "log_features": ["number_outpatient", "number_emergency", "number_inpatient"],
    "excluded_discharges": [11, 13, 14, 19, 20, 21],
    "source_batch_size": 256, "steps_per_epoch": 12, "momentum": 0.9,
    "delta": 1e-5, "primary_metric": "average_precision",
    "prediction_batch_size": 1024, "confidence_level": 0.95,
    "practical_ap_threshold": 0.005, "calibration_bins": 10,
    "evaluation_version": "evaluation_v3",
    "protocol_path": str(Path(__file__).resolve().parents[2] / "README.md"),
}

# These describe fixed code behavior and are retained in the public protocol.
FIXED_PROTOCOL = {
    "outcome": "readmitted == <30", "index_encounter": "smallest encounter_id before exclusions",
    "partition_sampling": "one seeded shuffle, without replacement, no stratification",
    "preprocessing_fit": "source_pool", "numeric_imputation": "median",
    "numeric_scaling": "standard", "categorical_encoding": "one-hot; ignore unknown",
    "categorical_imputation": "Missing", "hidden_sizes": [64, 32], "activation": "ReLU",
    "loss": "binary_cross_entropy_with_logits", "optimizer": "SGD",
    "weight_decay": 0, "nesterov": False, "target_optimizer": "fresh",
    "source_sampling": "shuffle each epoch; retain partial batch",
    "target_sampling": "Poisson", "accountant": "prv", "clipping": "flat",
    "privacy_unit": "patient", "privacy_adjacency": "add/remove",
    "privacy_scope": "per model", "secure_mode": False,
    "private_rng": "independent entropy-backed sampling/noise; not derived from master seed",
    "tuning_selection": "proxy-validation AP; target mean across SGD/DP x scratch/pretrained",
    "tuning_ties": "AP within 1e-6, then Brier, epochs, learning rate",
    "intervals": "Student t across repeats, paired for effects, conditional on fixed test set",
    "calibration_binning": "10 quantile bins per model/repeat; tied predictions stay together",
    "pr_summary": "mean step precision at a common recall grid; AP uses exact per-run points",
    "fixed_partitions_across_repeats": True, "cpu_threads": 1,
    "deterministic_algorithms": True,
}

SCHEMA = {
    "data": {f"{name}_size" for name in PARTITIONS},
    "source_conditions": {"pretrain_size", "shifted_positive_rate"},
    "privacy": {"epsilon", "clipping_norm"},
    "tuning": {"source_epochs", "source_learning_rates", "target_learning_rates", "target_epoch_candidates"},
}
SEED_ROLES = ("partition", "s0", "removed", "added", "tuning_sample", "tuning_initialization",
              "tuning_source_shuffle", "tuning_sampling", "tuning_noise", "initialization",
              "source_shuffle", "target_sampling")


def exact_keys(value, expected, name):
    require(isinstance(value, dict), f"{name} must be an object")
    missing, unknown = set(expected) - value.keys(), value.keys() - set(expected)
    require(not missing and not unknown,
            f"{name}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")


def positive_integer(value, name):
    require(type(value) is int and value > 0, f"{name} must be a positive integer")


def positive_number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{name} must be a positive finite number")


def resolve_config(experiment):
    """Validate user choices and expand defaults once for every pipeline stage."""
    exact_keys(experiment, {"seed", "repeats", *SCHEMA}, "experiment")
    for section, keys in SCHEMA.items():
        exact_keys(experiment[section], keys, section)
    seed, repeats = experiment["seed"], experiment["repeats"]
    require(type(seed) is int and 0 <= seed < 2**32, "seed must be an integer in [0, 2**32)")
    positive_integer(repeats, "repeats")
    require(repeats >= 2, "repeats must be at least 2 for repeat intervals")
    data = experiment["data"]
    source = experiment["source_conditions"]
    privacy = experiment["privacy"]
    tuning = experiment["tuning"]
    for key, value in data.items():
        positive_integer(value, f"data.{key}")
    for name in ("proxy_train", "target_train"):
        require(data[f"{name}_size"] % DEFAULTS["steps_per_epoch"] == 0,
                f"data.{name}_size must be divisible by {DEFAULTS['steps_per_epoch']}")
    positive_integer(source["pretrain_size"], "source_conditions.pretrain_size")
    positive_number(source["shifted_positive_rate"], "source_conditions.shifted_positive_rate")
    rate, size = source["shifted_positive_rate"], source["pretrain_size"]
    require(rate < 1, "shifted_positive_rate must be below 1")
    count = size * rate
    require(math.isclose(count, round(count), rel_tol=0, abs_tol=1e-9),
            "pretrain_size * shifted_positive_rate must be an integer")
    require(0 < round(count) < size, "Shifted source must contain both classes")
    require(size < data["source_pool_size"], "source_pool_size must exceed pretrain_size for replacements")
    for key, value in privacy.items():
        positive_number(value, f"privacy.{key}")
    positive_integer(tuning["source_epochs"], "tuning.source_epochs")
    for key in ("source_learning_rates", "target_learning_rates", "target_epoch_candidates"):
        candidates = tuning[key]
        require(isinstance(candidates, list) and candidates, f"tuning.{key} must be a nonempty list")
        for value in candidates:
            (positive_integer if key == "target_epoch_candidates" else positive_number)(value, f"tuning.{key}")
        require(len(candidates) == len(set(candidates)), f"tuning.{key} contains duplicates")
    seeds = {role: int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:4], "big") % 2**31
             for role in SEED_ROLES}
    return {
        **deepcopy(DEFAULTS), **deepcopy(tuning), "experiment": deepcopy(experiment),
        "fixed_protocol": deepcopy(FIXED_PROTOCOL), "protocol_version": PROTOCOL_VERSION,
        "seed": seed, "seeds": seeds, "repeats": repeats, "data": deepcopy(data),
        "pretrain_size": size, "source_positive_n": round(count),
        "shifted_positive_rate": rate, "epsilon_cap": privacy["epsilon"],
        "calibration_epsilon": privacy["epsilon"] * (7.99 / 8),
        "clipping_norm": privacy["clipping_norm"],
    }


def load_config(path):
    return resolve_config(read_json(path))
