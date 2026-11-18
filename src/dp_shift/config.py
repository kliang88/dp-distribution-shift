"""One experiment schema, fixed protocol defaults, and deterministic seed derivation."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math

from .data.utils import read_json, require
from .data.shifts import SHIFTS


DEFAULT_CONFIG = "configs/experiment.json"
PROTOCOL_VERSION = "source_shift_v1"
PARTITIONS = ("source_pool", "proxy_train", "proxy_validation", "target_train", "target_test")


@dataclass(frozen=True)
class Protocol:
    """Versioned scientific assumptions shared by every experiment."""

    numeric_features: tuple[str, ...] = (
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    )
    categorical_features: tuple[str, ...] = (
        "age", "gender", "race", "admission_type_id", "admission_source_id",
        "discharge_disposition_id", "diabetesMed", "change",
    )
    categorical_ids: tuple[str, ...] = ("admission_type_id", "admission_source_id", "discharge_disposition_id")
    log_features: tuple[str, ...] = ("number_outpatient", "number_emergency", "number_inpatient")
    excluded_discharges: tuple[int, ...] = (11, 13, 14, 19, 20, 21)
    source_batch_size: int = 256
    steps_per_epoch: int = 12
    momentum: float = 0.9
    delta: float = 1e-5
    primary_metric: str = "average_precision"
    prediction_batch_size: int = 1024
    confidence_level: float = 0.95
    evaluation_version: str = "evaluation_v4"


PROTOCOL = Protocol()

# These describe fixed code behavior and are retained in the public protocol.
FIXED_PROTOCOL = {
    "outcome": "readmitted == <30", "index_encounter": "smallest encounter_id before exclusions",
    "partition_sampling": "one seeded shuffle, without replacement, no stratification",
    "preprocessing_fit": "source_pool", "numeric_imputation": "median",
    "numeric_scaling": "standard",
    "categorical_encoding": "one-hot; fixed diagnosis and insulin categories; ignore unknown in source-fitted features",
    "categorical_imputation": "Missing", "hidden_sizes": [64, 32], "activation": "ReLU",
    "loss": "binary_cross_entropy_with_logits", "optimizer": "SGD",
    "weight_decay": 0, "nesterov": False, "target_optimizer": "fresh",
    "source_sampling": "shuffle each epoch; retain partial batch",
    "target_sampling": "Poisson", "accountant": "prv", "clipping": "flat",
    "privacy_unit": "patient", "privacy_adjacency": "add/remove",
    "privacy_scope": "per model", "secure_mode": False,
    "private_rng": "independent entropy-backed sampling/noise; not derived from master seed",
    "tuning_selection": "proxy-validation AUPRC; target mean across SGD/DP x scratch/pretrained",
    "tuning_ties": "AUPRC within 1e-6, then Brier, epochs, learning rate",
    "intervals": "Student t across repeats, paired for effects, conditional on fixed test set",
    "fixed_partitions_across_repeats": True, "cpu_threads": 1,
    "source_replacements": "Fixed shares use bidirectional swaps and seeded permutation prefixes. Diagnosis shifts "
                           "use readmission-matched swaps from outside S0, rounded to the nearest patient (ties to even).",
    "deterministic_algorithms": True,
}

SCHEMA = {
    "data": {f"{name}_size" for name in PARTITIONS},
    "source_conditions": {"pretrain_size", "shift_type", "shift_fraction"},
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


def normalize_experiment(experiment):
    """Expand a singleton study or normalize a persisted scalar configuration."""
    experiment = deepcopy(experiment)
    if isinstance(experiment, dict) and any(key in experiment for key in ("schema_version", "comparisons", "pretraining")):
        plan = study_plan(experiment)
        shifts, epsilons = plan["sweep"]["shifts"], plan["sweep"]["epsilons"]
        require(sum(map(len, shifts.values())) == len(epsilons) == 1,
                "This stage requires one comparison; run dp-shift experiment for the complete grid")
        kind, fractions = next(iter(shifts.items()))
        return comparison_config(plan["base"], kind, fractions[0], epsilons[0])
    source = experiment.get("source_conditions") if isinstance(experiment, dict) else None
    if isinstance(source, dict) and set(source) == {"pretrain_size", "shifted_positive_rate"}:
        source["shift_type"] = "readmission"
        source["shift_fraction"] = source.pop("shifted_positive_rate")
    return experiment


@dataclass(frozen=True)
class DataSizes:
    source_pool_size: int
    proxy_train_size: int
    proxy_validation_size: int
    target_train_size: int
    target_test_size: int


@dataclass(frozen=True)
class SourceConditions:
    pretrain_size: int
    shift_type: str
    shift_fraction: float


@dataclass(frozen=True)
class Privacy:
    epsilon: float
    clipping_norm: float


@dataclass(frozen=True)
class Tuning:
    source_epochs: int
    source_learning_rates: tuple[float, ...]
    target_learning_rates: tuple[float, ...]
    target_epoch_candidates: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated user choices, with immutable nested values."""

    seed: int
    repeats: int
    data: DataSizes
    source_conditions: SourceConditions
    privacy: Privacy
    tuning: Tuning

    def to_dict(self):
        result = asdict(self)
        for name in ("source_learning_rates", "target_learning_rates", "target_epoch_candidates"):
            result["tuning"][name] = list(result["tuning"][name])
        return result


@dataclass(frozen=True)
class Seeds:
    partition: int
    s0: int
    removed: int
    added: int
    tuning_sample: int
    tuning_initialization: int
    tuning_source_shuffle: int
    tuning_sampling: int
    tuning_noise: int
    initialization: int
    source_shuffle: int
    target_sampling: int


@dataclass(frozen=True)
class ResolvedConfig:
    requested: ExperimentConfig
    seeds: Seeds
    shift_member_count: int | None
    calibration_epsilon: float
    source_replacement_count: int | None = None

    def to_dict(self):
        """Keep the existing public metadata schema at serialization boundaries."""
        choices = self.requested.to_dict()
        source, privacy = choices["source_conditions"], choices["privacy"]
        protocol = {name: list(value) if isinstance(value, tuple) else value
                    for name, value in asdict(PROTOCOL).items()}
        return {
            **protocol, **choices["tuning"],
            "experiment": choices, "fixed_protocol": deepcopy(FIXED_PROTOCOL),
            "protocol_version": PROTOCOL_VERSION,
            "shift_definitions": {name: {"label": shift.label, "columns": list(shift.columns),
                                         "replacement_rate": shift.replacement_rate}
                                  for name, shift in SHIFTS.items()},
            "seed": choices["seed"], "repeats": choices["repeats"], "data": choices["data"],
            "seeds": asdict(self.seeds), **source,
            "shift_member_count": self.shift_member_count,
            "source_replacement_count": self.source_replacement_count,
            "epsilon_cap": privacy["epsilon"], "clipping_norm": privacy["clipping_norm"],
            "calibration_epsilon": self.calibration_epsilon,
        }


@dataclass(frozen=True)
class SelectedHyperparameters:
    source_lr: float
    target_lr: float | None = None
    target_epochs: int | None = None

    def __post_init__(self):
        positive_number(self.source_lr, "source_lr")
        require((self.target_lr is None) == (self.target_epochs is None),
                "target_lr and target_epochs must be supplied together")
        if self.target_lr is not None:
            positive_number(self.target_lr, "target_lr")
            positive_integer(self.target_epochs, "target_epochs")


@dataclass(frozen=True)
class TrainingSettings:
    config: ResolvedConfig
    selected: SelectedHyperparameters
    noise_multiplier: float | None = None


def resolve_config(experiment) -> ResolvedConfig:
    """Validate requested choices, then derive seeds and sampling/privacy values."""
    experiment = normalize_experiment(experiment)
    exact_keys(experiment, {"seed", "repeats", *SCHEMA}, "experiment")
    for section, keys in SCHEMA.items():
        exact_keys(experiment[section], keys, section)
    seed, repeats = experiment["seed"], experiment["repeats"]
    require(type(seed) is int and 0 <= seed < 2**32, "seed must be an integer in [0, 2**32)")
    positive_integer(repeats, "repeats")
    data = experiment["data"]
    source = experiment["source_conditions"]
    privacy = experiment["privacy"]
    tuning = experiment["tuning"]
    for key, value in data.items():
        positive_integer(value, f"data.{key}")
    for name in ("proxy_train", "target_train"):
        require(data[f"{name}_size"] % PROTOCOL.steps_per_epoch == 0,
                f"data.{name}_size must be divisible by {PROTOCOL.steps_per_epoch}")
    positive_integer(source["pretrain_size"], "source_conditions.pretrain_size")
    require(isinstance(source["shift_type"], str) and source["shift_type"] in SHIFTS, "Unknown shift_type")
    positive_number(source["shift_fraction"], "source_conditions.shift_fraction")
    rate, size = source["shift_fraction"], source["pretrain_size"]
    require(rate < 1, "shift_fraction must be below 1")
    count = size * rate
    # Round swaps to the nearest integer, using ties-to-even for half patients.
    rounded = round(count)
    replacement_rate = SHIFTS[source["shift_type"]].replacement_rate
    if replacement_rate:
        require(0 < rounded < size, "Replacement rate must round to between 1 and pretrain_size - 1 swaps")
    else:
        require(math.isclose(count, rounded, rel_tol=0, abs_tol=1e-9),
                "pretrain_size * shift_fraction must be an integer")
        require(0 < rounded < size, "Shifted source must contain both classes")
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
    requested = ExperimentConfig(seed, repeats, DataSizes(**data), SourceConditions(**source), Privacy(**privacy),
                                 Tuning(tuning["source_epochs"], tuple(tuning["source_learning_rates"]),
                                        tuple(tuning["target_learning_rates"]), tuple(tuning["target_epoch_candidates"])))
    seeds = Seeds(**{role: int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:4], "big") % 2**31
                     for role in SEED_ROLES})
    return ResolvedConfig(requested, seeds, shift_member_count=None if replacement_rate else rounded,
                          calibration_epsilon=privacy["epsilon"] * (7.99 / 8),
                          source_replacement_count=rounded if replacement_rate else None)


def load_config(path) -> ResolvedConfig:
    return resolve_config(read_json(path))


def validate_comparisons(shifts, epsilons, reference_epsilon):
    require(isinstance(shifts, dict) and shifts, "comparisons.shifts must be a nonempty object")
    for kind in shifts:
        require(kind in SHIFTS, f"Unknown shift type: {kind}")
    for name, values in [("comparisons.epsilons", epsilons), *shifts.items()]:
        require(isinstance(values, list) and values, f"{name} must be a nonempty list")
        for value in values:
            positive_number(value, name)
        require(len(set(values)) == len(values), f"{name} contains duplicates")
    positive_number(reference_epsilon, "tuning.reference_epsilon")


def comparison_config(base, kind, fraction, epsilon):
    """Build the scalar settings consumed by preparation, training, and evaluation."""
    result = normalize_experiment(base)
    result["source_conditions"].update(shift_type=kind, shift_fraction=fraction)
    result["privacy"]["epsilon"] = epsilon
    resolve_config(result)
    return result


def study_plan(choices):
    """Validate one self-contained study and expand its comparison grid."""
    exact_keys(choices, {"schema_version", "seed", "repeats", "data", "pretraining", "privacy", "tuning", "comparisons"}, "study")
    require(type(choices["schema_version"]) is int and choices["schema_version"] == 1, "Unsupported schema_version; expected 1")
    exact_keys(choices["pretraining"], {"sample_size", "epochs"}, "pretraining")
    exact_keys(choices["privacy"], {"clipping_norm"}, "privacy")
    exact_keys(choices["tuning"], {"reference_epsilon", "source_learning_rates", "target_learning_rates", "target_epoch_candidates"}, "tuning")
    exact_keys(choices["comparisons"], {"shifts", "epsilons"}, "comparisons")
    pretraining, tuning, comparisons = choices["pretraining"], choices["tuning"], choices["comparisons"]
    positive_integer(pretraining["sample_size"], "pretraining.sample_size")
    positive_integer(pretraining["epochs"], "pretraining.epochs")
    validate_comparisons(comparisons["shifts"], comparisons["epsilons"], tuning["reference_epsilon"])
    kind, fractions = next(iter(comparisons["shifts"].items()))
    base = {
        "seed": choices["seed"], "repeats": choices["repeats"], "data": deepcopy(choices["data"]),
        "source_conditions": {"pretrain_size": pretraining["sample_size"], "shift_type": kind, "shift_fraction": fractions[0]},
        "privacy": {"epsilon": tuning["reference_epsilon"], **choices["privacy"]},
        "tuning": {"source_epochs": pretraining["epochs"], **{name: deepcopy(tuning[name]) for name in
                   ("source_learning_rates", "target_learning_rates", "target_epoch_candidates")}},
    }
    base = resolve_config(base).requested.to_dict()
    for kind, fractions in comparisons["shifts"].items():
        for fraction in fractions:
            for epsilon in comparisons["epsilons"]:
                comparison_config(base, kind, fraction, epsilon)
    return {"base": base, "sweep": {**deepcopy(comparisons), "reference_epsilon": tuning["reference_epsilon"]}}
