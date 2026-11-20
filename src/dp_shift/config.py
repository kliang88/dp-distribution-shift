"""Configuration for the signed source-shift study."""

from dataclasses import dataclass
import hashlib
import math

from .data.utils import read_json, require


DEFAULT_CONFIG = "configs/experiment.json"
PARTITIONS = ("source_pool", "proxy_train", "proxy_validation", "target_train", "target_test")


@dataclass(frozen=True)
class Protocol:
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
    prediction_batch_size: int = 1024
    confidence_level: float = 0.95


PROTOCOL = Protocol()
SIGNED_LEVELS = (-8, -6, -4, -2, 2, 4, 6, 8)
PRIVACY_BUDGETS = (1., 2.5, 5., 7.5, 10.)


@dataclass(frozen=True)
class Config:
    seed: int
    repeats: int
    data: dict[str, int]
    pretrain_size: int
    source_epochs: int
    clipping_norm: float
    reference_epsilon: float
    source_learning_rates: tuple[float, ...]
    target_learning_rates: tuple[float, ...]
    target_epoch_candidates: tuple[int, ...]
    shift_pp: tuple[int, ...]
    epsilons: tuple[float, ...]

    def seed_for(self, role):
        return int.from_bytes(hashlib.sha256(f"{self.seed}:{role}".encode()).digest()[:4], "big") % 2**31


@dataclass(frozen=True)
class SelectedHyperparameters:
    source_lr: float
    target_lr: float | None = None
    target_epochs: int | None = None


@dataclass(frozen=True)
class TrainingSettings:
    config: Config
    selected: SelectedHyperparameters
    epsilon: float | None = None
    noise_multiplier: float | None = None


def parse_config(choices):
    def section(value, keys, name):
        require(isinstance(value, dict) and value.keys() == set(keys), f"Invalid keys in {name}")
        return value

    def positive_int(value):
        return type(value) is int and value > 0

    def positive_number(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    def candidates(value, predicate, name):
        require(isinstance(value, list) and value and all(map(predicate, value))
                and len(set(value)) == len(value), f"Invalid {name}")
        return tuple(sorted(value))

    section(choices, ("seed", "repeats", "data", "pretraining", "privacy", "tuning", "shift_pp", "epsilons"), "config")
    seed, repeats = choices["seed"], choices["repeats"]
    require(type(seed) is int and 0 <= seed < 2**32, "Invalid seed")
    require(positive_int(repeats), "Invalid repeat count")
    data = section(choices["data"], [f"{name}_size" for name in PARTITIONS], "data")
    require(all(map(positive_int, data.values())), "Partition sizes must be positive integers")
    for name in ("proxy_train_size", "target_train_size"):
        require(data[name] % PROTOCOL.steps_per_epoch == 0, f"{name} must be divisible by {PROTOCOL.steps_per_epoch}")
    pretraining = section(choices["pretraining"], ("sample_size", "epochs"), "pretraining")
    size, epochs = pretraining["sample_size"], pretraining["epochs"]
    require(positive_int(size) and size < data["source_pool_size"] and positive_int(epochs), "Invalid pretraining settings")
    privacy = section(choices["privacy"], ("clipping_norm",), "privacy")
    tuning = section(choices["tuning"], ("reference_epsilon", "source_learning_rates", "target_learning_rates",
                                         "target_epoch_candidates"), "tuning")
    require(positive_number(privacy["clipping_norm"]), "Invalid clipping norm")
    require(positive_number(tuning["reference_epsilon"]), "Invalid reference epsilon")
    levels = candidates(choices["shift_pp"], lambda value: type(value) is int and value in SIGNED_LEVELS, "shift_pp")
    require(all(size * level % 100 == 0 for level in levels), "Signed shifts must correspond to exact patient counts")
    return Config(seed, repeats, dict(data), size, epochs, privacy["clipping_norm"], tuning["reference_epsilon"],
                  candidates(tuning["source_learning_rates"], positive_number, "source learning rates"),
                  candidates(tuning["target_learning_rates"], positive_number, "target learning rates"),
                  candidates(tuning["target_epoch_candidates"], positive_int, "target epochs"), levels,
                  candidates(choices["epsilons"], lambda value: positive_number(value) and value in PRIVACY_BUDGETS,
                             "epsilons"))


def load_config(path):
    return parse_config(read_json(path))
