"""Experiment choices, cross-field constraints, and exact patient allocations."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_shift.cli import main
from dp_shift.config import PARTITIONS, load_config, resolve_config
from dp_shift.data.sampling import create_source_pair
from dp_shift.data.splits import create_split_manifests
from dp_shift.data.utils import read_json


CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment.json"


@pytest.fixture
def choices():
    return read_json(CONFIG)


@pytest.mark.parametrize("path,value,message", [
    (("lambda",), 1, "unknown keys"),
    (("privacy", "epslion"), 8, "unknown keys"),
    (("seed",), True, "seed must be"),
    (("seed",), -1, "seed must be"),
    (("seed",), 2**32, "seed must be"),
    (("repeats",), 1, "at least 2"),
    (("data", "source_pool_size"), 0, "positive integer"),
    (("data", "target_train_size"), 3001, "divisible by 12"),
    (("data", "proxy_train_size"), 3001, "divisible by 12"),
    (("data", "source_pool_size"), 16000, "must exceed pretrain_size"),
    (("source_conditions", "pretrain_size"), 16001, "must be an integer"),
    (("source_conditions", "shifted_positive_rate"), 1, "below 1"),
    (("source_conditions", "shifted_positive_rate"), 0, "positive finite"),
    (("privacy", "epsilon"), float("nan"), "positive finite"),
    (("privacy", "epsilon"), -1, "positive finite"),
    (("privacy", "clipping_norm"), 0, "positive finite"),
    (("tuning", "source_epochs"), 1.5, "positive integer"),
    (("tuning", "target_epoch_candidates"), [0], "positive integer"),
    (("tuning", "target_learning_rates"), [], "nonempty list"),
    (("tuning", "source_learning_rates"), [.01, .01], "duplicates"),
    (("tuning", "source_learning_rates"), [True], "positive finite"),
])
def test_invalid_choices_fail_before_output_creation(choices, path, value, message, tmp_path, capsys):
    section = choices
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    config, output = tmp_path / "experiment.json", tmp_path / "output"
    config.write_text(json.dumps(choices))
    with pytest.raises(SystemExit) as error:
        main(["all", "--config", str(config), "--output", str(output)])
    assert error.value.code == 1
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_missing_choice_is_rejected(choices):
    del choices["privacy"]["epsilon"]
    with pytest.raises(ValueError, match="missing keys.*epsilon"):
        resolve_config(choices)


def test_defaults_and_seeds_are_resolved_without_mutating_choices(choices):
    original = deepcopy(choices)
    first = resolve_config(choices)
    assert choices == original
    assert first == load_config(CONFIG)
    source = choices["source_conditions"]
    assert first["source_positive_n"] == round(source["pretrain_size"] * source["shifted_positive_rate"])
    assert first["calibration_epsilon"] == pytest.approx(choices["privacy"]["epsilon"] * (7.99 / 8))
    assert first["source_epochs"] == choices["tuning"]["source_epochs"]
    assert first["steps_per_epoch"] == 12
    assert len(set(first["seeds"].values())) == len(first["seeds"])
    choices["seed"] += 1
    second = resolve_config(choices)
    assert all(first["seeds"][role] != second["seeds"][role] for role in first["seeds"])
    first["numeric_features"].clear()
    assert resolve_config(original)["numeric_features"]


def small_config(choices):
    choices["data"] = dict(source_pool_size=200, proxy_train_size=24,
                            proxy_validation_size=40, target_train_size=24, target_test_size=40)
    choices["source_conditions"] = dict(pretrain_size=40, shifted_positive_rate=.05)
    choices["repeats"] = 2
    return resolve_config(choices)


def cohort(n):
    return pd.DataFrame(dict(patient_nbr=np.arange(n), encounter_id=np.arange(n) + 1000,
                             y=np.arange(n) % 2))


def test_exact_allocations_are_disjoint_reproducible_and_seeded(choices):
    config, patients = small_config(choices), cohort(400)
    frames = create_split_manifests(patients, config)
    for name in PARTITIONS:
        assert len(frames[name]) == config["data"][f"{name}_size"]
    assert len(frames["unused"]) == 72
    ids = [set(frames[name].patient_nbr) for name in (*PARTITIONS, "unused")]
    assert sum(map(len, ids)) == len(set.union(*ids)) == len(patients)
    again = create_split_manifests(patients, config)
    for name in frames:
        pd.testing.assert_frame_equal(frames[name], again[name])
    choices["seed"] += 1
    changed = create_split_manifests(patients, resolve_config(choices))
    assert not frames["target_train"].patient_nbr.equals(changed["target_train"].patient_nbr)


def test_partition_capacity_and_single_class_fail_early(choices):
    config = small_config(choices)
    with pytest.raises(ValueError, match="Requested partitions need"):
        create_split_manifests(cohort(300), config)
    patients = cohort(400)
    patients["y"] = 0
    with pytest.raises(ValueError, match="must contain both classes"):
        create_split_manifests(patients, config)


def test_source_shift_rejects_insufficient_replacement_patients(choices):
    config = small_config(choices)
    patients = cohort(41)
    # Any 40-patient S0 has about 20 positives to replace, but at most one spare negative.
    with pytest.raises(ValueError, match="Not enough unused negative patients"):
        create_source_pair(patients, 0, config)
