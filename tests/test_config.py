"""Experiment choices, cross-field constraints, and exact patient allocations."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import PARTITIONS, PROTOCOL, resolve_config, study_plan
from dp_shift.data.sampling import create_source_pair
from dp_shift.data.splits import create_split_manifests


@pytest.mark.parametrize("path,value,message", [
    (("schema_version",), True, "schema_version"),
    (("schema_version",), 2, "schema_version"),
    (("lambda",), 1, "unknown keys"),
    (("privacy", "epslion"), 8, "unknown keys"),
    (("seed",), True, "seed must be"),
    (("seed",), -1, "seed must be"),
    (("seed",), 2**32, "seed must be"),
    (("repeats",), 0, "positive integer"),
    (("data", "source_pool_size"), 0, "positive integer"),
    (("data", "target_train_size"), 25, "divisible by 12"),
    (("data", "proxy_train_size"), 25, "divisible by 12"),
    (("data", "source_pool_size"), 40, "must exceed pretrain_size"),
    (("pretraining", "sample_size"), 41, "must be an integer"),
    (("comparisons", "shifts", "readmission"), [1], "below 1"),
    (("comparisons", "shifts", "readmission"), [0], "positive finite"),
    (("comparisons", "shifts"), {"unknown": [.1]}, "Unknown shift type"),
    (("comparisons", "epsilons"), [float("nan")], "positive finite"),
    (("comparisons", "epsilons"), [-1], "positive finite"),
    (("comparisons", "epsilons"), [5, -1], "positive finite"),
    (("comparisons", "shifts", "readmission"), [.1, .11], "must be an integer"),
    (("comparisons", "shifts"), {}, "nonempty object"),
    (("comparisons", "shifts", "readmission"), .01, "nonempty list"),
    (("privacy", "clipping_norm"), 0, "positive finite"),
    (("pretraining", "epochs"), 1.5, "positive integer"),
    (("tuning", "target_epoch_candidates"), [0], "positive integer"),
    (("tuning", "target_learning_rates"), [], "nonempty list"),
    (("tuning", "source_learning_rates"), [.01, .01], "duplicates"),
    (("tuning", "source_learning_rates"), [True], "positive finite"),
])
def test_invalid_choices_are_rejected(choices, path, value, message):
    section = choices
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        study_plan(choices)


def test_missing_choice_is_rejected(choices):
    del choices["comparisons"]["epsilons"]
    with pytest.raises(ValueError, match="missing keys.*epsilons"):
        resolve_config(choices)


def test_single_and_multiple_comparisons_use_the_same_schema(choices):
    single = study_plan(choices)
    expanded = deepcopy(choices)
    expanded["comparisons"] = {"shifts": {"readmission": [.1, .2], "age_70_plus": [.5]}, "epsilons": [1., 4.]}
    multiple = study_plan(expanded)
    assert single["base"]["data"] == multiple["base"]["data"]
    assert single["base"]["tuning"] == multiple["base"]["tuning"]
    assert single["sweep"] == {**choices["comparisons"], "reference_epsilon": 4.}
    assert sum(map(len, multiple["sweep"]["shifts"].values())) * len(multiple["sweep"]["epsilons"]) == 6
    with pytest.raises(ValueError, match="requires one comparison"):
        resolve_config(expanded)


def test_defaults_and_seeds_are_resolved_without_mutating_choices(choices):
    original = deepcopy(choices)
    first = resolve_config(choices)
    assert choices == original
    assert first.shift_member_count == 4
    assert first.calibration_epsilon == pytest.approx(4. * (7.99 / 8))
    assert first.requested.tuning.source_epochs == choices["pretraining"]["epochs"]
    assert PROTOCOL.steps_per_epoch == 12
    assert len(set(asdict(first.seeds).values())) == len(asdict(first.seeds))
    choices["seed"] += 1
    second = resolve_config(choices)
    assert all(getattr(first.seeds, role) != getattr(second.seeds, role) for role in asdict(first.seeds))
    snapshot = first.to_dict()
    assert resolve_config(snapshot["experiment"]) == first
    snapshot["numeric_features"].clear()
    snapshot["experiment"]["tuning"]["source_learning_rates"].clear()
    assert first.requested.tuning.source_learning_rates
    assert PROTOCOL.numeric_features
    with pytest.raises(FrozenInstanceError):
        first.shift_member_count = 0
    with pytest.raises(FrozenInstanceError):
        first.requested.privacy.epsilon = 10
    with pytest.raises(TypeError):
        first.requested.tuning.source_learning_rates[0] = 10


def small_config(choices):
    choices["data"] = dict(source_pool_size=200, proxy_train_size=24,
                            proxy_validation_size=40, target_train_size=24, target_test_size=40)
    choices["pretraining"]["sample_size"] = 40
    choices["comparisons"]["shifts"] = {"readmission": [.05]}
    choices["repeats"] = 2
    return resolve_config(choices)


def cohort(n):
    return pd.DataFrame(dict(patient_nbr=np.arange(n), encounter_id=np.arange(n) + 1000,
                             y=np.arange(n) % 2))


def test_exact_allocations_are_disjoint_reproducible_and_seeded(choices):
    config, patients = small_config(choices), cohort(400)
    frames = create_split_manifests(patients, config)
    for name in PARTITIONS:
        assert len(frames[name]) == getattr(config.requested.data, f"{name}_size")
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


@pytest.mark.parametrize("rate,label", [(.05, "negative"), (.95, "positive")])
def test_source_shift_rejects_insufficient_replacement_patients(choices, rate, label):
    choices["comparisons"]["shifts"]["readmission"] = [rate]
    config = resolve_config(choices)
    patients = cohort(41)
    # A 40-patient sample needs 38 of one class, but the pool has only 20/21.
    with pytest.raises(ValueError, match=f"S1 requires 38 {label} patients; source pool has") as error:
        create_source_pair(patients, 0, config)
    assert "feasible fraction" in str(error.value)
