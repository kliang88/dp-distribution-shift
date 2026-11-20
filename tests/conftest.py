"""Small synthetic inputs shared by unit and integration tests."""

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import PROTOCOL, study_plan
from dp_shift.data.features import DIAGNOSIS_COLUMNS
from dp_shift.data.prepare import build
from dp_shift.data.utils import write_json


@pytest.fixture
def choices():
    return {
        "schema_version": 1,
        "seed": 42,
        "repeats": 2,
        "data": {
            "source_pool_size": 500,
            "proxy_train_size": 24,
            "proxy_validation_size": 100,
            "target_train_size": 24,
            "target_test_size": 80,
        },
        "pretraining": {"sample_size": 40, "epochs": 1},
        "privacy": {"clipping_norm": .7},
        "tuning": {
            "reference_epsilon": 4.,
            "source_learning_rates": [.01],
            "target_learning_rates": [.01],
            "target_epoch_candidates": [1],
        },
        "comparisons": {"shifts": {"readmission": [.1]}, "epsilons": [4.]},
    }


@pytest.fixture
def base_config(choices):
    return study_plan(choices)["base"]


@pytest.fixture
def experiment_inputs(tmp_path, monkeypatch, choices):
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    n = 1000
    records = pd.DataFrame({name: np.arange(n) % (i+3) for i, name in enumerate(list(PROTOCOL.numeric_features))})
    for name in PROTOCOL.categorical_features:
        records[name] = "1" if name in PROTOCOL.categorical_ids else "known"
    records[list(DIAGNOSIS_COLUMNS)] = "250.83"
    records["insulin"] = "No"
    records["age"] = np.where(np.arange(n) % 2 == 0, "[70-80)", "[50-60)")
    records["number_inpatient"] = np.where(np.arange(n) % 5 == 0, 2, 0)
    records["patient_nbr"], records["encounter_id"] = np.arange(n), np.arange(n) + 1000
    records["readmitted"] = np.where(np.arange(n) % 3 == 0, "<30", "NO")
    record_path = tmp_path / "data/records.csv"
    records.to_csv(record_path, index=False)
    mapping = tmp_path / "data/mapping.csv"
    mapping.write_text("discharge_disposition_id,description\n1,Home\n")
    config_path = tmp_path / "configs/experiment.json"
    write_json(config_path, choices)
    monkeypatch.setattr("dp_shift.data.prepare.DEFAULT_RECORDS", record_path)
    monkeypatch.setattr("dp_shift.data.prepare.DEFAULT_MAPPING", mapping)
    return tmp_path / "experiment", config_path


@pytest.fixture
def experiment(experiment_inputs):
    output, config_path = experiment_inputs
    data_root = output.with_name("prepared")
    build(config_path, data_root)
    return data_root, output, config_path
