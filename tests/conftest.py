"""Shared synthetic experiment setup; no real test-set inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import PROTOCOL
from dp_shift.data.prepare import build
from dp_shift.data.utils import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    experiment_config = read_json(ROOT / "configs/experiment.json")
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    n = 1000
    records = pd.DataFrame({name: np.arange(n) % (i+3) for i, name in enumerate(list(PROTOCOL.numeric_features))})
    for name in PROTOCOL.categorical_features:
        records[name] = "1" if name in PROTOCOL.categorical_ids else "known"
    records["age"] = np.where(np.arange(n) % 2 == 0, "[70-80)", "[50-60)")
    records["number_inpatient"] = np.where(np.arange(n) % 5 == 0, 2, 0)
    records["patient_nbr"], records["encounter_id"] = np.arange(n), np.arange(n) + 1000
    records["readmitted"] = np.where(np.arange(n) % 3 == 0, "<30", "NO")
    record_path = tmp_path / "data/records.csv"
    records.to_csv(record_path, index=False)
    mapping = tmp_path / "data/mapping.csv"
    mapping.write_text("discharge_disposition_id,description\n1,Home\n")
    experiment_config["privacy"] = dict(clipping_norm=.7)
    experiment_config["repeats"] = 2
    experiment_config["data"] = dict(source_pool_size=500, proxy_train_size=24,
                                    proxy_validation_size=100, target_train_size=24, target_test_size=80)
    experiment_config["pretraining"] = dict(sample_size=40, epochs=1)
    experiment_config["comparisons"] = dict(shifts={"readmission": [.1]}, epsilons=[4.])
    experiment_config["tuning"] = dict(reference_epsilon=4., source_learning_rates=[.01],
                                      target_learning_rates=[.01], target_epoch_candidates=[1])
    config_path = tmp_path / "configs/experiment.json"
    write_json(config_path, experiment_config)
    data_root, output = tmp_path / "prepared", tmp_path / "experiment"
    monkeypatch.setattr("dp_shift.data.prepare.DEFAULT_RECORDS", record_path)
    monkeypatch.setattr("dp_shift.data.prepare.DEFAULT_MAPPING", mapping)
    build(config_path, data_root)
    return data_root, output, config_path
