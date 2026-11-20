"""Synthetic choices and patient records shared by the tests."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import PRIVACY_BUDGETS, PROTOCOL, SIGNED_LEVELS
from dp_shift.data.shifts import SHIFTS
from dp_shift.data.utils import write_json
from dp_shift.evaluation.tables import COLUMNS


@pytest.fixture
def choices():
    return {
        "seed": 42,
        "repeats": 2,
        "data": dict(source_pool_size=500, proxy_train_size=24, proxy_validation_size=100,
                     target_train_size=24, target_test_size=80),
        "pretraining": {"sample_size": 50, "epochs": 1},
        "privacy": {"clipping_norm": .7},
        "tuning": dict(reference_epsilon=5., source_learning_rates=[.01], target_learning_rates=[.01],
                       target_epoch_candidates=[1]),
        "shift_pp": [-2, 2],
        "epsilons": [5.],
    }


@pytest.fixture
def experiment_inputs(tmp_path, choices):
    index = np.arange(900)
    records = pd.DataFrame({name: index % (i + 3) for i, name in enumerate(PROTOCOL.numeric_features)})
    for name in PROTOCOL.categorical_features:
        records[name] = "1" if name in PROTOCOL.categorical_ids else "known"
    records["diag_1"] = np.where(index % 4 < 2, "414", "250.83")
    records["diag_2"] = np.where(index % 5 < 2, "486", "250.83")
    records["diag_3"] = "?"
    records["insulin"] = "No"
    records["age"] = np.where(index % 2 == 0, "[70-80)", "[50-60)")
    records["patient_nbr"], records["encounter_id"] = index, index + 1000
    records["readmitted"] = np.where(index % 3 == 0, "<30", "NO")
    records_path = tmp_path / "records.csv"
    records.to_csv(records_path, index=False)
    config_path = tmp_path / "experiment.json"
    write_json(config_path, choices)
    return tmp_path / "output", config_path, records_path


@pytest.fixture
def signed_results():
    def scores(repeats=20, levels=SIGNED_LEVELS, epsilons=PRIVACY_BUDGETS):
        conditions = [("scratch", 0), ("baseline", 0), *product(SHIFTS, levels)]
        methods = [("SGD", None), *(("DP-SGD", epsilon) for epsilon in epsilons)]
        rows = []
        for repeat, (kind, pp), (method, epsilon) in product(range(1, repeats + 1), conditions, methods):
            row = {"repeat": repeat, "shift_type": kind, "shift_pp": pp, "method": method, "epsilon": epsilon,
                   "auprc": .19 if kind == "scratch" else .2 + pp / 1000, "auroc": .6, "brier_score": .12,
                   "target_train_size": 400, "target_test_size": 1000, "target_test_readmission_count": 96,
                   "target_lr": .01, "target_epochs": 2, "target_steps": 24, "seed": 42}
            if kind != "scratch":
                row.update(source_size=1000, source_lr=.03, source_epochs=3)
                counts = {"readmission": 100 + repeat, "age": 450, "circulatory": 300, "complexity": 700}
                if kind in counts:
                    counts[kind] += 10 * pp
                row.update({f"source_{trait}_count": count for trait, count in counts.items()})
            if epsilon is not None:
                row.update(epsilon_achieved=.99 * epsilon, delta=1e-5, clipping_norm=1., noise_multiplier=2 / epsilon)
            rows.append(row)
        return pd.DataFrame(rows, columns=COLUMNS)
    return scores
