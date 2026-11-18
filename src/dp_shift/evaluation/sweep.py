"""Combine comparison cells and verify pairing and private accounting."""

from pathlib import Path

import pandas as pd

from dp_shift.data.utils import file_hash, read_json, require
from dp_shift.stages.common import atomic_json
from . import tables
from .accounting import verify_privacy


def collect(output, comparison):
    stages = read_json(output / "protocol.json")["stages"]
    prepared, trained, evaluated = (stages[name] for name in ("prepare", "train", "evaluate"))
    record = {
        **dict(zip(tables.GRID, comparison)),
        "selected": trained["selected"],
        "membership_hashes": prepared["membership_hashes"],
        "source_hashes": prepared["source_hashes"],
        "source_pairs": prepared["source_pairs"],
        "characteristics": prepared["characteristics"],
        "preprocessing": prepared["preprocessing"],
        "partitions": prepared["partitions"],
        "test_reference": evaluated["test_reference"],
        "test_characteristics": evaluated["test_characteristics"],
        "privacy": evaluated["privacy"],
    }
    return tables.read(output / tables.RUNS_FILE), record


def validate(runs, saved):
    from dp_shift.sweep import grid

    plan = saved["plan"]
    selected = saved["reference_tuning"]["selected"]
    expected = set(grid(plan))
    records = {tuple(cell[name] for name in tables.GRID): cell for cell in saved["cells"]}
    require(len(saved["cells"]) == len(records) and set(records) == expected,
            "Missing or duplicate comparison records")
    require(set(tables.cells(runs).groups) == expected, "Comparison grid is incomplete")
    require(len(runs) == saved["evaluated_models"] and runs.repeat.nunique() == plan["base"]["repeats"],
            "Experiment model or repeat count differs")

    first = saved["cells"][0]
    source_hashes, ordinary_scores, noise_multipliers = {}, {}, {}
    ordinary_baseline = None
    for (kind, fraction, epsilon), group in tables.cells(runs):
        cell = records[kind, fraction, epsilon]
        require(cell["membership_hashes"] == first["membership_hashes"], "Experiment partitions differ")
        require(cell["preprocessing"] == first["preprocessing"], "Experiment preprocessing differs")
        reference = cell["test_reference"]
        require(reference == first["test_reference"] and group.test_size.eq(reference["n"]).all()
                and group.test_fraction.eq(cell["test_characteristics"][kind]["fraction"]).all(),
                "Experiment test sets differ")
        require(all(cell["selected"][name] == value for name, value in selected.items()),
                "Experiment training settings differ")
        require(cell["source_hashes"] == source_hashes.setdefault((kind, fraction), cell["source_hashes"]),
                "Source membership changed across epsilon values")
        require(all(cell["source_hashes"].get(key) == value for key, value in first["source_hashes"].items()
                    if key.endswith("_s0")), "Baseline source membership changed")

        score_columns = ["repeat", "pretraining", *tables.METRICS, "source_fraction", "source_size"]
        scores = group.loc[group.method.eq("sgd"), score_columns]
        scores = scores.sort_values(["repeat", "pretraining"], ignore_index=True)
        pd.testing.assert_frame_equal(scores, ordinary_scores.setdefault((kind, fraction), scores), check_exact=True)
        baseline = scores.loc[scores.pretraining.ne("shifted")].drop(columns="source_fraction").reset_index(drop=True)
        if ordinary_baseline is None:
            ordinary_baseline = baseline
        pd.testing.assert_frame_equal(baseline, ordinary_baseline, check_exact=True)

        sigma = cell["selected"]["noise_multiplier"]
        require(sigma == noise_multipliers.setdefault(epsilon, sigma), "Noise calibration changed across shifts")
        verify_privacy(group, cell["privacy"], cell["selected"], plan["base"]["privacy"]["clipping_norm"], epsilon)


def publish(output, saved, reference, datasets, records):
    runs = pd.concat(datasets, ignore_index=True)
    saved = {**saved, "reference_tuning": reference, "cells": records, "evaluated_models": len(runs),
             "definitions": {"auprc": "Non-interpolated average precision",
                             "auroc": "Area under the receiver operating characteristic curve",
                             "brier": "Mean squared error of predicted probabilities",
                             "comparison_epsilon": "Grid budget, with no privacy guarantee for SGD",
                             "privacy": "Per private model",
                             "intervals": "95% paired t intervals across repeats on a fixed test set"}}
    tables.validate(runs)
    validate(runs, saved)
    path = Path(output) / tables.RUNS_FILE
    tables.write(path, runs)
    saved.update(status="complete", files={tables.RUNS_FILE: file_hash(path)})
    atomic_json(Path(output) / "protocol.json", saved)
