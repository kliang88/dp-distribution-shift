"""Combine CSV cells and verify pairing and private accounting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.config import PROTOCOL, resolve_config
from dp_shift.data.utils import file_hash, read_json, require
from dp_shift.stages.common import atomic_json
from .summarize import validate_runs
from . import tables


GRID = ["shift_type", "shift_fraction", "grid_epsilon"]
PRIVACY = ["accountant", "accountant_history", "clipping_norm", "delta", "epsilon_achieved",
           "noise_multiplier", "sampling_rate", "steps", "checkpoint_sha256"]


def collect(output, kind, fraction, epsilon, prefix):
    output = Path(output)
    saved = read_json(output / "protocol.json")["stages"]
    data = tables.read(output)
    for table in data.values():
        table["run_id"] = prefix + table.run_id
    prepared, trained, evaluated = (saved[name] for name in ("prepare", "train", "evaluate"))
    record = {"shift_type": kind, "shift_fraction": fraction, "grid_epsilon": epsilon,
              "selected": trained["selected"], "membership_hashes": prepared["membership_hashes"],
              "source_hashes": prepared["source_hashes"], "source_pairs": prepared["source_pairs"],
              "characteristics": prepared.get("characteristics"),
              "preprocessing": prepared["preprocessing"], "partitions": prepared["partitions"],
              "test_reference": evaluated["test_reference"],
              "privacy": {prefix + key: value for key, value in evaluated["privacy"].items()}}
    return data, record


def accounted_runs(data, ledger):
    frames = [tables.internal(cell)[0] for _, cell in tables.cells(data)]
    runs = pd.concat(frames, ignore_index=True).drop(columns="epsilon_achieved")
    return runs.merge(tables.privacy_frame(data, ledger), on="run_id", how="left", validate="one_to_one").drop(columns="run_id")


def validate(runs, saved):
    from dp_shift.training.privacy import predicted_epsilon

    from dp_shift.sweep import normalize_plan, grid as plan_grid

    plan = normalize_plan(saved["plan"])
    config = resolve_config(plan["base"])
    selected = saved["reference_tuning"]["selected"]
    grid = set(plan_grid(plan))
    require(set(runs[GRID].drop_duplicates().itertuples(index=False, name=None)) == grid, "Sweep grid is incomplete")
    cells = {(cell.get("shift_type", "readmission"), cell.get("shift_fraction", cell.get("shifted_positive_rate")), cell["grid_epsilon"]): cell
             for cell in saved["cells"]}
    require(len(saved["cells"]) == len(cells) == len(grid) and set(cells) == grid, "Sweep cell records are incomplete")
    first = saved["cells"][0]
    expected_sources, ordinary, epsilons = {}, {}, {}
    ordinary_baseline = None
    for (kind, fraction, epsilon), group in runs.groupby(GRID, sort=True):
        validate_runs(group, config.requested.repeats)
        record = cells[kind, fraction, epsilon]
        require(record["membership_hashes"] == first["membership_hashes"], "Sweep partitions differ")
        require(record["preprocessing"] == first["preprocessing"], "Sweep preprocessing differs")
        require(record["test_reference"] == first["test_reference"], "Sweep test sets differ")
        require(all(record["selected"][key] == value for key, value in selected.items()), "Sweep training settings differ")
        require(record["source_hashes"] == expected_sources.setdefault((kind, fraction), record["source_hashes"]),
                "Source membership changed across epsilon values")
        for key, value in first["source_hashes"].items():
            if key.endswith("_s0"):
                require(record["source_hashes"][key] == value, "Baseline source membership changed")
        reference = record["test_reference"]
        require(np.allclose(group.test_prevalence, reference["test_prevalence"], rtol=0, atol=1e-12),
                "Sweep test prevalence differs")
        public = group.loc[group.method.eq("sgd")]
        require(public[PRIVACY].isna().all().all(), "SGD has unexpected privacy records")
        scores = public.drop(columns=[*GRID, *PRIVACY]).sort_values(["repeat_id", "path_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(scores, ordinary.setdefault((kind, fraction), scores), check_exact=True)
        baseline = scores.loc[scores.path_id.isin(["A", "B-S0"])].drop(columns=["source_fraction", "test_fraction"]).reset_index(drop=True)
        if ordinary_baseline is None:
            ordinary_baseline = baseline
        pd.testing.assert_frame_equal(baseline, ordinary_baseline, check_exact=True)
        private = group.loc[group.method.eq("dp_sgd")]
        require(private[PRIVACY].notna().all().all(), "Private models are missing accounting records")
        sigma = record["selected"]["noise_multiplier"]
        steps = selected["target_epochs"] * PROTOCOL.steps_per_epoch
        if epsilon not in epsilons:
            epsilons[epsilon] = (sigma, predicted_epsilon(sigma, 1 / PROTOCOL.steps_per_epoch, steps, PROTOCOL.delta))
        expected_sigma, achieved = epsilons[epsilon]
        require(sigma == expected_sigma, "Noise calibration changed across severities")
        require(np.isfinite(sigma) and sigma > 0 and np.isfinite(achieved) and achieved <= epsilon,
                "Sweep privacy budget exceeded")
        for row in private.itertuples():
            require(row.accountant == "prv" and row.noise_multiplier == sigma
                    and row.steps == steps and row.delta == PROTOCOL.delta
                    and row.clipping_norm == config.requested.privacy.clipping_norm
                    and row.sampling_rate == 1 / PROTOCOL.steps_per_epoch
                    and row.epsilon_achieved <= epsilon
                    and abs(row.epsilon_achieved - achieved) <= 1e-6, "Sweep privacy settings mismatch")
            history = json.loads(row.accountant_history)
            require(sum(entry[2] for entry in history) == steps
                    and all(entry[0] == sigma and entry[1] == row.sampling_rate and entry[2] > 0 for entry in history),
                    "Sweep accountant history mismatch")
    require(len(runs) == saved["evaluated_models"], "Sweep model count differs")


def verify_tables(output, saved):
    data = tables.read(output)
    ledger = {key: value for cell in saved["cells"] for key, value in cell["privacy"].items()}
    validate(accounted_runs(data, ledger), saved)


def publish(output, saved, reference, datasets, records):
    output = Path(output)
    data = {name: pd.concat([dataset[name] for dataset in datasets], ignore_index=True) for name in tables.COLUMNS}
    saved = {**saved, "reference_tuning": reference, "cells": records, "evaluated_models": len(data["runs"]),
             "resolved_base": resolve_config(saved["plan"]["base"]).to_dict(),
             "definitions": {"auprc": "Non-interpolated average precision; calculation unchanged",
                             "comparison_epsilon": "Comparison budget; SGD is nonprivate",
                             "dp_amplification": "DP shift penalty minus SGD shift penalty; each penalty is baseline minus shifted AUPRC",
                             "privacy": "Per private model, not jointly across the sweep",
                             "intervals": "95% Student t intervals across paired repeats, conditional on fixed partitions",
                             "duplicates": "Baselines repeat per cell; never pool cells as independent repeats"}}
    tables.validate(data)
    ledger = {key: value for cell in records for key, value in cell["privacy"].items()}
    validate(accounted_runs(data, ledger), saved)
    tables.write(output, data)
    verify_tables(output, saved)
    saved.update(status="complete", files={name: file_hash(output / name) for name in tables.FILES})
    atomic_json(output / "protocol.json", saved)
