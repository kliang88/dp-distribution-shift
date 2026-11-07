"""Publish two research tables, privacy records, and four reproducible figures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.config import load_config
from dp_shift.data.utils import read_json, require
from dp_shift.evaluation.metrics import validate_curves
from dp_shift.evaluation.plots import FIGURES, plot
from dp_shift.evaluation.provenance import freeze
from dp_shift.evaluation.run import verify_analysis
from dp_shift.evaluation.score import score_models
from dp_shift.evaluation.summarize import summary_tables, summarize, validate_runs
from dp_shift.training.model import configure_cpu
from .common import atomic_path, cleanup, completed, copy, finish, protocol, verify, workspace


def privacy_table(work):
    rows = []
    for path in sorted((work / "training").glob("repeat_*/*.json")):
        saved = read_json(path)
        metadata = saved["metadata"]
        privacy = metadata["privacy"]
        if privacy is None:
            continue
        history = privacy["accountant_history"]
        require(sum(entry[2] for entry in history) == privacy["steps"], "Accountant history is incomplete")
        rows.append({"repeat_id": metadata["repeat_id"], "path_id": metadata["path_id"], **privacy,
                     "accountant_history": json.dumps(history), "checkpoint_sha256": saved["checkpoint_sha256"]})
    return pd.DataFrame(rows)


def run(output, config, resume=False, progress=print):
    output = Path(output)
    existing = completed(output, "evaluate", config)
    if existing is not None:
        progress("Verified completed evaluation; no inference needed.")
        verify_results(output)
        cleanup(output)
        return existing
    verify(output, "train")
    work = workspace(output)
    evaluation = work / "evaluation_v3"
    configure_cpu()
    progress("Freezing evaluation, scoring models, and checking statistical results…")
    frozen = freeze(work / "data", work, evaluation, config)
    score_models(evaluation, resume=resume)
    summarize(evaluation)
    runs = pd.read_csv(evaluation / "runs.csv", float_precision="round_trip")
    effects = pd.read_csv(evaluation / "effects.csv", float_precision="round_trip")
    reference = read_json(evaluation / "test_reference.json")
    source_pairs = pd.DataFrame(protocol(output)["stages"]["prepare"]["source_pairs"])
    source_rates = source_pairs.groupby("condition").positive_rate.mean().to_dict()
    with np.load(evaluation / "curves.npz", allow_pickle=False) as curves:
        plot(evaluation, runs, curves, effects, reference, source_rates)
    result = verify_analysis(evaluation)
    progress("Publishing metrics, privacy accounting, processed test data, and PNG figures…")
    files = ["evaluation/privacy.csv"]
    with atomic_path(output / files[0]) as temporary:
        privacy_table(work).to_csv(temporary, index=False, float_format="%.17g")
    for name in ("metrics.csv", "effects.csv", "runs.csv", "curves.npz"):
        relative = f"evaluation/{name}" if name in {"metrics.csv", "effects.csv"} else f"evaluation/supporting/{name}"
        copy(evaluation / name, output / relative)
        files.append(relative)
    copy(work / "data/evaluation/target_test.npz", output / "data/target_test.npz")
    files.append("data/target_test.npz")
    for name in FIGURES:
        relative = f"evaluation/figures/{name}.png"
        copy(evaluation / f"figures/{name}.png", output / relative)
        files.append(relative)
    summary = {**result,
               "metrics": str(output / "evaluation/metrics.csv"),
               "effects": str(output / "evaluation/effects.csv"),
               "privacy": str(output / "evaluation/privacy.csv"),
               "figures": str(output / "evaluation/figures")}
    require(load_config(config) == frozen["config"], "Evaluation configuration changed during execution")
    finish(output, "evaluate", config, summary, files,
           resolved=frozen["config"],
           provenance={key: frozen[key] for key in ("code_hashes", "versions")},
           test_reference=reference, source_rates=source_rates,
           metric_definitions={"primary": "average_precision (non-interpolated)",
                               "metrics.csv": "Six conditions; mean and sample SD across repeats; n is repeats",
                               "effects.csv": "Seven paired AP differences in raw AP units",
                               "calibration_bias": "mean prediction minus observed prevalence; zero is ideal",
                               "intervals": "95% Student t across repeats, paired for effects, conditional on fixed test set",
                               "brier_reference": "Constant risk equal to observed test prevalence; descriptive reference",
                               "curves.npz": "Keys: repeat_NN/path_id/field; per-run PR, quantile bins, and risk histograms",
                               "privacy": "Per-model guarantee; not a joint collection guarantee"})
    verify_results(output)
    progress("Results verified; removing temporary checkpoints, predictions, and diagnostics…")
    cleanup(output)
    return summary


def verify_results(output):
    """Check summaries and privacy from retained records without re-running inference."""
    from dp_shift.training.privacy import predicted_epsilon

    output = Path(output)
    summary = verify(output, "evaluate")
    saved = protocol(output)["stages"]
    repeats = saved["prepare"]["config"]["repeats"]
    runs = pd.read_csv(output / "evaluation/supporting/runs.csv", float_precision="round_trip")
    validate_runs(runs, repeats)
    reference = saved["evaluate"]["test_reference"]
    require(len(runs) == summary["evaluated_models"] and reference["n"] == summary["test_patients"],
            "Evaluation counts disagree with retained data")
    require(np.allclose(runs.test_prevalence, reference["test_prevalence"], rtol=0, atol=1e-12),
            "Retained test prevalence mismatch")
    with np.load(output / "evaluation/supporting/curves.npz", allow_pickle=False) as curves:
        validate_curves(curves, runs, reference)
    for name, expected in summary_tables(runs, saved["evaluate"]["resolved"]).items():
        actual = pd.read_csv(output / "evaluation" / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-12)
    privacy = pd.read_csv(output / "evaluation/privacy.csv", float_precision="round_trip")
    expected_keys = {(repeat, path) for repeat in range(repeats) for path in ("C", "D-S0", "D-S1")}
    keys = list(zip(privacy.repeat_id, privacy.path_id))
    require(len(keys) == len(expected_keys) and set(keys) == expected_keys, "Missing or duplicate privacy records")
    settings = {**saved["train"]["resolved"], **saved["train"]["selected"]}
    steps = settings["steps_per_epoch"] * settings["target_epochs"]
    sigma, rate = settings["noise_multiplier"], 1 / settings["steps_per_epoch"]
    epsilon = predicted_epsilon(sigma, rate, steps, settings["delta"])
    for row in privacy.itertuples():
        require(row.accountant == "prv" and row.steps == steps and row.delta == settings["delta"]
                and row.noise_multiplier == sigma and row.sampling_rate == rate
                and row.clipping_norm == settings["clipping_norm"], "Privacy settings mismatch")
        history = json.loads(row.accountant_history)
        require(sum(entry[2] for entry in history) == steps
                and all(entry[0] == sigma and entry[1] == rate for entry in history), "Privacy history mismatch")
        require(abs(row.epsilon_achieved - epsilon) <= 1e-6
                and row.epsilon_achieved <= settings["epsilon_cap"], "Privacy epsilon mismatch")
    return summary
