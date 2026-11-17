"""Publish four readable CSV tables and retain accounting in protocol metadata."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.config import PROTOCOL, SelectedHyperparameters, TrainingSettings, load_config, resolve_config
from dp_shift.data.utils import read_json, require
from dp_shift.evaluation import tables
from dp_shift.evaluation.provenance import freeze
from dp_shift.evaluation.run import verify_analysis
from dp_shift.evaluation.score import score_models
from dp_shift.evaluation.summarize import summarize
from dp_shift.training.model import configure_cpu
from .common import cleanup, completed, finish, protocol, verify, workspace


def privacy_table(work):
    rows = []
    for path in sorted((work / "training").glob("repeat_*/*.json")):
        saved = read_json(path)
        metadata = saved["metadata"]
        privacy = metadata["privacy"]
        if privacy is None:
            continue
        require(sum(entry[2] for entry in privacy["accountant_history"]) == privacy["steps"],
                "Accountant history is incomplete")
        rows.append({"repeat_id": metadata["repeat_id"], "path_id": metadata["path_id"], **privacy,
                     "accountant_history": json.dumps(privacy["accountant_history"]),
                     "checkpoint_sha256": saved["checkpoint_sha256"]})
    return pd.DataFrame(rows)


def run(output, config, resume=False, progress=print):
    output = Path(output)
    existing = completed(output, "evaluate", config)
    if existing is not None:
        progress("Verifying retained CSV data; no inference needed.")
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
    result = verify_analysis(evaluation)
    runs = pd.read_csv(evaluation / "runs.csv", float_precision="round_trip")
    reference = read_json(evaluation / "test_reference.json")
    source_pairs = pd.DataFrame(protocol(output)["stages"]["prepare"]["source_pairs"])
    characteristics = read_json(work / "data/evaluation/characteristics.json")
    resolved = resolve_config(frozen["config"]["experiment"])
    with np.load(evaluation / "curves.npz", allow_pickle=False) as curves:
        data, ledger = tables.export(runs, curves, source_pairs, resolved, privacy_table(work),
                                     test_fraction=characteristics[frozen["config"]["shift_type"]]["fraction"])
    progress("Publishing scores, exact PR points, calibration bins, and risk histograms as CSV…")
    tables.validate(data)
    tables.write(output, data)
    tables.read(output)  # Check serialized values before deleting prediction caches.
    summary = {**result, "data": str(output / "data")}
    require(load_config(config).to_dict() == frozen["config"], "Evaluation configuration changed during execution")
    finish(output, "evaluate", config, summary, tables.FILES,
           resolved=frozen["config"], privacy=ledger, test_characteristics=characteristics,
           provenance={key: frozen[key] for key in ("code_hashes", "versions")},
           test_reference=reference,
           metric_definitions={"auprc": "Non-interpolated average precision; calculation unchanged",
                               "calibration_bias": "Mean predicted risk minus test prevalence",
                               "comparison_epsilon": "Grid budget; SGD is nonprivate",
                               "dp_amplification": "(DP baseline minus shifted AUPRC) minus (SGD baseline minus shifted AUPRC); positive means greater DP shift penalty",
                               "intervals": "95% Student t across paired repeats on a fixed test set",
                               "privacy": "Per private model, not jointly across released models"})
    verify_results(output)
    progress("CSV data verified; removing temporary arrays, checkpoints, and individual predictions…")
    cleanup(output)
    return summary


def verify_privacy(data, ledger, settings):
    from dp_shift.training.privacy import predicted_epsilon

    privacy = tables.privacy_frame(data, ledger)
    steps = PROTOCOL.steps_per_epoch * settings.selected.target_epochs
    sigma, rate = settings.noise_multiplier, 1 / PROTOCOL.steps_per_epoch
    epsilon = predicted_epsilon(sigma, rate, steps, PROTOCOL.delta)
    require(np.allclose(data["runs"].comparison_epsilon, settings.config.requested.privacy.epsilon, rtol=0, atol=1e-12),
            "Comparison epsilon mismatch")
    for row in privacy.itertuples():
        require(row.accountant == "prv" and row.steps == steps and row.delta == PROTOCOL.delta
                and row.noise_multiplier == sigma and row.sampling_rate == rate
                and row.clipping_norm == settings.config.requested.privacy.clipping_norm, "Privacy settings mismatch")
        history = json.loads(row.accountant_history)
        require(sum(entry[2] for entry in history) == steps
                and all(entry[0] == sigma and entry[1] == rate for entry in history), "Privacy history mismatch")
        require(abs(row.epsilon_achieved - epsilon) <= 1e-6
                and row.epsilon_achieved <= settings.config.requested.privacy.epsilon, "Privacy epsilon mismatch")


def verify_results(output):
    summary = verify(output, "evaluate")
    saved = protocol(output)["stages"]
    data = tables.read(output)
    scores, _, reference = tables.internal(data)
    require(len(scores) == summary["evaluated_models"] and reference["n"] == summary["test_patients"]
            and scores.repeat_id.nunique() == saved["prepare"]["config"]["repeats"], "Evaluation counts differ")
    expected = saved["evaluate"]["test_reference"]
    require(reference["n"] == expected["n"] and reference["positive_n"] == expected["positive_n"],
            "Test reference mismatch")
    trained = saved["train"]
    selected = trained["selected"]
    settings = TrainingSettings(resolve_config(trained["resolved"]["experiment"]),
                                SelectedHyperparameters(selected["source_lr"], selected["target_lr"], selected["target_epochs"]),
                                selected["noise_multiplier"])
    verify_privacy(data, saved["evaluate"]["privacy"], settings)
    return summary
