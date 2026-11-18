"""Score trained models and publish one CSV with the privacy ledger in metadata."""

from pathlib import Path

import pandas as pd

from dp_shift.config import PROTOCOL, load_config
from dp_shift.data.utils import read_json, require
from dp_shift.evaluation import tables
from dp_shift.evaluation.accounting import model_key, verify_privacy
from dp_shift.evaluation.provenance import freeze
from dp_shift.evaluation.score import score_models
from .common import cleanup, completed, finish, protocol, verify, workspace


def privacy_ledger(work):
    ledger = {}
    for path in sorted((work / "training").glob("repeat_*/*.json")):
        saved = read_json(path)
        metadata = saved["metadata"]
        if metadata["privacy"] is not None:
            key = model_key(metadata["repeat_id"], metadata["path_id"])
            ledger[key] = {**metadata["privacy"], "checkpoint_sha256": saved["checkpoint_sha256"]}
    return ledger


def model_rows(scores, prepared, config, reference, characteristics):
    source = config["source_conditions"]
    kind = source["shift_type"]
    samples = {(row["repeat_id"], row["condition"]): row for row in prepared["source_pairs"]}
    rows = []
    for (repeat, path_id), metrics in scores.items():
        method, initialization = tables.CONDITIONS[path_id]
        condition = {"baseline": "s0", "shifted": "s1"}.get(initialization)
        sample = samples[repeat, condition] if condition else None
        rows.append({
            "shift_type": kind, "shift_fraction": source["shift_fraction"],
            "comparison_epsilon": config["privacy"]["epsilon"], "repeat": repeat,
            "method": method, "pretraining": initialization, **metrics,
            "source_fraction": sample["shift_fraction"] if sample else None,
            "source_size": sample["n"] if sample else None,
            "test_fraction": characteristics[kind]["fraction"], "test_size": reference["n"],
        })
    return pd.DataFrame(rows, columns=tables.COLUMNS)


def verify_cell(runs, trained, evaluated):
    config = trained["resolved"]["experiment"]
    source, privacy = config["source_conditions"], config["privacy"]
    kind = source["shift_type"]
    require(runs.shift_type.eq(kind).all() and runs.shift_fraction.eq(source["shift_fraction"]).all()
            and runs.comparison_epsilon.eq(privacy["epsilon"]).all(), "Comparison settings mismatch")
    reference = evaluated["test_reference"]
    require(runs.repeat.nunique() == config["repeats"] and runs.test_size.eq(reference["n"]).all()
            and runs.test_fraction.eq(evaluated["test_characteristics"][kind]["fraction"]).all(),
            "Evaluation counts differ")
    verify_privacy(runs, evaluated["privacy"], trained["selected"], privacy["clipping_norm"], privacy["epsilon"])


def run(output, config, resume=False, progress=print):
    output = Path(output)
    existing = completed(output, "evaluate", config)
    if existing is not None:
        progress("Verifying retained CSV data; no inference needed.")
        return verify_and_cleanup(output)

    verify(output, "train")
    work = workspace(output)
    evaluation = work / PROTOCOL.evaluation_version
    progress("Scoring models from verified test predictions…")
    frozen = freeze(work / "data", work, evaluation, config)
    scores, reference = score_models(evaluation, resume=resume)
    stages = protocol(output)["stages"]
    characteristics = read_json(work / "data/evaluation/characteristics.json")
    runs = model_rows(scores, stages["prepare"], frozen["config"]["experiment"], reference, characteristics)
    evaluated = {"privacy": privacy_ledger(work), "test_characteristics": characteristics,
                 "test_reference": reference}
    tables.validate(runs)
    verify_cell(runs, stages["train"], evaluated)
    require(load_config(config).to_dict() == frozen["config"], "Evaluation configuration changed during execution")

    progress("Publishing model scores and cohort counts to runs.csv…")
    tables.write(output / tables.RUNS_FILE, runs)
    summary = {"evaluated_models": len(runs), "test_patients": reference["n"],
               "repeats": frozen["config"]["repeats"], "data": str(output / "data")}
    finish(output, "evaluate", config, summary, [tables.RUNS_FILE], **evaluated,
           provenance={key: frozen[key] for key in ("code_hashes", "versions")})
    progress("CSV data verified; removing temporary arrays, checkpoints, and individual predictions…")
    cleanup(output)
    return summary


def verify_and_cleanup(output):
    summary = verify(output, "evaluate")
    stages = protocol(output)["stages"]
    runs = tables.read(output / tables.RUNS_FILE)
    verify_cell(runs, stages["train"], stages["evaluate"])
    require(len(runs) == summary["evaluated_models"], "Evaluation model count differs")
    cleanup(output)
    return summary
