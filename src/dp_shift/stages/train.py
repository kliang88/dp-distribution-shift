"""Run the established training implementation inside the recovery workspace."""

from pathlib import Path

from dp_shift.config import load_config
from dp_shift.data.artifacts import verify_artifacts
from dp_shift.data.utils import read_json, require
from dp_shift.training.model import configure_cpu
from dp_shift.training.run import finalize, fit, pretrain
from dp_shift.training.tune import public_smoke, tune
from .common import completed, finish, verify, workspace


def run(output, config, progress=print):
    existing = completed(output, "train", config)
    if existing is not None:
        progress("Verified completed training.")
        return existing
    output = Path(output)
    verify(output, "prepare")
    work = workspace(output)
    data = work / "data"
    configure_cpu()
    progress("Verifying prepared data…")
    verify_artifacts(data)
    progress("Checking ordinary/private compatibility on public data…")
    smoke = public_smoke(data, work, config)
    progress("Selecting or verifying frozen public tuning settings…")
    frozen = tune(data, work, config)
    repeats = read_json(data / "config.json")["repeats"]
    for repeat in range(repeats):
        progress(f"Repeat {repeat + 1}/{repeats}: source pretraining (S0, S1)…")
        pretrain(data, work, [repeat])
        progress(f"Repeat {repeat + 1}/{repeats}: six target training paths…")
        fit(data, work, [repeat])
    progress("Verifying checkpoints, privacy accounting, and evaluation readiness…")
    result = finalize(data, work)
    summary = {key: value for key, value in result.items() if key != "evaluation_ready"}
    require(load_config(config) == frozen["training_config"], "Training configuration changed during execution")
    return finish(output, "train", config, summary, [],
                  resolved=frozen["training_config"],
                  selected={key: frozen["settings"][key] for key in
                            ("source_lr", "target_lr", "target_epochs", "noise_multiplier")},
                  candidates=frozen["candidate_results"], smoke=smoke,
                  logistic_diagnostic=read_json(work / "tuning/logistic_diagnostic.json"),
                  provenance={key: frozen[key] for key in ("code_hashes", "versions")})
