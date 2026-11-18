"""Exercise experiments and CSV-only plotting, including recovery."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dp_shift.cli import main, parser
from dp_shift.config import resolve_config
from dp_shift.data.utils import file_hash, read_json, write_json
from dp_shift.stages.prepare import run as prepare


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_defaults_to_the_unified_config():
    assert parser().parse_args(["experiment"]).config == "configs/experiment.json"


@pytest.mark.parametrize("command", ["experiment", "plot"])
def test_help_does_not_import_training_or_plotting(command):
    script = f"""
import sys
from dp_shift.cli import main
try:
    main([{command!r}, '--help'])
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'opacus', 'matplotlib'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
    assert ("--input" if command == "plot" else "--config") in result.stdout
    result = subprocess.run([sys.executable, "-m", "dp_shift", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "{experiment,plot}" in result.stdout


@pytest.mark.parametrize("command", ["sweep", "prepare", "train", "evaluate", "all", "verify"])
def test_removed_commands_are_not_exposed(command):
    with pytest.raises(SystemExit) as error:
        parser().parse_args([command])
    assert error.value.code == 2


def test_full_cli_reuses_training_resumes_evaluation_and_rejects_corruption(experiment, monkeypatch, capsys):
    from dp_shift.evaluation import score as scoring
    from dp_shift.evaluation import tables
    from dp_shift.evaluation.plots import FIGURES
    from dp_shift.stages import evaluate as publishing
    from dp_shift.training import train as training
    from dp_shift.training import tune as tuning

    data, output, config_path = experiment
    args = ["experiment", "--output", str(output), "--config", str(config_path)]
    cell = output / ".work/readmission/fraction_0.1/epsilon_4/result"
    original_predict = scoring.predict
    original_load = tuning.load_training_data
    public_partitions = set()

    def public_only(root, partition, repeat_id=None):
        assert partition in {"source_pool", "proxy_train", "proxy_validation"}
        public_partitions.add(partition)
        return original_load(root, partition, repeat_id)

    calls = []

    def interrupt(model, data):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("Synthetic interruption")
        return original_predict(model, data)

    with monkeypatch.context() as patch:
        patch.setattr(scoring, "predict", interrupt)
        patch.setattr(tuning, "load_training_data", public_only)
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 1
    assert "EXPERIMENT FAILED: Synthetic interruption" in capsys.readouterr().err
    assert public_partitions == {"source_pool", "proxy_train", "proxy_validation"}
    assert read_json(cell / "protocol.json")["stages"]["train"]["summary"]["target_jobs_complete"] == 12
    assert "evaluate" not in read_json(cell / "protocol.json")["stages"]
    for name in ["source_pool", "proxy_train", "proxy_validation", "target_train"]:
        with np.load(cell / f".work/data/arrays/{name}.npz") as actual, np.load(data / f"arrays/{name}.npz") as expected:
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    assert not (output / "data/target_test.npz").exists()
    first_cache = cell / ".work/evaluation_v4/predictions/repeat_00/A.npz"
    first_mtime = first_cache.stat().st_mtime_ns
    checkpoint = cell / ".work/training/repeat_00/A.pt"
    checkpoint_mtime = checkpoint.stat().st_mtime_ns
    readiness_hash = file_hash(cell / ".work/evaluation_ready.json")

    def fail_training(*args, **kwargs):
        pytest.fail("Completed training jobs must be reused")

    def interrupted_cleanup(root):
        # Check numerical equivalence before temporary evidence is discarded.
        for repeat in range(2):
            metadata = {path: read_json(root / f".work/training/repeat_{repeat:02d}/{path}.json")["metadata"]
                        for path in ("A", "B-S0", "B-S1", "C", "D-S0", "D-S1")}
            assert metadata["A"]["initial_state_sha256"] == metadata["C"]["initial_state_sha256"]
            for condition in ("S0", "S1"):
                for key in ("source_checkpoint_sha256", "initial_state_sha256"):
                    assert metadata[f"B-{condition}"][key] == metadata[f"D-{condition}"][key]
            assert len({row["target_membership_sha256"] for row in metadata.values()}) == 1
        assert first_cache.stat().st_mtime_ns == first_mtime
        assert checkpoint.stat().st_mtime_ns == checkpoint_mtime
        assert file_hash(root / ".work/evaluation_ready.json") == readiness_hash
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        exported = tables.read(root / tables.RUNS_FILE)
        assert len(exported) == 12 and exported.test_size.eq(80).all()
        with np.load(root / ".work/data/evaluation/target_test.npz") as test:
            for row in exported.itertuples():
                path_id = tables.PATHS[row.method, row.pretraining]
                cache = root / f".work/evaluation_v4/predictions/repeat_{row.repeat:02d}/{path_id}.npz"
                with np.load(cache) as predictions:
                    labels, probability = test["y"], predictions["probability"]
                    assert row.auprc == average_precision_score(labels, probability)
                    assert row.auroc == roc_auc_score(labels, probability)
                    assert row.brier == brier_score_loss(labels, probability)
        ledger = read_json(root / "protocol.json")["stages"]["evaluate"]["privacy"]
        assert len(ledger) == 6
        assert all(row["epsilon_achieved"] <= 4 and row["clipping_norm"] == .7 for row in ledger.values())
        assert not (root / "plots").exists()
        raise RuntimeError("Synthetic cleanup interruption")

    with monkeypatch.context() as patch:
        patch.setattr(training, "step_batch", fail_training)
        patch.setattr(publishing, "cleanup", interrupted_cleanup)
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 1
    assert "Synthetic cleanup interruption" in capsys.readouterr().err
    assert (output / ".work").exists()
    with monkeypatch.context() as patch:
        patch.setattr(scoring, "predict", lambda *a: pytest.fail("Completed evaluation must not run inference"))
        patch.setattr(training, "step_batch", fail_training)
        main(args)
        patch.setattr("dp_shift.data.prepare.DEFAULT_RECORDS", data.parent / "missing.csv")
        main(args)
    assert "EXPERIMENT COMPLETE" in capsys.readouterr().out
    assert not (output / ".work").exists()
    assert read_json(output / "protocol.json")["configuration"] == read_json(config_path)
    expected_files = {"protocol.json", tables.RUNS_FILE}
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == expected_files
    csv_hash = file_hash(output / tables.RUNS_FILE)
    with monkeypatch.context() as patch:
        patch.setattr(scoring, "predict", lambda *a: pytest.fail("Plotting must not run inference"))
        patch.setattr(training, "step_batch", fail_training)
        main(["plot", "--input", str(output)])
    assert {p.name for p in (output / "plots").iterdir()} == {f"{name}.png" for name in FIGURES}
    for name in FIGURES:
        assert (output / f"plots/{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert csv_hash == file_hash(output / tables.RUNS_FILE)

    runs_csv = output / "data/runs.csv"
    runs_csv.write_text(runs_csv.read_text() + "corrupted")
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "EXPERIMENT FAILED" in captured.err and "artifact changed" in captured.err
    assert "Traceback" not in captured.err


def test_legacy_outputs_are_not_adopted_or_removed(tmp_path, capsys):
    output = tmp_path / "old"
    output.mkdir()
    marker = output / "evaluation_ready.json"
    marker.write_text("legacy experiment")
    with pytest.raises(SystemExit):
        main(["experiment", "--output", str(output)])
    assert "choose a new --output" in capsys.readouterr().err
    assert marker.read_text() == "legacy experiment"
    assert not (output / ".work").exists()


def test_cleanup_refuses_unowned_work_directory(tmp_path):
    from dp_shift.stages.common import cleanup, finish, workspace

    output = tmp_path / "experiment"
    config = tmp_path / "config.json"
    write_json(config, {})
    work = workspace(output)
    for stage in ("prepare", "train", "evaluate"):
        finish(output, stage, config, {"complete": True}, [])
    marker = work / "important.txt"
    marker.write_text("retain this")
    write_json(work / "owner.json", {"output": "different experiment"})
    with pytest.raises(ValueError, match="unrecognized .work"):
        cleanup(output)
    assert marker.read_text() == "retain this"


def test_cleanup_recovers_if_deletion_is_interrupted(tmp_path, monkeypatch):
    from dp_shift.stages import common

    output = tmp_path / "experiment"
    config = tmp_path / "config.json"
    write_json(config, {})
    work = common.workspace(output)
    for stage in ("prepare", "train", "evaluate"):
        common.finish(output, stage, config, {"complete": True}, [])
    (work / "cache").mkdir()
    (work / "cache/predictions.txt").write_text("temporary")
    original = common.shutil.rmtree

    def interrupt(path):
        original(path)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(common.shutil, "rmtree", interrupt)
        with pytest.raises(KeyboardInterrupt):
            common.cleanup(output)
    assert (work / "owner.json").exists()
    common.cleanup(output)
    assert not work.exists()


def test_completed_preparation_rejects_config_changes(experiment):
    data, output, _ = experiment
    config = data.parent / "configs/experiment.json"
    prepare(output, config)
    original = file_hash(output / ".work/data/arrays/source_pool.npz")
    changed = read_json(config)
    changed["seed"] += 1
    write_json(config, changed)
    with pytest.raises(ValueError, match="different settings"):
        prepare(output, config)
    assert file_hash(output / ".work/data/arrays/source_pool.npz") == original


def test_missing_inputs_stop_with_concise_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["experiment", "--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out")])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "EXPERIMENT FAILED" in captured.err
    assert "COMPLETE" not in captured.out


def test_changed_experiment_stops_before_training(experiment, monkeypatch, capsys):
    from dp_shift import sweep
    from dp_shift.stages import train as training

    _, output, config_path = experiment
    args = ["experiment", "--output", str(output), "--config", str(config_path)]

    def interrupt(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(sweep, "reference_tuning", interrupt)
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 130
    capsys.readouterr()
    choices = read_json(config_path)
    choices["comparisons"]["epsilons"] = [2.]
    write_json(config_path, choices)
    monkeypatch.setattr(training, "public_smoke", lambda *a: pytest.fail("Training must not start"))
    with pytest.raises(SystemExit):
        main(args)
    assert "settings changed" in capsys.readouterr().err
    assert not (output / ".work/public_smoke").exists()


def test_preparation_records_resolved_protocol_and_rejects_changed_inputs(experiment):
    data, output, config_path = experiment
    records = data.parent / "data/records.csv"
    prepare(output, config_path)
    record = read_json(output / "protocol.json")["stages"]["prepare"]
    assert record["resolved"]["protocol_version"] == "source_shift_v1"
    assert record["resolved"]["experiment"] == resolve_config(read_json(config_path)).requested.to_dict()
    assert record["resolved"]["delta"] == 1e-5
    assert record["resolved"]["seeds"]["partition"] >= 0
    assert record["partitions"]["unused"]["n"] == 272
    assert record["partitions"]["target_train"]["n"] == 24
    assert len(record["membership_hashes"]) == 7
    assert len(record["source_pairs"]) == 4
    for row in record["source_pairs"]:
        if row["condition"] == "s1":
            assert row["positive_n"] == 4 and row["positive_rate"] == .1
    with records.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="inputs changed"):
        prepare(output, config_path)


def test_single_repeat_experiment_and_plots(experiment, monkeypatch):
    from dp_shift.evaluation import plots

    _, output, path = experiment
    choices = read_json(path)
    choices['repeats'] = 1
    write_json(path, choices)
    main(['experiment', '--config', str(path), '--output', str(output)])
    runs = pd.read_csv(output / 'data/runs.csv')
    assert len(runs) == 6 and runs.repeat.nunique() == 1
    original_save = plots.save
    captions = {}

    def inspect(figure, destination, name, caption, **kwargs):
        captions[name] = caption
        for axis in figure.axes:
            for container in axis.containers:
                if hasattr(container, 'has_yerr'):
                    assert not container.has_yerr
                if hasattr(container, 'errorbar'):
                    assert container.errorbar is None
        original_save(figure, destination, name, caption, **kwargs)

    monkeypatch.setattr(plots, 'save', inspect)
    main(['plot', '--input', str(output)])
    assert set(captions) == set(plots.FIGURES)
    for name in ('privacy_utility', 'source_target_prevalence', 'source_target_gap'):
        assert 'Single repeat. Confidence intervals require multiple repeats.' in captions.pop(name)
    assert all('Single repeat; uncertainty not estimated.' in caption for caption in captions.values())
