"""Exercise the stage CLI on synthetic patients, including reuse and failures."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dp_shift.cli import main
from dp_shift.data.utils import file_hash, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_help_does_not_import_training_or_plotting():
    script = """
import sys
from dp_shift.cli import main
try:
    main(['prepare', '--help'])
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'opacus', 'matplotlib'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--output" in result.stdout
    result = subprocess.run([sys.executable, "-m", "dp_shift", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ["prepare", "train", "evaluate", "all", "verify"])


def test_full_cli_reuses_training_resumes_evaluation_and_rejects_corruption(experiment, monkeypatch, capsys):
    from dp_shift.evaluation import score as scoring
    from dp_shift.evaluation.metrics import METRICS
    from dp_shift.stages import evaluate as publishing
    from dp_shift.training import train as training
    from dp_shift.training import tune as tuning

    data, output, config_path = experiment
    args = ["all", "--output", str(output), "--config", str(config_path)]
    main(["prepare", "--output", str(output), "--config", str(config_path)])
    assert not (output / "data/target_test.npz").exists()
    assert "train" not in read_json(output / "protocol.json")["stages"]
    for name in ["source_pool", "proxy_train", "proxy_validation", "target_train"]:
        with np.load(output / f"data/{name}.npz") as actual, np.load(data / f"arrays/{name}.npz") as expected:
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    with np.load(output / "data/source_pairs.npz") as pairs:
        assert len(pairs.files) == 4
        for repeat in range(2):
            for source in ("s0", "s1"):
                np.testing.assert_array_equal(pairs[f"repeat_{repeat:02d}_{source}"], np.load(
                    data / f"source_pairs/repeat_{repeat:02d}/{source}_indices.npy"))
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
    assert "EVALUATE FAILED: Synthetic interruption" in capsys.readouterr().err
    assert public_partitions == {"source_pool", "proxy_train", "proxy_validation"}
    assert read_json(output / "protocol.json")["stages"]["train"]["summary"]["target_jobs_complete"] == 12
    assert "evaluate" not in read_json(output / "protocol.json")["stages"]
    assert not (output / "data/target_test.npz").exists()
    first_cache = output / ".work/evaluation_v3/predictions/repeat_00/A.npz"
    first_mtime = first_cache.stat().st_mtime_ns
    checkpoint = output / ".work/training/repeat_00/A.pt"
    checkpoint_mtime = checkpoint.stat().st_mtime_ns
    readiness_hash = file_hash(output / ".work/evaluation_ready.json")
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--output", str(output), "--config", str(config_path)])
    assert error.value.code == 1
    assert "use dp-shift evaluate --resume" in capsys.readouterr().err

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
        original = pd.read_csv(root / ".work/evaluation_v3/runs.csv")
        metrics = pd.read_csv(root / "evaluation/metrics.csv")
        retained = pd.read_csv(root / "evaluation/supporting/runs.csv")
        np.testing.assert_allclose(retained[list(METRICS)], original[list(METRICS)], rtol=1e-12)
        assert len(metrics) == 6 and "record_type" not in metrics
        for row in metrics.itertuples():
            values = original.loc[original.path_id.eq(row.path_id), "average_precision"]
            assert row.average_precision_mean == pytest.approx(values.mean())
            assert row.average_precision_sd == pytest.approx(values.std(ddof=1))
        effects = pd.read_csv(root / "evaluation/effects.csv")
        assert len(effects) == 7
        with np.load(root / "evaluation/supporting/curves.npz", allow_pickle=False) as curves:
            assert len(curves.files) == 12 * 8
        for name in publishing.FIGURES:
            image = root / f"evaluation/figures/{name}.png"
            assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        privacy = pd.read_csv(root / "evaluation/privacy.csv")
        assert len(privacy) == 6 and privacy.epsilon_achieved.le(4).all()
        np.testing.assert_allclose(privacy.clipping_norm, .7)
        raise RuntimeError("Synthetic cleanup interruption")

    with monkeypatch.context() as patch:
        patch.setattr(training, "step_batch", fail_training)
        patch.setattr(publishing, "cleanup", interrupted_cleanup)
        with pytest.raises(SystemExit) as error:
            main(args + ["--resume"])
        assert error.value.code == 1
    assert "Synthetic cleanup interruption" in capsys.readouterr().err
    assert (output / ".work").exists()
    with monkeypatch.context() as patch:
        patch.setattr(scoring, "predict", lambda *a: pytest.fail("Completed evaluation must not run inference"))
        patch.setattr(training, "step_batch", fail_training)
        main(args)
        main(["verify", "all", "--output", str(output)])
    assert "EVALUATE VERIFIED" in capsys.readouterr().out
    assert not (output / ".work").exists()
    expected_files = {"protocol.json", "data/source_pairs.npz", "data/target_test.npz",
                      "evaluation/metrics.csv", "evaluation/effects.csv", "evaluation/privacy.csv",
                      "evaluation/supporting/runs.csv", "evaluation/supporting/curves.npz"}
    expected_files.update(f"data/{name}.npz" for name in
                          ["source_pool", "proxy_train", "proxy_validation", "target_train"])
    expected_files.update(f"evaluation/figures/{name}.png" for name in publishing.FIGURES)
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == expected_files

    metrics = output / "evaluation/metrics.csv"
    metrics.write_text(metrics.read_text() + "corrupted")
    with pytest.raises(SystemExit) as error:
        main(["verify", "evaluate", "--output", str(output)])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "EVALUATE FAILED" in captured.err and "artifact changed" in captured.err
    assert "Traceback" not in captured.err


def test_legacy_outputs_are_not_adopted_or_removed(tmp_path, capsys):
    output = tmp_path / "old"
    output.mkdir()
    marker = output / "evaluation_ready.json"
    marker.write_text("legacy experiment")
    with pytest.raises(SystemExit):
        main(["prepare", "--output", str(output)])
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


def test_completed_preparation_rejects_config_changes(experiment, capsys):
    data, output, _ = experiment
    config = data.parent / "configs/experiment.json"
    args = ["prepare", "--output", str(output), "--config", str(config)]
    main(args)
    original = file_hash(output / "data/source_pool.npz")
    changed = read_json(config)
    changed["seed"] += 1
    write_json(config, changed)
    with pytest.raises(SystemExit):
        main(args)
    assert "different settings" in capsys.readouterr().err
    assert file_hash(output / "data/source_pool.npz") == original


def test_missing_inputs_stop_with_concise_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out")])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "PREPARE FAILED" in captured.err
    assert "COMPLETE" not in captured.out


def test_changed_experiment_stops_before_training(experiment, monkeypatch, capsys):
    from dp_shift.stages import train as training

    data, output, config_path = experiment
    main(["prepare", "--output", str(output), "--config", str(config_path)])
    choices = read_json(config_path)
    choices["privacy"]["epsilon"] = 2.0
    write_json(config_path, choices)
    monkeypatch.setattr(training, "public_smoke", lambda *a: pytest.fail("Training must not start"))
    with pytest.raises(SystemExit):
        main(["train", "--output", str(output), "--config", str(config_path)])
    assert "different settings" in capsys.readouterr().err
    assert not (output / ".work/public_smoke").exists()


def test_preparation_records_resolved_protocol_and_rejects_changed_inputs(experiment, capsys):
    data, output, config_path = experiment
    records = data.parent / "data/records.csv"
    args = ["prepare", "--output", str(output), "--config", str(config_path)]
    main(args)
    record = read_json(output / "protocol.json")["stages"]["prepare"]
    assert record["resolved"]["protocol_version"] == "prevalence_shift_v3"
    assert record["resolved"]["experiment"] == read_json(config_path)
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
    with pytest.raises(SystemExit):
        main(args)
    assert "inputs changed" in capsys.readouterr().err
