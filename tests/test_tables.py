"""Check the public score CSV format and plotting without training."""

import csv
from io import StringIO
import subprocess
import sys

import pandas as pd
import pytest

from dp_shift.evaluation import tables


@pytest.fixture
def runs():
    return pd.DataFrame([
        dict(shift_type="readmission", shift_fraction=.04, comparison_epsilon=5, repeat=repeat,
             method=method, pretraining=initialization, auprc=.12345678901234567, auroc=.61, brier=.12,
             source_fraction={"none": None, "baseline": .09, "shifted": .04}[initialization],
             source_size=None if initialization == "none" else 100, test_fraction=.086, test_size=200)
        for repeat in range(2) for method, initialization in tables.CONDITIONS.values()
    ])


def test_csv_preserves_scores_and_formats_cohort_sizes_as_integers(runs, tmp_path):
    path = tmp_path / "runs.csv"
    tables.write(path, runs)
    expected = runs.sort_values(tables.KEY, ignore_index=True)
    pd.testing.assert_frame_equal(tables.read(path), expected, check_exact=True)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == tables.COLUMNS
    assert {row["source_size"] for row in rows} == {"", "100"}
    assert {row["test_size"] for row in rows} == {"200"}
    assert all(not row["source_fraction"] for row in rows if row["pretraining"] == "none")


@pytest.mark.parametrize("column,value,match", [
    ("auprc", 1.1, "AUPRC"),
    ("auroc", -0.1, "AUROC"),
    ("brier", float("nan"), "BRIER"),
    ("method", "unknown", "Unknown method"),
    ("repeat", .5, "repeat numbers"),
    ("source_size", 40, "Scratch source fields"),
    ("test_fraction", .1, "Test fraction changed"),
    ("test_size", 5000, "Test size changed"),
    ("comparison_epsilon", -1, "comparison epsilon"),
])
def test_invalid_model_rows_are_rejected(runs, column, value, match):
    runs[column] = runs[column].astype(object if isinstance(value, str) else float)
    runs.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        tables.validate(runs)


def test_missing_and_duplicate_rows_are_rejected(runs):
    with pytest.raises(ValueError, match="Missing model"):
        tables.validate(runs.iloc[:-1])
    with pytest.raises(ValueError, match="Duplicate model"):
        tables.validate(pd.concat([runs, runs.iloc[:1]]))


def test_legacy_csv_schema_is_rejected(runs):
    old = runs.drop(columns="test_size").assign(run_id=range(len(runs)))
    with pytest.raises(ValueError, match="Invalid runs.csv schema"):
        tables.read(StringIO(old.to_csv(index=False)))


def test_plot_uses_only_runs_csv_without_training_packages_or_protocol(runs, tmp_path):
    path = tmp_path / "runs.csv"
    tables.write(path, runs)
    original = path.read_bytes()
    script = '''
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "opacus"} or name.startswith("dp_shift.training"):
        raise AssertionError("Plotting imported training code: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from dp_shift.cli import main
import sys
main(["plot", "--input", sys.argv[1]])
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "plots").glob("*.png"))) == 6
    assert {file.name for file in tmp_path.iterdir()} == {"runs.csv", "plots"}
    assert path.read_bytes() == original


def test_plot_rejects_a_destination_inside_data(tmp_path):
    from dp_shift.evaluation.report import run

    with pytest.raises(ValueError, match="separate"):
        run(tmp_path, tmp_path / "data/plots")
    with pytest.raises(ValueError, match="separate"):
        run(tmp_path, tmp_path)


@pytest.mark.parametrize("options", [{"epsilon": 999}, {"fraction": .999}])
def test_unknown_plot_selections_are_rejected(runs, monkeypatch, options):
    from dp_shift.evaluation import report

    monkeypatch.setattr(tables, "read", lambda path: runs)
    with pytest.raises(ValueError, match="must be one of"):
        report.run("unused", **options)


def test_single_repeat_plots_omit_confidence_intervals(runs, monkeypatch):
    from dp_shift.evaluation import plots, report

    single = runs.loc[runs.repeat.eq(0)]
    monkeypatch.setattr(tables, "read", lambda path: single)
    captions = {}

    def inspect(figure, destination, name, caption, **kwargs):
        try:
            captions[name] = caption
            for axis in figure.axes:
                for container in axis.containers:
                    assert not getattr(container, "has_yerr", False)
                    assert getattr(container, "errorbar", None) is None
        finally:
            plots.plt.close(figure)

    monkeypatch.setattr(plots, "save", inspect)
    report.run("unused")
    assert set(captions) == set(plots.FIGURES)
    for name in ("privacy_utility", "source_target_prevalence", "source_target_gap"):
        assert "Single repeat. Confidence intervals require multiple repeats." in captions.pop(name)
    assert all("Single repeat; uncertainty not estimated." in caption for caption in captions.values())
