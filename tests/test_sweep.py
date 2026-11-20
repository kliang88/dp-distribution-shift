"""Exercise repeat-level reuse and publication on synthetic patients."""

from pathlib import Path

import pandas as pd
import pytest

from dp_shift import sweep
from dp_shift.config import load_config
from dp_shift.evaluation import tables
from dp_shift.training import run as training


def test_resume_reuses_preparation_tuning_and_completed_repeats(experiment_inputs, monkeypatch):
    output, config, records = experiment_inputs
    target_calls, source_calls = [], []
    train_target, pretrain_source = training.train_target, training.pretrain_source
    compute = sweep.compute_repeat

    def count_target(*args):
        target_calls.append(args[3])
        return train_target(*args)

    def count_source(*args):
        source_calls.append(1)
        return pretrain_source(*args)

    def interrupt_second(work, repeat, plan):
        if repeat == 2:
            raise KeyboardInterrupt
        return compute(work, repeat, plan)

    monkeypatch.setattr(training, "train_target", count_target)
    monkeypatch.setattr(training, "pretrain_source", count_source)
    with monkeypatch.context() as patch:
        patch.setattr(sweep, "compute_repeat", interrupt_second)
        with pytest.raises(KeyboardInterrupt):
            sweep.run(output, config, records)
    assert (output / ".work/repeat_01.json").is_file()
    assert not (output / "results.csv").exists()

    def fail(*args):
        pytest.fail("Resuming should reuse the prepared arrays and tuning settings")

    monkeypatch.setattr(sweep, "prepare_splits", fail)
    monkeypatch.setattr(sweep, "select_hyperparameters", fail)
    result = sweep.run(output, config, records)
    assert len(source_calls) == 18
    assert target_calls.count("sgd") == target_calls.count("dp_sgd") == 20
    assert result["evaluated_models"] == 40
    assert list(output.iterdir()) == [output / "results.csv"]
    scores = tables.read(output / "results.csv")
    tables.validate(scores, load_config(config))
    assert scores.loc[scores.shift_type.eq("scratch"), tables.SOURCE].isna().all().all()
    assert scores.loc[scores.method.eq("DP-SGD"), "epsilon_achieved"].between(0, 5).all()
    for _, repeat in scores.loc[scores.method.eq("SGD")].groupby("repeat"):
        baseline = repeat.loc[repeat.shift_type.eq("baseline")].iloc[0]
        for kind in ("readmission", "age", "circulatory", "complexity"):
            shifted = repeat.loc[repeat.shift_type.eq(kind)]
            count = f"source_{kind}_count"
            assert (shifted[count] - baseline[count]).eq(shifted.shift_pp * baseline.source_size / 100).all()
            if kind != "readmission":
                assert shifted.source_readmission_count.eq(baseline.source_readmission_count).all()


def test_parallel_repeats_match_serial_sgd(experiment_inputs):
    output, config, records = experiment_inputs
    serial = output.with_name("serial")
    sweep.run(serial, config, records, workers=1)
    sweep.run(output, config, records, workers=2)
    one, many = (tables.read(path / "results.csv") for path in (serial, output))
    pd.testing.assert_frame_equal(one[tables.MODEL_KEYS], many[tables.MODEL_KEYS])
    pd.testing.assert_frame_equal(one.loc[one.method.eq("SGD")], many.loc[many.method.eq("SGD")], check_exact=True)
    assert Path(output / "results.csv").read_text().splitlines()[0] == ",".join(tables.COLUMNS)
