"""Exercise a paired grid, recovery, and the shared CSV publication contract."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_shift.cli import main
from dp_shift.config import resolve_config
from dp_shift.data.sampling import create_source_pair
from dp_shift.data.utils import file_hash, read_json, write_json
from dp_shift.sweep import experiment, load_plan, verify


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment.json"


def sweep_config(path, base=BASE_CONFIG):
    choices = read_json(base)
    choices["comparisons"] = {"shifts": {"readmission": [.1, .6]}, "epsilons": [4., 8.]}
    choices["tuning"]["reference_epsilon"] = 4.
    write_json(path, choices)
    return choices


@pytest.mark.parametrize("section,key,value", [
    ("comparisons", "epsilons", []), ("comparisons", "epsilons", [True]),
    ("comparisons", "epsilons", [1, 1.]), ("comparisons", "epsilons", [-1]),
    ("tuning", "reference_epsilon", 0),
    ("comparisons", "shifts", {"readmission": [1]}),
    ("comparisons", "shifts", {"readmission": [.010001]}),
    (None, "base_config", "missing.json"), (None, "unknown", 1),
])
def test_invalid_sweep_stops_before_output_creation(tmp_path, section, key, value, capsys):
    path, output = tmp_path / "study.json", tmp_path / "output"
    choices = sweep_config(path)
    (choices if section is None else choices[section])[key] = value
    write_json(path, choices)
    with pytest.raises(SystemExit) as error:
        main(["experiment", "--config", str(path), "--output", str(output)])
    assert error.value.code == 1
    assert "EXPERIMENT FAILED" in capsys.readouterr().err
    assert not output.exists()


def test_self_contained_config_and_cell_settings_do_not_mutate_choices(tmp_path):
    path = tmp_path / "study.json"
    choices = sweep_config(path)
    original = deepcopy(choices)
    plan = load_plan(path)
    cell = experiment(plan["base"], .04, 1.)
    assert read_json(path) == original
    assert cell["tuning"] == plan["base"]["tuning"]
    assert cell["privacy"]["epsilon"] == 1.
    assert "base_config" not in plan["sweep"]


@pytest.mark.parametrize("mild_rate,severe_rate", [(.2, .1), (.75, .9)])
def test_replacements_are_nested_across_severities_and_independent_of_epsilon(mild_rate, severe_rate):
    base = load_plan(BASE_CONFIG)["base"]
    base["source_conditions"]["pretrain_size"] = 40
    patients = pd.DataFrame(dict(patient_nbr=np.arange(500), encounter_id=np.arange(500) + 1000,
                                 y=np.arange(500) % 2))
    for repeat in range(3):
        mild = create_source_pair(patients, repeat, resolve_config(experiment(base, mild_rate, 4.)))
        severe = create_source_pair(patients, repeat, resolve_config(experiment(base, severe_rate, 4.)))
        other_epsilon = create_source_pair(patients, repeat, resolve_config(experiment(base, severe_rate, 8.)))
        pd.testing.assert_frame_equal(mild["s0"], severe["s0"])
        for name in ("removed", "added"):
            assert set(mild[name].patient_nbr) < set(severe[name].patient_nbr)
        for name in severe:
            pd.testing.assert_frame_equal(severe[name], other_epsilon[name])
        assert mild["s1"].y.sum() == round(40 * mild_rate)
        assert severe["s1"].y.sum() == round(40 * severe_rate)


def test_equal_prevalence_keeps_the_exact_baseline_sample():
    base = load_plan(BASE_CONFIG)["base"]
    base["source_conditions"]["pretrain_size"] = 40
    patients = pd.DataFrame(dict(patient_nbr=np.arange(500), encounter_id=np.arange(500) + 1000,
                                 y=np.arange(500) % 2))
    config = resolve_config(experiment(base, .1, 4.))
    first = create_source_pair(patients, 0, config)
    config = replace(config, shift_member_count=int(first["s0"].y.sum()))
    equal = create_source_pair(patients, 0, config)
    pd.testing.assert_frame_equal(equal["s0"], equal["s1"])
    assert equal["removed"].empty and equal["added"].empty


@pytest.mark.parametrize("target", [10, 30])
def test_shift_can_use_all_available_patients_of_the_needed_class(target):
    base = load_plan(BASE_CONFIG)["base"]
    base["source_conditions"]["pretrain_size"] = 40
    patients = pd.DataFrame(dict(patient_nbr=np.arange(60), encounter_id=np.arange(60) + 1000,
                                 y=np.arange(60) % 2))
    config = resolve_config(experiment(base, target / 40, 4.))
    pair = create_source_pair(patients, 0, config)
    assert len(pair["s1"]) == 40 and pair["s1"].patient_nbr.is_unique
    assert pair["s1"].y.sum() == target
    label = int(target == 30)
    assert set(patients.loc[patients.y.eq(label), "patient_nbr"]) <= set(pair["s1"].patient_nbr)


def test_infeasible_grid_fails_before_tuning(experiment, monkeypatch):
    from dp_shift import sweep

    _, output, base = experiment
    path = base.parent / "sweep.json"
    choices = sweep_config(path, base)
    choices["comparisons"]["shifts"]["readmission"] = [.2, .9]
    choices["pretraining"]["sample_size"] = 400
    write_json(path, choices)
    monkeypatch.setattr(sweep, "reference_tuning", lambda *args: pytest.fail("Infeasible grid reached tuning"))
    with pytest.raises(ValueError, match="positive patients; source pool has"):
        sweep.run(output, path, progress=lambda message: None)
    assert not output.exists()


def test_grid_recovers_verifies_and_retains_csv_data(experiment, monkeypatch, capsys):
    from dp_shift import sweep
    from dp_shift.evaluation.sweep import validate, accounted_runs
    from dp_shift.evaluation import tables
    from dp_shift.evaluation.summarize import summary_tables
    from dp_shift.stages import train

    _, output, base = experiment
    path = base.parent / "sweep.json"
    sweep_config(path, base)
    args = ["experiment", "--config", str(path), "--output", str(output)]
    original_train = train.run
    calls = []

    def interrupt_second_cell(cell, *args, **kwargs):
        calls.append(cell)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original_train(cell, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train, "run", interrupt_second_cell)
        with pytest.raises(SystemExit) as error:
            main(args)
    assert error.value.code == 130
    assert "EXPERIMENT INTERRUPTED" in capsys.readouterr().err
    assert read_json(output / "protocol.json")["status"] == "running"
    first = output / ".work/readmission/fraction_0.1/epsilon_4/result"
    stages = read_json(first / "protocol.json")["stages"]
    assert set(stages) == {"prepare", "train", "evaluate"}
    assert stages["train"]["config"]["tuning"] == load_plan(path)["base"]["tuning"]
    assert stages["train"]["candidates"] == []
    assert stages["train"]["logistic_diagnostic"] is None
    assert not (first / ".work").exists()
    first_hash = file_hash(first / "data/runs.csv")

    def check_reused_cell(cell, *args, **kwargs):
        if cell == first:
            assert file_hash(first / "data/runs.csv") == first_hash
        return original_train(cell, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(train, "run", check_reused_cell)
        main(args)
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {"protocol.json", *tables.FILES}
    saved = read_json(output / "protocol.json")
    assert saved["reference_tuning"]["selected"] == dict(source_lr=.01, target_lr=.01, target_epochs=1)
    assert len(saved["reference_tuning"]["candidates"]) == 5
    assert len(saved["cells"]) == 4
    for cell in saved["cells"]:
        assert cell["selected"]["target_epochs"] == 1
        assert cell["selected"]["source_lr"] == cell["selected"]["target_lr"] == .01
    assert len({tuple(sorted(c["membership_hashes"].items())) for c in saved["cells"]}) == 1
    data = tables.read(output)
    runs = data["runs"]
    for _, group in runs.groupby(["repeat", "comparison_epsilon"]):
        baseline = group.loc[group.pretraining.eq("baseline"), "source_prevalence"].iloc[0]
        shifted = group.loc[group.pretraining.eq("shifted")]
        assert shifted.loc[shifted.shift_fraction.eq(.1), "source_prevalence"].lt(baseline).all()
        assert shifted.loc[shifted.shift_fraction.eq(.6), "source_prevalence"].gt(baseline).all()
    ledger = {key: value for cell in saved["cells"] for key, value in cell["privacy"].items()}
    internal = accounted_runs(data, ledger)
    assert len(runs) == 4 * 6 * 2
    assert list(runs.columns) == tables.COLUMNS["runs"]
    private = runs.loc[runs.method.eq("dp_sgd")]
    assert private.epsilon_achieved.le(private.comparison_epsilon).all()
    assert len({row["noise_multiplier"] for row in ledger.values()}) == 2
    assert runs.loc[runs.method.eq("sgd"), "epsilon_achieved"].isna().all()
    assert runs.run_id.is_unique
    assert not (output / "plots").exists()
    for _, cell in tables.cells(data):
        scores, curves, reference = tables.internal(cell)
        assert reference["n"] == 80
        assert len(curves) == 12 * 8
        effects = summary_tables(scores, 2)["effects.csv"]
        ap = scores.pivot(index="repeat_id", columns="path_id", values="average_precision")
        interaction = (ap["D-S0"] - ap["D-S1"]) - (ap["B-S0"] - ap["B-S1"])
        assert effects.loc[effects.effect.eq("dp_amplification"), "mean"].iloc[0] == pytest.approx(interaction.mean())
    hashes = {name: file_hash(output / name) for name in tables.FILES}
    main(["plot", "--input", str(output / "data")])
    assert len(list((output / "plots").rglob("*.png"))) == 5
    assert (output / "plots/dp_amplification.png").is_file()
    assert hashes == {name: file_hash(output / name) for name in tables.FILES}
    assert verify(output)["evaluated_models"] == 48
    with monkeypatch.context() as patch:
        patch.setattr(sweep, "reference_tuning", lambda *args: pytest.fail("Completed sweep tuned again"))
        main(args)

    # Verification uses retained scores/accounting, even without source files.
    changed = internal.copy()
    changed.loc[changed.method.eq("dp_sgd"), "epsilon_achieved"] = 100.
    with pytest.raises(ValueError, match="privacy settings"):
        validate(changed, saved)
    changed_saved = deepcopy(saved)
    changed_saved["cells"][1]["membership_hashes"]["target_train"] = "wrong"
    with pytest.raises(ValueError, match="partitions differ"):
        validate(internal, changed_saved)
    (output / "data/runs.csv").write_text("corrupted\n")
    with pytest.raises(ValueError, match="artifact changed"):
        verify(output)
