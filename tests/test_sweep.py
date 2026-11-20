"""Exercise a paired grid, recovery, and the shared CSV publication contract."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dp_shift.cli import main
from dp_shift.config import resolve_config, study_plan
from dp_shift.data.sampling import create_source_pair
from dp_shift.data.utils import file_hash, read_json, write_json
from dp_shift.sweep import experiment, verify


@pytest.mark.parametrize("section,key,value", [
    ("comparisons", "epsilons", []), ("comparisons", "epsilons", [True]),
    ("comparisons", "epsilons", [1, 1.]),
    ("tuning", "reference_epsilon", 0),
    (None, "base_config", "missing.json"),
])
def test_invalid_sweep_options_are_rejected(choices, section, key, value):
    (choices if section is None else choices[section])[key] = value
    with pytest.raises(ValueError, match=key):
        study_plan(choices)


def test_self_contained_config_and_cell_settings_do_not_mutate_choices(choices):
    original = deepcopy(choices)
    plan = study_plan(choices)
    cell = experiment(plan["base"], .2, 1.)
    assert choices == original
    assert cell["tuning"] == plan["base"]["tuning"]
    assert cell["privacy"]["epsilon"] == 1.
    assert plan["base"]["privacy"]["epsilon"] == 4.
    assert "base_config" not in plan["sweep"]


@pytest.mark.parametrize("mild_rate,severe_rate", [(.2, .1), (.75, .9)])
def test_replacements_are_nested_across_severities_and_independent_of_epsilon(base_config, mild_rate, severe_rate):
    patients = pd.DataFrame(dict(patient_nbr=np.arange(500), encounter_id=np.arange(500) + 1000,
                                 y=np.arange(500) % 2))
    for repeat in range(3):
        mild = create_source_pair(patients, repeat, resolve_config(experiment(base_config, mild_rate, 4.)))
        severe = create_source_pair(patients, repeat, resolve_config(experiment(base_config, severe_rate, 4.)))
        other_epsilon = create_source_pair(patients, repeat, resolve_config(experiment(base_config, severe_rate, 8.)))
        pd.testing.assert_frame_equal(mild["s0"], severe["s0"])
        for name in ("removed", "added"):
            assert set(mild[name].patient_nbr) < set(severe[name].patient_nbr)
        for name in severe:
            pd.testing.assert_frame_equal(severe[name], other_epsilon[name])
        assert mild["s1"].y.sum() == round(40 * mild_rate)
        assert severe["s1"].y.sum() == round(40 * severe_rate)


def test_equal_prevalence_keeps_the_exact_baseline_sample(base_config):
    patients = pd.DataFrame(dict(patient_nbr=np.arange(500), encounter_id=np.arange(500) + 1000,
                                 y=np.arange(500) % 2))
    config = resolve_config(experiment(base_config, .1, 4.))
    first = create_source_pair(patients, 0, config)
    config = replace(config, shift_member_count=int(first["s0"].y.sum()))
    equal = create_source_pair(patients, 0, config)
    pd.testing.assert_frame_equal(equal["s0"], equal["s1"])
    assert equal["removed"].empty and equal["added"].empty


@pytest.mark.parametrize("target", [10, 30])
def test_shift_can_use_all_available_patients_of_the_needed_class(base_config, target):
    patients = pd.DataFrame(dict(patient_nbr=np.arange(60), encounter_id=np.arange(60) + 1000,
                                 y=np.arange(60) % 2))
    config = resolve_config(experiment(base_config, target / 40, 4.))
    pair = create_source_pair(patients, 0, config)
    assert len(pair["s1"]) == 40 and pair["s1"].patient_nbr.is_unique
    assert pair["s1"].y.sum() == target
    label = int(target == 30)
    assert set(patients.loc[patients.y.eq(label), "patient_nbr"]) <= set(pair["s1"].patient_nbr)


def test_infeasible_grid_fails_before_tuning(experiment_inputs, choices, monkeypatch):
    from dp_shift import sweep

    output, path = experiment_inputs
    choices["comparisons"]["shifts"]["readmission"] = [.2, .9]
    choices["pretraining"]["sample_size"] = 400
    write_json(path, choices)
    monkeypatch.setattr(sweep, "reference_tuning", lambda *args: pytest.fail("Infeasible grid reached tuning"))
    with pytest.raises(ValueError, match="positive patients; source pool has"):
        sweep.run(output, path, progress=lambda message: None)
    assert not output.exists()


def test_parallel_experiment_matches_serial_sgd_and_reuses_completed_cells(experiment_inputs, choices, capsys, monkeypatch):
    from dp_shift import sweep

    output, path = experiment_inputs
    choices["repeats"] = 1
    choices["comparisons"]["epsilons"] = [4., 8.]
    write_json(path, choices)
    serial = output.with_name("serial")
    args = ["experiment", "--config", str(path)]

    main([*args, "--output", str(serial)])
    main([*args, "--output", str(output), "--workers", "2"])
    assert "Running comparisons with 2 CPU workers" in capsys.readouterr().out
    expected = pd.read_csv(serial / "data/runs.csv")
    actual = pd.read_csv(output / "data/runs.csv")
    from dp_shift.evaluation.tables import KEY

    pd.testing.assert_frame_equal(actual[KEY], expected[KEY], check_exact=True)
    pd.testing.assert_frame_equal(actual.loc[actual.method.eq("sgd")],
                                  expected.loc[expected.method.eq("sgd")], check_exact=True)
    assert sweep.verify(output)["evaluated_models"] == len(expected)

    saved = read_json(output / "protocol.json")
    expected_saved = read_json(serial / "protocol.json")
    for cell, expected_cell in zip(saved["cells"], expected_saved["cells"]):
        assert cell["membership_hashes"] == expected_cell["membership_hashes"]
        assert cell["source_hashes"] == expected_cell["source_hashes"]

    monkeypatch.setattr(sweep, "run_cell", lambda *args, **kwargs: pytest.fail("Completed cells ran again"))
    main([*args, "--output", str(output), "--workers", "1"])
    assert read_json(output / "protocol.json") == saved


def test_grid_recovers_verifies_and_retains_csv_data(experiment_inputs, choices, monkeypatch, capsys):
    from dp_shift import sweep
    from dp_shift.evaluation.sweep import validate
    from dp_shift.evaluation import tables
    from dp_shift.stages import train

    output, path = experiment_inputs
    choices["comparisons"] = {"shifts": {"readmission": [.1, .6]}, "epsilons": [4., 8.]}
    write_json(path, choices)
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
    assert stages["train"]["config"]["tuning"] == study_plan(choices)["base"]["tuning"]
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
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {"protocol.json", tables.RUNS_FILE}
    saved = read_json(output / "protocol.json")
    assert saved["reference_tuning"]["selected"] == dict(source_lr=.01, target_lr=.01, target_epochs=1)
    assert len(saved["reference_tuning"]["candidates"]) == 5
    assert len(saved["cells"]) == 4
    for cell in saved["cells"]:
        assert cell["selected"]["target_epochs"] == 1
        assert cell["selected"]["source_lr"] == cell["selected"]["target_lr"] == .01
    assert len({tuple(sorted(c["membership_hashes"].items())) for c in saved["cells"]}) == 1
    runs = tables.read(output / tables.RUNS_FILE)
    for _, group in runs.groupby(["repeat", "comparison_epsilon"]):
        baseline = group.loc[group.pretraining.eq("baseline"), "source_fraction"].iloc[0]
        shifted = group.loc[group.pretraining.eq("shifted")]
        assert shifted.loc[shifted.shift_fraction.eq(.1), "source_fraction"].lt(baseline).all()
        assert shifted.loc[shifted.shift_fraction.eq(.6), "source_fraction"].gt(baseline).all()
    assert len(runs) == 4 * 6 * 2
    assert list(runs.columns) == tables.COLUMNS
    assert runs.test_size.eq(80).all()
    assert not runs.duplicated(tables.KEY).any()
    for cell in saved["cells"]:
        assert len(cell["privacy"]) == 6
        assert all(row["epsilon_achieved"] <= cell["comparison_epsilon"] for row in cell["privacy"].values())
    assert not (output / "plots").exists()
    csv_hash = file_hash(output / tables.RUNS_FILE)
    main(["plot", "--input", str(output / "data")])
    assert len(list((output / "plots").rglob("*.png"))) == 6
    assert (output / "plots/dp_amplification.png").is_file()
    assert csv_hash == file_hash(output / tables.RUNS_FILE)
    assert verify(output)["evaluated_models"] == 48
    with monkeypatch.context() as patch:
        patch.setattr(sweep, "reference_tuning", lambda *args: pytest.fail("Completed sweep tuned again"))
        main(args)

    # Verification uses retained scores/accounting, even without source files.
    changed = deepcopy(saved)
    next(iter(changed["cells"][0]["privacy"].values()))["epsilon_achieved"] = 100.
    with pytest.raises(ValueError, match="Privacy settings"):
        validate(runs, changed)
    changed_saved = deepcopy(saved)
    changed_saved["cells"][1]["membership_hashes"]["target_train"] = "wrong"
    with pytest.raises(ValueError, match="partitions differ"):
        validate(runs, changed_saved)
    (output / "data/runs.csv").write_text("corrupted\n")
    with pytest.raises(ValueError, match="artifact changed"):
        verify(output)
