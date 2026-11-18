"""Registered traits, generic paired sampling, compatibility, and mixed sweeps."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import resolve_config
from dp_shift.data.sampling import create_source_pair
from dp_shift.data.shifts import SHIFTS, Shift, members
from dp_shift.data.utils import file_hash, read_json, write_json
from dp_shift.sweep import load_plan, preflight


BASE = Path(__file__).resolve().parents[1] / "configs/experiment.json"


def patients():
    index = np.arange(500)
    return pd.DataFrame(dict(patient_nbr=index, encounter_id=index + 1000, y=(index % 3 == 0).astype(int),
                             age=np.where(index % 2 == 0, "[70-80)", "[50-60)"),
                             number_inpatient=np.where(index % 5 == 0, 2, 0)))


def config(kind, fraction):
    choices = load_plan(BASE)["base"]
    choices["source_conditions"] = dict(pretrain_size=40, shift_type=kind, shift_fraction=fraction)
    return resolve_config(choices)


def test_indicators_use_raw_values_and_exact_thresholds():
    frame = pd.DataFrame(dict(age=["[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                              number_inpatient=[0, 1, 2, 3], y=[0, 1, 0, 1]))
    assert members(frame, "age_70_plus").tolist() == [False, True, True, True]
    assert members(frame, "prior_inpatient_2_plus").tolist() == [False, False, True, True]
    assert members(frame, "readmission").tolist() == [False, True, False, True]


def test_diagnosis_indicators_use_distinct_eligible_categories():
    frame = pd.DataFrame({
        "diag_1": ["414.01", "401", None, "V42.0", "300", "invalid", "250.83"],
        "diag_2": ["427", "486", "?", "E888.9", "301", "250", "250.01"],
        "diag_3": ["invalid", None, "E888", None, "414", "486", "?"],
    })
    assert members(frame, "primary_diagnosis_circulatory").tolist() == [True, True, False, False, False, False, False]
    assert members(frame, "clinical_complexity_2plus").tolist() == [False, True, False, True, False, True, False]


def test_diagnosis_swaps_round_counts_and_use_only_labels_with_available_matches():
    kind = "primary_diagnosis_circulatory"
    choices = config(kind, .0625)  # 2.5 patients rounds to 2.
    pool = patients()
    baseline = pool.sample(n=40, random_state=choices.seeds.s0)
    in_baseline = pool.index.isin(baseline.index)
    pool["diag_1"] = np.where(in_baseline & pool.y.eq(0), None, "414")
    pool.loc[~in_baseline & pool.y.eq(1), "diag_1"] = "250"
    pair = create_source_pair(pool, 0, choices)
    assert len(pair["removed"]) == len(pair["added"]) == 2
    assert pair["removed"].y.eq(0).all() and pair["added"].y.eq(0).all()
    assert pair["s0"].y.sum() == pair["s1"].y.sum()
    assert set(pair["added"].patient_nbr).isdisjoint(pair["s0"].patient_nbr)
    assert members(pair["s1"], kind).sum() - members(pair["s0"], kind).sum() == 2
    repeated = create_source_pair(pool, 0, choices)
    for name in pair:
        pd.testing.assert_frame_equal(pair[name], repeated[name])

    pool.loc[~in_baseline & pool.y.eq(0), "diag_1"] = "250"
    with pytest.raises(ValueError, match="at most 0 are possible"):
        create_source_pair(pool, 0, choices)


@pytest.mark.parametrize("kind,column,value", [
    ("age_70_plus", "age", None), ("age_70_plus", "age", "unknown"),
    ("prior_inpatient_2_plus", "number_inpatient", None),
    ("prior_inpatient_2_plus", "number_inpatient", -.1),
    ("prior_inpatient_2_plus", "number_inpatient", 1.5),
    ("prior_inpatient_2_plus", "number_inpatient", float("inf")),
])
def test_invalid_selected_trait_is_rejected(kind, column, value):
    frame = patients()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        members(frame, kind)


@pytest.mark.parametrize("kind,mild,severe", [("readmission", .5, .7), ("age_70_plus", .7, .9),
                                            ("prior_inpatient_2_plus", .4, .6), ("age_70_plus", .3, .1),
                                            ("prior_inpatient_2_plus", .1, .05)])
def test_one_sampler_preserves_labels_pairing_nesting_and_noop(kind, mild, severe):
    pool = patients()
    for repeat in range(2):
        first = create_source_pair(pool, repeat, config(kind, mild))
        second = create_source_pair(pool, repeat, config(kind, severe))
        baseline = create_source_pair(pool, repeat, config("readmission", .1))["s0"]
        pd.testing.assert_frame_equal(first["s0"], baseline)
        pd.testing.assert_frame_equal(second["s0"], baseline)
        for name in ("added", "removed"):
            assert set(first[name].patient_nbr) < set(second[name].patient_nbr)
        assert int(members(second["s1"], kind).sum()) == round(40 * severe)
        pd.testing.assert_series_equal(second["s1"].set_index("patient_nbr").y,
                                       pool.set_index("patient_nbr").loc[second["s1"].patient_nbr, "y"], check_names=False)
        assert set(second["s1"].y) == {0, 1}
        equal = config(kind, mild)
        equal = replace(equal, shift_member_count=int(members(baseline, kind).sum()))
        pair = create_source_pair(pool, repeat, equal)
        pd.testing.assert_frame_equal(pair["s0"], pair["s1"])
        assert pair["removed"].empty and pair["added"].empty


def test_legacy_scalar_config_normalizes_without_mutation():
    old = load_plan(BASE)["base"]
    old["source_conditions"] = dict(pretrain_size=20000, shifted_positive_rate=.01)
    original = deepcopy(old)
    assert resolve_config(old).requested.to_dict()["source_conditions"] == dict(pretrain_size=20000, shift_type="readmission", shift_fraction=.01)
    assert old == original


def test_enrichment_must_preserve_both_readmission_outcomes():
    pool = patients()
    choices = config("age_70_plus", .9)
    selected = create_source_pair(pool, 0, choices)
    pool["y"] = pool.patient_nbr.isin(selected["removed"].patient_nbr).astype(int)
    with pytest.raises(ValueError, match="both classes"):
        create_source_pair(pool, 0, choices)


def test_new_registry_entry_can_use_a_nonpredictor_raw_column(experiment, monkeypatch):
    from dp_shift.data.prepare import input_fingerprints
    from dp_shift.data.preprocessing import fit_source_transformer
    from dp_shift.data.schema import load_and_validate_raw

    _, _, path = experiment
    records = path.parent.parent / "data/records.csv"
    frame = pd.read_csv(records)
    frame["custom_flag"] = np.where(np.arange(len(frame)) % 2 == 0, "yes", "no")
    frame.to_csv(records, index=False)
    monkeypatch.setitem(SHIFTS, "custom", Shift("Custom group", ("custom_flag",), lambda data: data.custom_flag.eq("yes")))
    choices = read_json(path)
    choices["pretraining"]["sample_size"] = 40
    choices["comparisons"]["shifts"] = {"custom": [.7]}
    resolved = resolve_config(choices)
    parsed = load_and_validate_raw(records, resolved, input_fingerprints()["records"]["sha256"])
    assert "custom_flag" in parsed
    pair = create_source_pair(parsed, 0, resolved)
    assert members(pair["s1"], "custom").sum() == 28
    transformer = fit_source_transformer(parsed)
    assert all("custom_flag" not in name for name in transformer.get_feature_names_out())


def test_default_grid_is_feasible_on_local_patient_data():
    preflight(load_plan(BASE))


def test_preflight_validates_all_selected_traits_before_training(experiment):
    data_root, _, path = experiment
    records = path.parent.parent / "data/records.csv"
    target_ids = pd.read_csv(data_root / "manifests/target_test.csv").patient_nbr
    frame = pd.read_csv(records)
    frame.loc[frame.patient_nbr.eq(target_ids.iloc[0]), "age"] = "unknown"
    frame.to_csv(records, index=False)
    sweep = path.parent / "sweep.json"
    choices = read_json(path)
    choices["comparisons"]["shifts"] = {"age_70_plus": [.7]}
    write_json(sweep, choices)
    with pytest.raises(ValueError, match="valid ten-year age bins"):
        preflight(load_plan(sweep))


def test_infeasible_diagnosis_shift_is_audited_before_training(experiment, monkeypatch):
    from dp_shift.data.audit import SHIFT_AUDIT_FILE
    from dp_shift.sweep import run

    _, output, path = experiment
    choices = read_json(path)
    choices["comparisons"]["shifts"] = {"primary_diagnosis_circulatory": [.2]}
    write_json(path, choices)
    monkeypatch.setattr("dp_shift.sweep.reference_tuning", lambda *_: pytest.fail("Tuned an infeasible sweep"))
    with pytest.raises(ValueError, match="infeasible source draws"):
        run(output, path)
    audit = pd.read_csv(output / SHIFT_AUDIT_FILE)
    assert len(audit) == 4
    assert set(audit.condition) == {"s0", "s1"}
    assert audit.status.eq("infeasible").all()
    assert audit.requested_replacements.eq(8).all() and audit.replacements.eq(0).all()
    assert audit.shifted_subgroup_fraction.isna().all()
    assert audit.reason.str.contains("at most 0 are possible").all()


def test_mixed_sweep_preserves_controls_and_creates_five_faceted_figures(experiment, monkeypatch):
    from dp_shift.cli import main
    from dp_shift.data.audit import SHIFT_AUDIT_FILE
    from dp_shift.evaluation import tables, plots
    from dp_shift.sweep import run, verify

    _, output, path = experiment
    sweep = path.parent / "mixed.json"
    choices = read_json(path)
    choices["comparisons"]["shifts"] = {
        "readmission": [.1], "age_70_plus": [.7],
        "primary_diagnosis_circulatory": [.1], "clinical_complexity_2plus": [.2],
    }
    write_json(sweep, choices)
    records = path.parent.parent / "data/records.csv"
    frame = pd.read_csv(records, dtype={"diag_1": str, "diag_2": str})
    frame.loc[frame.patient_nbr.mod(4).eq(0), "diag_1"] = "414"
    frame.loc[frame.patient_nbr.mod(5).eq(0), "diag_2"] = "486"
    frame.to_csv(records, index=False)
    run(output, sweep, progress=lambda message: None)
    saved = read_json(output / "protocol.json")
    assert saved["evaluated_models"] == 48
    assert len({tuple(sorted(cell["membership_hashes"].items())) for cell in saved["cells"]}) == 1
    assert len({tuple(sorted({key: value for key, value in cell["source_hashes"].items() if key.endswith("_s0")}.items()))
                for cell in saved["cells"]}) == 1
    assert len({tuple(sorted(cell["selected"].items())) for cell in saved["cells"]}) == 1
    runs = tables.read(output / tables.RUNS_FILE)
    assert set(runs.shift_type) == set(choices["comparisons"]["shifts"])
    assert len(tables.cells(runs)) == 4
    assert not (output / ".work").exists()
    for (kind, rate, _), cell in tables.cells(runs):
        shifted = cell.loc[cell.pretraining.eq("shifted")]
        if SHIFTS[kind].replacement_rate:
            baseline = cell.loc[cell.pretraining.eq("baseline")]
            np.testing.assert_allclose(shifted.source_fraction.to_numpy() - baseline.source_fraction.to_numpy(), rate)
        else:
            np.testing.assert_allclose(shifted.source_fraction, shifted.shift_fraction)
    audit = pd.read_csv(output / SHIFT_AUDIT_FILE)
    assert audit.status.eq("complete").all() and len(audit) == 8
    assert audit.baseline_readmission_count.eq(audit.shifted_readmission_count).all()
    assert (audit.shifted_subgroup_count - audit.baseline_subgroup_count).eq(audit.replacements).all()
    assert file_hash(output / SHIFT_AUDIT_FILE) == saved["files"][SHIFT_AUDIT_FILE]
    age = runs.loc[runs.shift_type.eq("age_70_plus")]
    assert age.test_fraction.eq(saved["cells"][1]["test_characteristics"]["age_70_plus"]["fraction"]).all()
    assert not np.isclose(age.test_fraction.iloc[0], saved["cells"][1]["test_reference"]["test_prevalence"])
    csv_hash = file_hash(output / tables.RUNS_FILE)
    captured = {}
    original_save = plots.save

    def inspect(figure, directory, name, caption, **kwargs):
        captured[name] = len(figure.axes)
        if name in {"pretraining_benefit", "dp_amplification"}:
            label = "Source patients replaced (%)" if name == "dp_amplification" else "Source replacement rate"
            assert [axis.get_xlabel() for axis in figure.axes][2:] == [label] * 2
            for axis in figure.axes[2:]:
                assert axis.containers[0].lines[0].get_xdata()[0] == 0
        original_save(figure, directory, name, caption, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(plots, "save", inspect)
        main(["plot", "--input", str(output)])
    assert set(captured) == set(plots.FIGURES)
    assert set(captured.values()) == {4}
    main(["plot", "--input", str(output), "--output", str(output / "age_plots"), "--shift", "age_70_plus", "--fraction", ".7"])
    assert len(list((output / "age_plots").glob("*.png"))) == 5
    assert csv_hash == file_hash(output / tables.RUNS_FILE)
    assert verify(output)["evaluated_models"] == 48
