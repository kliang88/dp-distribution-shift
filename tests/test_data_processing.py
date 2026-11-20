"""Cohort construction, raw validation, and source-only preprocessing."""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest

from dp_shift.config import PROTOCOL
from dp_shift.data.cohort import build_cohort
from dp_shift.data.features import DIAGNOSIS_COLUMNS, FIXED_CATEGORIES, diagnosis_category
from dp_shift.data.preprocessing import fit_source_transformer, transform_features
from dp_shift.data.schema import load_and_validate_raw
from dp_shift.data.utils import file_hash


def make_frame(n):
    frame = pd.DataFrame({name: np.arange(n, dtype=float) + i
                          for i, name in enumerate(list(PROTOCOL.numeric_features))})
    for name in PROTOCOL.categorical_features:
        frame[name] = "1" if name in PROTOCOL.categorical_ids else "known"
    frame["age"] = "[60-70)"
    frame[list(DIAGNOSIS_COLUMNS)] = "250.83"
    frame["insulin"] = "No"
    frame["patient_nbr"] = np.arange(n, dtype=np.int64) + 1
    frame["encounter_id"] = np.arange(n, dtype=np.int64) + 1000
    frame["y"] = (np.arange(n) % 3 == 0).astype(np.int64)
    return frame


def save_raw(frame, path):
    frame = frame.copy()
    frame["readmitted"] = np.where(frame.y.eq(1), "<30", "NO")
    frame.drop(columns="y").to_csv(path, index=False)


def test_index_selection_precedes_exclusion():
    frame = make_frame(4)
    frame["patient_nbr"] = [1, 1, 2, 2]
    frame["encounter_id"] = [10, 20, 30, 40]
    frame["discharge_disposition_id"] = ["11", "1", "1", "1"]
    cohort, excluded, _ = build_cohort(frame.sample(frac=1, random_state=10))
    assert cohort.patient_nbr.tolist() == [2]
    assert cohort.encounter_id.tolist() == [30]
    assert excluded.patient_nbr.tolist() == [1]


def test_raw_parser_preserves_large_integer_ids_and_literal_categories(tmp_path):
    frame = make_frame(2)
    frame["patient_nbr"] = [9007199254740993, 9007199254740994]
    frame["race"] = ["None", "?"]
    frame["admission_type_id"] = ["1.0", "2"]
    frame["diag_1"] = ["001.2", "V45.85"]
    path = tmp_path / "records.csv"
    save_raw(frame, path)
    result = load_and_validate_raw(path, file_hash(path))
    assert result.patient_nbr.tolist() == frame.patient_nbr.tolist()
    assert result.admission_type_id.tolist() == ["1", "2"]
    assert result.race.iloc[0] == "None" and pd.isna(result.race.iloc[1])
    assert result.diag_1.tolist() == ["001.2", "V45.85"]
    assert result.y.tolist() == [1, 0]
    assert "readmitted" not in result


@pytest.mark.parametrize("feature", FIXED_CATEGORIES)
def test_raw_parser_requires_expanded_features(tmp_path, feature):
    path = tmp_path / "records.csv"
    save_raw(make_frame(2).drop(columns=feature), path)
    with pytest.raises(ValueError, match=f"Missing required columns: .*{feature}"):
        load_and_validate_raw(path, file_hash(path))


@pytest.mark.parametrize("code,category", [
    ("001.2", "Infectious"), ("139", "Infectious"), ("140", "Neoplasm"),
    ("249", "Other"), ("250.83", "Diabetes"), ("251", "Other"), ("272.4", "Other"),
    ("414.01", "Circulatory"), ("460", "Respiratory"), ("520", "Digestive"),
    ("580", "Genitourinary"), ("710", "Musculoskeletal"), ("786.50", "Symptoms"),
    ("999", "Injury"), (" v45.85 ", "V_code"), ("V91", "V_code"), ("E888.9", "E_code"),
    (None, "Missing"), (np.nan, "Missing"), ("?", "Missing"), (" n/a ", "Missing"),
    ("000", "Invalid"), ("ABC", "Invalid"), ("250.X", "Invalid"), ("1000", "Invalid"),
    ("V00", "Invalid"), ("V99", "Invalid"), ("E888.99", "Invalid"),
])
def test_diagnosis_category(code, category):
    assert diagnosis_category(code) == category


@pytest.mark.parametrize("column,value,error", [
    ("encounter_id", "1000", "Duplicate encounter"),
    ("patient_nbr", "?", "Missing patient"),
    ("readmitted", "unexpected", "unexpected readmitted"),
    ("time_in_hospital", "bad", "Malformed numerical"),
    ("time_in_hospital", "-1", "Negative or nonfinite"),
    ("time_in_hospital", "inf", "Negative or nonfinite"),
])
def test_raw_parser_rejects_invalid_data(tmp_path, column, value, error):
    path = tmp_path / "records.csv"
    save_raw(make_frame(2), path)
    frame = pd.read_csv(path, dtype=str)
    frame.loc[1, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        load_and_validate_raw(path, file_hash(path))


def test_raw_fingerprint_and_duplicate_headers(tmp_path):
    path = tmp_path / "records.csv"
    save_raw(make_frame(2), path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_and_validate_raw(path, "wrong fingerprint")
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + ",patient_nbr\n" + "\n".join(line + ",1" for line in lines[1:]) + "\n")
    with pytest.raises(ValueError, match="Duplicate CSV column"):
        load_and_validate_raw(path, file_hash(path))


def test_preprocessing_uses_source_statistics_and_survives_unseen_categories():
    source = make_frame(4)
    source["number_outpatient"] = [0., 1., 3., 7.]
    source.loc[0, "time_in_hospital"] = np.nan
    transformer = fit_source_transformer(source)
    numeric = transformer.named_transformers_["numeric"]
    imputer, scaler = numeric.named_steps["impute"], numeric.named_steps["scale"]
    before = [value.copy() for value in (imputer.statistics_, scaler.mean_, scaler.scale_)]
    # Compute the expected statistics independently in the prescribed order.
    values = source[list(PROTOCOL.numeric_features)].to_numpy().copy()
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isnan(values), medians, values)
    for name in PROTOCOL.log_features:
        index = list(PROTOCOL.numeric_features).index(name)
        values[:, index] = np.log1p(values[:, index])
    np.testing.assert_allclose(before[0], medians)
    np.testing.assert_allclose(before[1], values.mean(axis=0))
    np.testing.assert_allclose(before[2], values.std(axis=0))
    np.testing.assert_allclose(transform_features(transformer, source)[:, :8],
                               (values - values.mean(axis=0)) / values.std(axis=0), atol=1e-6)
    target = make_frame(2)
    target["race"] = ["unseen", np.nan]
    target.loc[0, "time_in_hospital"] = 1e8
    target.loc[1, "number_outpatient"] = np.nan
    transformed = transform_features(transformer, target)
    assert np.isfinite(transformed).all() and transformed.shape[1] == len(transformer.get_feature_names_out())
    for actual, expected in zip((imputer.statistics_, scaler.mean_, scaler.scale_), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    serialized = BytesIO()
    joblib.dump(transformer, serialized)
    serialized.seek(0)
    np.testing.assert_array_equal(transformed, transform_features(joblib.load(serialized), target))
    source["num_medications"] = np.nan
    with pytest.raises(ValueError, match="entire source numerical"):
        fit_source_transformer(source)


def test_fixed_categories_encode_values_absent_from_source():
    source = make_frame(4)
    transformer = fit_source_transformer(source)
    feature_names = transformer.get_feature_names_out().tolist()
    target = make_frame(2)
    target["diag_1"] = ["V45.85", "414.01"]
    target["diag_2"] = ["E888.9", "272.4"]
    target["diag_3"] = ["bad", "?"]
    target["insulin"] = ["Up", "unknown"]
    transformed = transform_features(transformer, target)
    expected = {"diag_1": ["V_code", "Circulatory"], "diag_2": ["E_code", "Other"],
                "diag_3": ["Invalid", "Missing"], "insulin": ["Up", "Missing"]}
    for feature, categories in FIXED_CATEGORIES.items():
        names = [f"fixed_categorical__{feature}_{category}" for category in categories]
        indices = [feature_names.index(name) for name in names]
        np.testing.assert_array_equal(transformed[:, indices],
                                      np.equal.outer(expected[feature], categories))
