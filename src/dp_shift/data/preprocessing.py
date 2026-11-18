"""One serializable transformer fitted only on the public source pool."""

import hashlib

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from dp_shift.config import PROTOCOL

from .features import FIXED_CATEGORIES, normalize_fixed_categories
from .utils import require


def log_columns(values, indices):
    values = np.array(values, dtype=float, copy=True)
    values[:, indices] = np.log1p(values[:, indices])
    return values


def fit_source_transformer(source_pool):
    numeric = list(PROTOCOL.numeric_features)
    require(not source_pool[numeric].isna().all().any(), "An entire source numerical column is missing")
    indices = [numeric.index(name) for name in PROTOCOL.log_features]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log_columns, kw_args={"indices": indices},
                                        feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
            ("encode", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False,
                                     dtype=np.float32)),
        ]), list(PROTOCOL.categorical_features)),
        ("fixed_categorical", Pipeline([
            ("normalize", FunctionTransformer(normalize_fixed_categories, feature_names_out="one-to-one")),
            ("encode", OneHotEncoder(categories=[list(values) for values in FIXED_CATEGORIES.values()],
                                     sparse_output=False, dtype=np.float32)),
        ]), list(FIXED_CATEGORIES)),
    ], remainder="drop", sparse_threshold=0)
    transformer.fit(source_pool[[*numeric, *PROTOCOL.categorical_features, *FIXED_CATEGORIES]])
    return transformer


def transform_features(transformer, frame):
    values = np.asarray(transformer.transform(frame), dtype=np.float32)
    require(values.ndim == 2 and np.isfinite(values).all(), "Transformed features are not finite")
    return values


def transformer_metadata(transformer, source_pool):
    numeric = transformer.named_transformers_["numeric"]
    encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
    fixed_encoder = transformer.named_transformers_["fixed_categorical"].named_steps["encode"]
    names = transformer.get_feature_names_out().tolist()
    return {
        "feature_names": names,
        "input_dimension": len(names),
        "numeric_features": list(PROTOCOL.numeric_features),
        "log_features": list(PROTOCOL.log_features),
        "medians": numeric.named_steps["impute"].statistics_.tolist(),
        "means": numeric.named_steps["scale"].mean_.tolist(),
        "scales": numeric.named_steps["scale"].scale_.tolist(),
        "categories": {name: values.tolist() for name, values in zip(
            (*PROTOCOL.categorical_features, *FIXED_CATEGORIES),
            (*encoder.categories_, *fixed_encoder.categories_), strict=True)},
        "fit_patient_n": len(source_pool),
        "fit_patient_ids_sha256": hashlib.sha256(
            source_pool.patient_nbr.to_numpy(dtype="<i8").tobytes()).hexdigest(),
    }
