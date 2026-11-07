"""Scalar scores and compact, reproducible curve data for each model run."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from dp_shift.data.utils import require


PAPER_METRICS = ("average_precision", "auroc", "brier", "calibration_bias")
METRICS = (*PAPER_METRICS, "mean_prediction", "test_prevalence")
CONDITIONS = {
    "A": ("sgd", "scratch"), "B-S0": ("sgd", "s0"), "B-S1": ("sgd", "s1"),
    "C": ("dp_sgd", "scratch"), "D-S0": ("dp_sgd", "s0"), "D-S1": ("dp_sgd", "s1"),
}
CURVE_FIELDS = ("precision", "recall", "calibration_n", "calibration_positive_n",
                "calibration_mean_prediction", "calibration_observed_rate",
                "histogram_edges", "histogram_n")


def model_label(path):
    method, initialization = CONDITIONS[path]
    return f"{'SGD' if method == 'sgd' else 'DP-SGD'} · {initialization.upper() if initialization != 'scratch' else 'Scratch'}"


def validate_probabilities(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    require(y.ndim == probability.ndim == 1 and y.shape == probability.shape and len(y) > 0,
            "Labels and probabilities must be nonempty aligned vectors")
    require(np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2, "Evaluation requires both binary classes")
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
            "Probabilities must be finite and in [0,1]")


def metric_values(y, probability):
    validate_probabilities(y, probability)
    y, p = np.asarray(y, dtype=np.float64), np.asarray(probability, dtype=np.float64)
    return {"average_precision": float(average_precision_score(y, p)),
            "auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "calibration_bias": float(p.mean() - y.mean()),
            "mean_prediction": float(p.mean()), "test_prevalence": float(y.mean())}


def curve_key(repeat, path, field):
    return f"repeat_{repeat:02d}/{path}/{field}"


def curve_values(y, probability, n_bins=10):
    """Quantile reliability bins, exact PR points, and fixed-bin risk distributions.

    Equal predictions stay together; duplicate quantile boundaries collapse.
    A constant predictor therefore produces one reliability point, not empty bins.
    No predictions are averaged across models or repeats.
    """
    validate_probabilities(y, probability)
    y, p = np.asarray(y, dtype=np.int64), np.asarray(probability, dtype=np.float64)
    require(type(n_bins) is int and n_bins > 0, "n_bins must be a positive integer")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    assigned = np.searchsorted(edges[1:-1], p, side="left")
    _, assigned = np.unique(assigned, return_inverse=True)
    count = np.bincount(assigned)
    positives = np.bincount(assigned, weights=y).astype(np.int64)
    precision, recall, _ = precision_recall_curve(y, p)
    histogram, histogram_edges = np.histogram(p, bins=np.linspace(0, 1, 51))
    return {
        "precision": precision, "recall": recall,
        "calibration_n": count, "calibration_positive_n": positives,
        "calibration_mean_prediction": np.bincount(assigned, weights=p) / count,
        "calibration_observed_rate": positives / count,
        "histogram_edges": histogram_edges, "histogram_n": histogram,
    }


def validate_curves(curves, runs, reference):
    """Check retained plotting data after prediction caches have been removed."""
    expected = {curve_key(row.repeat_id, row.path_id, field)
                for row in runs.itertuples() for field in CURVE_FIELDS}
    require(set(curves) == expected, "Missing or unexpected curve arrays")
    for row in runs.itertuples():
        data = {field: curves[curve_key(row.repeat_id, row.path_id, field)] for field in CURVE_FIELDS}
        precision, recall = data["precision"], data["recall"]
        require(precision.ndim == recall.ndim == 1 and precision.shape == recall.shape
                and len(precision) >= 2, "Invalid PR curve shape")
        require(np.isfinite(precision).all() and np.isfinite(recall).all()
                and ((precision >= 0) & (precision <= 1)).all()
                and ((recall >= 0) & (recall <= 1)).all()
                and (np.diff(recall) <= 0).all(), "Invalid PR curve values")
        require(recall[0] == 1 and recall[-1] == 0 and precision[-1] == 1,
                "Invalid PR curve endpoints")
        require(np.isclose(-np.sum(np.diff(recall) * precision[:-1]), row.average_precision),
                "PR curve and AP disagree")
        n, positive_n = data["calibration_n"], data["calibration_positive_n"]
        predicted, observed = data["calibration_mean_prediction"], data["calibration_observed_rate"]
        require(n.ndim == 1 and 0 < len(n) <= 10 and n.dtype.kind in "iu"
                and (n > 0).all() and n.sum() == reference["n"], "Invalid calibration bin counts")
        require(positive_n.shape == predicted.shape == observed.shape == n.shape
                and positive_n.dtype.kind in "iu" and ((positive_n >= 0) & (positive_n <= n)).all()
                and positive_n.sum() == reference["positive_n"], "Invalid calibration positive counts")
        require(np.isfinite(predicted).all() and ((predicted >= 0) & (predicted <= 1)).all()
                and (np.diff(predicted) >= 0).all() and np.allclose(observed, positive_n / n),
                "Invalid reliability curve")
        require(np.isclose(np.average(predicted, weights=n), row.mean_prediction),
                "Calibration bins and mean prediction disagree")
        require(np.array_equal(data["histogram_edges"], np.linspace(0, 1, 51))
                and data["histogram_n"].shape == (50,) and data["histogram_n"].dtype.kind in "iu"
                and (data["histogram_n"] >= 0).all() and data["histogram_n"].sum() == reference["n"],
                "Invalid prediction histogram")
