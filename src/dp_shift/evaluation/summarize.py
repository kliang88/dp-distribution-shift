"""Two research tables, with uncertainty across paired repeats on a fixed test set."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from dp_shift.config import PROTOCOL
from dp_shift.data.artifacts import PATH_IDS
from dp_shift.data.utils import require
from .metrics import CONDITIONS, METRICS, PAPER_METRICS


CONTRASTS = {
    "gain_sgd_baseline": ("Pretraining", "SGD: S0 − scratch"),
    "gain_sgd_shifted": ("Pretraining", "SGD: S1 − scratch"),
    "gain_dp_baseline": ("Pretraining", "DP-SGD: S0 − scratch"),
    "gain_dp_shifted": ("Pretraining", "DP-SGD: S1 − scratch"),
    "shift_penalty_sgd": ("Shift penalty", "SGD: S0 − S1"),
    "shift_penalty_dp": ("Shift penalty", "DP-SGD: S0 − S1"),
    "dp_amplification": ("DP amplification", "Shift penalty: DP-SGD − SGD"),
}


def validate_runs(runs, repeats):
    expected = {(repeat, path) for repeat in range(repeats) for path in PATH_IDS}
    actual = list(zip(runs.repeat_id, runs.path_id))
    require(len(actual) == len(set(actual)) == len(expected) and set(actual) == expected,
            "Missing or duplicate evaluation paths")
    require(runs.status.eq("complete").all(), "Incomplete evaluation path")
    require(np.isfinite(runs[list(METRICS)].to_numpy()).all(), "Missing/nonfinite model metrics")
    require(np.allclose(runs.test_prevalence, runs.test_prevalence.iloc[0], rtol=0, atol=1e-12),
            "Models were scored on different test prevalences")
    for path, (method, initialization) in CONDITIONS.items():
        group = runs.loc[runs.path_id.eq(path)]
        require(group.method.eq(method).all() and group.initialization.eq(initialization).all(),
                f"Incorrect model condition: {path}")
    for name in ("average_precision", "auroc", "brier", "mean_prediction", "test_prevalence"):
        require(runs[name].between(0, 1).all(), f"Metric outside [0,1]: {name}")
    require(np.allclose(runs.calibration_bias, runs.mean_prediction - runs.test_prevalence,
                        rtol=0, atol=1e-12), "Calibration bias disagrees with mean prediction")


def paired_effects(runs, repeats):
    validate_runs(runs, repeats)
    rows = []
    for repeat, group in runs.groupby("repeat_id", sort=True):
        ap = group.set_index("path_id").average_precision
        penalty_sgd, penalty_dp = ap["B-S0"] - ap["B-S1"], ap["D-S0"] - ap["D-S1"]
        rows.append({"repeat_id": int(repeat),
                     "gain_sgd_baseline": ap["B-S0"] - ap["A"],
                     "gain_sgd_shifted": ap["B-S1"] - ap["A"],
                     "gain_dp_baseline": ap["D-S0"] - ap["C"],
                     "gain_dp_shifted": ap["D-S1"] - ap["C"],
                     "shift_penalty_sgd": penalty_sgd, "shift_penalty_dp": penalty_dp,
                     "dp_amplification": penalty_dp - penalty_sgd})
    return pd.DataFrame(rows)


def interval(values, confidence=.95):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) >= 1 and np.isfinite(values).all(), "Invalid repeat values")
    if len(values) == 1:
        return {"n": 1, "mean": float(values[0]), "sd": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    margin = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "ci_lower": mean - margin, "ci_upper": mean + margin}


def classify_amplification(lower, upper, threshold=.005):
    require(np.isfinite([lower, upper]).all() and lower <= upper and threshold > 0, "Invalid interval or threshold")
    if upper < -threshold:
        return "negative"
    if lower > threshold:
        return "positive"
    if lower >= -threshold and upper <= threshold:
        return "small"
    return "inconclusive"


def effect_summary(effects):
    rows = []
    for name, (group, label) in CONTRASTS.items():
        summary = interval(effects[name], PROTOCOL.confidence_level)
        interpretation = np.nan
        if summary["n"] > 1 and name == "dp_amplification":
            interpretation = classify_amplification(summary["ci_lower"], summary["ci_upper"], PROTOCOL.practical_ap_threshold)
        elif summary["n"] > 1 and name in {"gain_sgd_baseline", "gain_dp_baseline"}:
            interpretation = "useful" if summary["ci_lower"] > PROTOCOL.practical_ap_threshold else "not_established"
        rows.append({"effect": name, "group": group, "contrast": label, **summary,
                     "interpretation": interpretation})
    return pd.DataFrame(rows)


def metric_summary(runs):
    rows = []
    for path, (method, initialization) in CONDITIONS.items():
        group = runs.loc[runs.path_id.eq(path)]
        row = {"path_id": path, "method": method, "initialization": initialization, "n": len(group)}
        for metric in PAPER_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_tables(runs, repeats):
    effects = paired_effects(runs, repeats)
    return {"metrics.csv": metric_summary(runs), "effects.csv": effect_summary(effects)}


def summarize(output):
    from .provenance import atomic_csv, require_open, verify_scored

    output = Path(output)
    require_open(output)
    frozen = verify_scored(output)
    runs = pd.read_csv(output / "runs.csv", float_precision="round_trip")
    for name, frame in summary_tables(runs, frozen["config"]["repeats"]).items():
        atomic_csv(output / name, frame)
    return {"summarized_models": len(runs), "paired_repeats": frozen["config"]["repeats"]}
