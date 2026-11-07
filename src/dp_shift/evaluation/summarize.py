"""Two research tables, with uncertainty across paired repeats on a fixed test set."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from dp_shift.data.artifacts import PATH_IDS
from dp_shift.data.utils import require
from .metrics import CONDITIONS, METRICS, PAPER_METRICS
from .provenance import atomic_csv, require_open, verify_scored


CONTRASTS = {
    "ap_gain_sgd_s0": ("Pretraining", "SGD: S0 − scratch"),
    "ap_gain_sgd_s1": ("Pretraining", "SGD: S1 − scratch"),
    "ap_gain_dp_s0": ("Pretraining", "DP-SGD: S0 − scratch"),
    "ap_gain_dp_s1": ("Pretraining", "DP-SGD: S1 − scratch"),
    "ap_shift_sgd": ("Source shift", "SGD: S1 − S0"),
    "ap_shift_dp": ("Source shift", "DP-SGD: S1 − S0"),
    "ap_dp_interaction": ("Interaction", "Source shift: DP-SGD − SGD"),
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
        shift_sgd, shift_dp = ap["B-S1"] - ap["B-S0"], ap["D-S1"] - ap["D-S0"]
        rows.append({"repeat_id": int(repeat),
                     "ap_gain_sgd_s0": ap["B-S0"] - ap["A"],
                     "ap_gain_sgd_s1": ap["B-S1"] - ap["A"],
                     "ap_gain_dp_s0": ap["D-S0"] - ap["C"],
                     "ap_gain_dp_s1": ap["D-S1"] - ap["C"],
                     "ap_shift_sgd": shift_sgd, "ap_shift_dp": shift_dp,
                     "ap_dp_interaction": shift_dp - shift_sgd})
    return pd.DataFrame(rows)


def interval(values, confidence=.95):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all(), "Invalid repeat values")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    margin = float(t.ppf((1 + confidence) / 2, len(values) - 1) * sd / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "ci_lower": mean - margin, "ci_upper": mean + margin}


def classify_interaction(lower, upper, threshold=.005):
    require(np.isfinite([lower, upper]).all() and lower <= upper and threshold > 0, "Invalid interval or threshold")
    if upper < -threshold:
        return "negative"
    if lower > threshold:
        return "positive"
    if lower >= -threshold and upper <= threshold:
        return "small"
    return "inconclusive"


def effect_summary(effects, config):
    rows = []
    for name, (group, label) in CONTRASTS.items():
        summary = interval(effects[name], config["confidence_level"])
        interpretation = np.nan
        if name == "ap_dp_interaction":
            interpretation = classify_interaction(summary["ci_lower"], summary["ci_upper"], config["practical_ap_threshold"])
        elif name in {"ap_gain_sgd_s0", "ap_gain_dp_s0"}:
            interpretation = "useful" if summary["ci_lower"] > config["practical_ap_threshold"] else "not_established"
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


def summary_tables(runs, config):
    effects = paired_effects(runs, config["repeats"])
    return {"metrics.csv": metric_summary(runs), "effects.csv": effect_summary(effects, config)}


def summarize(output):
    output = Path(output)
    require_open(output)
    frozen = verify_scored(output)
    runs = pd.read_csv(output / "runs.csv", float_precision="round_trip")
    for name, frame in summary_tables(runs, frozen["config"]).items():
        atomic_csv(output / name, frame)
    return {"summarized_models": len(runs), "paired_repeats": frozen["config"]["repeats"]}
