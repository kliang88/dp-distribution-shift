"""Descriptive raw-feature summaries; no performance evaluation."""

import pandas as pd

from .cohort import describe


def feature_summaries(pair, repeat_id, config):
    rows = []
    for condition in ["s0", "s1"]:
        for label in ["all", 0, 1]:
            frame = pair[condition]
            if label != "all":
                frame = frame.loc[frame.y.eq(label)]
            base = {"repeat_id": repeat_id, "condition": condition, "label": label, "scale": "raw"}
            for name in config["numeric_features"]:
                values = frame[name]
                rows.append({**base, "feature": name, "kind": "numeric", "n": len(frame),
                             "missing_n": int(values.isna().sum()), "mean": values.mean(),
                             "std": values.std(ddof=1)})
            for name in config["categorical_features"]:
                counts = frame[name].fillna("Missing").value_counts().sort_index()
                for category, count in counts.items():
                    rows.append({**base, "feature": name, "kind": "categorical", "n": len(frame),
                                 "category": category, "count": int(count), "frequency": count / len(frame)})
    return rows


def write_audits(root, frames, pairs, config):
    directory = root / "audit"
    directory.mkdir()
    pd.DataFrame([{"partition": name, **describe(frame)} for name, frame in frames.items()]).to_csv(
        directory / "partition_summary.csv", index=False)
    rows, features = [], []
    for repeat, pair in enumerate(pairs):
        for condition in ["s0", "s1"]:
            rows.append({"repeat_id": repeat, "condition": condition, **describe(pair[condition]),
                         "replaced_n": len(pair["removed"]),
                         "shared_n": config["pretrain_size"] - len(pair["removed"])})
        features.extend(feature_summaries(pair, repeat, config))
    pd.DataFrame(rows).to_csv(directory / "source_pair_summary.csv", index=False)
    pd.DataFrame(features).to_csv(directory / "source_feature_summaries.csv", index=False)
    pd.DataFrame([{"partition": name, "feature": feature, "missing_n": int(frame[feature].isna().sum())}
                  for name, frame in frames.items()
                  for feature in config["numeric_features"] + config["categorical_features"]]).to_csv(
        directory / "feature_missingness.csv", index=False)
