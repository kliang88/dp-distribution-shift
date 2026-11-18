"""Descriptive raw-feature summaries; no performance evaluation."""

import csv

import pandas as pd

from dp_shift.config import PARTITIONS, PROTOCOL
from dp_shift.stages.common import atomic_path

from .cohort import describe
from .features import DIAGNOSIS_COLUMNS, diagnosis_category
from .sampling import replacement_seed
from .shifts import describe as characteristics, members
from .utils import write_json


SHIFT_AUDIT_FILE = "source_shift_audit.csv"


def replacement_audit_rows(config, repeat, s0, s1=None, replacements=0, reason=""):
    source = config.requested.source_conditions
    size = len(s0)
    baseline_count = int(members(s0, source.shift_type).sum())
    shifted_count = int(members(s1, source.shift_type).sum()) if s1 is not None else None
    baseline_readmissions = int(s0.y.sum())
    shifted_readmissions = int(s1.y.sum()) if s1 is not None else None
    row = {
        "repeat": repeat, "shift_type": source.shift_type, "rate": source.shift_fraction,
        "source_N": size, "requested_replacements": config.source_replacement_count, "replacements": replacements,
        "baseline_subgroup_count": baseline_count, "shifted_subgroup_count": shifted_count,
        "baseline_subgroup_fraction": baseline_count / size,
        "shifted_subgroup_fraction": shifted_count / size if shifted_count is not None else None,
        "baseline_readmission_count": baseline_readmissions, "shifted_readmission_count": shifted_readmissions,
        "baseline_readmission_fraction": baseline_readmissions / size,
        "shifted_readmission_fraction": shifted_readmissions / size if shifted_readmissions is not None else None,
        "baseline_seed": config.seeds.s0 + repeat, "seed": replacement_seed(config, repeat),
        "status": "infeasible" if reason else "complete", "reason": reason,
    }
    return [{**row, "condition": condition} for condition in ("s0", "s1")]


def write_replacement_audit(path, rows):
    with atomic_path(path) as temporary, temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_summaries(pair, repeat_id):
    rows = []
    for condition in ["s0", "s1"]:
        for label in ["all", 0, 1]:
            frame = pair[condition]
            if label != "all":
                frame = frame.loc[frame.y.eq(label)]
            base = {"repeat_id": repeat_id, "condition": condition, "label": label, "scale": "raw"}
            for name in PROTOCOL.numeric_features:
                values = frame[name]
                rows.append({**base, "feature": name, "kind": "numeric", "n": len(frame),
                             "missing_n": int(values.isna().sum()), "mean": values.mean(),
                             "std": values.std(ddof=1)})
            for name in PROTOCOL.categorical_features:
                counts = frame[name].fillna("Missing").value_counts().sort_index()
                for category, count in counts.items():
                    rows.append({**base, "feature": name, "kind": "categorical", "n": len(frame),
                                 "category": category, "count": int(count), "frequency": count / len(frame)})
    return rows


def diagnosis_category_rows(frame, scope, partition=None, repeat=None, condition=None):
    rows = []
    for feature in DIAGNOSIS_COLUMNS:
        counts = frame[feature].value_counts(dropna=False)
        counts = counts.groupby(counts.index.map(diagnosis_category)).sum()
        for category in ("Missing", "Invalid", "Other"):
            count = int(counts.get(category, 0))
            rows.append({"scope": scope, "partition": partition, "repeat": repeat, "condition": condition,
                         "feature": feature, "category": category, "count": count,
                         "total": len(frame), "fraction": count / len(frame)})
    return rows


def write_audits(root, frames, pairs, config):
    directory = root / "audit"
    directory.mkdir()
    pd.DataFrame([{"partition": name, **describe(frame)} for name, frame in frames.items()]).to_csv(
        directory / "partition_summary.csv", index=False)
    rows, features, collateral = [], [], []
    diagnoses = diagnosis_category_rows(frames["cohort"], "cohort")
    for name in PARTITIONS:
        diagnoses.extend(diagnosis_category_rows(frames[name], "partition", partition=name))
    for repeat, pair in enumerate(pairs):
        for condition in ["s0", "s1"]:
            diagnoses.extend(diagnosis_category_rows(pair[condition], "source_sample", repeat=repeat,
                                                    condition=condition))
            mask = members(pair[condition], config.requested.source_conditions.shift_type)
            rows.append({"repeat_id": repeat, "condition": condition, **describe(pair[condition]),
                         "shift_type": config.requested.source_conditions.shift_type,
                         "shift_count": int(mask.sum()),
                         "shift_fraction": float(mask.mean()),
                         "replaced_n": len(pair["removed"]),
                         "shared_n": config.requested.source_conditions.pretrain_size - len(pair["removed"])})
            collateral.append({"repeat": repeat, "condition": condition, "traits": characteristics(pair[condition])})
        features.extend(feature_summaries(pair, repeat))
    pd.DataFrame(rows).to_csv(directory / "source_pair_summary.csv", index=False)
    if config.source_replacement_count is not None:
        shift_rows = [row for repeat, pair in enumerate(pairs)
                      for row in replacement_audit_rows(config, repeat, pair["s0"], pair["s1"], len(pair["removed"]))]
        write_replacement_audit(directory / SHIFT_AUDIT_FILE, shift_rows)
    write_json(directory / "characteristics.json", {"source_pool": characteristics(frames["source_pool"]), "samples": collateral})
    pd.DataFrame(features).to_csv(directory / "source_feature_summaries.csv", index=False)
    diagnosis_table = pd.DataFrame(diagnoses)
    diagnosis_table["repeat"] = diagnosis_table["repeat"].astype("Int64")
    diagnosis_table.to_csv(directory / "diagnosis_category_counts.csv", index=False)
    pd.DataFrame([{"partition": name, "feature": feature, "missing_n": int(frame[feature].isna().sum())}
                  for name, frame in frames.items()
                  for feature in PROTOCOL.numeric_features + PROTOCOL.categorical_features]).to_csv(
        directory / "feature_missingness.csv", index=False)
