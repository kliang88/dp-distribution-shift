"""Prepare verified training data in the temporary workspace."""

from pathlib import Path

import pandas as pd

from dp_shift.config import load_config
from dp_shift.data.prepare import build, input_fingerprints
from dp_shift.data.utils import file_hash, read_json, require
from .common import completed, finish, protocol, workspace


def run(output, config, progress=print, *, records=None, mapping=None):
    existing = completed(output, "prepare", config)
    if existing is not None:
        saved = protocol(output)["stages"]["prepare"]["resolved"]
        require(saved["inputs"] == input_fingerprints(records, mapping),
                "Preparation inputs changed; choose a new --output directory")
        progress("Verified existing prepared data.")
        return existing
    output = Path(output)
    work = workspace(output)
    progress("Preparing and verifying patient splits, source pairs, and features…")
    summary = build(config, work / "data", records=records, mapping=mapping)
    saved_config = read_json(work / "data/config.json")
    require(load_config(config).requested.to_dict() == saved_config["experiment"], "Preparation configuration changed during execution")
    return finish(output, "prepare", config, summary, [],
                  resolved=saved_config,
                  source_pairs=pd.read_csv(work / "data/audit/source_pair_summary.csv").to_dict("records"),
                  characteristics=read_json(work / "data/audit/characteristics.json"),
                  source_hashes={f"repeat_{repeat:02d}_{source}": file_hash(
                      work / f"data/source_pairs/repeat_{repeat:02d}/{source}.csv")
                      for repeat in range(saved_config["repeats"]) for source in ("s0", "s1")},
                  partitions=read_json(work / "data/partition_summary.json"),
                  membership_hashes={path.stem: file_hash(path) for path in
                                     sorted((work / "data/manifests").glob("*.csv"))},
                  provenance=read_json(work / "data/provenance.json"),
                  preprocessing=read_json(work / "data/preprocessing/metadata.json"),
                  cohort=read_json(work / "data/cohort_audit.json"))
