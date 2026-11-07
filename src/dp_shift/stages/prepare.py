"""Prepare verified splits and publish only the five training data bundles."""

from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.config import load_config
from dp_shift.data.artifacts import TRAIN_PARTITIONS
from dp_shift.data.prepare import build, input_fingerprints
from dp_shift.data.utils import file_hash, read_json, require
from .common import atomic_path, completed, copy, finish, protocol, workspace


def run(output, config, progress=print):
    existing = completed(output, "prepare", config)
    if existing is not None:
        saved = protocol(output)["stages"]["prepare"]["resolved"]
        require(saved["inputs"] == input_fingerprints(),
                "Preparation inputs changed; choose a new --output directory")
        progress("Verified existing prepared data.")
        return existing
    output = Path(output)
    work = workspace(output)
    progress("Preparing and verifying patient splits, source pairs, and features…")
    summary = build(config, work / "data")
    saved_config = read_json(work / "data/config.json")
    files = []
    for name in TRAIN_PARTITIONS:
        relative = f"data/{name}.npz"
        copy(work / f"data/arrays/{name}.npz", output / relative)
        files.append(relative)
    indices = {f"repeat_{repeat:02d}_{source}": np.load(
        work / f"data/source_pairs/repeat_{repeat:02d}/{source}_indices.npy", allow_pickle=False)
        for repeat in range(saved_config["repeats"]) for source in ("s0", "s1")}
    with atomic_path(output / "data/source_pairs.npz") as temporary:
        np.savez_compressed(temporary, **indices)
    files.append("data/source_pairs.npz")
    require(load_config(config)["experiment"] == saved_config["experiment"], "Preparation configuration changed during execution")
    return finish(output, "prepare", config, summary, files,
                  resolved=saved_config,
                  source_pairs=pd.read_csv(work / "data/audit/source_pair_summary.csv").to_dict("records"),
                  partitions=read_json(work / "data/partition_summary.json"),
                  membership_hashes={path.stem: file_hash(path) for path in
                                     sorted((work / "data/manifests").glob("*.csv"))},
                  provenance=read_json(work / "data/provenance.json"),
                  preprocessing=read_json(work / "data/preprocessing/metadata.json"),
                  cohort=read_json(work / "data/cohort_audit.json"))
