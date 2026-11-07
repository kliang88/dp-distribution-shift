"""Atomic, plain-state checkpoints with explicit provenance checks."""

import os
from pathlib import Path
import tempfile

import torch

from dp_shift.data.utils import file_hash, read_json, require, write_json
from .model import build_model, model_state, state_hash
from .train import check_parameters


def atomic_torch_save(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def load_checkpoint(path, expected=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    for key, value in (expected or {}).items():
        require(metadata.get(key) == value, f"Checkpoint {key} mismatch: {path}")
    require(state_hash(saved["state_dict"]) == saved["state_sha256"], "Checkpoint state fingerprint mismatch")
    model = build_model(metadata["input_dimension"], 0)
    model.load_state_dict(saved["state_dict"], strict=True)
    check_parameters(model)
    return model, metadata


def save_checkpoint(path, model, metadata):
    state = model_state(model)
    atomic_torch_save(path, {"state_dict": state, "state_sha256": state_hash(state), "metadata": metadata})
    load_checkpoint(path, metadata)


def completed_job(path, expected):
    path = Path(path)
    status_path = path.with_suffix(".json")
    if not status_path.exists():
        require(not path.exists(), f"Checkpoint without status: {path}")
        return False
    status = read_json(status_path)
    require(status["status"] == "complete", f"Incomplete/failed job requires investigation: {path}")
    require(file_hash(path) == status["checkpoint_sha256"], f"Checkpoint file fingerprint mismatch: {path}")
    _, metadata = load_checkpoint(path, expected)
    require(status["metadata"] == metadata, f"Checkpoint/status metadata mismatch: {path}")
    if "scheduled_steps" in metadata:
        require(metadata["steps"] == metadata["scheduled_steps"], f"Incomplete optimizer schedule: {path}")
    return True


def execute_job(path, metadata, train, internal_path):
    """Reuse complete jobs; stop on partial/failed jobs instead of silently restarting."""
    path = Path(path)
    if completed_job(path, metadata):
        return load_checkpoint(path, metadata)[1]
    atomic_json(path.with_suffix(".json"), {"status": "running", "metadata": metadata})
    try:
        model, diagnostics = train()
        final_metadata = {**metadata, "steps": diagnostics["steps"], "privacy": diagnostics.get("privacy")}
        atomic_torch_save(internal_path, diagnostics, private=True)
        save_checkpoint(path, model, final_metadata)
        atomic_json(path.with_suffix(".json"), {"status": "complete", "metadata": final_metadata,
                                              "checkpoint_sha256": file_hash(path)})
        return final_metadata
    except Exception as error:
        atomic_json(path.with_suffix(".json"), {"status": "failed", "metadata": metadata,
                                              "error": f"{type(error).__name__}: {error}"})
        raise
