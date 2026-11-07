"""Publish minimal results; keep recoverable implementation artifacts in .work."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from dp_shift.data.utils import file_hash, read_json, require, write_json


FORMAT = "dp-shift-minimal-v3"
STAGES = ("prepare", "train", "evaluate")


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_path(path) as temporary:
        write_json(temporary, value)


def copy(source, target):
    with atomic_path(target) as temporary:
        shutil.copyfile(source, temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol(output):
    result = read_json(Path(output) / "protocol.json")
    require(result.get("format") == FORMAT,
            "Output uses another format; choose a new --output directory")
    return result


def workspace(output):
    """Create only owned workspaces; never adopt or remove legacy output directories."""
    output = Path(output).resolve()
    if (output / "protocol.json").exists():
        protocol(output)
    else:
        require(not output.exists() or not any(output.iterdir()),
                "Output is nonempty and has no minimal protocol; choose a new --output directory")
        atomic_json(output / "protocol.json", {"format": FORMAT, "stages": {}})
    work = output / ".work"
    owner = {"format": FORMAT, "output": str(output)}
    require(not work.is_symlink(), "Refusing a symlinked .work directory")
    if work.exists():
        require(read_json(work / "owner.json") == owner, "Unrecognized .work directory")
    else:
        work.mkdir(mode=0o700)
        atomic_json(work / "owner.json", owner)
    return work


def finish(output, stage, config, summary, files, **information):
    output = Path(output)
    saved = protocol(output)
    previous = STAGES[:STAGES.index(stage)]
    require(all(name in saved["stages"] for name in previous), "Earlier stage is incomplete")
    record = {"config": read_json(config), "summary": summary, **information,
              "depends_on": {name: digest(saved["stages"][name]) for name in previous},
              "files": {name: file_hash(output / name) for name in sorted(files)}}
    require(stage not in saved["stages"], f"Stage already complete: {stage}")
    saved["stages"][stage] = record
    atomic_json(output / "protocol.json", saved)
    return summary


def verify(output, stage):
    """Verify published artifacts without checkpoints, raw inputs, or inference."""
    output = Path(output)
    saved = protocol(output)
    require(stage in saved["stages"], f"{stage} is incomplete; run dp-shift {stage}")
    for name in STAGES[:STAGES.index(stage) + 1]:
        record = saved["stages"][name]
        expected = {prior: digest(saved["stages"][prior]) for prior in STAGES[:STAGES.index(name)]}
        require(record["depends_on"] == expected, f"Frozen {name} dependencies changed")
        for relative, fingerprint in record["files"].items():
            path = Path(relative)
            require(not path.is_absolute() and ".." not in path.parts, "Invalid artifact path")
            require(file_hash(output / path) == fingerprint, f"Published artifact changed: {relative}")
    return saved["stages"][stage]["summary"]


def completed(output, stage, config):
    path = Path(output) / "protocol.json"
    if not path.exists():
        return None
    saved = protocol(output)
    choices = read_json(config)
    for name, record in saved["stages"].items():
        require(record["config"] == choices,
                f"Completed {name} uses different settings; choose a new --output directory")
    if stage not in saved["stages"]:
        return None
    return verify(output, stage)


def cleanup(output):
    """Discard owned temporary models/caches only after published results verify."""
    verify(output, "evaluate")
    work = Path(output).resolve() / ".work"
    if work.exists():
        require(not work.is_symlink(), "Refusing to remove a symlinked .work directory")
        owner = work / "owner.json"
        # A crash between removing the last marker and rmdir leaves an empty directory.
        if not owner.exists() and not any(work.iterdir()):
            work.rmdir()
            return
        require(read_json(owner) == {"format": FORMAT, "output": str(Path(output).resolve())},
                "Refusing to remove an unrecognized .work directory")
        # Keep ownership evidence until the end so interrupted deletion can resume.
        for child in work.iterdir():
            if child == owner:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        owner.unlink()
        work.rmdir()
