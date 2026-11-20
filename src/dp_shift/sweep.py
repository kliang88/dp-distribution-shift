"""Prepare and tune once, then run independently resumable signed repeats."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version
from multiprocessing import get_context
import os
from pathlib import Path
import shutil

from filelock import FileLock
import numpy as np
import pandas as pd

from .config import SelectedHyperparameters, TrainingSettings, parse_config
from .data.prepare import DEFAULT_RECORDS, prepare_splits
from .data.sampling import plan_swaps
from .data.shifts import SHIFTS
from .data.utils import atomic_output, file_hash, read_json, require, write_json
from .evaluation import tables
from .training.model import configure_cpu
from .training.run import run_repeat
from .training.tune import select_hyperparameters, settings


ARRAYS = ("source_pool", "target_train", "target_test")
_worker_state = None


def save_json(path, value):
    with atomic_output(path) as temporary:
        write_json(temporary, value)


def load_arrays(work, name):
    # Copy-on-write maps share physical pages and let Torch wrap NumPy without copying.
    return {field: np.load(work / f"{name}_{field}.npy", mmap_mode="c") for field in ("X", "y")}


def load_masks(work):
    return {name: np.load(work / f"{name}_mask.npy", mmap_mode="r") for name in SHIFTS}


def initialize_worker(work, ordinary, private):
    global _worker_state
    configure_cpu()
    _worker_state = (*[load_arrays(work, name) for name in ARRAYS], load_masks(work), ordinary, private)


def compute_repeat(work, repeat, plan):
    rows = run_repeat(repeat, plan, *_worker_state)
    save_json(work / f"repeat_{repeat:02d}.json", rows)
    return repeat


def compute_pending(work, pending, ordinary, private, workers, progress):
    if workers == 1:
        initialize_worker(work, ordinary, private)
        try:
            for repeat, plan in pending:
                compute_repeat(work, repeat, plan)
                progress(f"Completed repeat {repeat}")
        finally:
            clear_worker()
        return

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=initialize_worker, initargs=(work, ordinary, private)) as executor:
        futures = [executor.submit(compute_repeat, work, repeat, plan) for repeat, plan in pending]
        try:
            for future in as_completed(futures):
                progress(f"Completed repeat {future.result()}")
        finally:
            for future in futures:
                future.cancel()


def clear_worker():
    global _worker_state
    _worker_state = None


def run(output, config_path, records=None, *, workers=1, progress=print):
    require(type(workers) is int and workers > 0, "workers must be a positive integer")
    output = Path(output).resolve()
    require(not (output / tables.RESULTS_FILE).exists(), "results.csv already exists; choose a new output directory")
    work = output / ".work"
    require(not work.is_symlink(), "Invalid experiment workspace")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    with FileLock(work / "run.lock", timeout=0):
        return run_study(output, work, config_path, DEFAULT_RECORDS if records is None else records, workers, progress)


def run_study(output, work, config_path, records, workers, progress):
    choices = read_json(config_path)
    config = parse_config(choices)
    result = output / tables.RESULTS_FILE
    require(not result.exists(), "results.csv already exists; choose a new output directory")
    context_path = work / "context.json"
    root = Path(__file__).parent
    context = {"config": choices, "records_sha256": file_hash(records),
               "code": {str(path.relative_to(root)): file_hash(path) for path in root.rglob("*.py")},
               "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "opacus")}}
    if context_path.exists():
        require(read_json(context_path) == context, "Study inputs, configuration, or code changed; choose a new output directory")
    else:
        require(all(path.name == "run.lock" for path in work.iterdir()), "Unrecognized experiment workspace")
        save_json(context_path, context)

    selected_path = work / "selected.json"
    if selected_path.exists():
        saved = read_json(selected_path)
        selected = SelectedHyperparameters(**saved["selected"])
        plans = plan_swaps(np.load(work / "source_pool_y.npy", mmap_mode="r"), load_masks(work), config)
        ordinary = settings(config, selected)
        private = tuple(TrainingSettings(config, selected, epsilon, noise)
                        for epsilon, noise in zip(config.epsilons, saved["noise"], strict=True))
    else:
        progress("Preparing data and checking all signed swaps…")
        arrays, masks = prepare_splits(records, context["records_sha256"], config)
        plans = plan_swaps(arrays["source_pool"]["y"], masks, config)
        progress("Selecting training settings on public proxy data…")
        configure_cpu()
        selected = select_hyperparameters(config, arrays["source_pool"], arrays["proxy_train"], arrays["proxy_validation"])
        ordinary = settings(config, selected)
        private = tuple(settings(config, selected, epsilon) for epsilon in config.epsilons)
        for name in ARRAYS:
            for field, values in arrays[name].items():
                np.save(work / f"{name}_{field}.npy", values, allow_pickle=False)
        for name, mask in masks.items():
            np.save(work / f"{name}_mask.npy", mask, allow_pickle=False)
        save_json(selected_path, {"selected": asdict(selected), "noise": [item.noise_multiplier for item in private]})
        del arrays, masks

    progress(f"Source learning rate {selected.source_lr:g}, target learning rate {selected.target_lr:g}, "
             f"target epochs {selected.target_epochs}")
    pending = [(repeat, plan) for repeat, plan in enumerate(plans, 1)
               if not (work / f"repeat_{repeat:02d}.json").exists()]
    if pending:
        count = min(workers, len(pending))
        progress(f"Running {len(pending)} repeats with {count} CPU worker(s)…")
        compute_pending(work, pending, ordinary, private, count, progress)
    frame = pd.DataFrame([row for repeat in range(1, config.repeats + 1)
                          for row in read_json(work / f"repeat_{repeat:02d}.json")], columns=tables.COLUMNS)
    tables.validate(frame, config)
    with atomic_output(result) as temporary:
        frame.astype(dict.fromkeys(tables.INTEGER_COLUMNS, "Int64")).to_csv(temporary, index=False)
    shutil.rmtree(work)
    return {"evaluated_models": len(frame), "repeats": config.repeats, "output": str(result)}
