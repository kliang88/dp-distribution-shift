"""Run scalar experiments over a grid, then retain only combined research outputs."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
import os
from pathlib import Path
import signal

from .config import (SelectedHyperparameters, comparison_config, resolve_config, study_plan)
from .data.shifts import SHIFTS, members
from .data.utils import file_hash, read_json, require
from .stages.common import atomic_json, owned_workspace, remove_workspace


FORMAT = "dp-shift-experiment-runs-v2"


def experiment(base, fraction, epsilon, *, shift_type=None):
    """Expand a grid cell into the existing single-experiment schema."""
    kind = base["source_conditions"]["shift_type"] if shift_type is None else shift_type
    return comparison_config(base, kind, fraction, epsilon)


def load_plan(path):
    return study_plan(read_json(path))


def grid(plan):
    for kind, fractions in plan["sweep"]["shifts"].items():
        for fraction in fractions:
            for epsilon in plan["sweep"]["epsilons"]:
                yield kind, fraction, epsilon


def preflight(plan):
    """Check every source draw and partition before any tuning or target training."""
    from .data.cohort import build_cohort
    from .data import prepare
    from .data.sampling import create_source_pair
    from .data.schema import load_and_validate_raw
    from .data.splits import create_split_manifests

    config = resolve_config(plan["base"])
    inputs = prepare.input_fingerprints()
    raw = load_and_validate_raw(prepare.DEFAULT_RECORDS, config, inputs["records"]["sha256"])
    for kind in plan["sweep"]["shifts"]:
        members(raw, kind)
    cohort, _, _ = build_cohort(raw)
    pool = create_split_manifests(cohort, config)["source_pool"]
    for kind, fractions in plan["sweep"]["shifts"].items():
        for fraction in fractions:
            config = resolve_config(experiment(plan["base"], fraction, plan["sweep"]["reference_epsilon"], shift_type=kind))
            for repeat in range(config.requested.repeats):
                create_source_pair(pool, repeat, config)


def implementation_hashes():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))}


def save_config(path, choices):
    if path.exists():
        require(read_json(path) == choices, "Stored sweep configuration changed; choose a new --output directory")
    else:
        atomic_json(path, choices)


def reference_tuning(work, plan):
    from .data.prepare import build
    from .training.tune import tune

    reference = work / "reference"
    choices = plan["sweep"]
    config = reference / "config.json"
    kind = next(iter(choices["shifts"]))
    save_config(config, experiment(plan["base"], choices["shifts"][kind][0], choices["reference_epsilon"], shift_type=kind))
    build(config, reference / "data")
    frozen = tune(reference / "data", reference / "training", config)
    return {"selected": frozen["selected"],
            "candidates": frozen["candidate_results"]}


def run_cell(comparison, directory, base, selected, inputs, progress=None):
    from .stages import prepare, train, evaluate

    kind, fraction, epsilon = comparison
    label = f"{kind}, fraction {fraction:g}, epsilon {epsilon:g}"

    def report(message):
        if progress is None:
            print(f"[experiment] {label}: {message}", flush=True)
        else:
            progress(f"{label}: {message}")

    config = directory / "config.json"
    cell = directory / "result"
    try:
        save_config(config, experiment(base, fraction, epsilon, shift_type=kind))
        prepare.run(cell, config, progress=report, records=inputs["records"], mapping=inputs["mapping"])
        train.run(cell, config, progress=report, selected=selected)
        evaluate.run(cell, config, progress=report, resume=True)
    except Exception as error:
        if progress is not None:
            raise
        raise RuntimeError(f"{label}: {error}") from error


def configure_worker():
    # The parent handles Ctrl-C so active cells can finish writing their outputs.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def run_parallel(cells, base, selected, inputs, workers, progress):
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                   initializer=configure_worker)
    remaining = iter(cells.items())

    def submit_next():
        job = next(remaining, None)
        if job is not None:
            comparison, directory = job
            pending.add(executor.submit(run_cell, comparison, directory, base, selected, inputs))

    pending = set()
    try:
        for _ in range(workers):
            submit_next()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            for _ in done:
                submit_next()
    except BaseException:
        progress("Waiting for active comparisons to finish before exiting…")
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run(output, config, progress=print, *, workers=1):
    require(type(workers) is int and workers >= 1, "workers must be a positive integer")

    from .data.prepare import input_fingerprints
    from .evaluation.sweep import collect, publish
    from .evaluation.provenance import versions
    from .training.model import configure_cpu

    output = Path(output)
    configuration = read_json(config)
    plan = study_plan(configuration)
    manifest = output / "protocol.json"
    if manifest.exists():
        saved = read_json(manifest)
        require(saved.get("format") == FORMAT, "Output uses another format; choose a new --output directory")
        require(saved["plan"] == plan, "Sweep settings changed; choose a new --output directory")
        if saved["status"] == "complete":
            result = verify(output)
            remove_workspace(output, FORMAT)
            return result
    else:
        require(not output.exists() or not any(output.iterdir()), "Output must be empty or belong to this study; choose a new --output directory")
        saved = None
    cells = {(kind, fraction, epsilon): Path(kind) / f"fraction_{fraction:g}" / f"epsilon_{epsilon:g}"
             for kind, fraction, epsilon in grid(plan)}
    require(len(set(cells.values())) == len(cells), "Comparison paths collide after formatting fractions or epsilons")
    context = {"inputs": input_fingerprints(), "code_hashes": implementation_hashes(), "versions": versions()}
    if saved is not None:
        require(saved["context"] == context, "Sweep inputs, code, or dependencies changed; choose a new --output directory")
    progress("Checking partitions and source feasibility across the complete grid…")
    preflight(plan)
    if saved is None:
        saved = {"format": FORMAT, "status": "running", "configuration": configuration,
                 "plan": plan, "context": context}
        atomic_json(manifest, saved)
    work = owned_workspace(output, FORMAT)
    configure_cpu()
    progress("Selecting or verifying shared settings on public reference data…")
    reference = reference_tuning(work, plan)
    selected = SelectedHyperparameters(**reference["selected"])
    inputs = {name: info["path"] for name, info in context["inputs"].items()}
    cells = {comparison: work / path for comparison, path in cells.items()}
    workers = min(workers, len(cells))
    if workers == 1:
        for comparison, directory in cells.items():
            run_cell(comparison, directory, plan["base"], selected, inputs, progress=progress)
    else:
        progress(f"Running comparisons with {workers} CPU workers…")
        run_parallel(cells, plan["base"], selected, inputs, workers, progress)

    runs, records = [], []
    for comparison, directory in cells.items():
        scores, record = collect(directory / "result", comparison)
        runs.append(scores)
        records.append(record)
    progress("Publishing combined model scores to runs.csv…")
    publish(output, saved, reference, runs, records)
    result = verify(output)
    remove_workspace(output, FORMAT)
    return result


def verify(output):
    """Verify retained CSV data without inputs, temporary data, or inference."""
    from .evaluation.sweep import validate
    from .evaluation.tables import RUNS_FILE, read

    output = Path(output)
    saved = read_json(output / "protocol.json")
    require(saved.get("format") == FORMAT and saved.get("status") == "complete", "Sweep is incomplete")
    require(set(saved["files"]) == {RUNS_FILE}, "Experiment output manifest is incomplete")
    require(file_hash(output / RUNS_FILE) == saved["files"][RUNS_FILE], "Published sweep artifact changed: runs.csv")
    validate(read(output / RUNS_FILE), saved)
    return {"combinations": len(saved["cells"]), "evaluated_models": saved["evaluated_models"],
            "repeats": saved["plan"]["base"]["repeats"], "output": str(output)}
