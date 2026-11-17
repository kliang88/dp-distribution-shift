"""Run scalar experiments over a grid, then retain only combined research outputs."""

from copy import deepcopy
from pathlib import Path

from .config import (PROTOCOL, SelectedHyperparameters, comparison_config, exact_keys,
                     resolve_config, study_plan, validate_comparisons)
from .data.shifts import SHIFTS, members
from .data.utils import file_hash, read_json, require
from .stages.common import atomic_json, owned_workspace, remove_workspace


FORMAT = "dp-shift-sweep-csv-v1"


def experiment(base, fraction, epsilon, *, shift_type=None):
    """Expand a grid cell into the existing single-experiment schema."""
    kind = base["source_conditions"]["shift_type"] if shift_type is None else shift_type
    return comparison_config(base, kind, fraction, epsilon)


def normalize_plan(plan):
    """Normalize legacy prevalence-only choices once at the input boundary."""
    choices = deepcopy(plan["sweep"])
    if "shifted_positive_rates" in choices and "shifts" not in choices:
        choices["shifts"] = {"readmission": choices.pop("shifted_positive_rates")}
    # Older retained protocols name their former base file; no file is read here.
    choices.pop("base_config", None)
    exact_keys(choices, {"shifts", "epsilons", "reference_epsilon"}, "comparisons")
    validate_comparisons(choices["shifts"], choices["epsilons"], choices["reference_epsilon"])
    base = resolve_config(plan["base"]).requested.to_dict()
    for kind, fractions in choices["shifts"].items():
        for fraction in fractions:
            experiment(base, fraction, choices["reference_epsilon"], shift_type=kind)
    return {"sweep": choices, "base": base}


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


def run(output, config, progress=print):
    from .data.prepare import input_fingerprints
    from .evaluation.sweep import collect, publish
    from .evaluation.provenance import versions
    from .stages import prepare, train, evaluate
    from .training.model import configure_cpu

    output = Path(output)
    configuration = read_json(config)
    plan = study_plan(configuration)
    manifest = output / "protocol.json"
    if manifest.exists():
        saved = read_json(manifest)
        require(saved.get("format") == FORMAT, "Output uses another format; choose a new --output directory")
        require(normalize_plan(saved["plan"]) == plan, "Sweep settings changed; choose a new --output directory")
        if saved["status"] == "complete":
            result = verify(output)
            remove_workspace(output, FORMAT)
            return result
    else:
        require(not output.exists() or not any(output.iterdir()), "Output must be empty or belong to this study; choose a new --output directory")
        saved = None
    context = {"inputs": input_fingerprints(), "code_hashes": implementation_hashes(), "versions": versions(),
               "readme_sha256": file_hash(PROTOCOL.protocol_path)}
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
    runs, records = [], []
    for kind, fraction, epsilon in grid(plan):
        directory = work / kind / f"fraction_{fraction:g}" / f"epsilon_{epsilon:g}"
        cell_config = directory / "config.json"
        save_config(cell_config, experiment(plan["base"], fraction, epsilon, shift_type=kind))
        cell = directory / "result"

        def report(message):
            progress(f"{kind}, fraction {fraction:g}, epsilon {epsilon:g}: {message}")

        prepare.run(cell, cell_config, progress=report)
        train.run(cell, cell_config, progress=report, selected=SelectedHyperparameters(**reference["selected"]))
        evaluate.run(cell, cell_config, progress=report, resume=True)
        scores, record = collect(cell, kind, fraction, epsilon, prefix=f"{kind}_{fraction:g}_e{epsilon:g}_")
        runs.append(scores)
        records.append(record)
    progress("Publishing combined scores and curve data as four CSV tables…")
    publish(output, saved, reference, runs, records)
    result = verify(output)
    remove_workspace(output, FORMAT)
    return result


def verify(output):
    """Verify retained CSV data without inputs, temporary data, or inference."""
    from .evaluation.sweep import verify_tables
    from .evaluation.tables import FILES

    output = Path(output)
    saved = read_json(output / "protocol.json")
    require(saved.get("format") == FORMAT and saved.get("status") == "complete", "Sweep is incomplete")
    require(set(saved["files"]) == set(FILES), "Sweep output manifest is incomplete")
    for name in FILES:
        require(file_hash(output / name) == saved["files"][name], f"Published sweep artifact changed: {name}")
    verify_tables(output, saved)
    return {"combinations": len(saved["cells"]), "evaluated_models": saved["evaluated_models"],
            "repeats": saved["plan"]["base"]["repeats"], "output": str(output)}
