"""A single CLI for the readmission prevalence-shift experiment."""

import argparse
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG, load_config

DEFAULT_OUTPUT = "results/prevalence_shift_v3"
STAGES = ("prepare", "train", "evaluate")


def parser():
    cli = argparse.ArgumentParser(prog="dp-shift", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name, help_text in {
        "prepare": "Prepare and verify training data (test arrays deferred)",
        "train": "Check, tune, train all repeats, verify, and finalize",
        "evaluate": "Publish metrics, privacy, and PNG figures; remove completed work files",
        "all": "Run prepare, train, and evaluate in order",
        "verify": "Verify an existing stage without fitting or inference",
    }.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", default=DEFAULT_OUTPUT,
                             help="Experiment root, consistent across all stages (default: %(default)s)")
        if name != "verify":
            command.add_argument("--config", default=DEFAULT_CONFIG, help="Experiment config (default: %(default)s)")
        if name in {"evaluate", "all"}:
            command.add_argument("--resume", action="store_true",
                                 help="Reuse verified caches after interrupted evaluation; partial training still stops")
        if name == "verify":
            command.add_argument("stage", choices=["prepare", "train", "evaluate", "all"])
    return cli


def verified(stage, output):
    if stage == "evaluate":
        from .stages.evaluate import verify_results
        return verify_results(output)
    from .stages.common import verify
    return verify(output, stage)


def display(title, rows):
    print(f"\n{title}", flush=True)
    formatted = {label.replace("_", " ").capitalize(): f"{value:.6g}" if isinstance(value, float) else value
                 for label, value in rows.items()}
    width = max(map(len, formatted))
    for label, value in formatted.items():
        print(f"  {label:<{width}}  {value}", flush=True)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    output = Path(args.output)
    active_stage = args.command

    def progress(message):
        print(f"[{active_stage}] {message}", flush=True)

    try:
        if args.command == "verify":
            stages = list(STAGES) if args.stage == "all" else [args.stage]
            for active_stage in stages:
                progress("Verifying existing artifacts…")
                display(f"{active_stage.upper()} VERIFIED", verified(active_stage, output))
            return
        stages = list(STAGES) if args.command == "all" else [args.command]
        load_config(args.config)  # Validate all choices before creating output or starting work.
        for active_stage in stages:
            config = args.config
            # Lazy imports keep prepare and --help independent of torch/Opacus/matplotlib.
            if active_stage == "prepare":
                from .stages.prepare import run
            elif active_stage == "train":
                from .stages.train import run
            else:
                from .stages.evaluate import run
            options = {"resume": args.resume} if active_stage == "evaluate" else {}
            result = run(output, config, progress=progress, **options)
            display(f"{active_stage.upper()} COMPLETE", result)
    except (ValueError, OSError, KeyError, RuntimeError, AssertionError, ImportError) as error:
        message = str(error)
        if "use score --resume" in message:
            message = message.replace("use score --resume", "use dp-shift evaluate --resume")
        print(f"\n{active_stage.upper()} FAILED: {message}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("Install the stage dependencies described in README.md.", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print(f"\n{active_stage.upper()} INTERRUPTED. Completed jobs are retained. "
              "Partial training jobs require investigation; resume evaluation with "
              "dp-shift evaluate --resume.", file=sys.stderr)
        raise SystemExit(130) from None
