"""Run readmission experiments or create plots from retained CSV data."""

import argparse
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG
from .data.shifts import SHIFTS

DEFAULT_OUTPUT = "results/prevalence_shift_v3"


def parser():
    cli = argparse.ArgumentParser(prog="dp-shift", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("experiment", help="Run every configured comparison and publish verified results")
    experiment.add_argument("--output", default=DEFAULT_OUTPUT,
                            help="Results directory (default: %(default)s)")
    experiment.add_argument("--config", default=DEFAULT_CONFIG,
                            help="Configuration (default: %(default)s)")
    experiment.add_argument("--workers", type=int, default=1,
                            help="Number of CPU processes for comparison cells (default: %(default)s)")
    plot = commands.add_parser("plot", help="Create figures from retained CSV data without training")
    plot.add_argument("--input", required=True, help="Results directory, data/ directory, or runs.csv")
    plot.add_argument("--output", help="Figure directory (default: <results>/plots)")
    plot.add_argument("--epsilon", type=float, help="Budget for overall performance (default: smallest available)")
    plot.add_argument("--shift", choices=list(SHIFTS), help="Plot one shift type (default: all available)")
    plot.add_argument("--fraction", "--prevalence", dest="fraction", type=float,
                      help="Source share or replacement rate for performance and privacy–utility (one shift type)")
    return cli


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
    output = Path(args.output) if args.output is not None else None
    active_stage = args.command

    def progress(message):
        print(f"[{active_stage}] {message}", flush=True)

    try:
        if args.command == "plot":
            from .evaluation.report import run
            display("PLOTS COMPLETE", run(args.input, output, progress=progress,
                                         epsilon=args.epsilon, shift=args.shift, fraction=args.fraction))
            return
        from .sweep import run
        display(f"{args.command.upper()} COMPLETE",
                run(output, args.config, progress=progress, workers=args.workers))
    except (ValueError, OSError, KeyError, RuntimeError, AssertionError, ImportError) as error:
        message = str(error)
        print(f"\n{active_stage.upper()} FAILED: {message}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("Install dependencies with python -m pip install -r requirements.txt.", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print(f"\n{active_stage.upper()} INTERRUPTED. Run the same command to reuse completed comparisons "
              "and resume evaluation. Partial training jobs require investigation.", file=sys.stderr)
        raise SystemExit(130) from None
