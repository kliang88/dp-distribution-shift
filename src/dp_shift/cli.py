"""Run the signed source-shift study or plot saved results."""

import argparse
from pathlib import Path
import sys

from .config import DEFAULT_CONFIG


DEFAULT_OUTPUT = "results/signed_source_shift"


def parser():
    cli = argparse.ArgumentParser(prog="dp-shift", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("experiment", help="Run the signed study and write results.csv")
    experiment.add_argument("--output", default=DEFAULT_OUTPUT)
    experiment.add_argument("--config", default=DEFAULT_CONFIG)
    experiment.add_argument("--records", help="Input patient records CSV")
    experiment.add_argument("--workers", type=int, default=1, help="Number of CPU workers for repeats")
    plot = commands.add_parser("plot", help="Plot signed response, privacy interaction, and negative transfer")
    slides = commands.add_parser("plot-slides", help="Create slides 6 and 7 from results.csv")
    for command in (plot, slides):
        command.add_argument("--input", required=True, help="Signed results directory or results.csv")
        command.add_argument("--output", type=Path, help="Figure directory")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)

    def progress(message):
        print(f"[{args.command}] {message}", flush=True)

    try:
        if args.command == "experiment":
            from .sweep import run
            result = run(args.output, args.config, args.records, workers=args.workers, progress=progress)
        else:
            from .evaluation import plots, slides
            command = plots if args.command == "plot" else slides
            result = command.run(args.input, args.output, progress=progress)
        print("\n" + args.command.upper() + " COMPLETE", flush=True)
        for label, value in result.items():
            print(f"  {label.replace('_', ' ').capitalize():<20} {value}", flush=True)
    except (ValueError, OSError, KeyError, RuntimeError, AssertionError, ImportError) as error:
        print(f"\n{args.command.upper()} FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print(f"\n{args.command.upper()} INTERRUPTED", file=sys.stderr)
        raise SystemExit(130) from None
