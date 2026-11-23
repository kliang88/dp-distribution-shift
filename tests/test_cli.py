"""The CLI exposes the signed study, signed plots, and slides."""

import pytest

from dp_shift.cli import main, parser


def test_defaults_use_the_signed_study():
    args = parser().parse_args(["experiment"])
    assert args.config == "configs/experiment.json"
    assert args.output == "results/signed_source_shift"
    assert args.workers == 1


def test_signed_plot_rejects_legacy_options():
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["plot", "--input", "results", "--fraction", ".1"])
    assert error.value.code == 2


def test_separate_slide_command_is_removed():
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["plot-slides", "--input", "results"])
    assert error.value.code == 2


def test_missing_results_is_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["plot", "--input", str(tmp_path)])
    assert error.value.code == 1
    assert "PLOT FAILED" in capsys.readouterr().err
