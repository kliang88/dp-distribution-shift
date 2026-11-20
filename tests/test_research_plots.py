"""Inspect plotted values to check selections, paired contrasts, and repeat counts."""

import numpy as np
from scipy.stats import t

from dp_shift.cli import main
from dp_shift.config import SIGNED_LEVELS
from dp_shift.evaluation import plots


def test_signed_plots_use_the_signed_grid_and_one_heatmap_scale(monkeypatch, tmp_path, signed_results):
    frame = signed_results(repeats=2)
    frame.to_csv(tmp_path / "results.csv", index=False)
    captured = {}
    original_save = plots.save

    def inspect(figure, output, name, caption, **kwargs):
        captured[name] = len(figure.axes)
        if name == "privacy_shift_interaction":
            images = [axis.images[0] for axis in figure.axes[:4]]
            assert len({image.get_clim() for image in images}) == 1
            assert all(image.get_array().shape == (5, 8) for image in images)
        elif name == "signed_shift_response":
            for axis in figure.axes:
                assert len(axis.containers) == 4
                np.testing.assert_array_equal(axis.containers[0].lines[0].get_xdata(), np.arange(-8, 9, 2))
        else:
            for axis in figure.axes:
                assert len(axis.containers) == 5
                for curve in axis.containers:
                    np.testing.assert_array_equal(curve.lines[0].get_xdata(), SIGNED_LEVELS)
                    np.testing.assert_allclose(curve.lines[0].get_ydata(), 1 + np.asarray(SIGNED_LEVELS) / 10)
        original_save(figure, output, name, caption, **kwargs)

    monkeypatch.setattr(plots, "save", inspect)
    main(["plot", "--input", str(tmp_path)])
    assert captured == {"signed_shift_response": 4, "privacy_shift_interaction": 5, "negative_transfer": 4}
    assert len(list((tmp_path / "plots").glob("*.png"))) == 3


def test_slide_plots_use_epsilon_five_and_average_severities_before_intervals(monkeypatch, tmp_path, signed_results):
    frame = signed_results(epsilons=(1., 5.))
    kind_index = {kind: index for index, kind in enumerate(("readmission", "age", "complexity", "circulatory"))}
    severity = frame.shift_pp.map({level: index for index, level in enumerate(SIGNED_LEVELS)})
    scale = np.where(frame.method.eq("DP-SGD"), 1, .5)
    gain = (.015 + .003 * frame.shift_type.map(kind_index) + .004 * severity
            + .0003 * (severity + 1) * (frame.repeat - 10.5) + .02 * frame.epsilon.eq(1))
    frame["auprc"] = .2 + .001 * frame.repeat + scale * np.select(
        [frame.shift_type.eq("scratch"), frame.shift_type.eq("baseline")], [0, .020], default=gain)
    frame["auroc"] = .4 + 1.5 * frame.auprc
    path = tmp_path / "results.csv"
    frame.to_csv(path, index=False)

    captured = {}
    original_save = plots.save

    def inspect(figure, output, name, caption, **kwargs):
        assert caption == "ε = 5, n = 20 paired repeats, 95% CI"
        captured[name] = figure.axes[0]
        original_save(figure, output, name, caption, **kwargs)

    monkeypatch.setattr(plots, "save", inspect)
    main(["plot-slides", "--input", str(path)])
    assert len(captured) == 4
    expected_margin = 100 * .0003 * t.ppf(.975, 19) * np.arange(1, 21).std(ddof=1) / np.sqrt(20)
    for suffix, metric_label, metric_scale in (("", "AUPRC", 1), ("_auroc", "AUROC", 1.5)):
        readmission = captured[f"slide6_readmission_prevalence{suffix}"]
        gain = captured[f"slide7_pretraining_gain{suffix}"]
        assert metric_label in readmission.get_ylabel()
        assert metric_label in gain.get_ylabel()
        assert gain.get_ylim()[0] < 0
        target_line = next(line for line in readmission.lines if line.get_label().startswith("Target"))
        np.testing.assert_allclose(target_line.get_xdata(), [9.6, 9.6])
        bars = [series for series in gain.containers if hasattr(series, "patches")]
        for line, group, method_scale, offset in zip(readmission.containers, bars, (1, .5), (-.18, .18), strict=True):
            scale = metric_scale * method_scale
            shift_means = 100 * scale * (-.005 + .004 * np.arange(8))
            gain_means = 100 * scale * (.029 + .003 * np.arange(4))
            np.testing.assert_allclose(line.lines[0].get_xdata(), 11.05 + np.asarray(SIGNED_LEVELS))
            np.testing.assert_allclose(line.lines[0].get_ydata(), shift_means)
            np.testing.assert_allclose([bar.get_x() + bar.get_width() / 2 for bar in group], np.arange(4) + offset)
            np.testing.assert_allclose([bar.get_height() for bar in group], gain_means)
            for series, means, margin in (
                (line, shift_means, expected_margin * scale * np.arange(1, 9)),
                (group.errorbar, gain_means, expected_margin * scale * 4.5),
            ):
                bounds = np.asarray(series.lines[2][0].get_segments())[:, :, 1]
                np.testing.assert_allclose(bounds, np.column_stack((means - margin, means + margin)))
    output = tmp_path / "plots/slides"
    assert {path.name for path in output.iterdir()} == {f"{name}.png" for name in captured}
    for name in captured:
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
