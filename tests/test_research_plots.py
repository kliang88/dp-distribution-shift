"""Inspect plotted values to check selections, paired contrasts, and repeat counts."""

import numpy as np
from scipy.stats import t

from dp_shift.cli import main
from dp_shift.config import SIGNED_LEVELS
from dp_shift.evaluation import plots


def test_average_dp_penalty_averages_conditions_before_repeat_intervals(monkeypatch, tmp_path, signed_results):
    frame = signed_results(repeats=4)
    severity = frame.shift_pp.abs()
    penalty = .001 * frame.repeat * severity * frame.epsilon.fillna(0)
    shifted = frame.shift_type.isin(plots.LABELS) & frame.method.eq("DP-SGD")
    frame.loc[shifted, "auprc"] -= penalty[shifted]
    frame["auroc"] = .4 + 2 * frame.auprc
    captured = []

    def inspect(figure, output, name, caption, **kwargs):
        assert len(figure.axes) == 2
        # E[|shift|] = 5, E[epsilon] = 5.2; only four repeat means determine the CI.
        repeat_means = .001 * np.arange(1, 5) * 5 * 5.2 * 100
        mean = repeat_means.mean()
        margin = t.ppf(.975, 3) * repeat_means.std(ddof=1) / 2
        for axis, scale, metric in zip(figure.axes, (1, 2), ("AUPRC", "AUROC")):
            assert axis.get_title() == metric
            assert not axis.patches
            np.testing.assert_allclose([series.lines[0].get_ydata()[0] for series in axis.containers], mean * scale)
            bounds = np.asarray([series.lines[2][0].get_segments()[0][:, 1] for series in axis.containers])
            np.testing.assert_allclose(bounds, np.tile([mean - margin, mean + margin], (4, 1)) * scale)
        assert "n = 4 paired repeats" in caption
        assert "ε = 1, 2.5, 5, 7.5, 10" in caption
        captured.append(name)
        plots.plt.close(figure)

    monkeypatch.setattr(plots, "save", inspect)
    plots.average_dp_penalty_pair(frame, tmp_path)
    assert captured == ["average_dp_penalty"]


def test_signed_plots_use_the_signed_grid_and_one_heatmap_scale(monkeypatch, tmp_path, signed_results):
    frame = signed_results(repeats=2)
    frame["auprc"] += np.where(frame.method.eq("DP-SGD"), frame.shift_pp / 2000, 0)
    frame["auroc"] = .3 + 2 * frame.auprc
    frame.to_csv(tmp_path / "results.csv", index=False)
    captured = {}
    original_save = plots.save

    def inspect(figure, output, name, caption, **kwargs):
        if name.startswith(("readmission_prevalence", "average_pretraining_gain", "average_dp_penalty")):
            return original_save(figure, output, name, caption, **kwargs)
        captured[name] = len(figure.axes)
        if name == "source_shift_effects":
            assert len(figure.axes) == 3
            for axis, metric in zip(figure.axes, ("auprc", "auroc", "brier_score")):
                for curve, kind in zip(axis.containers, plots.LABELS):
                    levels = curve.lines[0].get_xdata()
                    np.testing.assert_array_equal(levels, np.arange(-8, 9, 2))
                    selected = frame.loc[frame.method.eq("DP-SGD") & frame.epsilon.eq(10)]
                    control = selected.loc[selected.shift_type.eq("baseline")].set_index("repeat")[metric]
                    for level, value in zip(levels, curve.lines[0].get_ydata()):
                        shifted = selected.loc[selected.shift_type.eq(kind) & selected.shift_pp.eq(level)]
                        expected = 0 if level == 0 else (shifted.set_index("repeat")[metric] - control).mean() * 100
                        np.testing.assert_allclose(value, expected, atol=1e-12)
            assert "ε = 10" in caption
            return original_save(figure, output, name, caption, **kwargs)
        scale = 2 if name.endswith("_auroc") else 1
        metric = "AUROC" if scale == 2 else "AUPRC"
        if name.startswith("privacy_shift_interaction_"):
            images = [axis.images[0] for axis in figure.axes[:4]]
            assert len({image.get_clim() for image in images}) == 1
            assert all(image.get_array().shape == (5, 8) for image in images)
            for image in images:
                np.testing.assert_allclose(image.get_array(), np.tile(-scale * np.asarray(SIGNED_LEVELS) / 20, (5, 1)), atol=1e-12)
            assert metric in figure.axes[-1].get_ylabel()
        elif name.startswith("source_shift_"):
            for axis in figure.axes:
                assert len(axis.containers) == 3
                assert [curve.get_label() for curve in axis.containers] == ["SGD", "DP-SGD ε = 1", "DP-SGD ε = 10"]
                sgd = axis.containers[0].lines[0]
                assert sgd.get_color() == "#b07a25"
                for curve, epsilon in zip(axis.containers[1:], (1, 10)):
                    assert curve.lines[0].get_color() == plots.DP_COLORS[epsilon]
                assert sgd.get_linestyle() == "-"
                assert sgd.get_marker() == "o"
                assert sgd.get_linewidth() == axis.containers[1].lines[0].get_linewidth()
                np.testing.assert_array_equal(axis.containers[0].lines[0].get_xdata(), np.arange(-8, 9, 2))
                np.testing.assert_allclose(axis.containers[0].lines[0].get_ydata(), scale * np.arange(-8, 9, 2) / 10, atol=1e-12)
        else:
            for axis in figure.axes:
                assert len(axis.containers) == 3
                assert [curve.get_label() for curve in axis.containers] == ["DP-SGD ε = 1", "DP-SGD ε = 5", "DP-SGD ε = 10"]
                for curve in axis.containers:
                    np.testing.assert_array_equal(curve.lines[0].get_xdata(), SIGNED_LEVELS)
                    np.testing.assert_allclose(curve.lines[0].get_ydata(), scale * (1 + 1.5 * np.asarray(SIGNED_LEVELS) / 10))
        original_save(figure, output, name, caption, **kwargs)

    monkeypatch.setattr(plots, "save", inspect)
    main(["plot", "--input", str(tmp_path)])
    assert captured == {"source_shift_effects": 3, **{f"{name}_{metric}": axes for metric in ("auprc", "auroc")
                        for name, axes in (("source_shift", 4), ("privacy_shift_interaction", 5),
                                           ("pretraining_vs_scratch", 4))}}
    assert len(list((tmp_path / "plots").glob("*.png"))) == 10


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
        if not name.startswith(("readmission_prevalence", "average_pretraining_gain")):
            return original_save(figure, output, name, caption, **kwargs)
        assert caption == "ε = 5, n = 20 paired repeats, 95% CI"
        captured[name] = figure.axes
        original_save(figure, output, name, caption, **kwargs)

    monkeypatch.setattr(plots, "save", inspect)
    main(["plot", "--input", str(path), "--output", str(tmp_path / "custom-plots")])
    assert len(captured) == 2
    expected_margin = 100 * .0003 * t.ppf(.975, 19) * np.arange(1, 21).std(ddof=1) / np.sqrt(20)
    for index, metric_label, metric_scale in ((0, "AUPRC", 1), (1, "AUROC", 1.5)):
        readmission = captured["readmission_prevalence"][index]
        gain = captured["average_pretraining_gain"][index]
        assert readmission.get_title() == metric_label
        assert gain.get_title() == metric_label
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
    output = tmp_path / "custom-plots"
    assert {path.name for path in output.glob("*.png")} == {
        *{f"{name}.png" for name in captured},
        "average_dp_penalty.png",
        *{f"{name}_{metric}.png" for metric in ("auprc", "auroc")
          for name in ("source_shift", "privacy_shift_interaction", "pretraining_vs_scratch")},
    }
    assert not list(output.glob("*.pdf"))
    for name in captured:
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
