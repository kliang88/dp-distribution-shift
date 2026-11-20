"""Inspect plotted values to check selections, paired contrasts, and repeat counts."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_shift.evaluation.tables import CONDITIONS
from dp_shift.evaluation.summarize import paired_effects


def test_slide_plots_use_epsilon_five_and_average_severities_before_intervals(monkeypatch, tmp_path):
    from scipy.stats import t

    from dp_shift.cli import main
    from dp_shift.evaluation import plots, tables

    levels = {
        "readmission": (.01, .04, .14, .17),
        "age_70_plus": (.6, .7, .8, .9),
        "clinical_complexity_2plus": (.05, .10, .15, .20),
        "primary_diagnosis_circulatory": (.05, .10, .15, .20),
    }
    repeats = np.arange(20)
    rows = []
    for shift_index, (kind, fractions) in enumerate(levels.items()):
        for (severity, fraction), epsilon, repeat in product(enumerate(fractions), (1, 5), repeats):
            for method, pretraining in CONDITIONS.values():
                scale = 1 if method == "dp_sgd" else .5
                scratch = .2 + .001 * repeat
                gain = (.015 + .003 * shift_index + .004 * severity
                        + .0003 * (severity + 1) * (repeat - 9.5) + .02 * (epsilon == 1))
                scores = {"none": scratch, "baseline": scratch + .020 * scale, "shifted": scratch + gain * scale}
                shifted_fraction = fraction + .2 if shift_index >= 2 else fraction
                shares = {"none": np.nan, "baseline": .2, "shifted": shifted_fraction}
                auprc = scores[pretraining]
                auroc = .4 + 1.5 * auprc
                rows.append((kind, fraction, epsilon, repeat, method, pretraining, auprc, auroc, .2,
                             shares[pretraining], np.nan if pretraining == "none" else 100, .0863, 10000))
    tables.write(tmp_path / tables.RUNS_FILE, pd.DataFrame(rows, columns=tables.COLUMNS))

    captured = {}
    original_save = plots.save

    def inspect(figure, output, name, caption, **kwargs):
        assert caption == "ε = 5, n = 20 repeats, 95% CI"
        captured[name] = figure.axes[0]
        original_save(figure, output, name, caption, **kwargs)

    monkeypatch.setattr(plots, "save", inspect)
    main(["plot-slides", "--input", str(tmp_path)])
    assert len(captured) == 4
    expected_margin = 100 * .0003 * t.ppf(.975, 19) * repeats.std(ddof=1) / np.sqrt(20)
    for suffix, metric_label, metric_scale in (("", "AUPRC", 1), ("_auroc", "AUROC", 1.5)):
        readmission = captured[f"slide6_readmission_prevalence{suffix}"]
        gain = captured[f"slide7_pretraining_gain{suffix}"]
        assert metric_label in readmission.get_ylabel()
        assert metric_label in gain.get_ylabel()
        assert gain.get_ylim()[0] < 0
        bars = [series for series in gain.containers if hasattr(series, "patches")]
        for line, group, method_scale, offset in zip(readmission.containers, bars, (1, .5), (-.18, .18), strict=True):
            scale = metric_scale * method_scale
            shift_means = 100 * scale * (-.005 + .004 * np.arange(4))
            gain_means = 100 * scale * (.021 + .003 * np.arange(4))
            np.testing.assert_allclose(line.lines[0].get_xdata(), [1, 4, 14, 17])
            np.testing.assert_allclose(line.lines[0].get_ydata(), shift_means)
            np.testing.assert_allclose([bar.get_x() + bar.get_width() / 2 for bar in group], np.arange(4) + offset)
            np.testing.assert_allclose([bar.get_height() for bar in group], gain_means)
            assert all(bar.get_y() == 0 for bar in group)
            for series, means, margin in (
                (line, shift_means, expected_margin * scale * np.arange(1, 5)),
                (group.errorbar, gain_means, expected_margin * scale * 2.5),
            ):
                bounds = np.asarray(series.lines[2][0].get_segments())[:, :, 1]
                np.testing.assert_allclose(bounds, np.column_stack((means - margin, means + margin)))
    output = tmp_path / "plots/slides"
    assert {path.name for path in output.iterdir()} == {f"{name}.png" for name in captured}
    for name in captured:
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")


def test_figures_use_paired_repeats_and_positive_amplification(monkeypatch):
    from dp_shift.evaluation import plots

    raw = []
    for prevalence in (.01, .2):
        for epsilon in (1., 5.):
            scores = {"A": .1, "B-S0": .13, "B-S1": .12 if prevalence == .01 else .11,
                      "C": .08, "D-S0": .12 if epsilon == 1 else .15,
                      "D-S1": (.09 if prevalence == .01 else .07) if epsilon == 1 else (.12 if prevalence == .01 else .11)}
            if epsilon == 5 and prevalence == .01:
                scores["D-S0"] += .01
            for repeat in range(3):
                for path, (method, initialization) in CONDITIONS.items():
                    # Shared repeat offsets change model CIs but cancel in paired contrasts.
                    value = scores[path] + repeat * .01
                    source = {"none": np.nan, "baseline": .08 + repeat * .01, "shifted": prevalence}[initialization]
                    raw.append(dict(repeat=repeat, method=method, pretraining=initialization,
                                    shift_type="readmission", shift_fraction=prevalence, comparison_epsilon=epsilon, auprc=value,
                                    source_fraction=source, source_size=20000 if initialization != "none" else np.nan,
                                    test_fraction=.1, test_size=100))
    runs = pd.DataFrame(raw)
    captured = {}
    captions = {}

    def inspect(figure, output, name, caption, **kwargs):
        captured[name] = figure
        captions[name] = caption

    monkeypatch.setattr(plots, "save", inspect)
    try:
        count = plots.plot(runs, paired_effects(runs), 100, Path("plots"), epsilon=5., fractions={"readmission": .2})
        assert count == 6 and set(captured) == set(plots.FIGURES)
        overall = captured["overall_performance"].axes[0]
        bars = [container for container in overall.containers if hasattr(container, "patches")]
        assert len(bars) == 2
        np.testing.assert_allclose([bar.get_height() for bar in bars[0]], [.11, .14, .12])
        np.testing.assert_allclose([bar.get_height() for bar in bars[1]], [.09, .16, .12])
        assert all(container.errorbar.has_yerr for container in bars)
        assert overall.texts[0].get_text() == "20% source readmission rate"

        benefit = captured["pretraining_benefit"]
        gain_axis = benefit.axes[0]
        assert [label.get_text() for label in benefit.legends[0].get_texts()] == ["SGD", "DP-SGD ε=1", "DP-SGD ε=5"]
        assert any(label.get_text() == "Unshifted source 9.0%" for label in gain_axis.texts)
        assert any("No benefit from pretraining" in label.get_text() for label in benefit.texts)
        gain, reference = gain_axis.containers[:2]
        np.testing.assert_allclose(gain.lines[0].get_xdata(), [.01, .2])
        np.testing.assert_allclose(gain.lines[0].get_ydata(), [2, 1])
        np.testing.assert_allclose(reference.lines[0].get_xdata(), [.09])
        np.testing.assert_allclose(reference.lines[0].get_ydata(), [3])
        assert reference.lines[0].get_marker() == "s"
        assert reference.lines[0].get_linestyle() == "None"
        np.testing.assert_allclose(gain_axis.containers[5].lines[0].get_ydata(), [7])
        # Shared repeat offsets cancel in both the shifted and unshifted gains.
        for container in (gain, reference):
            assert container.has_yerr
            for segment in container.lines[2][0].get_segments():
                assert segment[0, 1] == pytest.approx(segment[1, 1])

        amplification = captured["dp_amplification"].axes
        assert len(amplification) == 1
        np.testing.assert_allclose(amplification[0].containers[0].lines[0].get_ydata(), [2, 0, 3])
        np.testing.assert_allclose(amplification[0].containers[1].lines[0].get_ydata(), [3, 0, 2])
        tradeoff = captured["privacy_utility"].axes[0]
        np.testing.assert_allclose(tradeoff.containers[1].lines[0].get_xdata(), [1, 5])
        np.testing.assert_allclose(tradeoff.containers[1].lines[0].get_ydata(), [.13, .16])
        np.testing.assert_allclose(tradeoff.containers[2].lines[0].get_ydata(), [.08, .12])
        distribution, gap = (captured[name].axes[0] for name in ("source_target_prevalence", "source_target_gap"))
        assert [label.get_text() for label in distribution.get_xticklabels()] == ["Unshifted", "1%", "20%"]
        assert any(text.get_text() == "Target: 10.0%" for text in distribution.texts)
        assert gap.get_xticklabels()[0].get_text() == "Readmission Prevalence\n1% / 20%"
        np.testing.assert_allclose(distribution.lines[-1].get_ydata(), [.1, .1])
        np.testing.assert_allclose(gap.lines[-1].get_ydata(), [0, 0])
        assert distribution.get_ylim()[0] == 0
        source_bars = [container for container in distribution.containers if hasattr(container, "patches")]
        gap_bars = [container for container in gap.containers if hasattr(container, "patches")]
        for source, difference, expected in zip(source_bars, gap_bars, [[.09], [.01, .2]], strict=True):
            np.testing.assert_allclose([bar.get_height() for bar in source], expected)
            np.testing.assert_allclose([bar.get_height() for bar in difference], (np.array(expected) - .1) * 100)
            assert all(bar.get_y() == 0 for bar in [*source, *difference])
            assert source.errorbar.has_yerr and difference.errorbar.has_yerr
            bounds = np.asarray(source.errorbar.lines[2][0].get_segments())[:, :, 1]
            gap_bounds = np.asarray(difference.errorbar.lines[2][0].get_segments())[:, :, 1]
            np.testing.assert_allclose(gap_bounds, (bounds - .1) * 100)
        assert gap_bars[0].errorbar.lines[2][0].get_segments()[0][0, 1] < -1
        for name in ("source_target_prevalence", "source_target_gap"):
            assert "Source n=20,000 per cohort. Fixed target test n=100." in captions[name]
            assert "95% t intervals across 3 repeats" in captions[name]
    finally:
        for figure in captured.values():
            plots.plt.close(figure)


def test_performance_figures_use_planned_levels_and_matched_panel_scores(monkeypatch):
    from dp_shift.evaluation import plots, report, tables

    scenarios = {
        "readmission": ((.01, .17), .01, "1% source readmission rate"),
        "age_70_plus": ((.7, .9), .9, "90% of source patients aged ≥70"),
        "primary_diagnosis_circulatory": ((.05, .2), .2, "19.802% source replacement · Circulatory primary diagnosis"),
        "clinical_complexity_2plus": ((.05, .2), .2, "19.802% source replacement · ≥2 diagnosis categories"),
    }
    rows = []
    for index, (kind, (levels, _, _)) in enumerate(scenarios.items()):
        replacement = index >= 2
        for level, budget, repeat, (method, init) in product(levels, [.5, 1.], range(2), CONDITIONS.values()):
            subgroup = .3 + round(level * 101) / 101 if replacement else level
            score = .20 + .02 * plots.INITIALIZATIONS.index(init) + .01 * repeat
            score += (.01 * index + .02 * budget) if method == "dp_sgd" else 0
            score += .10 * level if init == "shifted" else 0
            rows.append(dict(shift_type=kind, shift_fraction=level, comparison_epsilon=budget, repeat=repeat,
                             method=method, pretraining=init, auprc=score, auroc=.5, brier=.2,
                             source_fraction={"none": np.nan, "baseline": .3, "shifted": subgroup}[init],
                             source_size=np.nan if init == "none" else 101, test_fraction=.1, test_size=100))
    runs = pd.DataFrame(rows, columns=tables.COLUMNS)
    monkeypatch.setattr(tables, "read", lambda path: runs)
    captured = {"overall_performance": [], "privacy_utility": []}

    def inspect(figure, directory, name, caption, **kwargs):
        if name in captured:
            captured[name].append((figure, caption))
        else:
            plots.plt.close(figure)

    monkeypatch.setattr(plots, "save", inspect)
    try:
        report.run("unused", progress=lambda message: None)
        figure, caption = captured["overall_performance"][0]
        assert len(figure.axes) == 4
        assert figure.axes[0].get_gridspec().get_geometry() == (2, 2)
        assert len({axis.get_ylim() for axis in figure.axes}) == 1
        lower, upper = figure.axes[0].get_ylim()
        assert lower == 0 and upper > max(axis.dataLim.ymax for axis in figure.axes)
        assert len(figure.legends) == 1
        assert [label.get_text() for label in figure.legends[0].get_texts()] == ["SGD", "DP-SGD (ε = 1)"]
        assert "DP-SGD scratch and unshifted scores vary across panels" in caption
        assert "2 repeats. 95% t intervals" in caption
        for axis, (kind, (_, level, subtitle)) in zip(figure.axes, scenarios.items()):
            assert axis.texts[0].get_text() == subtitle
            assert axis.get_legend() is None
            matching = runs.loc[runs.shift_type.eq(kind) & runs.shift_fraction.eq(level) & runs.comparison_epsilon.eq(1)]
            bars = [container for container in axis.containers if hasattr(container, "patches")]
            assert len(bars) == 2
            for method, series in zip(["sgd", "dp_sgd"], bars):
                means = matching.loc[matching.method.eq(method)].groupby("pretraining").auprc.mean()
                np.testing.assert_allclose([bar.get_height() for bar in series], means.loc[list(plots.INITIALIZATIONS)])

        tradeoff, caption = captured["privacy_utility"][0]
        assert len(tradeoff.axes) == 4
        assert tradeoff.axes[0].get_gridspec().get_geometry() == (2, 2)
        assert len({axis.get_ylim() for axis in tradeoff.axes}) == 1
        lower, upper = tradeoff.axes[0].get_ylim()
        assert 0 < lower < min(axis.dataLim.ymin for axis in tradeoff.axes)
        assert upper > max(axis.dataLim.ymax for axis in tradeoff.axes)
        assert len(tradeoff.legends) == 1
        assert [label.get_text() for label in tradeoff.legends[0].get_texts()] == [
            "Scratch", "Unshifted pretraining", "Shifted pretraining"]
        assert "independent sampling and noise" in caption
        assert "2 repeats. 95% t intervals" in caption
        for axis, (kind, (_, level, subtitle)) in zip(tradeoff.axes, scenarios.items()):
            assert axis.texts[0].get_text() == subtitle
            assert axis.get_legend() is None
            matching = runs.loc[runs.shift_type.eq(kind) & runs.shift_fraction.eq(level) & runs.method.eq("dp_sgd")]
            assert len(axis.containers) == len(plots.INITIALIZATIONS)
            for initialization, series in zip(plots.INITIALIZATIONS, axis.containers):
                means = matching.loc[matching.pretraining.eq(initialization)].groupby("comparison_epsilon").auprc.mean()
                np.testing.assert_allclose(series.lines[0].get_xdata(), means.index)
                np.testing.assert_allclose(series.lines[0].get_ydata(), means)
                assert series.has_yerr

        report.run("unused", progress=lambda message: None, shift="age_70_plus", fraction=.7, epsilon=.5)
        single, caption = captured["overall_performance"][1]
        assert len(single.axes) == 1
        assert single.axes[0].texts[0].get_text() == "70% of source patients aged ≥70"
        assert single.legends[0].get_texts()[1].get_text() == "DP-SGD (ε = 0.5)"
        assert "scores vary across panels" not in caption
        single, caption = captured["privacy_utility"][1]
        assert len(single.axes) == 1
        assert single.axes[0].texts[0].get_text() == "70% of source patients aged ≥70"
        np.testing.assert_allclose(single.axes[0].containers[2].lines[0].get_xdata(), [.5, 1])
        np.testing.assert_allclose(single.axes[0].containers[2].lines[0].get_ydata(), [.335, .345])
        assert "scores can vary across panels" not in caption
    finally:
        for figures in captured.values():
            for figure, _ in figures:
                plots.plt.close(figure)


def test_amplification_shares_scale_and_legend_across_the_four_shift_panels(monkeypatch):
    from dp_shift.evaluation import plots

    baselines = {"readmission": .09, "age_70_plus": .44,
                 "primary_diagnosis_circulatory": 0, "clinical_complexity_2plus": 0}
    fractions = {"readmission": [.01, .17], "age_70_plus": [.5, .9],
                 "primary_diagnosis_circulatory": [.05, .1], "clinical_complexity_2plus": [.05, .2]}
    budgets = [1., 5., 10.]
    rows = []
    for index, (kind, levels) in enumerate(fractions.items()):
        for level, fraction in enumerate(levels):
            for budget in budgets:
                for repeat in range(3):
                    rows.append(dict(shift_type=kind, shift_fraction=fraction, comparison_epsilon=budget,
                                     repeat=repeat,
                                     dp_amplification=(index + 1) * (level + 1) * .01 + budget * .001 + (repeat - 1) * .001))
    captured = []
    monkeypatch.setattr(plots, "save", lambda figure, *args, **kwargs: captured.append(figure))
    try:
        plots.dp_amplification(pd.DataFrame(rows), Path("plots"), budgets,
                               dict(zip(budgets, ["red", "green", "blue"])), baselines)
        figure = captured[0]
        axes = figure.axes
        assert len(axes) == 4
        assert axes[0].get_gridspec().get_geometry() == (2, 2)
        assert len(figure.legends) == 1
        assert figure.legends[0].get_title().get_text() == "DP-SGD privacy budget"
        assert [label.get_text() for label in figure.legends[0].get_texts()] == ["ε=1", "ε=5", "ε=10"]
        assert sum("No additional DP effect" in text.get_text() for text in figure.texts) == 1
        for index, (axis, (kind, baseline)) in enumerate(zip(axes, baselines.items())):
            title, description, xlabel = plots.SOURCE_SHIFT_PANELS[kind]
            assert axis.get_title(loc="left") == title
            assert description in {text.get_text() for text in axis.texts}
            assert axis.get_xlabel() == xlabel
            assert axis.get_ylim() == axes[0].get_ylim()
            assert axis.get_legend() is None
            zero_guides = [line for line in axis.lines if line.get_linestyle() == "--"]
            assert len(zero_guides) == 1
            np.testing.assert_allclose(zero_guides[0].get_ydata(), [0, 0])
            baselines_guides = [line for line in axis.lines if line.get_linestyle() == ":"]
            if baseline:
                assert len(baselines_guides) == 1
                np.testing.assert_allclose(baselines_guides[0].get_xdata(), [baseline, baseline])
            else:
                assert not baselines_guides
                assert any(label.get_text() == "0%\nUnshifted source" for label in axis.get_xticklabels())
            for budget, container in zip(budgets, axis.containers):
                expected = sorted([(baseline, 0), *[(fraction, (index + 1) * (level + 1) + budget * .1)
                                                   for level, fraction in enumerate(fractions[kind])]])
                np.testing.assert_allclose(container.lines[0].get_xdata(), [x for x, _ in expected])
                np.testing.assert_allclose(container.lines[0].get_ydata(), [y for _, y in expected])
                for (x, _), segment in zip(expected, container.lines[2][0].get_segments()):
                    if x == baseline:
                        np.testing.assert_allclose(segment[:, 1], 0)
                    else:
                        assert segment[1, 1] > segment[0, 1]
        assert len({axis.get_xlim() for axis in axes}) == 4
    finally:
        for figure in captured:
            plots.plt.close(figure)
