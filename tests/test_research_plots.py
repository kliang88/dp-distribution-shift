"""Inspect plotted values to check selections, paired contrasts, and repeat counts."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from dp_shift.evaluation.tables import CONDITIONS
from dp_shift.evaluation.summarize import paired_effects


def test_five_figures_use_paired_repeats_and_positive_amplification(monkeypatch, tmp_path):
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
                    source = {"none": np.nan, "baseline": .09, "shifted": prevalence}[initialization]
                    raw.append(dict(repeat=repeat, method=method, pretraining=initialization,
                                    shift_type="readmission", shift_fraction=prevalence, comparison_epsilon=epsilon, auprc=value,
                                    source_fraction=source, source_size=20000 if initialization != "none" else np.nan,
                                    test_fraction=.1, test_size=100))
    runs = pd.DataFrame(raw)
    captured = {}
    monkeypatch.setattr(plots, "save", lambda figure, output, name, caption, **kwargs: captured.update({name: figure}))
    try:
        count = plots.plot(runs, paired_effects(runs), 100, tmp_path, epsilon=5., fractions={"readmission": .2})
        assert count == 5 and set(captured) == set(plots.FIGURES)
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
        distribution = captured["source_target_prevalence"].axes[0]
        np.testing.assert_allclose([bar.get_height() for bar in distribution.containers[0]], [.1, .09, .01, .2])
        assert "n=100" in distribution.get_xticklabels()[0].get_text()
        assert "n=20,000" in distribution.get_xticklabels()[1].get_text()
    finally:
        for figure in captured.values():
            plots.plt.close(figure)


def test_overall_performance_uses_planned_levels_and_matched_panel_scores(monkeypatch, tmp_path):
    from dp_shift.evaluation import plots, report, tables

    scenarios = {
        "readmission": ((.01, .17), .01, "1% source readmission rate"),
        "age_70_plus": ((.7, .8), .8, "80% of source patients aged ≥70"),
        "primary_diagnosis_circulatory": ((.05, .1), .05, "4.9505% source replacement · Circulatory primary diagnosis"),
        "clinical_complexity_2plus": ((.05, .1), .05, "4.9505% source replacement · ≥2 diagnosis categories"),
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
    path = tmp_path / "runs.csv"
    tables.write(path, runs)
    captured = []

    def inspect(figure, directory, name, caption, **kwargs):
        if name == "overall_performance":
            captured.append((figure, caption))
        else:
            plots.plt.close(figure)

    monkeypatch.setattr(plots, "save", inspect)
    try:
        report.run(path, progress=lambda message: None)
        figure, caption = captured[0]
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

        report.run(path, progress=lambda message: None, shift="age_70_plus", fraction=.7, epsilon=.5)
        single, caption = captured[1]
        assert len(single.axes) == 1
        assert single.axes[0].texts[0].get_text() == "70% of source patients aged ≥70"
        assert single.legends[0].get_texts()[1].get_text() == "DP-SGD (ε = 0.5)"
        assert "scores vary across panels" not in caption
    finally:
        for figure, _ in captured:
            plots.plt.close(figure)


def test_amplification_shares_scale_and_legend_across_the_four_shift_panels(monkeypatch, tmp_path):
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
        plots.dp_amplification(pd.DataFrame(rows), tmp_path, budgets,
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
