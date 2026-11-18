"""Inspect plotted values to check selections, paired contrasts, and repeat counts."""

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
        np.testing.assert_allclose([bar.get_height() for bar in bars[0]], [.11, .14, .12])
        np.testing.assert_allclose([bar.get_height() for bar in bars[1]], [.09, .16, .12])
        assert "20.0%" in overall.get_title()

        gain = captured["pretraining_benefit"].axes[0].containers[0]
        np.testing.assert_allclose(gain.lines[0].get_xdata(), [.01, .09, .2])
        np.testing.assert_allclose(gain.lines[0].get_ydata(), [2, 3, 1])
        # Three paired repetitions, rather than pooling copies across epsilon cells.
        for segment in gain.lines[2][0].get_segments():
            assert segment[0, 1] == pytest.approx(segment[1, 1])

        amplification = captured["dp_amplification"].axes
        assert len(amplification) == 1
        np.testing.assert_allclose(amplification[0].containers[0].lines[0].get_ydata(), [2, 0, 3])
        np.testing.assert_allclose(amplification[0].containers[1].lines[0].get_ydata(), [2, 0, 2])
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
            title, description, xlabel = plots.AMPLIFICATION_PANELS[kind]
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
