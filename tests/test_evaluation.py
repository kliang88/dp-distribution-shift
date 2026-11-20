"""Check that signed plots compare paired scores in the intended direction."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from dp_shift.evaluation.summarize import interaction, interval, signed_effects
from dp_shift.evaluation.tables import METRICS, MODEL_KEYS


def test_baseline_join_pairs_by_repeat_method_and_privacy_budget():
    rows = [
        (1, "baseline", 0, "SGD", None, .20, .6, .1),
        (1, "baseline", 0, "DP-SGD", 5., .18, .6, .1),
        (1, "baseline", 0, "DP-SGD", 10., .19, .6, .1),
        (1, "scratch", 0, "SGD", None, .17, .6, .1),
        (1, "scratch", 0, "DP-SGD", 5., .15, .6, .1),
        (1, "scratch", 0, "DP-SGD", 10., .17, .6, .1),
        (1, "age", -8, "SGD", None, .18, .6, .1),
        (1, "age", -8, "DP-SGD", 5., .13, .6, .1),
        (1, "age", -8, "DP-SGD", 10., .18, .6, .1),
        (2, "baseline", 0, "SGD", None, .30, .6, .1),
        (2, "scratch", 0, "SGD", None, .34, .6, .1),
        (2, "age", -8, "SGD", None, .31, .6, .1),
    ]
    scores = pd.DataFrame(rows, columns=[*MODEL_KEYS, *METRICS])
    effects = signed_effects(scores)
    np.testing.assert_allclose(effects.gain, [-.02, -.05, -.01, .01])
    extra = interaction(effects).set_index("epsilon").interaction
    assert extra.to_dict() == pytest.approx({5.: .03, 10.: -.01})
    np.testing.assert_allclose(signed_effects(scores, reference="scratch").gain, [.01, -.02, .01, -.03])


def test_interval_uses_repeat_level_sample_variance():
    mean, half_width = interval([1, 3])
    assert mean == 2 and half_width == pytest.approx(t.ppf(.975, 1))
    mean, half_width = interval([.01])
    assert mean == .01 and np.isnan(half_width)
