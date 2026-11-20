"""Check binary traits and exact paired signed samples."""

import numpy as np
import pandas as pd
import pytest

from dp_shift.config import SIGNED_LEVELS, parse_config
from dp_shift.data.sampling import plan_swaps, shifted_indices
from dp_shift.data.shifts import SHIFTS


def test_indicators_use_age_bins_and_distinct_recognized_diagnoses():
    frame = pd.DataFrame({"y": [0, 1, 0, 1], "age": ["[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                          "diag_1": ["414", "250.83", "401", "?"], "diag_2": ["427", "486", "?", "invalid"],
                          "diag_3": ["?", "250", "250.01", "300"]})
    assert SHIFTS["readmission"](frame).tolist() == [False, True, False, True]
    assert SHIFTS["age"](frame).tolist() == [False, True, True, True]
    assert SHIFTS["circulatory"](frame).tolist() == [True, False, True, False]
    assert SHIFTS["complexity"](frame).tolist() == [False, True, True, False]
    frame.loc[0, "age"] = "unknown"
    with pytest.raises(ValueError, match="valid ten-year age bins"):
        SHIFTS["age"](frame)


def test_all_signed_shifts_are_exact_nested_and_label_matched(choices):
    choices["shift_pp"] = list(SIGNED_LEVELS)
    config = parse_config(choices)
    rows = np.arange(500)
    labels = rows % 3 == 0
    masks = {"readmission": labels, "age": rows % 2 == 0,
             "circulatory": rows % 5 < 2, "complexity": rows % 7 < 5}
    plans = plan_swaps(labels, masks, config)
    for baseline, orders in plans:
        for kind, mask in masks.items():
            for direction in (-1, 1):
                last = set()
                for magnitude in (2, 4, 6, 8):
                    count = config.pretrain_size * magnitude // 100
                    shifted = shifted_indices(baseline, orders[kind, direction], count)
                    assert len(np.unique(shifted)) == len(baseline)
                    assert int(mask[shifted].sum()) - int(mask[baseline].sum()) == direction * count
                    added = set(shifted) - set(baseline)
                    assert last < added and len(added) == count
                    last = added
                    if kind != "readmission":
                        assert labels[shifted].sum() == labels[baseline].sum()
    duplicate_plans = plan_swaps(labels, masks, config)
    np.testing.assert_array_equal(plans[0][0], duplicate_plans[0][0])
    np.testing.assert_array_equal(plans[0][1]["age", -1], duplicate_plans[0][1]["age", -1])


def test_infeasible_feature_swap_is_rejected_before_training(choices):
    config = parse_config(choices)
    labels = np.arange(500) % 3 == 0
    with pytest.raises(ValueError, match=r"Repeat 1, age, direction -1: requested 1 swaps, capacity 0"):
        plan_swaps(labels, {"age": np.ones_like(labels)}, config)
