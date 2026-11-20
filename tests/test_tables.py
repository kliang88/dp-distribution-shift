"""Validate the study's results contract in memory."""

import pandas as pd
import pytest

from dp_shift.evaluation import tables


def test_full_design_has_shared_baseline_and_scratch_controls_per_method(signed_results):
    frame = signed_results()
    tables.validate(frame)
    assert len(frame) == 4080
    assert len(frame.loc[frame.shift_type.eq("baseline")]) == 120
    assert len(frame.loc[frame.shift_type.eq("scratch")]) == 120
    assert frame.loc[frame.method.eq("SGD"), "epsilon"].isna().all()


@pytest.mark.parametrize("change", [
    lambda frame: pd.concat([frame, frame.iloc[[0]]]),
    lambda frame: frame.iloc[1:],
    lambda frame: frame.assign(shift_pp=0),
    lambda frame: frame.assign(epsilon=5),
    lambda frame: frame.assign(auprc=-.1),
    lambda frame: frame.assign(epsilon_achieved=6),
    lambda frame: frame.assign(target_test_readmission_count=1100),
    lambda frame: frame.assign(source_size=1000),
    lambda frame: frame.assign(unexpected="extra"),
])
def test_invalid_results_fail(change, signed_results):
    with pytest.raises(ValueError):
        tables.validate(change(signed_results(2, (-2, 2), (5.,))))
