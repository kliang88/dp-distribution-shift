"""Validate the signed study configuration without files."""

import pytest

from dp_shift.config import parse_config


def test_signed_config_sorts_and_copies_choices(choices):
    choices["shift_pp"].reverse()
    config = parse_config(choices)
    assert config.shift_pp == (-2, 2)
    assert choices["shift_pp"] == [2, -2]
    assert config.seed_for("partition") == parse_config(choices).seed_for("partition")
    assert config.seed_for("partition") != config.seed_for("s0")
    assert config.data is not choices["data"]


@pytest.mark.parametrize("field,value", [("seed", True), ("repeats", 0), ("shift_pp", [0]),
                                         ("shift_pp", [2, 2]), ("shift_pp", [-3, 4]),
                                         ("epsilons", [6.]), ("epsilons", [True]), ("epsilons", [])])
def test_invalid_signed_config_is_rejected(choices, field, value):
    choices[field] = value
    with pytest.raises(ValueError):
        parse_config(choices)


def test_config_rejects_fractional_patient_swaps_and_nonintegral_batches(choices):
    choices["pretraining"]["sample_size"] = 41
    with pytest.raises(ValueError, match="exact patient counts"):
        parse_config(choices)
    choices["pretraining"]["sample_size"] = 50
    choices["data"]["target_train_size"] = 23
    with pytest.raises(ValueError, match="divisible"):
        parse_config(choices)
