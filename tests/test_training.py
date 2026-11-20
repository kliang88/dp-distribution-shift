"""Small checks for optimization, privacy, and proxy selection."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from dp_shift.config import PROTOCOL, SelectedHyperparameters, parse_config
from dp_shift.training.loaders import generator, poisson_loader, source_loader
from dp_shift.training.model import build_model, configure_cpu, model_state
from dp_shift.training.privacy import accounted_steps, predicted_epsilon, private_generators, wrap_private
from dp_shift.training.train import make_optimizer, pretrain_source, step_batch, train_target
from dp_shift.training.tune import best_candidate, settings as training_settings


@pytest.fixture(autouse=True)
def cpu():
    configure_cpu()


@pytest.fixture
def settings(choices):
    config = replace(parse_config(choices), source_epochs=2)
    return training_settings(config, SelectedHyperparameters(.01, .01, 1), 5.)


def tiny_data(n=24, d=3):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(n, d)).astype(np.float32)
    return {"X": x, "y": (x[:, 0] > 0).astype(np.int64), "input_dimension": d}


def test_model_accepts_empty_and_nonempty_batches():
    model = build_model(3, 42)
    for size in [0, 1, 7]:
        assert model(torch.zeros(size, 3)).squeeze(-1).shape == (size,)


def test_source_training_learns_and_keeps_partial_batches(settings):
    data = tiny_data(65)
    loader = source_loader(data, 32, generator(1))
    batches = list(loader)
    assert [len(y) for _, y in batches] == [32, 32, 1]
    expected_rows = sorted(map(tuple, data["X"].tolist()))
    assert sorted(map(tuple, torch.cat([x for x, _ in batches]).tolist())) == expected_rows
    model = build_model(3, 1)
    x, y = torch.from_numpy(data["X"]), torch.tensor(data["y"], dtype=torch.float32)
    initial_loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y).item()
    settings = training_settings(replace(settings.config, source_epochs=30), SelectedHyperparameters(.1))
    trained, diagnostic = pretrain_source(model, data, settings, generator(1))
    final_loss = F.binary_cross_entropy_with_logits(trained(x).squeeze(-1), y).item()
    assert final_loss < initial_loss / 2
    assert diagnostic["steps"] == 30


def test_expected_batch_gradient_matches_patient_gradient_sum():
    model = build_model(3, 4)
    initial = model_state(model)
    data = tiny_data()
    x, y = torch.from_numpy(data["X"][:3]), torch.tensor(data["y"][:3], dtype=torch.float32)
    gradients = {name: torch.zeros_like(value) for name, value in initial.items()}
    for row in range(3):
        model.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(x[row:row+1]).squeeze(-1), y[row:row+1]).backward()
        for name, parameter in model.named_parameters():
            gradients[name] += parameter.grad.detach() / 2
    model.load_state_dict(initial)
    step_batch(model, make_optimizer(model, .01), x, y, expected_batch_size=2)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name] - .01 * gradients[name], rtol=1e-5, atol=1e-7)


def test_private_gradient_parity_and_empty_momentum_step():
    data = tiny_data()
    ordinary, private = build_model(3, 10), build_model(3, 10)
    opt = make_optimizer(ordinary, .01)
    loader = poisson_loader(data, 12, generator(1))
    private, dp_opt, _, engine = wrap_private(private, make_optimizer(private, .01), loader,
                                             noise_multiplier=0., clipping_norm=1e6, noise_generator=generator(2))
    x, y = torch.from_numpy(data["X"][:3]), torch.tensor(data["y"][:3], dtype=torch.float32)
    for batch_x, batch_y in [(x, y), (x[:0], y[:0])]:
        step_batch(ordinary, opt, batch_x, batch_y, 2)
        step_batch(private, dp_opt, batch_x, batch_y, 2, private=True)
        for left, right in zip(ordinary.parameters(), private.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
    assert accounted_steps(engine.accountant) == 2


def test_empty_private_batch_adds_noise_and_is_accounted():
    data = tiny_data()
    model = build_model(3, 1)
    before = model_state(model)
    loader = poisson_loader(data, 12, generator(1))
    model, optimizer, _, engine = wrap_private(model, make_optimizer(model, .01), loader,
                                              noise_multiplier=1., clipping_norm=1., noise_generator=generator(2))
    loss = step_batch(model, optimizer, torch.empty(0, 3), torch.empty(0), 2, private=True)
    assert loss == 0 and accounted_steps(engine.accountant) == 1
    assert all(parameter.grad_sample.shape[0] == 0 for parameter in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model._module.state_dict().items())


def test_private_sampling_and_noise_use_independent_entropy():
    sampling, noise = private_generators()
    other_sampling, other_noise = private_generators()
    states = {generator.get_state().numpy().tobytes() for generator in (sampling, noise, other_sampling, other_noise)}
    assert len(states) == 4


def test_private_training_accounts_for_each_step_and_stays_within_budget(settings):
    _, result = train_target(build_model(3, 1), tiny_data(), settings, "dp_sgd", generator(1), generator(2))
    epsilon = predicted_epsilon(settings.noise_multiplier, 1/12, 12, PROTOCOL.delta)
    assert result["steps"] == result["privacy"]["steps"] == 12
    assert result["privacy"]["epsilon_achieved"] == pytest.approx(epsilon)
    assert epsilon <= settings.epsilon
    assert sum(count for _, _, count in result["privacy"]["accountant_history"]) == 12
    invalid = replace(settings, noise_multiplier=0.)
    with pytest.raises(ValueError, match="positive noise"):
        train_target(build_model(3, 1), tiny_data(), invalid, "dp_sgd", generator(1), generator(2))


def test_proxy_tuning_breaks_ties_using_brier_epochs_and_learning_rate():
    candidates = [(.2, .1, 8, .03), (.2000005, .1, 8, .01),
                  (.2, .11, 8, .003), (.2, .1, 16, .003), (.19, .01, 8, .003)]
    assert best_candidate(candidates) == candidates[1]
    with pytest.raises(ValueError, match="No valid"):
        best_candidate([])
