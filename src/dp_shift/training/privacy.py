"""Opacus setup, independently checked accounting, and local private RNG states."""

from functools import lru_cache
import math
import secrets
import warnings

from opacus import PrivacyEngine
from opacus.accountants import PRVAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator

from dp_shift.config import PROTOCOL, TrainingSettings
from dp_shift.data.utils import require
from .loaders import generator


def accounted_steps(accountant):
    return sum(count for _, _, count in accountant.history)


@lru_cache(maxsize=64)
def history_epsilon(history, delta):
    accountant = PRVAccountant()
    accountant.history = list(history)
    return float(accountant.get_epsilon(delta=delta))


def predicted_epsilon(noise_multiplier, sample_rate, steps, delta):
    return history_epsilon(((noise_multiplier, sample_rate, steps),), delta)


@lru_cache(maxsize=32)
def calibrate_noise(epochs, steps_per_epoch=12, delta=1e-5, epsilon_cap=8., target_epsilon=7.99):
    require(0 < target_epsilon <= epsilon_cap and 0 < delta < 1, "Invalid privacy budget")
    steps = epochs * steps_per_epoch
    sigma = float(get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                      sample_rate=1 / steps_per_epoch, steps=steps,
                                      accountant="prv", epsilon_tolerance=.001))
    while predicted_epsilon(sigma, 1 / steps_per_epoch, steps, delta) > epsilon_cap:
        sigma *= 1.001
    return sigma


def wrap_private(model, optimizer, loader, noise_multiplier, clipping_norm, noise_generator):
    ModuleValidator.validate(model, strict=True)
    with warnings.catch_warnings():
        # This mode is explicitly part of the public-data simulation configuration.
        warnings.filterwarnings("ignore", message="Secure RNG turned off.*", category=UserWarning)
        engine = PrivacyEngine(accountant="prv", secure_mode=False)
    model, optimizer, private_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=clipping_norm,
        noise_generator=noise_generator, clipping="flat", loss_reduction="mean",
        grad_sample_mode="hooks", poisson_sampling=True)
    require(len(private_loader) == len(loader), "Private wrapping changed the loader length")
    require(private_loader.sample_rate == loader.sample_rate, "Private wrapping changed the sample rate")
    expected = len(loader.dataset) // len(loader)
    require(optimizer.expected_batch_size == expected, "Incorrect private expected batch size")
    return model, optimizer, private_loader, engine


def privacy_summary(engine, settings, steps):
    actual_steps = accounted_steps(engine.accountant)
    require(actual_steps == steps, "Accountant and optimizer step counts disagree")
    history = tuple(engine.accountant.history)
    epsilon = history_epsilon(history, PROTOCOL.delta)
    require(math.isfinite(epsilon) and epsilon <= settings.epsilon, "Achieved epsilon exceeds the budget")
    return {"accountant": "prv", "epsilon_achieved": epsilon, "delta": PROTOCOL.delta,
            "noise_multiplier": settings.noise_multiplier, "clipping_norm": settings.config.clipping_norm,
            "sampling_rate": 1 / PROTOCOL.steps_per_epoch, "steps": actual_steps,
            "accountant_history": [list(entry) for entry in engine.accountant.history]}


def private_generators():
    return generator(secrets.randbits(63)), generator(secrets.randbits(63))
