"""Two direct training loops; no test-data access or checkpoint selection."""

import math

import torch
from torch.nn import functional as F

from dp_shift.config import PROTOCOL, TrainingSettings
from dp_shift.data.utils import require
from .loaders import poisson_loader, source_loader
from .privacy import privacy_summary, wrap_private


def make_optimizer(model, learning_rate, momentum=.9):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                           dampening=0, weight_decay=0, nesterov=False)


def check_parameters(model):
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite model parameters")


def step_batch(model, optimizer, x, y, expected_batch_size, private=False):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x).squeeze(-1)
    if private:
        loss = F.binary_cross_entropy_with_logits(logits, y) if len(y) else logits.sum()
    else:
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / expected_batch_size
    require(torch.isfinite(loss).item(), "Nonfinite training loss")
    loss.backward()
    optimizer.step()
    check_parameters(model)
    return float(loss.detach())


def pretrain_source(model, data, settings: TrainingSettings, shuffle_generator):
    loader = source_loader(data, PROTOCOL.source_batch_size, shuffle_generator)
    optimizer = make_optimizer(model, settings.selected.source_lr, PROTOCOL.momentum)
    epochs = settings.config.source_epochs
    model.train()
    steps = 0
    for _ in range(epochs):
        for x, y in loader:
            # Ordinary mean BCE, including the final partial batch.
            step_batch(model, optimizer, x, y, len(y))
            steps += 1
    expected = epochs * math.ceil(len(loader.dataset) / PROTOCOL.source_batch_size)
    require(steps == expected, "Incorrect source-pretraining step count")
    return model, {"steps": steps}


def train_target(model, data, settings: TrainingSettings, method, sampling_generator, noise_generator=None):
    require(method in {"sgd", "dp_sgd"}, "Unknown target optimizer")
    require(settings.selected.target_lr is not None and settings.selected.target_epochs is not None,
            "Target training requires selected hyperparameters")
    private = method == "dp_sgd"
    loader = poisson_loader(data, PROTOCOL.steps_per_epoch, sampling_generator)
    optimizer = make_optimizer(model, settings.selected.target_lr, PROTOCOL.momentum)
    engine = None
    if private:
        clipping_norm = settings.config.clipping_norm
        require(settings.noise_multiplier is not None and settings.noise_multiplier > 0
                and clipping_norm > 0,
                "DP training requires positive noise and clipping")
        require(noise_generator is not None, "DP training requires a separate noise generator")
        model, optimizer, loader, engine = wrap_private(
            model, optimizer, loader, settings.noise_multiplier, clipping_norm, noise_generator)
    model.train()
    expected_batch = len(loader.dataset) // PROTOCOL.steps_per_epoch
    steps = 0
    for _ in range(settings.selected.target_epochs):
        for x, y in loader:
            step_batch(model, optimizer, x, y, expected_batch, private)
            steps += 1
    require(steps == settings.selected.target_epochs * PROTOCOL.steps_per_epoch, "Incorrect target step count")
    diagnostics = {"steps": steps}
    if private:
        diagnostics["privacy"] = privacy_summary(engine, settings, steps)
        model = model.to_standard_module()
    return model, diagnostics
