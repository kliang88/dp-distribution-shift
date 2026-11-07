"""Two direct training loops; no test-data access or checkpoint selection."""

import math

import torch
from torch.nn import functional as F

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


def pretrain_source(model, data, settings, shuffle_generator):
    loader = source_loader(data, settings["source_batch_size"], shuffle_generator)
    optimizer = make_optimizer(model, settings["source_lr"], settings["momentum"])
    model.train()
    losses, steps = [], 0
    for _ in range(settings["source_epochs"]):
        total = 0.
        for x, y in loader:
            # Ordinary mean BCE, including the final partial batch.
            total += step_batch(model, optimizer, x, y, len(y)) * len(y)
            steps += 1
        losses.append(total / len(loader.dataset))
    expected = settings["source_epochs"] * math.ceil(len(loader.dataset) / settings["source_batch_size"])
    require(steps == expected, "Incorrect source-pretraining step count")
    return model, {"steps": steps, "epoch_losses": losses}


def train_target(model, data, settings, method, sampling_generator, noise_generator=None):
    require(method in {"sgd", "dp_sgd"}, "Unknown target optimizer")
    private = method == "dp_sgd"
    loader = poisson_loader(data, settings["steps_per_epoch"], sampling_generator)
    optimizer = make_optimizer(model, settings["target_lr"], settings["momentum"])
    engine = None
    if private:
        require(settings["noise_multiplier"] > 0 and settings["clipping_norm"] > 0,
                "DP training requires positive noise and clipping")
        require(noise_generator is not None, "DP training requires a separate noise generator")
        model, optimizer, loader, engine = wrap_private(
            model, optimizer, loader, settings["noise_multiplier"], settings["clipping_norm"], noise_generator)
    model.train()
    expected_batch = len(loader.dataset) // settings["steps_per_epoch"]
    batch_sizes, losses, steps = [], [], 0
    for _ in range(settings["target_epochs"]):
        total = 0.
        for x, y in loader:
            total += step_batch(model, optimizer, x, y, expected_batch, private)
            batch_sizes.append(len(y))
            steps += 1
        losses.append(total / settings["steps_per_epoch"])
    require(steps == settings["target_epochs"] * settings["steps_per_epoch"], "Incorrect target step count")
    diagnostics = {"steps": steps, "batch_sizes": batch_sizes, "epoch_losses": losses}
    if private:
        diagnostics["privacy"] = privacy_summary(engine, settings, steps)
        diagnostics["accountant_state"] = engine.accountant.state_dict()
        model = model.to_standard_module()
    return model, diagnostics
