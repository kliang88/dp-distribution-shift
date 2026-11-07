"""Explicit tensor conversion and the two sampling schemes."""

import numpy as np
import torch
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader, TensorDataset

from dp_shift.data.utils import require


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def tensor_dataset(data):
    x = torch.as_tensor(data["X"], dtype=torch.float32)
    y = torch.as_tensor(data["y"], dtype=torch.float32)
    require(x.ndim == 2 and y.shape == (len(x),) and len(x) > 0, "Invalid training array shapes")
    require(torch.isfinite(x).all() and torch.isfinite(y).all(), "Nonfinite training data")
    require(torch.all((y == 0) | (y == 1)), "Training labels must be binary")
    return TensorDataset(x, y)


def source_loader(data, batch_size, rng):
    return DataLoader(tensor_dataset(data), batch_size=batch_size, shuffle=True,
                      drop_last=False, num_workers=0, generator=rng)


def poisson_loader(data, steps_per_epoch, rng):
    dataset = tensor_dataset(data)
    require(len(dataset) % steps_per_epoch == 0,
            "Patient count must be divisible by steps_per_epoch for exact expected-batch normalization")
    loader = DPDataLoader(dataset, sample_rate=1 / steps_per_epoch, generator=rng, num_workers=0)
    require(len(loader) == steps_per_epoch, "Poisson loader has the wrong number of steps")
    require(np.isclose(loader.sample_rate, 1 / steps_per_epoch), "Incorrect Poisson sample rate")
    return loader
