"""One MLP shared by every training path."""

import torch
from torch import nn


def build_model(input_dimension, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(input_dimension, 64), nn.ReLU(),
                             nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
