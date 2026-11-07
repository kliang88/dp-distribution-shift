"""One MLP shared by every training path."""

import hashlib

import torch
from torch import nn


def build_model(input_dimension, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(input_dimension, 64), nn.ReLU(),
                             nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
