"""Select one set of training parameters on the public proxy splits."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss

from dp_shift.config import PROTOCOL, SelectedHyperparameters, TrainingSettings
from dp_shift.data.utils import require
from .loaders import generator
from .model import build_model, model_state
from .privacy import calibrate_noise
from .train import pretrain_source, train_target


def settings(config, selected, epsilon=None):
    noise = None
    if epsilon is not None:
        noise = calibrate_noise(selected.target_epochs, PROTOCOL.steps_per_epoch, PROTOCOL.delta,
                                epsilon, epsilon * (7.99 / 8))
    return TrainingSettings(config, selected, epsilon, noise)


@torch.inference_mode()
def predict(model, data):
    model.eval()
    size = PROTOCOL.prediction_batch_size
    result = np.empty(len(data["y"]), dtype=np.float32)
    for start in range(0, len(result), size):
        features = torch.as_tensor(data["X"][start:start + size], dtype=torch.float32)
        result[start:start + size] = torch.sigmoid(model(features).squeeze(-1)).numpy()
    require(np.isfinite(result).all(), "Nonfinite predictions")
    return result


def proxy_scores(model, validation):
    probability = predict(model, validation)
    labels = validation["y"]
    return float(average_precision_score(labels, probability)), float(brier_score_loss(labels, probability))


def best_candidate(candidates):
    """Select on AP, with ties broken by Brier score, epochs, then learning rate."""
    require(bool(candidates), "No valid public tuning candidates")
    best = max(row[0] for row in candidates)
    return min((row for row in candidates if row[0] >= best - 1e-6), key=lambda row: row[1:4])


def select_hyperparameters(config, source, proxy, validation):
    indices = np.random.RandomState(config.seed_for("tuning_sample")).choice(
        len(source["y"]), config.pretrain_size, replace=False)
    indices.sort()
    public_source = {name: array[indices] for name, array in source.items()}
    dimension = source["X"].shape[1]
    seed = config.seed_for("tuning_initialization")
    candidates = []
    for lr in config.source_learning_rates:
        model, _ = pretrain_source(build_model(dimension, seed), public_source, settings(config, SelectedHyperparameters(lr)),
                                   generator(config.seed_for("tuning_source_shuffle")))
        candidates.append((*proxy_scores(model, validation), 0, lr, model_state(model)))
    _, _, _, source_lr, initial = best_candidate(candidates)

    candidates = []
    for lr in config.target_learning_rates:
        for epochs in config.target_epoch_candidates:
            selected = SelectedHyperparameters(source_lr, lr, epochs)
            ordinary = settings(config, selected)
            private = settings(config, selected, config.reference_epsilon)
            scores = []
            for method, chosen in (("sgd", ordinary), ("dp_sgd", private)):
                for pretrained in (False, True):
                    model = build_model(dimension, seed)
                    if pretrained:
                        model.load_state_dict(initial)
                    sampling = generator(config.seed_for("tuning_sampling") + (method == "dp_sgd"))
                    noise = generator(config.seed_for(f"tuning_noise:{lr}:{epochs}:{pretrained}")) if method == "dp_sgd" else None
                    trained, _ = train_target(model, proxy, chosen, method, sampling, noise)
                    scores.append(proxy_scores(trained, validation))
            candidates.append((*np.mean(scores, axis=0), epochs, lr))
    _, _, epochs, lr = best_candidate(candidates)
    return SelectedHyperparameters(source_lr, lr, epochs)
