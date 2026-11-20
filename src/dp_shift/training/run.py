"""Train and score one repeat, keeping only one source cohort in memory."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dp_shift.data.sampling import shifted_indices
from dp_shift.data.shifts import SHIFTS
from .loaders import generator
from .model import build_model, model_state
from .privacy import private_generators
from .train import pretrain_source, train_target
from .tune import predict


def run_repeat(repeat, plan, source, target, test, masks, ordinary, private):
    config = ordinary.config
    dimension = source["X"].shape[1]
    baseline, swaps = plan
    seed = config.seed_for("initialization") + repeat - 1
    shuffle_seed = config.seed_for("source_shuffle") + repeat - 1
    sampling_seed = config.seed_for("target_sampling") + repeat - 1
    labels = test["y"]
    shared = {"repeat": repeat, "seed": config.seed, "target_train_size": len(target["y"]),
              "target_test_size": len(labels), "target_test_readmission_count": int(labels.sum()),
              "target_lr": ordinary.selected.target_lr, "target_epochs": ordinary.selected.target_epochs}
    rows = []

    def evaluate(kind, level, source_info, initial=None):
        for chosen in (ordinary, *private):
            private_run = chosen.epsilon is not None
            method = "dp_sgd" if private_run else "sgd"
            model = build_model(dimension, seed)
            if initial is not None:
                model.load_state_dict(initial)
            sampling, noise = private_generators() if private_run else (generator(sampling_seed), None)
            trained, diagnostics = train_target(model, target, chosen, method, sampling, noise)
            probabilities = predict(trained, test)
            privacy = diagnostics.get("privacy", {})
            rows.append({**shared, **source_info, "shift_type": kind, "shift_pp": level,
                         "method": "DP-SGD" if private_run else "SGD", "epsilon": chosen.epsilon,
                         "auprc": float(average_precision_score(labels, probabilities)),
                         "auroc": float(roc_auc_score(labels, probabilities)),
                         "brier_score": float(brier_score_loss(labels, probabilities)),
                         "target_steps": diagnostics["steps"],
                         "epsilon_achieved": privacy.get("epsilon_achieved"), "delta": privacy.get("delta"),
                         "clipping_norm": privacy.get("clipping_norm"), "noise_multiplier": privacy.get("noise_multiplier")})

    def evaluate_pretrained(kind, level, indices):
        source_info = {"source_size": len(indices), "source_lr": ordinary.selected.source_lr,
                       "source_epochs": config.source_epochs,
                       **{f"source_{name}_count": np.count_nonzero(mask[indices]) for name, mask in masks.items()}}
        cohort = {name: values[indices] for name, values in source.items()}
        pretrained, _ = pretrain_source(build_model(dimension, seed), cohort, ordinary, generator(shuffle_seed))
        del cohort
        evaluate(kind, level, source_info, model_state(pretrained))

    evaluate("scratch", 0, {})
    evaluate_pretrained("baseline", 0, baseline)
    for kind in SHIFTS:
        for level in config.shift_pp:
            count = abs(level) * config.pretrain_size // 100
            evaluate_pretrained(kind, level, shifted_indices(baseline, swaps[kind, int(np.sign(level))], count))
    return rows
