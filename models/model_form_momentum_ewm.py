# %%
"""Recent-form momentum model: shrunk EWM of prior outings' target_y.

Why this works when workload features alone do not: target_y has a strong
within-pitcher lag-1 autocorrelation (~+0.46 on the collected LAD 2025 relief
data), because the label's EWMA baseline adapts slowly to genuine form drift.
The single best predictor of today's baseline-relative performance is the
pitcher's own recent target history, shrunk toward the neutral anchor 0.5 by
evidence count so short histories stay conservative.

The default averages over a small (halflife, shrink_k) grid instead of picking
one setting: with only a few hundred outings, hyperparameter selection is
noisier than hyperparameter averaging.

All inputs are prior-outing values (shift(1) within pitcher), so the model
respects the project's prior-data-only leakage principle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.evaluate import evaluate_predictions
from lib.modeling import clip_predictions, get_git_commit
from lib.momentum_features import (
    add_prior_target_features,
    combine_train_test,
    ewm_grid_ensemble_prediction,
    shrunk_ewm_prediction,
)


CONFIG = {
    "anchor": 0.5,
    "ensemble": True,
    "ensemble_grid": [(2, 5), (3, 5), (5, 5), (8, 5), (2, 10), (3, 10), (5, 10), (8, 10)],
    # used when ensemble is False
    "halflife": 5,
    "shrink_k": 10,
}


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})

    combined, is_test = combine_train_test(train_df, test_df)
    halflives = sorted({hl for hl, _ in cfg["ensemble_grid"]} | {int(cfg["halflife"])})
    combined = add_prior_target_features(combined, halflives=halflives)

    test_rows = combined.loc[is_test].sort_values("_test_pos")
    anchor = float(cfg["anchor"])
    if cfg.get("ensemble", True):
        predictions = ewm_grid_ensemble_prediction(test_rows, cfg["ensemble_grid"], anchor)
    else:
        predictions = shrunk_ewm_prediction(test_rows, int(cfg["halflife"]), float(cfg["shrink_k"]), anchor)

    predictions = clip_predictions(predictions)
    metrics = evaluate_predictions(test_df, predictions)

    return {
        "model_name": "form_momentum_ewm",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": {"type": "shrunk_ewm_ensemble", "grid": cfg["ensemble_grid"], "anchor": anchor},
        "git_commit": get_git_commit(),
    }
