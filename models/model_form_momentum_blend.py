# %%
"""Blend of the shrunk-EWM ensemble and a ridge over recent-form features.

The EWM ensemble (see model_form_momentum_ewm) is the robust base. The ridge
adds a pooled linear view over multiple momentum horizons plus the expanding
mean and evidence count, which lets the data choose how much weight short vs
long form deserves. A fixed 50/50 blend had the best forward-chaining CV mean
and worst-fold error during screening on the collected LAD 2025 relief data.

All features derive from prior outings only (shift(1) within pitcher).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.evaluate import evaluate_predictions
from lib.modeling import NumpyRidgeRegressor, clip_predictions, get_git_commit
from lib.momentum_features import (
    add_prior_target_features,
    combine_train_test,
    ewm_grid_ensemble_prediction,
)


CONFIG = {
    "anchor": 0.5,
    "ensemble_grid": [(2, 5), (3, 5), (5, 5), (8, 5), (2, 10), (3, 10), (5, 10), (8, 10)],
    "ridge_features": ["prior_y_ewm2", "prior_y_ewm5", "prior_y_ewm12", "prior_y_expmean", "prior_y_count"],
    "ridge_alpha": 50.0,
    "blend_weight_ewm": 0.5,
}


# %%
def _make_ridge(alpha: float):
    try:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=alpha)
    except Exception:
        return NumpyRidgeRegressor(alpha=alpha)


# %%
def _design(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    x_train = train[columns].replace([np.inf, -np.inf], np.nan)
    x_test = test[columns].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median().fillna(0.0)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1.0)
    return (
        ((x_train - mean) / std).to_numpy(dtype=float),
        ((x_test - mean) / std).to_numpy(dtype=float),
    )


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})

    combined, is_test = combine_train_test(train_df, test_df)
    halflives = sorted(
        {hl for hl, _ in cfg["ensemble_grid"]}
        | {int(col.replace("prior_y_ewm", "")) for col in cfg["ridge_features"] if col.startswith("prior_y_ewm")}
    )
    combined = add_prior_target_features(combined, halflives=halflives)

    train_rows = combined.loc[~is_test]
    test_rows = combined.loc[is_test].sort_values("_test_pos")

    ewm_pred = ewm_grid_ensemble_prediction(test_rows, cfg["ensemble_grid"], float(cfg["anchor"]))

    x_train, x_test = _design(train_rows, test_rows, list(cfg["ridge_features"]))
    ridge = _make_ridge(float(cfg["ridge_alpha"]))
    ridge.fit(x_train, train_rows["target_y"].to_numpy(dtype=float))
    ridge_pred = ridge.predict(x_test)

    w = float(cfg["blend_weight_ewm"])
    predictions = clip_predictions(w * ewm_pred + (1.0 - w) * ridge_pred)
    metrics = evaluate_predictions(test_df, predictions)

    return {
        "model_name": "form_momentum_blend",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": ridge,
        "git_commit": get_git_commit(),
    }
