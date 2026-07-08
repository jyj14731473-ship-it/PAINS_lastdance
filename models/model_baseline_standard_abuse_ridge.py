# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.evaluate import evaluate_predictions
from lib.feature_sets import get_feature_set
from lib.modeling import NumpyRidgeRegressor, clip_predictions, get_git_commit
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "feature_set": "standard_abuse_only",
    "alpha": 1.0,
    "log1p_features": True,
    "standardize": True,
}


# %%
def _available_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    features = [col for col in get_feature_set(feature_set) if col in df.columns]
    if not features:
        raise ValueError(f"No available features for feature set: {feature_set}")
    validate_no_result_features(features)
    return features


# %%
def _numeric_design(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], cfg: dict):
    train = train_df[features].replace([np.inf, -np.inf], np.nan).copy()
    test = test_df[features].replace([np.inf, -np.inf], np.nan).copy()
    non_numeric = [col for col in features if not pd.api.types.is_numeric_dtype(train[col])]
    if non_numeric:
        raise ValueError(f"Standard abuse ridge baseline only accepts numeric features: {non_numeric}")

    medians = train.median(numeric_only=True).fillna(0.0)
    train = train.fillna(medians)
    test = test.fillna(medians)

    if cfg.get("log1p_features", True):
        train = np.log1p(train.clip(lower=0.0))
        test = np.log1p(test.clip(lower=0.0))

    train_array = train.to_numpy(dtype=float)
    test_array = test.to_numpy(dtype=float)

    if cfg.get("standardize", True):
        mean = train_array.mean(axis=0)
        std = train_array.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        train_array = (train_array - mean) / std
        test_array = (test_array - mean) / std

    return train_array, test_array


# %%
def _make_estimator(cfg: dict):
    try:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(cfg["alpha"]))
    except Exception:
        return NumpyRidgeRegressor(alpha=float(cfg["alpha"]))


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})

    features = _available_features(train_df, cfg["feature_set"])
    x_train, x_test = _numeric_design(train_df, test_df, features, cfg)
    y_train = train_df["target_y"].to_numpy(dtype=float)

    sample_weight = train_df.get("BF", None)
    if sample_weight is not None:
        sample_weight = sample_weight.to_numpy(dtype=float)

    model = _make_estimator(cfg)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)

    predictions = clip_predictions(model.predict(x_test))
    metrics = evaluate_predictions(test_df, predictions)
    cfg["feature_columns"] = features
    cfg["transform"] = {
        "log1p_features": bool(cfg.get("log1p_features", True)),
        "standardize": bool(cfg.get("standardize", True)),
    }

    return {
        "model_name": "baseline_standard_abuse_ridge",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": model,
        "git_commit": get_git_commit(),
    }
