# %%
from __future__ import annotations

import numpy as np

from lib.design_matrix import build_tabular_design
from lib.evaluate import evaluate_predictions
from lib.feature_sets import get_feature_set
from lib.modeling import NumpyRidgeRegressor, clip_predictions, get_git_commit
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "feature_set": "custom_abuse_only",
    "alpha": 1.0,
    "standardize_numeric": True,
}


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
    features = [col for col in get_feature_set(cfg["feature_set"]) if col in train_df.columns]
    validate_no_result_features(features)
    if not features:
        raise ValueError("control_01_custom_abuse_ridge has no available custom abuse features.")

    x_train, x_test, encoded_features, transform_summary = build_tabular_design(
        train_df,
        test_df,
        features,
        log1p_prefixes=("custom_abuse",),
        standardize_numeric=bool(cfg["standardize_numeric"]),
    )
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
    cfg["encoded_feature_count"] = len(encoded_features)
    cfg["transform"] = transform_summary

    return {
        "model_name": "control_01_custom_abuse_ridge",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": model,
        "git_commit": get_git_commit(),
    }
