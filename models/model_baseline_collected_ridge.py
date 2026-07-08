# %%
from __future__ import annotations

import numpy as np

from lib.design_matrix import build_tabular_design
from lib.evaluate import evaluate_predictions
from lib.feature_sets import expand_feature_set
from lib.modeling import NumpyRidgeRegressor, clip_predictions, get_git_commit
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "feature_set": "collected_pitcher_max",
    "alpha": 1.0,
    "standardize_numeric": True,
}


# %%
def _make_estimator(config: dict):
    try:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(config["alpha"]))
    except Exception:
        return NumpyRidgeRegressor(alpha=float(config["alpha"]))


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})

    feature_columns = expand_feature_set(train_df, cfg["feature_set"])
    validate_no_result_features(feature_columns)
    if not feature_columns:
        raise ValueError("baseline_collected_ridge has no available features.")

    x_train, x_test, encoded_features, transform_summary = build_tabular_design(
        train_df,
        test_df,
        feature_columns,
        log1p_prefixes=("standard_abuse", "custom_abuse"),
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
    cfg["feature_columns"] = feature_columns
    cfg["encoded_feature_count"] = len(encoded_features)
    cfg["transform"] = transform_summary

    return {
        "model_name": "baseline_collected_ridge",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": model,
        "git_commit": get_git_commit(),
    }
