# %%
from __future__ import annotations

import numpy as np

from lib.design_matrix import build_tabular_design
from lib.evaluate import evaluate_predictions
from lib.feature_interactions import DEFAULT_INTERACTION_PAIRS, add_numeric_interactions
from lib.feature_sets import expand_feature_set
from lib.modeling import NumpyRidgeRegressor, clip_predictions, get_git_commit
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "feature_set": "standard_abuse_plus_pitcher",
    "interaction_pairs": DEFAULT_INTERACTION_PAIRS,
    "random_state": 42,
    "n_estimators": 250,
    "learning_rate": 0.04,
    "max_depth": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "standardize_numeric": False,
}


# %%
def _make_estimator(cfg: dict):
    try:
        import xgboost as xgb

        class NativeXGBoostRegressor:
            def __init__(self, config):
                self.config = config
                self.model = None

            def fit(self, x, y, sample_weight=None):
                dtrain = xgb.DMatrix(x, label=y, weight=sample_weight)
                params = {
                    "objective": "reg:squarederror",
                    "eta": float(self.config["learning_rate"]),
                    "max_depth": int(self.config["max_depth"]),
                    "subsample": float(self.config["subsample"]),
                    "colsample_bytree": float(self.config["colsample_bytree"]),
                    "seed": int(self.config["random_state"]),
                    "nthread": 1,
                    "verbosity": 0,
                }
                self.model = xgb.train(params, dtrain, num_boost_round=int(self.config["n_estimators"]))
                return self

            def predict(self, x):
                if self.model is None:
                    raise RuntimeError("Model is not fitted.")
                return self.model.predict(xgb.DMatrix(x))

        return NativeXGBoostRegressor(cfg)
    except Exception:
        try:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=8,
                random_state=int(cfg["random_state"]),
                n_jobs=1,
            )
        except Exception:
            return NumpyRidgeRegressor(alpha=2.0)


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})

    train_aug, train_interactions = add_numeric_interactions(train_df, cfg["interaction_pairs"])
    test_aug, test_interactions = add_numeric_interactions(test_df, cfg["interaction_pairs"])
    interaction_features = [col for col in train_interactions if col in test_interactions]

    base_features = expand_feature_set(train_aug, cfg["feature_set"])
    features = base_features + interaction_features
    validate_no_result_features(features)
    if not features:
        raise ValueError("control_03_interaction_xgboost has no available features.")

    x_train, x_test, encoded_features, transform_summary = build_tabular_design(
        train_aug,
        test_aug,
        features,
        log1p_prefixes=("standard_abuse",),
        standardize_numeric=bool(cfg["standardize_numeric"]),
    )
    y_train = train_aug["target_y"].to_numpy(dtype=float)
    sample_weight = train_aug.get("BF", None)
    if sample_weight is not None:
        sample_weight = sample_weight.to_numpy(dtype=float)

    model = _make_estimator(cfg)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)

    predictions = clip_predictions(model.predict(x_test))
    metrics = evaluate_predictions(test_aug, predictions)
    cfg["feature_columns"] = features
    cfg["interaction_features"] = interaction_features
    cfg["encoded_feature_count"] = len(encoded_features)
    cfg["transform"] = transform_summary

    return {
        "model_name": "control_03_interaction_xgboost",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": model,
        "git_commit": get_git_commit(),
    }
