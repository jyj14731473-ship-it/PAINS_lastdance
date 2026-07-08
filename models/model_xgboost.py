# %%
from __future__ import annotations

import numpy as np

from lib.evaluate import evaluate_predictions
from lib.modeling import (
    NumpyRidgeRegressor,
    available_features,
    clip_predictions,
    get_git_commit,
    make_design_matrices,
)
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "random_state": 42,
    "n_estimators": 350,
    "learning_rate": 0.035,
    "max_depth": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


# %%
def _make_estimator(config: dict):
    try:
        import xgboost as xgb

        class NativeXGBoostRegressor:
            def __init__(self, cfg):
                self.cfg = cfg
                self.model = None

            def fit(self, x, y, sample_weight=None):
                dtrain = xgb.DMatrix(x, label=y, weight=sample_weight)
                params = {
                    "objective": "reg:squarederror",
                    "eta": float(self.cfg["learning_rate"]),
                    "max_depth": int(self.cfg["max_depth"]),
                    "subsample": float(self.cfg["subsample"]),
                    "colsample_bytree": float(self.cfg["colsample_bytree"]),
                    "seed": int(self.cfg["random_state"]),
                    "nthread": 1,
                    "verbosity": 0,
                }
                self.model = xgb.train(params, dtrain, num_boost_round=int(self.cfg["n_estimators"]))
                return self

            def predict(self, x):
                if self.model is None:
                    raise RuntimeError("Model is not fitted.")
                return self.model.predict(xgb.DMatrix(x))

        return NativeXGBoostRegressor(config)
    except Exception:
        try:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=8,
                random_state=int(config["random_state"]),
                n_jobs=1,
            )
        except Exception:
            return NumpyRidgeRegressor(alpha=2.0)


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})
    feature_columns = available_features(train_df, cfg.get("features"))
    validate_no_result_features(feature_columns)

    x_train, x_test, encoded_features = make_design_matrices(train_df, test_df, feature_columns)
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

    return {
        "model_name": "xgboost",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": model,
        "git_commit": get_git_commit(),
    }
