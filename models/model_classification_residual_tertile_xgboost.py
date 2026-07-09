# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.evaluate import evaluate_classification
from lib.feature_sets import expand_feature_set
from lib.modeling import get_git_commit, make_design_matrices
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "feature_set": "personalized_workload_max",
    "target_source": "residual",
    "lower_quantile": 1.0 / 3.0,
    "upper_quantile": 2.0 / 3.0,
    "use_bf_weight": True,
    "class_weight": "balanced",
    "random_state": 42,
    "n_estimators": 350,
    "learning_rate": 0.035,
    "max_depth": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


# %%
def _classify_residual(values: pd.Series, low_cut: float, high_cut: float) -> np.ndarray:
    residual = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    classes = np.full(len(residual), 1, dtype=int)
    classes[residual <= low_cut] = 0
    classes[residual >= high_cut] = 2
    return classes


# %%
def _make_sample_weight(train_df: pd.DataFrame, y_train: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    if cfg.get("use_bf_weight", True) and "BF" in train_df.columns:
        weight = pd.to_numeric(train_df["BF"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        weight = np.where(weight > 0, weight, 1.0)
    else:
        weight = np.ones(len(y_train), dtype=float)

    multipliers = {str(cls): 1.0 for cls in [0, 1, 2]}
    if cfg.get("class_weight") == "balanced":
        total = float(weight.sum())
        for cls in [0, 1, 2]:
            class_total = float(weight[y_train == cls].sum())
            if class_total > 0:
                multipliers[str(cls)] = total / (3.0 * class_total)
                weight[y_train == cls] *= multipliers[str(cls)]

    return weight, multipliers


# %%
def _align_proba(proba: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == 3 and classes is None:
        return proba

    aligned = np.zeros((len(proba), 3), dtype=float)
    if classes is None:
        classes = np.arange(proba.shape[1])
    for src_idx, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls <= 2:
            aligned[:, cls] = proba[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return aligned / row_sum


# %%
def _make_estimator(config: dict):
    class PriorProbabilityClassifier:
        def __init__(self):
            self.proba = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)

        def fit(self, x, y, sample_weight=None):
            weight = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
            totals = np.array([weight[np.asarray(y) == cls].sum() for cls in [0, 1, 2]], dtype=float)
            if totals.sum() > 0:
                self.proba = totals / totals.sum()
            return self

        def predict_proba(self, x):
            return np.tile(self.proba, (len(x), 1))

    try:
        import xgboost as xgb

        class NativeXGBoostClassifier:
            def __init__(self, cfg):
                self.cfg = cfg
                self.model = None

            def fit(self, x, y, sample_weight=None):
                dtrain = xgb.DMatrix(x, label=y, weight=sample_weight)
                params = {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
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

            def predict_proba(self, x):
                if self.model is None:
                    raise RuntimeError("Model is not fitted.")
                proba = self.model.predict(xgb.DMatrix(x))
                return np.asarray(proba, dtype=float).reshape(-1, 3)

        return NativeXGBoostClassifier(config)
    except Exception:
        try:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=50,
                max_features=0.7,
                random_state=int(config["random_state"]),
                n_jobs=1,
            )
        except Exception:
            return PriorProbabilityClassifier()


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})
    target_source = str(cfg["target_source"])
    if target_source not in train_df.columns or target_source not in test_df.columns:
        raise ValueError(f"classification_residual_tertile_xgboost requires column: {target_source}")

    low_cut = float(train_df[target_source].quantile(float(cfg["lower_quantile"])))
    high_cut = float(train_df[target_source].quantile(float(cfg["upper_quantile"])))
    y_train = _classify_residual(train_df[target_source], low_cut, high_cut)
    y_test = _classify_residual(test_df[target_source], low_cut, high_cut)

    feature_columns = expand_feature_set(train_df, cfg["feature_set"])
    validate_no_result_features(feature_columns)
    if not feature_columns:
        raise ValueError("classification_residual_tertile_xgboost has no available features.")

    x_train, x_test, encoded_features = make_design_matrices(train_df, test_df, feature_columns)
    sample_weight, class_weight_multipliers = _make_sample_weight(train_df, y_train, cfg)

    model = _make_estimator(cfg)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)

    if hasattr(model, "predict_proba"):
        proba = _align_proba(model.predict_proba(x_test), getattr(model, "classes_", None))
    else:
        raw_pred = np.asarray(model.predict(x_test), dtype=float)
        proba = np.zeros((len(raw_pred), 3), dtype=float)
        proba[:, 1] = 1.0
    predictions = np.argmax(proba, axis=1)

    eval_df = test_df.copy()
    eval_df["residual_class"] = y_test
    metrics = evaluate_classification(
        eval_df,
        predictions,
        predicted_proba=proba,
        target_col="residual_class",
    )
    cfg["feature_columns"] = feature_columns
    cfg["encoded_feature_count"] = len(encoded_features)
    cfg["class_labels"] = {
        "0": "risk: residual <= train lower quantile",
        "1": "normal: between train quantiles",
        "2": "good: residual >= train upper quantile",
    }
    cfg["residual_class_thresholds"] = {
        "low_cut": low_cut,
        "high_cut": high_cut,
    }
    cfg["class_weight_multipliers"] = class_weight_multipliers
    cfg["train_class_counts"] = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()}
    cfg["test_class_counts"] = {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()}

    return {
        "model_name": "classification_residual_tertile_xgboost",
        "task": "classification",
        "target_col": "residual_class",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "predicted_proba": proba,
        "model_object": model,
        "git_commit": get_git_commit(),
    }
