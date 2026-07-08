# %%
from __future__ import annotations

import numpy as np

from lib.evaluate import evaluate_predictions
from lib.modeling import NumpyRidgeRegressor, clip_predictions, get_git_commit
from lib.sanity_checks import validate_no_result_features


CONFIG = {
    "features": ["ACWR"],
    "lam": 0.6,
    "n_splines": 10,
}


# %%
def _matrix(df, features: list[str]) -> np.ndarray:
    x = df[features].replace([np.inf, -np.inf], np.nan).copy()
    for col in features:
        median = x[col].median()
        x[col] = x[col].fillna(0.0 if not np.isfinite(median) else median)
    return x.to_numpy(dtype=float)


# %%
def _fit_gam(x_train, y_train, sample_weight, cfg):
    try:
        from pygam import LinearGAM, s

        model = LinearGAM(s(0, n_splines=int(cfg["n_splines"])), lam=float(cfg["lam"]))
        model.fit(x_train, y_train, weights=sample_weight)
        return model
    except Exception:
        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import SplineTransformer, StandardScaler
            from sklearn.linear_model import Ridge

            model = make_pipeline(
                SplineTransformer(n_knots=6, degree=3, include_bias=False),
                StandardScaler(with_mean=False),
                Ridge(alpha=float(cfg["lam"])),
            )
            model.fit(x_train, y_train, ridge__sample_weight=sample_weight)
            return model
        except Exception:
            return NumpyRidgeRegressor(alpha=float(cfg["lam"])).fit(x_train, y_train, sample_weight)


# %%
def run(config: dict, train_df, test_df) -> dict:
    cfg = CONFIG.copy()
    cfg.update(config or {})
    features = [col for col in cfg["features"] if col in train_df.columns]
    validate_no_result_features(features)
    if not features:
        raise ValueError("model_gam_acwr has no ACWR feature.")

    x_train = _matrix(train_df, features)
    x_test = _matrix(test_df, features)
    y_train = train_df["target_y"].to_numpy(dtype=float)
    sample_weight = train_df.get("BF", None)
    if sample_weight is not None:
        sample_weight = sample_weight.to_numpy(dtype=float)

    model = _fit_gam(x_train, y_train, sample_weight, cfg)
    predictions = clip_predictions(model.predict(x_test))
    metrics = evaluate_predictions(test_df, predictions)

    return {
        "model_name": "gam_acwr",
        "config": cfg,
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "model_object": model,
        "git_commit": get_git_commit(),
    }
