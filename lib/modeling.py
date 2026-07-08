# %%
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "acute_workload_7d",
    "chronic_workload_28d",
    "ACWR",
    "rest_days",
    "back_to_back",
    "age",
    "standard_abuse_prev1",
    "standard_abuse_sum_3d",
    "standard_abuse_sum_7d",
    "standard_abuse_sum_14d",
    "standard_abuse_sum_28d",
    "standard_abuse_mean_last3",
    "standard_abuse_mean_last5",
    "standard_abuse_max_7d",
    "standard_abuse_acute_7d",
    "standard_abuse_chronic_28d",
    "standard_abuse_acwr",
    "standard_abuse_high_count_7d",
    "standard_abuse_streak_prior",
    "release_speed_ma5",
    "release_speed_slope5",
    "release_speed_z",
    "effective_speed_ma5",
    "effective_speed_slope5",
    "effective_speed_z",
    "release_spin_rate_ma5",
    "release_spin_rate_slope5",
    "release_spin_rate_z",
    "release_extension_ma5",
    "release_extension_slope5",
    "release_extension_z",
    "release_pos_x_ma5",
    "release_pos_x_slope5",
    "release_pos_x_z",
    "release_pos_y_ma5",
    "release_pos_y_slope5",
    "release_pos_y_z",
    "release_pos_z_ma5",
    "release_pos_z_slope5",
    "release_pos_z_z",
    "arm_angle_ma5",
    "arm_angle_slope5",
    "arm_angle_z",
    "spin_axis_ma5",
    "spin_axis_slope5",
    "spin_axis_z",
    "pfx_x_ma5",
    "pfx_x_slope5",
    "pfx_x_z",
    "pfx_z_ma5",
    "pfx_z_slope5",
    "pfx_z_z",
    "plate_x_ma5",
    "plate_x_slope5",
    "plate_x_z",
    "plate_z_ma5",
    "plate_z_slope5",
    "plate_z_z",
    "zone_ma5",
    "zone_slope5",
    "zone_z",
    "api_break_z_with_gravity_ma5",
    "api_break_z_with_gravity_slope5",
    "api_break_z_with_gravity_z",
    "api_break_x_arm_ma5",
    "api_break_x_arm_slope5",
    "api_break_x_arm_z",
    "role",
]


# %%
def _dynamic_default_features(df: pd.DataFrame) -> list[str]:
    dynamic = [
        col
        for col in df.columns
        if col.startswith("pitch_mix_") and (col.endswith("_ma5") or col.endswith("_slope5") or col.endswith("_z"))
    ]
    return DEFAULT_FEATURE_COLUMNS + sorted(dynamic)


# %%
def get_git_commit(repo_dir: str | Path | None = None) -> str:
    """Return the current git commit, handling empty repos gracefully."""
    git_exe = shutil.which("git") or r"C:\Program Files\Git\bin\git.exe"
    try:
        completed = subprocess.run(
            [git_exe, "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
    except Exception:
        return "uncommitted-or-no-commit"
    return completed.stdout.strip()


# %%
def available_features(
    df: pd.DataFrame,
    requested: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return requested model features that exist in df."""
    requested = list(requested) if requested is not None else _dynamic_default_features(df)
    exclude_set = set(exclude or [])
    return [col for col in requested if col in df.columns and col not in exclude_set]


# %%
def _fill_numeric(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    median = train.replace([np.inf, -np.inf], np.nan).median()
    if not np.isfinite(median):
        median = 0.0
    return (
        train.replace([np.inf, -np.inf], np.nan).fillna(median),
        test.replace([np.inf, -np.inf], np.nan).fillna(median),
    )


# %%
def make_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build aligned numeric matrices with one-hot categorical handling."""
    features = list(feature_columns)
    train = train_df[features].copy()
    test = test_df[features].copy()

    for col in features:
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col], test[col] = _fill_numeric(train[col], test[col])
        else:
            train[col] = train[col].astype("string").fillna("unknown")
            test[col] = test[col].astype("string").fillna("unknown")

    combined = pd.concat([train, test], axis=0, keys=["train", "test"])
    encoded = pd.get_dummies(combined, columns=[c for c in features if not pd.api.types.is_numeric_dtype(combined[c])])
    train_encoded = encoded.xs("train").astype(float)
    test_encoded = encoded.xs("test").astype(float)
    return train_encoded.to_numpy(), test_encoded.to_numpy(), list(train_encoded.columns)


# %%
@dataclass
class NumpyRidgeRegressor:
    """Small dependency-free weighted ridge fallback."""

    alpha: float = 1.0
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_design = np.column_stack([np.ones(len(x)), x])
        if sample_weight is None:
            w = np.ones(len(x_design))
        else:
            w = np.asarray(sample_weight, dtype=float)
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        xw = x_design * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        penalty = np.eye(x_design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(xw.T @ xw + penalty) @ xw.T @ yw
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        x_design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
        return x_design @ self.coef_


# %%
@dataclass
class WeightedMeanRegressor:
    """Constant baseline fallback."""

    value_: float = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            self.value_ = float(np.nanmean(y))
        else:
            w = np.asarray(sample_weight, dtype=float)
            mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
            self.value_ = float(np.average(y[mask], weights=w[mask])) if mask.any() else float(np.nanmean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value_, dtype=float)


# %%
def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
