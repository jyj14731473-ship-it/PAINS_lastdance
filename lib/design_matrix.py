# %%
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# %%
def build_tabular_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Iterable[str],
    log1p_prefixes: tuple[str, ...] = ("standard_abuse", "custom_abuse"),
    standardize_numeric: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Build train/test matrices with train-fitted numeric transforms and one-hot categoricals."""
    features = list(feature_columns)
    train = train_df[features].copy()
    test = test_df[features].copy()

    numeric_cols = [col for col in features if pd.api.types.is_numeric_dtype(train[col])]
    categorical_cols = [col for col in features if col not in numeric_cols]

    transform_summary = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "log1p_prefixes": list(log1p_prefixes),
        "standardize_numeric": bool(standardize_numeric),
    }

    for col in numeric_cols:
        train_col = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_col = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = train_col.median()
        if not np.isfinite(median):
            median = 0.0
        train_col = train_col.fillna(median)
        test_col = test_col.fillna(median)

        if any(col.startswith(prefix) for prefix in log1p_prefixes) and not col.endswith("_pitcher_z"):
            train_col = np.log1p(train_col.clip(lower=0.0))
            test_col = np.log1p(test_col.clip(lower=0.0))

        if standardize_numeric:
            mean = float(train_col.mean())
            std = float(train_col.std(ddof=0))
            if not np.isfinite(std) or std == 0:
                std = 1.0
            train_col = (train_col - mean) / std
            test_col = (test_col - mean) / std

        train[col] = train_col
        test[col] = test_col

    for col in categorical_cols:
        train[col] = train[col].astype("string").fillna("unknown")
        test[col] = test[col].astype("string").fillna("unknown")

    combined = pd.concat([train, test], axis=0, keys=["train", "test"])
    encoded = pd.get_dummies(combined, columns=categorical_cols)
    train_encoded = encoded.xs("train").astype(float)
    test_encoded = encoded.xs("test").astype(float)
    return train_encoded.to_numpy(), test_encoded.to_numpy(), list(train_encoded.columns), transform_summary
