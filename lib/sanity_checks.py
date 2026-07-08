# %%
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_OR_RESULT_COLUMNS = {
    "target_y",
    "y",
    "residual",
    "baseline_skill",
    "shrunk_xwOBA",
    "outing_xwOBA",
    "estimated_woba_using_speedangle",
    "estimated_woba_using_speedangle_mean",
    "fip",
    "era",
}


# %%
def validate_feature_dates(
    df: pd.DataFrame,
    feature_asof_col: str = "feature_asof_date",
    game_date_col: str = "game_date",
) -> dict:
    """Ensure feature timestamps do not point after the outing date."""
    if feature_asof_col not in df.columns or game_date_col not in df.columns:
        return {"checked": False, "violations": 0}

    asof = pd.to_datetime(df[feature_asof_col], errors="coerce")
    game_date = pd.to_datetime(df[game_date_col], errors="coerce")
    violations = (asof > game_date).fillna(False)
    count = int(violations.sum())
    if count:
        examples = df.loc[violations, [feature_asof_col, game_date_col]].head(5)
        raise ValueError(
            "Feature leakage detected: feature_asof_date is after game_date. "
            f"Examples:\n{examples}"
        )
    return {"checked": True, "violations": 0}


# %%
def validate_no_result_features(feature_columns: Iterable[str]) -> dict:
    """Block target/result columns from entering model X."""
    columns = list(feature_columns)
    bad = [col for col in columns if col in LABEL_OR_RESULT_COLUMNS]
    if bad:
        raise ValueError(f"Result/label columns cannot be model features: {bad}")
    return {"checked": True, "blocked_columns": bad}


# %%
def validate_label_distribution(
    df: pd.DataFrame,
    target_col: str = "target_y",
    normal_col: str = "is_normal_condition",
    tolerance: float = 0.08,
    min_rows: int = 30,
    strict: bool = False,
) -> dict:
    """Check that normal-condition outings center near 0.5."""
    if target_col not in df.columns or normal_col not in df.columns:
        return {"checked": False, "reason": "missing target or normal-condition column"}

    normal = df.loc[df[normal_col].fillna(False), target_col].dropna()
    if len(normal) < min_rows:
        return {"checked": False, "reason": "too few normal-condition rows", "n": len(normal)}

    mean = float(normal.mean())
    distance = abs(mean - 0.5)
    result = {"checked": True, "n": len(normal), "mean": mean, "distance": distance}
    if distance > tolerance:
        message = (
            f"Normal-condition target_y mean is {mean:.3f}, "
            f"outside 0.5 +/- {tolerance:.3f}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return result


# %%
def summarize_outliers(df: pd.DataFrame, columns: Iterable[str], z_threshold: float = 6.0) -> dict:
    """Return a light-weight count of extreme numeric values."""
    summary: dict[str, int] = {}
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        std = series.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            summary[col] = 0
            continue
        z = (series - series.mean()) / std
        summary[col] = int((z.abs() > z_threshold).sum())
    return summary


# %%
def run_feature_sanity_checks(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
    strict: bool = False,
) -> dict:
    """Run common leakage checks after feature construction."""
    checks = {"feature_dates": validate_feature_dates(df)}
    if feature_columns is not None:
        checks["result_features"] = validate_no_result_features(feature_columns)
    checks["outliers"] = summarize_outliers(
        df,
        [
            "acute_workload_7d",
            "chronic_workload_28d",
            "ACWR",
            "rest_days",
            "release_speed_z",
            "release_spin_rate_z",
        ],
    )
    if strict and any(v > 0 for v in checks["outliers"].values()):
        raise ValueError(f"Extreme feature outliers found: {checks['outliers']}")
    return checks


# %%
def run_label_sanity_checks(df: pd.DataFrame, strict: bool = False) -> dict:
    """Run common checks after label construction."""
    checks = {"feature_dates": validate_feature_dates(df)}
    checks["label_distribution"] = validate_label_distribution(df, strict=strict)
    return checks
