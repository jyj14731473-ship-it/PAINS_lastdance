# %%
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


STANDARD_ABUSE_PREFIX = "standard_abuse"
CUSTOM_ABUSE_PREFIX = "custom_abuse"
DEFAULT_DAY_WINDOWS = (3, 7, 14, 28)
DEFAULT_OUTING_WINDOWS = (3, 5)
STANDARD_REST_WEIGHTS = {
    "rest_ge_4": 1.0,
    "rest_3": 1.5,
    "rest_2": 2.0,
    "rest_le_1": 3.0,
    "streak_ge_3": 5.0,
}


# %%
def _sort_columns(df: pd.DataFrame, date_col: str) -> list[str]:
    columns = [date_col]
    for col in ["game_pk", "pitcher"]:
        if col in df.columns and col not in columns:
            columns.append(col)
    return columns


# %%
def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required workload-index columns: {missing}")


# %%
def _prior_sum_by_days(
    df: pd.DataFrame,
    value_col: str,
    days: int,
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
) -> pd.Series:
    output = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(pitcher_col, sort=False):
        group = group.sort_values(_sort_columns(group, date_col))
        dates = pd.to_datetime(group[date_col]).to_numpy(dtype="datetime64[ns]")
        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        sums = []
        for current in dates:
            start = current - np.timedelta64(days, "D")
            left = np.searchsorted(dates, start, side="left")
            right = np.searchsorted(dates, current, side="left")
            sums.append(cumulative[right] - cumulative[left])
        output.loc[group.index] = sums
    return output


# %%
def _prior_max_by_days(
    df: pd.DataFrame,
    value_col: str,
    days: int,
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
) -> pd.Series:
    output = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(pitcher_col, sort=False):
        group = group.sort_values(_sort_columns(group, date_col))
        dates = pd.to_datetime(group[date_col]).to_numpy(dtype="datetime64[ns]")
        values = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        max_values = []
        for pos, current in enumerate(dates):
            start = current - np.timedelta64(days, "D")
            left = np.searchsorted(dates, start, side="left")
            window = values[left:pos]
            max_values.append(float(np.max(window)) if len(window) else 0.0)
        output.loc[group.index] = max_values
    return output


# %%
def _prior_count_by_days(
    df: pd.DataFrame,
    flag_col: str,
    days: int,
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
) -> pd.Series:
    return _prior_sum_by_days(df, flag_col, days, pitcher_col=pitcher_col, date_col=date_col)


# %%
def add_consecutive_appearance_streak(
    df: pd.DataFrame,
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
    rest_col: str = "rest_days",
    streak_col: str = "consecutive_appearance_streak",
) -> pd.DataFrame:
    """Count consecutive-day appearances, including the current outing date."""
    _require_columns(df, [pitcher_col, date_col, rest_col])
    out = df.copy()
    out[streak_col] = 1

    for _, group in out.groupby(pitcher_col, sort=False):
        group = group.sort_values(_sort_columns(group, date_col))
        rest = pd.to_numeric(group[rest_col], errors="coerce")
        streaks = []
        streak = 1
        for pos, rest_days in enumerate(rest):
            if pos == 0:
                streak = 1
            elif pd.notna(rest_days) and rest_days <= 1:
                streak += 1
            else:
                streak = 1
            streaks.append(streak)
        out.loc[group.index, streak_col] = streaks

    return out


# %%
def standard_rest_weight(
    rest_days: pd.Series,
    streak: pd.Series,
) -> pd.Series:
    """Apply the standard rest-day abuse weights supplied by the project."""
    rest = pd.to_numeric(rest_days, errors="coerce")
    streak = pd.to_numeric(streak, errors="coerce").fillna(1)
    weights = pd.Series(1.0, index=rest.index)
    weights.loc[rest == 3] = 1.5
    weights.loc[rest == 2] = 2.0
    weights.loc[rest <= 1] = 3.0
    weights.loc[(rest <= 1) & (streak >= 3)] = 5.0
    return weights


# %%
def configured_rest_weight(
    rest_days: pd.Series,
    streak: pd.Series,
    weights: dict | None = None,
) -> pd.Series:
    """Apply configurable rest-day weights for custom abuse-index variants."""
    cfg = STANDARD_REST_WEIGHTS.copy()
    if weights:
        cfg.update(weights)

    rest = pd.to_numeric(rest_days, errors="coerce")
    streak = pd.to_numeric(streak, errors="coerce").fillna(1)
    out = pd.Series(float(cfg["rest_ge_4"]), index=rest.index)
    out.loc[rest == 3] = float(cfg["rest_3"])
    out.loc[rest == 2] = float(cfg["rest_2"])
    out.loc[rest <= 1] = float(cfg["rest_le_1"])
    out.loc[(rest <= 1) & (streak >= 3)] = float(cfg["streak_ge_3"])
    return out


# %%
def add_standard_abuse_index(
    df: pd.DataFrame,
    pitch_count_col: str = "pitch_count",
    rest_col: str = "rest_days",
    prefix: str = STANDARD_ABUSE_PREFIX,
) -> pd.DataFrame:
    """Create outing-level standard abuse score from pitch count and rest weight."""
    _require_columns(df, [pitch_count_col, rest_col, "pitcher", "game_date"])
    out = add_consecutive_appearance_streak(df, rest_col=rest_col, streak_col=f"{prefix}_streak")
    out[f"{prefix}_rest_weight"] = standard_rest_weight(out[rest_col], out[f"{prefix}_streak"])
    pitch_count = pd.to_numeric(out[pitch_count_col], errors="coerce").fillna(0.0)
    out[f"{prefix}_score"] = pitch_count * out[f"{prefix}_rest_weight"]
    return out


# %%
def add_custom_abuse_index(
    df: pd.DataFrame,
    pitch_count_col: str = "pitch_count",
    rest_col: str = "rest_days",
    prefix: str = CUSTOM_ABUSE_PREFIX,
    rest_weights: dict | None = None,
) -> pd.DataFrame:
    """Create outing-level custom abuse score with configurable rest weights."""
    _require_columns(df, [pitch_count_col, rest_col, "pitcher", "game_date"])
    out = add_consecutive_appearance_streak(df, rest_col=rest_col, streak_col=f"{prefix}_streak")
    out[f"{prefix}_rest_weight"] = configured_rest_weight(
        out[rest_col],
        out[f"{prefix}_streak"],
        rest_weights,
    )
    pitch_count = pd.to_numeric(out[pitch_count_col], errors="coerce").fillna(0.0)
    out[f"{prefix}_score"] = pitch_count * out[f"{prefix}_rest_weight"]
    return out


# %%
def add_standard_abuse_rolling_features(
    df: pd.DataFrame,
    day_windows: Iterable[int] = DEFAULT_DAY_WINDOWS,
    outing_windows: Iterable[int] = DEFAULT_OUTING_WINDOWS,
    high_threshold: float = 75.0,
    prefix: str = STANDARD_ABUSE_PREFIX,
) -> pd.DataFrame:
    """Add prior-only rolling features derived from the standard abuse score."""
    score_col = f"{prefix}_score"
    if score_col not in df.columns:
        df = add_standard_abuse_index(df, prefix=prefix)
    out = df.copy()

    for _, group in out.groupby("pitcher", sort=False):
        group = group.sort_values(_sort_columns(group, "game_date"))
        prior_score = pd.to_numeric(group[score_col], errors="coerce").shift(1)
        out.loc[group.index, f"{prefix}_prev1"] = prior_score
        for window in outing_windows:
            out.loc[group.index, f"{prefix}_mean_last{window}"] = prior_score.rolling(
                window=window,
                min_periods=1,
            ).mean()

    high_flag_col = f"{prefix}_high_flag"
    out[high_flag_col] = (pd.to_numeric(out[score_col], errors="coerce").fillna(0.0) >= high_threshold).astype(int)

    for days in day_windows:
        out[f"{prefix}_sum_{days}d"] = _prior_sum_by_days(out, score_col, days)
    out[f"{prefix}_max_7d"] = _prior_max_by_days(out, score_col, 7)
    out[f"{prefix}_high_count_7d"] = _prior_count_by_days(out, high_flag_col, 7)

    if f"{prefix}_sum_7d" in out.columns:
        out[f"{prefix}_acute_7d"] = out[f"{prefix}_sum_7d"] / 7.0
    if f"{prefix}_sum_28d" in out.columns:
        out[f"{prefix}_chronic_28d"] = out[f"{prefix}_sum_28d"] / 28.0
    if {f"{prefix}_acute_7d", f"{prefix}_chronic_28d"}.issubset(out.columns):
        chronic = out[f"{prefix}_chronic_28d"].replace(0, np.nan)
        out[f"{prefix}_acwr"] = (out[f"{prefix}_acute_7d"] / chronic).replace([np.inf, -np.inf], np.nan)

    out[f"{prefix}_streak_prior"] = np.nan
    for _, group in out.groupby("pitcher", sort=False):
        group = group.sort_values(_sort_columns(group, "game_date"))
        out.loc[group.index, f"{prefix}_streak_prior"] = group[f"{prefix}_streak"].shift(1)

    return out.drop(columns=[high_flag_col])


# %%
def add_standard_abuse_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Public entry point for standard abuse-index features."""
    config = config or {}
    return add_standard_abuse_rolling_features(
        df,
        day_windows=config.get("standard_abuse_day_windows", DEFAULT_DAY_WINDOWS),
        outing_windows=config.get("standard_abuse_outing_windows", DEFAULT_OUTING_WINDOWS),
        high_threshold=float(config.get("standard_abuse_high_threshold", 75.0)),
        prefix=str(config.get("standard_abuse_prefix", STANDARD_ABUSE_PREFIX)),
    )


# %%
def add_custom_abuse_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Public entry point for the editable/custom abuse-index feature family."""
    config = config or {}
    prefix = str(config.get("custom_abuse_prefix", CUSTOM_ABUSE_PREFIX))
    out = add_custom_abuse_index(
        df,
        prefix=prefix,
        rest_weights=config.get("custom_abuse_rest_weights"),
    )
    return add_standard_abuse_rolling_features(
        out,
        day_windows=config.get("custom_abuse_day_windows", DEFAULT_DAY_WINDOWS),
        outing_windows=config.get("custom_abuse_outing_windows", DEFAULT_OUTING_WINDOWS),
        high_threshold=float(config.get("custom_abuse_high_threshold", 75.0)),
        prefix=prefix,
    )
