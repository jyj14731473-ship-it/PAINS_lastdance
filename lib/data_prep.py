# %%
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from lib.sanity_checks import run_feature_sanity_checks
from lib.workload_index import add_custom_abuse_features, add_standard_abuse_features


DEFAULT_CONFIG = {
    "acute_days": 7,
    "chronic_days": 28,
    "recent_outings": 5,
    "first_rest_days": 7,
    "cluster_count": 6,
    "random_state": 42,
    "standard_abuse_high_threshold": 75.0,
    "custom_abuse_high_threshold": 75.0,
    "strict_sanity": False,
}

TRACKING_COLUMNS = [
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "arm_angle",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "api_break_z_with_gravity",
    "api_break_x_arm",
]


# %%
def _merge_config(config: dict | None) -> dict:
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


# %%
def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# %%
def _rolling_sum_by_days(
    df: pd.DataFrame,
    value_col: str,
    days: int,
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
) -> pd.Series:
    """Sum prior values in the previous N calendar days, excluding current outing."""
    output = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(pitcher_col, sort=False):
        group = group.sort_values(date_col)
        dates = pd.to_datetime(group[date_col]).to_numpy(dtype="datetime64[ns]")
        values = group[value_col].fillna(0.0).to_numpy(dtype=float)
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        sums = []
        for pos, current in enumerate(dates):
            start = current - np.timedelta64(days, "D")
            left = np.searchsorted(dates, start, side="left")
            right = np.searchsorted(dates, current, side="left")
            sums.append(cumulative[right] - cumulative[left])
        output.loc[group.index] = sums
    return output


# %%
def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    y = values[mask]
    x = np.arange(len(values), dtype=float)[mask]
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return np.nan
    return float(np.sum(x * (y - y.mean())) / denom)


# %%
def _add_prior_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int,
    pitcher_col: str = "pitcher",
) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        ma_col = f"{col}_ma{window}"
        slope_col = f"{col}_slope{window}"
        z_col = f"{col}_z"
        df[ma_col] = np.nan
        df[slope_col] = np.nan
        df[z_col] = np.nan

        for _, group in df.groupby(pitcher_col, sort=False):
            group = group.sort_values("game_date")
            prior = group[col].shift(1)
            ma = prior.rolling(window=window, min_periods=1).mean()
            slope = prior.rolling(window=window, min_periods=2).apply(_slope, raw=True)
            expanding_mean = prior.expanding(min_periods=3).mean()
            expanding_std = prior.expanding(min_periods=3).std(ddof=0)
            z = (ma - expanding_mean) / expanding_std.replace(0, np.nan)
            df.loc[group.index, ma_col] = ma
            df.loc[group.index, slope_col] = slope
            df.loc[group.index, z_col] = z
    return df


# %%
def _infer_role(df: pd.DataFrame) -> pd.Series:
    if "role" in df.columns:
        return df["role"].fillna("unknown").astype(str)

    required = {"SV", "HLD", "G", "GS"}
    if required.issubset(df.columns):
        games = df["G"].replace(0, np.nan)
        start_rate = df["GS"] / games
        leverage_rate = (df["SV"].fillna(0) + df["HLD"].fillna(0)) / games
        return np.select(
            [start_rate >= 0.25, leverage_rate >= 0.30, leverage_rate >= 0.12],
            ["starter", "high_leverage", "setup"],
            default="relief",
        )

    return pd.Series("unknown", index=df.index)


# %%
def _add_cluster_id(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cluster_features = [
        col
        for col in [
            "arm_angle_ma5",
            "release_speed_ma5",
            "spin_axis_ma5",
            "release_spin_rate_ma5",
        ]
        if col in df.columns
    ]
    cluster_features.extend([col for col in df.columns if col.startswith("pitch_mix_") and col.endswith("_ma5")])

    if not cluster_features:
        df["cluster_id"] = 0
        return df

    x = df[cluster_features].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    n_clusters = min(int(config["cluster_count"]), max(1, len(df) // 20))
    if n_clusters <= 1:
        df["cluster_id"] = 0
        return df

    df["cluster_id"] = _simple_kmeans_labels(
        x.to_numpy(dtype=float),
        n_clusters=n_clusters,
        random_state=int(config["random_state"]),
    )
    return df


# %%
def _simple_kmeans_labels(x: np.ndarray, n_clusters: int, random_state: int, iterations: int = 30) -> np.ndarray:
    """Small numpy KMeans for stable offline-style cluster IDs."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    scale = x.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    z = (x - x.mean(axis=0)) / scale
    initial_idx = rng.choice(len(z), size=n_clusters, replace=False)
    centers = z[initial_idx].copy()
    labels = np.zeros(len(z), dtype=int)

    for _ in range(iterations):
        distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            mask = labels == cluster
            if mask.any():
                centers[cluster] = z[mask].mean(axis=0)
    return labels


# %%
def prepare_features(outings_df: pd.DataFrame, config: dict | None = None, run_checks: bool = True) -> pd.DataFrame:
    """Create common leakage-safe features shared by every model."""
    config = _merge_config(config)
    df = outings_df.copy()
    _require_columns(df, ["pitcher", "game_date"])

    if "game_pk" not in df.columns:
        df["game_pk"] = np.arange(len(df))
    if "pitch_count" not in df.columns:
        if "pitches" in df.columns:
            df["pitch_count"] = df["pitches"]
        else:
            raise ValueError("Missing pitch_count/pitches column.")
    if "BF" not in df.columns:
        df["BF"] = df.get("batters_faced", np.nan)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    df["feature_asof_date"] = df["game_date"] - pd.Timedelta(days=1)

    if "rest_days" not in df.columns:
        rest = df.groupby("pitcher")["game_date"].diff().dt.days
        df["rest_days"] = rest.fillna(config["first_rest_days"])
    else:
        df["rest_days"] = pd.to_numeric(df["rest_days"], errors="coerce").fillna(config["first_rest_days"])

    df["day_of_season"] = df["game_date"].dt.dayofyear
    df["back_to_back"] = (df["rest_days"] <= 1).astype(int)

    acute_sum = _rolling_sum_by_days(df, "pitch_count", int(config["acute_days"]))
    chronic_sum = _rolling_sum_by_days(df, "pitch_count", int(config["chronic_days"]))
    df["acute_workload_7d"] = acute_sum
    df["chronic_workload_28d"] = chronic_sum
    acute_rate = acute_sum / float(config["acute_days"])
    chronic_rate = chronic_sum / float(config["chronic_days"])
    df["ACWR"] = (acute_rate / chronic_rate.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df = add_standard_abuse_features(df, config)
    df = add_custom_abuse_features(df, config)

    if "age" not in df.columns:
        df["age"] = np.nan

    pitch_mix_cols = [col for col in df.columns if col.startswith("pitch_mix_")]
    df = _add_prior_rolling_features(df, TRACKING_COLUMNS + pitch_mix_cols, int(config["recent_outings"]))
    df = _add_cluster_id(df, config)
    df["role"] = _infer_role(df)

    if "opponent_batter_prior_xwOBA" not in df.columns:
        df["opponent_batter_prior_xwOBA"] = np.nan
    if "same_hand_ratio" not in df.columns:
        df["same_hand_ratio"] = np.nan
    if "lefty_batter_ratio" not in df.columns:
        df["lefty_batter_ratio"] = np.nan

    if run_checks:
        run_feature_sanity_checks(df, strict=bool(config["strict_sanity"]))
    return df


# %%
if __name__ == "__main__":
    from lib.demo_data import make_demo_outings

    demo = make_demo_outings()
    features = prepare_features(demo)
    print(features.head())
