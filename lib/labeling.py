# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.sanity_checks import run_label_sanity_checks


DEFAULT_CONFIG = {
    "eb_k": 10.0,
    "baseline_k": 8.0,
    "baseline_min_normal": 8,
    "half_life_outings": 18,
    "normal_acwr_min": 0.8,
    "normal_acwr_max": 1.3,
    "default_xwoba_prior": 0.320,
    "default_xwoba_std": 0.040,
    "strict_sanity": False,
}


# %%
def _merge_config(config: dict | None) -> dict:
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


# %%
def _sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


# %%
def _find_outing_xwoba_source(df: pd.DataFrame) -> str:
    candidates = [
        "outing_xwOBA",
        "estimated_woba_using_speedangle_mean",
        "estimated_woba_using_speedangle",
        "xwoba",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No xwOBA source column found. Expected one of: "
        "outing_xwOBA, estimated_woba_using_speedangle_mean, "
        "estimated_woba_using_speedangle, xwoba."
    )


# %%
def _prior_expanding_mean(df: pd.DataFrame, value_col: str, group_col: str = "pitcher") -> pd.Series:
    output = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(group_col, sort=False):
        group = group.sort_values("game_date")
        output.loc[group.index] = group[value_col].shift(1).expanding(min_periods=1).mean()
    return output


# %%
def _prior_expanding_median(df: pd.DataFrame, value_col: str, group_col: str = "pitcher") -> pd.Series:
    output = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(group_col, sort=False):
        group = group.sort_values("game_date")
        output.loc[group.index] = group[value_col].shift(1).expanding(min_periods=1).median()
    return output


# %%
def _league_prior_by_date(df: pd.DataFrame, value_col: str, default: float) -> pd.Series:
    by_date = (
        df.assign(_value=df[value_col])
        .groupby("game_date")["_value"]
        .agg(["sum", "count"])
        .sort_index()
    )
    prior_sum = by_date["sum"].cumsum().shift(1)
    prior_count = by_date["count"].cumsum().shift(1)
    prior_mean = (prior_sum / prior_count).replace([np.inf, -np.inf], np.nan).fillna(default)
    return df["game_date"].map(prior_mean).fillna(default)


# %%
def _previous_season_prior(df: pd.DataFrame, value_col: str, normal_col: str) -> pd.Series:
    seasons = df["game_date"].dt.year
    normal_df = df.loc[df[normal_col].fillna(False)].copy()
    if normal_df.empty:
        return pd.Series(np.nan, index=df.index)

    season_means = (
        normal_df.assign(season=normal_df["game_date"].dt.year)
        .groupby(["pitcher", "season"])[value_col]
        .mean()
    )

    priors = []
    for pitcher, season in zip(df["pitcher"], seasons):
        priors.append(season_means.get((pitcher, season - 1), np.nan))
    return pd.Series(priors, index=df.index, dtype=float)


# %%
def _baseline_ewma(df: pd.DataFrame, value_col: str, normal_col: str, half_life: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ewma = pd.Series(index=df.index, dtype=float)
    normal_count = pd.Series(index=df.index, dtype=float)
    normal_std = pd.Series(index=df.index, dtype=float)

    for _, group in df.groupby("pitcher", sort=False):
        group = group.sort_values("game_date")
        normal_values = group[value_col].where(group[normal_col].fillna(False))
        prior_normal = normal_values.shift(1)
        ewma.loc[group.index] = prior_normal.ewm(halflife=half_life, adjust=False, ignore_na=True).mean()
        normal_count.loc[group.index] = group[normal_col].astype(int).cumsum().shift(1).fillna(0)
        normal_std.loc[group.index] = prior_normal.expanding(min_periods=3).std(ddof=0)
    return ewma, normal_count, normal_std


# %%
def create_labels(features_df: pd.DataFrame, config: dict | None = None, run_checks: bool = True) -> pd.DataFrame:
    """Create baseline-relative target_y labels from leakage-safe features."""
    config = _merge_config(config)
    df = features_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)

    source = _find_outing_xwoba_source(df)
    df["outing_xwOBA"] = pd.to_numeric(df[source], errors="coerce")

    prior_personal = _prior_expanding_mean(df, "outing_xwOBA").fillna(config["default_xwoba_prior"])
    df["personal_prior_xwOBA"] = prior_personal
    bf = pd.to_numeric(df.get("BF", 1), errors="coerce").fillna(1).clip(lower=1)
    eb_k = float(config["eb_k"])
    df["shrunk_xwOBA"] = (bf * df["outing_xwOBA"] + eb_k * prior_personal) / (bf + eb_k)

    prior_median_rest = _prior_expanding_median(df, "rest_days").fillna(df["rest_days"].median())
    df["pitcher_prior_median_rest"] = prior_median_rest
    df["is_normal_condition"] = (
        df["ACWR"].between(float(config["normal_acwr_min"]), float(config["normal_acwr_max"]), inclusive="both")
        & (df["rest_days"] >= (prior_median_rest - 1))
    )

    ewma, normal_count, normal_std = _baseline_ewma(
        df,
        "shrunk_xwOBA",
        "is_normal_condition",
        int(config["half_life_outings"]),
    )
    league_prior = _league_prior_by_date(df, "shrunk_xwOBA", float(config["default_xwoba_prior"]))
    prev_season_prior = _previous_season_prior(df, "shrunk_xwOBA", "is_normal_condition")
    prior = prev_season_prior.fillna(league_prior).fillna(float(config["default_xwoba_prior"]))

    k2 = float(config["baseline_k"])
    df["normal_condition_count_prior"] = normal_count
    shrunk_baseline = (normal_count * ewma.fillna(prior) + k2 * prior) / (normal_count + k2)
    df["baseline_skill"] = np.where(
        normal_count >= int(config["baseline_min_normal"]),
        ewma.fillna(prior),
        shrunk_baseline,
    )

    league_std = df.loc[df["is_normal_condition"], "shrunk_xwOBA"].std(ddof=0)
    if not np.isfinite(league_std) or league_std <= 0:
        league_std = float(config["default_xwoba_std"])
    scale = normal_std.fillna(league_std).clip(lower=0.010, upper=0.150)

    df["residual"] = df["baseline_skill"] - df["shrunk_xwOBA"]
    df["target_y"] = _sigmoid(df["residual"] / scale)

    if run_checks:
        run_label_sanity_checks(df, strict=bool(config["strict_sanity"]))
    return df


# %%
if __name__ == "__main__":
    from lib.data_prep import prepare_features
    from lib.demo_data import make_demo_outings

    demo = make_demo_outings()
    labeled = create_labels(prepare_features(demo))
    print(labeled[["pitcher", "game_date", "ACWR", "is_normal_condition", "target_y"]].head())
