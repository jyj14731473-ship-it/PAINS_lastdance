# %%
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_HALFLIVES = (2, 3, 5, 8, 12)


# %%
def add_prior_target_features(
    df: pd.DataFrame,
    halflives: Iterable[int] = DEFAULT_HALFLIVES,
    target_col: str = "target_y",
    pitcher_col: str = "pitcher",
    date_col: str = "game_date",
) -> pd.DataFrame:
    """Add leakage-safe recent-form features from prior outings' target values.

    Every feature is built from shift(1) within pitcher, so the current
    outing's own target never enters its features. At prediction time the
    targets of earlier (already finished) outings are known, so these are
    valid forecasting inputs under the project's prior-data-only principle.

    Added columns:
      prior_y_count    number of prior outings with a known target
      prior_y_expmean  expanding mean of prior targets
      prior_y_ewm{h}   exponentially weighted mean of prior targets, halflife h
    """
    sort_cols = [pitcher_col, date_col]
    if "game_pk" in df.columns:
        sort_cols.append("game_pk")
    out = df.sort_values(sort_cols).copy()

    parts = []
    for _, group in out.groupby(pitcher_col, sort=False):
        prior = pd.to_numeric(group[target_col], errors="coerce").shift(1)
        feats = pd.DataFrame(index=group.index)
        feats["prior_y_count"] = prior.notna().cumsum().astype(float)
        feats["prior_y_expmean"] = prior.expanding(min_periods=1).mean()
        for halflife in halflives:
            feats[f"prior_y_ewm{halflife}"] = prior.ewm(halflife=halflife, ignore_na=True).mean()
        parts.append(feats)

    features = pd.concat(parts).loc[out.index]
    return pd.concat([out, features], axis=1)


# %%
def shrunk_ewm_prediction(
    frame: pd.DataFrame,
    halflife: int,
    shrink_k: float,
    anchor: float = 0.5,
) -> np.ndarray:
    """Shrink the prior-target EWM toward the anchor by evidence count.

    prediction = (n * ewm + k * anchor) / (n + k), where n is the number of
    prior outings. Pitchers without history predict exactly the anchor.
    """
    count = frame["prior_y_count"].to_numpy(dtype=float)
    ewm = frame[f"prior_y_ewm{halflife}"].to_numpy(dtype=float)
    ewm = np.where(np.isnan(ewm), anchor, ewm)
    return (count * ewm + shrink_k * anchor) / (count + shrink_k)


# %%
def ewm_grid_ensemble_prediction(
    frame: pd.DataFrame,
    grid: Iterable[tuple[int, float]],
    anchor: float = 0.5,
) -> np.ndarray:
    """Average shrunk-EWM predictions over a (halflife, shrink_k) grid.

    Averaging over hyperparameters instead of selecting one reduces the
    variance of the choice, which matters with only a few hundred outings.
    """
    predictions = [shrunk_ewm_prediction(frame, hl, k, anchor) for hl, k in grid]
    return np.mean(predictions, axis=0)


# %%
def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Stack train+test so prior-target features can flow across the boundary.

    Returns the combined frame and a boolean mask marking test rows, in a
    positional order that can be mapped back to the original test_df order.
    """
    train = train_df.copy()
    test = test_df.copy()
    train["_row_role"] = "train"
    test["_row_role"] = "test"
    test["_test_pos"] = np.arange(len(test))
    combined = pd.concat([train, test], ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    return combined, combined["_row_role"].eq("test")
