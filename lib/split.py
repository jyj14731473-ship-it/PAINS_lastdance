# %%
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSplit:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    test_start: pd.Timestamp


# %%
def rolling_origin_split(
    df: pd.DataFrame,
    date_col: str = "game_date",
    embargo_days: int = 3,
) -> TimeSplit:
    """Chronological split: latest season back half is final test."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.sort_values(date_col).reset_index(drop=True)
    seasons = sorted(data[date_col].dt.year.dropna().unique())

    if len(seasons) >= 2:
        test_season = seasons[-1]
        season_dates = data.loc[data[date_col].dt.year == test_season, date_col]
        test_start = season_dates.min() + (season_dates.max() - season_dates.min()) / 2
    else:
        test_start = data[date_col].quantile(0.80)

    embargo_start = test_start - pd.Timedelta(days=embargo_days)
    train_df = data.loc[data[date_col] < embargo_start].copy()
    validation_df = data.loc[(data[date_col] >= embargo_start) & (data[date_col] < test_start)].copy()
    test_df = data.loc[data[date_col] >= test_start].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")

    return TimeSplit(train_df=train_df, validation_df=validation_df, test_df=test_df, test_start=test_start)
