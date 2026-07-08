# %%
from __future__ import annotations

from dataclasses import dataclass
from math import floor

import pandas as pd


@dataclass
class TimeSplit:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    test_start: pd.Timestamp
    strategy: str = "rolling_origin"
    metadata: dict | None = None


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

    return TimeSplit(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        test_start=test_start,
        strategy="rolling_origin",
        metadata={"embargo_days": embargo_days},
    )


# %%
def monthly_train_test_split(
    df: pd.DataFrame,
    date_col: str = "game_date",
    train_fraction: float = 0.80,
) -> TimeSplit:
    """Split each calendar month chronologically into train/test rows."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    sort_cols = [date_col]
    for col in ["game_pk", "pitcher"]:
        if col in data.columns:
            sort_cols.append(col)
    data = data.sort_values(sort_cols).reset_index(drop=True)
    data["_split_month"] = data[date_col].dt.to_period("M")

    train_parts = []
    test_parts = []
    month_counts: dict[str, dict[str, int]] = {}

    for month, group in data.groupby("_split_month", sort=True):
        group = group.drop(columns=["_split_month"])
        if len(group) < 2:
            train_part = group
            test_part = group.iloc[0:0].copy()
        else:
            n_train = floor(len(group) * train_fraction)
            n_train = min(max(n_train, 1), len(group) - 1)
            train_part = group.iloc[:n_train].copy()
            test_part = group.iloc[n_train:].copy()

        train_parts.append(train_part)
        test_parts.append(test_part)
        month_counts[str(month)] = {"train": int(len(train_part)), "test": int(len(test_part))}

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else data.iloc[0:0].drop(columns=["_split_month"])
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else data.iloc[0:0].drop(columns=["_split_month"])
    validation_df = data.iloc[0:0].drop(columns=["_split_month"]).copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Monthly split produced an empty train or test set.")

    return TimeSplit(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        test_start=test_df[date_col].min(),
        strategy="monthly",
        metadata={
            "train_fraction": train_fraction,
            "month_counts": month_counts,
        },
    )
