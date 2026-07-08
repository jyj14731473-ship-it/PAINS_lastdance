# %%
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# %%
def _read_many(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No parquet/csv files found in {input_dir}")
    frames = []
    for path in paths:
        if path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


# %%
def _batters_faced(group: pd.DataFrame) -> int:
    if "at_bat_number" in group.columns:
        return int(group["at_bat_number"].nunique())
    if {"batter", "inning"}.issubset(group.columns):
        return int(group[["batter", "inning"]].drop_duplicates().shape[0])
    return int(max(1, round(len(group) / 4)))


# %%
def _pitch_mix(pitches: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if "pitch_type" not in pitches.columns:
        return pd.DataFrame(columns=keys)
    counts = pitches.groupby(keys + ["pitch_type"]).size().rename("n").reset_index()
    totals = counts.groupby(keys)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    mix = counts.pivot_table(index=keys, columns="pitch_type", values="share", fill_value=0)
    mix.columns = [f"pitch_mix_{col}" for col in mix.columns]
    return mix.reset_index()


# %%
def build_outings(pitches: pd.DataFrame, player_bio: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate Statcast pitch-level rows into pitcher-game outings."""
    required = {"pitcher", "game_pk", "game_date"}
    missing = required - set(pitches.columns)
    if missing:
        raise ValueError(f"Missing required pitch columns: {sorted(missing)}")

    df = pitches.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    keys = ["pitcher", "game_pk", "game_date"]

    agg_spec = {
        "pitch_count": ("pitcher", "size"),
        "release_speed": ("release_speed", "mean"),
        "release_spin_rate": ("release_spin_rate", "mean"),
        "arm_angle": ("arm_angle", "mean"),
        "spin_axis": ("spin_axis", "mean"),
        "estimated_woba_using_speedangle_mean": ("estimated_woba_using_speedangle", "mean"),
    }
    existing_agg = {name: spec for name, spec in agg_spec.items() if spec[0] in df.columns}
    outings = df.groupby(keys, as_index=False).agg(**existing_agg)

    bf = df.groupby(keys).apply(_batters_faced, include_groups=False).rename("BF").reset_index()
    outings = outings.merge(bf, on=keys, how="left")

    mix = _pitch_mix(df, keys)
    if not mix.empty:
        outings = outings.merge(mix, on=keys, how="left")

    if {"p_throws", "stand"}.issubset(df.columns):
        matchup = df.assign(
            same_hand=(df["p_throws"].astype(str).str[0] == df["stand"].astype(str).str[0]).astype(float),
            lefty_batter=(df["stand"].astype(str).str.upper().str[0] == "L").astype(float),
        ).groupby(keys, as_index=False).agg(
            same_hand_ratio=("same_hand", "mean"),
            lefty_batter_ratio=("lefty_batter", "mean"),
        )
        outings = outings.merge(matchup, on=keys, how="left")

    if "batter_prior_xwOBA" in df.columns:
        batter_quality = df.groupby(keys, as_index=False).agg(
            opponent_batter_prior_xwOBA=("batter_prior_xwOBA", "mean")
        )
        outings = outings.merge(batter_quality, on=keys, how="left")

    outings = outings.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    outings["rest_days"] = outings.groupby("pitcher")["game_date"].diff().dt.days
    outings["day_of_season"] = outings["game_date"].dt.dayofyear

    if player_bio is not None and {"pitcher", "birth_date"}.issubset(player_bio.columns):
        bio = player_bio[["pitcher", "birth_date"]].copy()
        bio["birth_date"] = pd.to_datetime(bio["birth_date"], errors="coerce")
        outings = outings.merge(bio, on="pitcher", how="left")
        outings["age"] = (outings["game_date"] - outings["birth_date"]).dt.days / 365.25
        outings = outings.drop(columns=["birth_date"])
    elif "age" not in outings.columns:
        outings["age"] = np.nan

    return outings


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Build outing-level data from Statcast pitch data.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pitches = _read_many(args.input_dir)
    outings = build_outings(pitches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outings.to_parquet(args.output, index=False)
    print(f"Wrote {len(outings):,} outings to {args.output}")


# %%
if __name__ == "__main__":
    main()
