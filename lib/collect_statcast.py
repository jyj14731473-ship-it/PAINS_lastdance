# %%
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# %%
def _date_chunks(start: str, end: str, chunk_days: int):
    current = datetime.fromisoformat(start).date()
    end_date = datetime.fromisoformat(end).date()
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield current.isoformat(), chunk_end.isoformat()
        current = chunk_end + timedelta(days=1)


# %%
def filter_pitching_team(frame: pd.DataFrame, team: str) -> pd.DataFrame:
    """Keep pitches thrown by one MLB team from Statcast game rows."""
    if frame.empty:
        filtered = frame.copy()
        filtered["pitching_team"] = team.upper()
        return filtered

    required = {"home_team", "away_team", "inning_topbot"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot filter pitching team; missing Statcast columns: {sorted(missing)}")

    team = team.upper()
    half = frame["inning_topbot"].astype("string").str.lower()
    home_pitching = frame["home_team"].astype("string").str.upper().eq(team) & half.str.startswith("top")
    away_pitching = frame["away_team"].astype("string").str.upper().eq(team) & half.str.startswith("bot")
    filtered = frame.loc[home_pitching | away_pitching].copy()
    filtered["pitching_team"] = team
    return filtered


# %%
def collect_statcast(
    start: str,
    end: str,
    out_dir: str | Path,
    chunk_days: int = 7,
    pitching_team: str | None = None,
) -> list[Path]:
    """Pull Statcast pitch-level data in chunks and cache parquet files."""
    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise RuntimeError("Install pybaseball first: pip install pybaseball") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        path = out / f"statcast_{chunk_start}_{chunk_end}.parquet"
        if path.exists():
            written.append(path)
            continue
        print(f"Pulling Statcast {chunk_start}..{chunk_end}")
        frame = statcast(start_dt=chunk_start, end_dt=chunk_end, team=pitching_team)
        if pitching_team:
            frame = filter_pitching_team(frame, pitching_team)
        frame.to_parquet(path, index=False)
        written.append(path)
    return written


# %%
def collect_pitching_stats(years: list[int], out_dir: str | Path) -> Path:
    """Cache FanGraphs pitching stats for role labeling."""
    try:
        from pybaseball import pitching_stats
    except ImportError as exc:
        raise RuntimeError("Install pybaseball first: pip install pybaseball") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in years:
        stats = pitching_stats(year)
        stats["season"] = year
        frames.append(stats)
    combined = pd.concat(frames, ignore_index=True)
    path = out / f"pitching_stats_{min(years)}_{max(years)}.parquet"
    combined.to_parquet(path, index=False)
    return path


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MLB Statcast pitch-level data.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/statcast", type=Path)
    parser.add_argument("--chunk-days", default=7, type=int)
    parser.add_argument("--pitching-team", default=None, help="Optional team abbreviation, e.g. LAD.")
    args = parser.parse_args()

    paths = collect_statcast(args.start, args.end, args.out_dir, args.chunk_days, args.pitching_team)
    print(f"Cached {len(paths)} parquet files under {args.out_dir}")


# %%
if __name__ == "__main__":
    main()
