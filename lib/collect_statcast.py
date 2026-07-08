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
def collect_statcast(start: str, end: str, out_dir: str | Path, chunk_days: int = 7) -> list[Path]:
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
        frame = statcast(start_dt=chunk_start, end_dt=chunk_end)
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
    args = parser.parse_args()

    paths = collect_statcast(args.start, args.end, args.out_dir, args.chunk_days)
    print(f"Cached {len(paths)} parquet files under {args.out_dir}")


# %%
if __name__ == "__main__":
    main()
