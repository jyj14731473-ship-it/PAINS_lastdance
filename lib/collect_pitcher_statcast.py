from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def collect_pitchers(
    pitcher_ids: list[int], start: str, end: str, output_dir: Path, max_retries: int = 3
) -> list[Path]:
    """Collect complete Statcast histories for a small list of pitchers."""
    from pybaseball import statcast_pitcher

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pitcher_id in pitcher_ids:
        path = output_dir / f"statcast_pitcher_{pitcher_id}_{start}_{end}.parquet"
        if not path.exists():
            for attempt in range(1, max_retries + 1):
                try:
                    frame = statcast_pitcher(start, end, pitcher_id)
                    break
                except Exception:
                    if attempt == max_retries:
                        raise
                    time.sleep(2**attempt)
            frame.to_parquet(path, index=False)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect complete Statcast histories by pitcher.")
    parser.add_argument("--pitcher-ids", required=True, nargs="+", type=int)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = collect_pitchers(args.pitcher_ids, args.start, args.end, args.output_dir)
    print(f"Cached {len(paths)} pitcher files under {args.output_dir}")


if __name__ == "__main__":
    main()
