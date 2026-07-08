# %%
from __future__ import annotations

import numpy as np
import pandas as pd


# %%
def make_demo_outings(
    n_pitchers: int = 48,
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024),
    outings_per_pitcher_season: int = 34,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic outing data for smoke tests and demos."""
    rng = np.random.default_rng(random_state)
    rows = []
    game_pk = 100000
    pitch_types = ["FF", "SL", "CH"]

    for pitcher_idx in range(n_pitchers):
        pitcher = 600000 + pitcher_idx
        skill = rng.normal(0.315, 0.025)
        velo = rng.normal(94, 2.5)
        spin = rng.normal(2350, 170)
        birth_year = rng.integers(1988, 1999)
        role = rng.choice(["relief", "setup", "high_leverage"], p=[0.55, 0.30, 0.15])

        for season in seasons:
            dates = pd.to_datetime(f"{season}-04-01") + pd.to_timedelta(
                np.sort(rng.choice(np.arange(0, 182), size=outings_per_pitcher_season, replace=False)),
                unit="D",
            )
            prior_date = None
            for date in dates:
                rest_days = 7 if prior_date is None else max(0, int((date - prior_date).days))
                prior_date = date
                pitch_count = int(np.clip(rng.normal(19, 7) + (role == "high_leverage") * 2, 4, 45))
                bf = int(np.clip(round(pitch_count / rng.normal(4.2, 0.6)), 1, 12))
                fatigue = max(0, 2 - rest_days) * 0.010 + max(0, pitch_count - 25) * 0.001
                xwoba = np.clip(skill + fatigue + rng.normal(0, 0.035), 0.180, 0.520)
                mix = rng.dirichlet([5, 3, 2])
                row = {
                    "pitcher": pitcher,
                    "game_pk": game_pk,
                    "game_date": date,
                    "pitch_count": pitch_count,
                    "BF": bf,
                    "rest_days": rest_days,
                    "release_speed": velo + rng.normal(0, 0.9) - fatigue * 18,
                    "release_spin_rate": spin + rng.normal(0, 65) - fatigue * 700,
                    "arm_angle": rng.normal(42, 7),
                    "spin_axis": rng.normal(185, 25),
                    "estimated_woba_using_speedangle_mean": xwoba,
                    "opponent_batter_prior_xwOBA": rng.normal(0.320, 0.025),
                    "same_hand_ratio": rng.beta(8, 8),
                    "lefty_batter_ratio": rng.beta(6, 7),
                    "age": season - birth_year + rng.uniform(0.1, 0.9),
                    "role": role,
                }
                for pitch_type, share in zip(pitch_types, mix):
                    row[f"pitch_mix_{pitch_type}"] = share
                rows.append(row)
                game_pk += 1

    return pd.DataFrame(rows)
