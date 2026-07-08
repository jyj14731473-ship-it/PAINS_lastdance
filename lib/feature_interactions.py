# %%
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_INTERACTION_PAIRS = [
    ("standard_abuse_sum_7d", "release_speed_z"),
    ("standard_abuse_sum_7d", "release_spin_rate_z"),
    ("standard_abuse_acwr", "release_speed_z"),
    ("standard_abuse_acwr", "arm_angle_z"),
    ("standard_abuse_prev1", "back_to_back"),
]


# %%
def interaction_name(left: str, right: str) -> str:
    return f"interaction__{left}__x__{right}"


# %%
def add_numeric_interactions(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]] = DEFAULT_INTERACTION_PAIRS,
) -> tuple[pd.DataFrame, list[str]]:
    """Add simple pairwise numeric interactions when both source columns exist."""
    out = df.copy()
    created: list[str] = []
    for left, right in pairs:
        if left not in out.columns or right not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[left]) or not pd.api.types.is_numeric_dtype(out[right]):
            continue
        name = interaction_name(left, right)
        out[name] = out[left] * out[right]
        created.append(name)
    return out, created
