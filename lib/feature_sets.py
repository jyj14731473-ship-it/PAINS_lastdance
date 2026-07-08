# %%
from __future__ import annotations


# Standard abuse-index rolling features. These are prior-only features generated
# from lib.workload_index.add_standard_abuse_features().
STANDARD_ABUSE_FEATURES = [
    "standard_abuse_prev1",
    "standard_abuse_sum_3d",
    "standard_abuse_sum_7d",
    "standard_abuse_sum_14d",
    "standard_abuse_sum_28d",
    "standard_abuse_mean_last3",
    "standard_abuse_mean_last5",
    "standard_abuse_max_7d",
    "standard_abuse_acute_7d",
    "standard_abuse_chronic_28d",
    "standard_abuse_acwr",
    "standard_abuse_high_count_7d",
    "standard_abuse_streak_prior",
]

CUSTOM_ABUSE_FEATURES = [
    name.replace("standard_abuse", "custom_abuse")
    for name in STANDARD_ABUSE_FEATURES
]

ROLE_FEATURES = ["role"]

PITCHER_ROLLING_FEATURES = [
    "release_speed_ma5",
    "release_speed_slope5",
    "release_speed_z",
    "effective_speed_ma5",
    "effective_speed_slope5",
    "effective_speed_z",
    "release_spin_rate_ma5",
    "release_spin_rate_slope5",
    "release_spin_rate_z",
    "release_extension_ma5",
    "release_extension_slope5",
    "release_extension_z",
    "release_pos_x_ma5",
    "release_pos_x_slope5",
    "release_pos_x_z",
    "release_pos_y_ma5",
    "release_pos_y_slope5",
    "release_pos_y_z",
    "release_pos_z_ma5",
    "release_pos_z_slope5",
    "release_pos_z_z",
    "arm_angle_ma5",
    "arm_angle_slope5",
    "arm_angle_z",
    "spin_axis_ma5",
    "spin_axis_slope5",
    "spin_axis_z",
    "pfx_x_ma5",
    "pfx_x_slope5",
    "pfx_x_z",
    "pfx_z_ma5",
    "pfx_z_slope5",
    "pfx_z_z",
    "plate_x_ma5",
    "plate_x_slope5",
    "plate_x_z",
    "plate_z_ma5",
    "plate_z_slope5",
    "plate_z_z",
    "zone_ma5",
    "zone_slope5",
    "zone_z",
    "api_break_z_with_gravity_ma5",
    "api_break_z_with_gravity_slope5",
    "api_break_z_with_gravity_z",
    "api_break_x_arm_ma5",
    "api_break_x_arm_slope5",
    "api_break_x_arm_z",
]

CURRENT_WORKLOAD_FEATURES = [
    "acute_workload_7d",
    "chronic_workload_28d",
    "ACWR",
    "rest_days",
    "back_to_back",
]

FEATURE_SETS = {
    "standard_abuse_only": STANDARD_ABUSE_FEATURES,
    "custom_abuse_only": CUSTOM_ABUSE_FEATURES,
    "standard_abuse_with_role": STANDARD_ABUSE_FEATURES + ROLE_FEATURES,
    "custom_abuse_with_role": CUSTOM_ABUSE_FEATURES + ROLE_FEATURES,
    "standard_abuse_plus_pitcher": (
        CURRENT_WORKLOAD_FEATURES + STANDARD_ABUSE_FEATURES + PITCHER_ROLLING_FEATURES + ROLE_FEATURES
    ),
    "custom_abuse_plus_pitcher": CUSTOM_ABUSE_FEATURES + PITCHER_ROLLING_FEATURES + ROLE_FEATURES,
    "current_workload_only": CURRENT_WORKLOAD_FEATURES,
    "collected_pitcher_max": CURRENT_WORKLOAD_FEATURES + STANDARD_ABUSE_FEATURES + PITCHER_ROLLING_FEATURES + ROLE_FEATURES,
}


# %%
def get_feature_set(name: str) -> list[str]:
    try:
        return list(FEATURE_SETS[name])
    except KeyError as exc:
        available = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown feature set '{name}'. Available: {available}") from exc


# %%
def expand_feature_set(df, name: str) -> list[str]:
    features = get_feature_set(name)
    dynamic_pitch_mix = [
        col
        for col in df.columns
        if col.startswith("pitch_mix_") and (col.endswith("_ma5") or col.endswith("_slope5") or col.endswith("_z"))
    ]
    if name in {"standard_abuse_plus_pitcher", "custom_abuse_plus_pitcher", "collected_pitcher_max"}:
        features = features + sorted(dynamic_pitch_mix)
    return [col for col in dict.fromkeys(features) if col in df.columns]
