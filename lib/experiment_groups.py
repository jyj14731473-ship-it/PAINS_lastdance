# %%
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentGroup:
    group_id: str
    description: str
    feature_set: str
    model_family: str
    model_file: str
    status: str = "planned"


BASELINE_GROUP = ExperimentGroup(
    group_id="baseline_collected_xgboost",
    description="Maximal collected pitcher-safe features: workload, standard abuse, pitcher rolling, pitch mix, role.",
    feature_set="collected_pitcher_max",
    model_family="xgboost",
    model_file="model_baseline_collected_xgboost.py",
    status="ready",
)

CONTROL_GROUPS = [
    ExperimentGroup(
        group_id="control_01_custom_abuse_formula",
        description="Reserved for the revised abuse-index formula only.",
        feature_set="custom_abuse_only",
        model_family="ridge",
        model_file="model_control_01_custom_abuse_ridge.py",
        status="ready",
    ),
    ExperimentGroup(
        group_id="control_02_abuse_plus_pitcher_features",
        description="Reserved for abuse-index features plus pitcher-only rolling features.",
        feature_set="standard_abuse_plus_pitcher",
        model_family="ridge",
        model_file="model_control_02_abuse_plus_pitcher_ridge.py",
        status="ready",
    ),
    ExperimentGroup(
        group_id="control_03_interaction_tabular",
        description="Reserved for interaction/tabular feature engineering.",
        feature_set="standard_abuse_plus_pitcher",
        model_family="xgboost",
        model_file="model_control_03_interaction_xgboost.py",
        status="ready",
    ),
]

EXPERIMENT_GROUPS = [BASELINE_GROUP, *CONTROL_GROUPS]
