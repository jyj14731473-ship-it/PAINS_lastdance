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
    status: str = "ready"


BASELINE_GROUP = ExperimentGroup(
    group_id="classification_residual_tertile_xgboost",
    description="3범주 분류 베이스라인: 본인 baseline 대비 하/중/상 등판을 예측한다.",
    feature_set="personalized_workload_max",
    model_family="xgboost_multiclass",
    model_file="model_classification_residual_tertile_xgboost.py",
)

EXPERIMENT_GROUPS = [BASELINE_GROUP]
