# %%
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# %%
def evaluate_classification(
    test_df: pd.DataFrame,
    predicted_class,
    predicted_proba=None,
    target_col: str = "residual_class",
    risk_class: int = 0,
    good_class: int = 2,
) -> dict:
    """Compute decision-oriented metrics for residual class models."""
    y = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(predicted_class, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(int)
    pred = pred[mask].astype(int)
    if len(y) == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "risk_precision": np.nan,
            "risk_recall": np.nan,
            "risk_f1": np.nan,
        }

    classes = np.array([0, 1, 2], dtype=int)
    recalls = []
    f1s = []
    precisions = {}
    class_recalls = {}
    class_f1s = {}
    for cls in classes:
        tp = float(np.sum((y == cls) & (pred == cls)))
        fp = float(np.sum((y != cls) & (pred == cls)))
        fn = float(np.sum((y == cls) & (pred != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)
        precisions[int(cls)] = precision
        class_recalls[int(cls)] = recall
        class_f1s[int(cls)] = f1

    metrics = {
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "risk_precision": float(precisions[risk_class]),
        "risk_recall": float(class_recalls[risk_class]),
        "risk_f1": float(class_f1s[risk_class]),
        "good_precision": float(precisions[good_class]),
        "good_recall": float(class_recalls[good_class]),
        "risk_rate": float(np.mean(y == risk_class)),
        "predicted_risk_rate": float(np.mean(pred == risk_class)),
        "predicted_normal_rate": float(np.mean(pred == 1)),
        "predicted_good_rate": float(np.mean(pred == good_class)),
    }

    if predicted_proba is not None:
        proba = np.asarray(predicted_proba, dtype=float)[mask]
        eps = 1e-12
        proba = np.clip(proba, eps, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(-np.mean(np.log(proba[np.arange(len(y)), y])))
        if proba.shape[1] > risk_class:
            risk_score = proba[:, risk_class]
            cutoff = max(1, int(np.ceil(len(y) * 0.20)))
            top_idx = np.argsort(-risk_score)[:cutoff]
            top_risk_rate = float(np.mean(y[top_idx] == risk_class))
            baseline_rate = metrics["risk_rate"]
            metrics["top20_risk_rate"] = top_risk_rate
            metrics["top20_risk_lift"] = (
                float(top_risk_rate / baseline_rate) if baseline_rate > 0 else np.nan
            )
            if "residual" in test_df.columns:
                residual = pd.to_numeric(test_df["residual"], errors="coerce").to_numpy(dtype=float)[mask]
                metrics["top20_risk_mean_residual"] = float(np.nanmean(residual[top_idx]))

    return metrics


# %%
def log_experiment(
    result: dict,
    run_id: str,
    n_train: int,
    n_test: int,
    log_path: str | Path = "experiments/experiments_log.csv",
) -> None:
    """Append one classification result to the experiment ledger."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    metrics = result["metrics"]
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": result["model_name"],
        "task": result.get("task", "classification"),
        "target_col": result.get("target_col", "residual_class"),
        "git_commit": result.get("git_commit", ""),
        "config": json.dumps(result.get("config", {}), sort_keys=True, ensure_ascii=False),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "risk_precision": metrics.get("risk_precision"),
        "risk_recall": metrics.get("risk_recall"),
        "risk_f1": metrics.get("risk_f1"),
        "top20_risk_rate": metrics.get("top20_risk_rate"),
        "top20_risk_lift": metrics.get("top20_risk_lift"),
        "n_train": n_train,
        "n_test": n_test,
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
