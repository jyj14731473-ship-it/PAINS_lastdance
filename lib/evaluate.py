# %%
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# %%
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


# %%
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


# %%
def evaluate_predictions(test_df: pd.DataFrame, predictions, target_col: str = "target_y") -> dict:
    """Compute common metrics for every model."""
    pred = np.asarray(predictions, dtype=float)
    y = test_df[target_col].to_numpy(dtype=float)
    metrics = {"rmse": rmse(y, pred), "mae": mae(y, pred)}

    if "BF" in test_df.columns:
        low_bf = test_df["BF"].fillna(0) <= test_df["BF"].median()
        metrics["rmse_low_bf"] = rmse(y[low_bf], pred[low_bf]) if low_bf.any() else np.nan
    else:
        metrics["rmse_low_bf"] = np.nan

    return metrics


# %%
def acwr_residual_summary(test_df: pd.DataFrame, predictions, target_col: str = "target_y") -> pd.DataFrame:
    pred = np.asarray(predictions, dtype=float)
    out = test_df[["ACWR", target_col]].copy()
    out["prediction"] = pred
    out["residual"] = out[target_col] - out["prediction"]
    bins = [-np.inf, 0.8, 1.3, 1.5, np.inf]
    labels = ["<0.8", "0.8-1.3", "1.3-1.5", ">1.5"]
    out["ACWR_bin"] = pd.cut(out["ACWR"], bins=bins, labels=labels)
    return out.groupby("ACWR_bin", observed=False)["residual"].describe().reset_index()


# %%
def log_experiment(
    result: dict,
    run_id: str,
    n_train: int,
    n_test: int,
    log_path: str | Path = "experiments/experiments_log.csv",
) -> None:
    """Append one model result to the text experiment ledger."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": result["model_name"],
        "git_commit": result.get("git_commit", ""),
        "config": json.dumps(result.get("config", {}), sort_keys=True, ensure_ascii=False),
        "rmse": result["metrics"].get("rmse"),
        "mae": result["metrics"].get("mae"),
        "rmse_low_bf": result["metrics"].get("rmse_low_bf"),
        "n_train": n_train,
        "n_test": n_test,
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# %%
def save_metric_bar_chart(results: list[dict], output_path: str | Path) -> None:
    """Save RMSE/MAE bar chart when matplotlib is available."""
    try:
        output = Path(output_path)
        config_dir = output.parent / ".mplconfig"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    names = [r["model_name"] for r in results]
    rmse_values = [r["metrics"]["rmse"] for r in results]
    mae_values = [r["metrics"]["mae"] for r in results]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, rmse_values, width, label="RMSE")
    ax.bar(x + width / 2, mae_values, width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, max(rmse_values + mae_values) * 1.25)
    ax.legend()
    ax.set_title("Model Comparison")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


# %%
def save_acwr_boxplot(summary_frames: dict[str, pd.DataFrame], output_path: str | Path) -> None:
    """Save a compact residual-by-ACWR summary plot."""
    try:
        output = Path(output_path)
        config_dir = output.parent / ".mplconfig"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, frame in summary_frames.items():
        if {"ACWR_bin", "mean"}.issubset(frame.columns):
            ax.plot(frame["ACWR_bin"].astype(str), frame["mean"], marker="o", label=name)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean residual")
    ax.set_xlabel("ACWR bin")
    ax.legend()
    ax.set_title("Residual by ACWR Bin")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


# %%
def permutation_importance_shap_fallback(
    model,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Iterable[str],
    metric=rmse,
    random_state: int = 42,
) -> pd.DataFrame:
    """A small SHAP fallback: permutation importance with the same output intent."""
    rng = np.random.default_rng(random_state)
    baseline = metric(y, model.predict(x))
    rows = []
    for idx, feature in enumerate(feature_names):
        shuffled = x.copy()
        shuffled[:, idx] = rng.permutation(shuffled[:, idx])
        rows.append({"feature": feature, "importance": metric(y, model.predict(shuffled)) - baseline})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)
