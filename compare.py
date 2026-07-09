# %%
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from lib.data_prep import prepare_features
from lib.demo_data import make_demo_outings
from lib.evaluate import log_experiment
from lib.labeling import create_labels
from lib.split import monthly_train_test_split, rolling_origin_split

BASELINE_MODEL_NAMES = ["model_classification_residual_tertile_xgboost"]


# %%
def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# %%
def discover_model_modules(models_dir: str | Path = "models", model_names: list[str] | None = None):
    paths = sorted(Path(models_dir).glob("model_*.py"))
    if model_names is None:
        model_names = BASELINE_MODEL_NAMES
    if model_names:
        requested = {name if name.startswith("model_") else f"model_{name}" for name in model_names}
        paths = [path for path in paths if path.stem in requested]
        missing = sorted(requested - {path.stem for path in paths})
        if missing:
            raise ValueError(f"Requested model files not found: {missing}")
    return [_load_module(path) for path in paths]


# %%
def _json_ready(value: Any):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


# %%
def save_dataframe_artifacts(df: pd.DataFrame, output_dir: str | Path, name: str, preview_rows: int = 200) -> None:
    """Save a full table, a human-readable preview, and a compact schema summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output / f"{name}.parquet", index=False)
    df.head(preview_rows).to_csv(output / f"{name}_preview.csv", index=False, encoding="utf-8-sig")

    summary = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(value) for column, value in df.isna().sum().items()},
        "preview_rows": [
            {column: _json_ready(value) for column, value in row.items()}
            for row in df.head(5).to_dict(orient="records")
        ],
    }
    (output / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# %%
def save_prediction_artifact(test_df: pd.DataFrame, result: dict, output_dir: str | Path) -> None:
    """Save per-row predictions for later decision review."""
    metadata_columns = [
        "game_date",
        "game_pk",
        "team",
        "home_team",
        "away_team",
        "opponent",
        "pitcher",
        "pitcher_name",
        "player_name",
        "BF",
        "IP",
        "target_y",
        "residual",
        "shrunk_xwOBA",
        "outing_xwOBA",
        "baseline_skill",
    ]
    keep_columns = [column for column in metadata_columns if column in test_df.columns]
    predictions = pd.DataFrame(index=test_df.index)
    if keep_columns:
        predictions = test_df.loc[:, keep_columns].copy()

    task = result.get("task", "classification")
    raw_predictions = result.get("predictions")
    if raw_predictions is not None:
        if task == "classification":
            thresholds = result.get("config", {}).get("residual_class_thresholds", {})
            low_cut = thresholds.get("low_cut")
            high_cut = thresholds.get("high_cut")
            if "residual" in predictions.columns and low_cut is not None and high_cut is not None:
                residual = pd.to_numeric(predictions["residual"], errors="coerce")
                predictions["actual_class"] = 1
                predictions.loc[residual <= float(low_cut), "actual_class"] = 0
                predictions.loc[residual >= float(high_cut), "actual_class"] = 2
                predictions["actual_label"] = predictions["actual_class"].map(
                    {0: "하/risk", 1: "중/normal", 2: "상/good"}
                )
            predictions["predicted_class"] = raw_predictions
            predictions["predicted_label"] = predictions["predicted_class"].map(
                {0: "하/risk", 1: "중/normal", 2: "상/good"}
            )
        else:
            predictions["prediction"] = raw_predictions

    proba = result.get("predicted_proba")
    if proba is not None:
        proba_df = pd.DataFrame(
            proba,
            columns=["proba_risk", "proba_normal", "proba_good"][: proba.shape[1]],
            index=test_df.index,
        )
        predictions = pd.concat([predictions, proba_df], axis=1)

    safe_name = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(result.get("model_name", "model"))
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    predictions.to_csv(Path(output_dir) / f"predictions_{safe_name}.csv", index=False, encoding="utf-8-sig")


# %%
def filter_pitcher_sample(
    df: pd.DataFrame,
    min_outings: int = 0,
    min_bf: int = 0,
    min_ip: float = 0,
    pitcher_col: str = "pitcher",
) -> pd.DataFrame:
    """Keep pitchers with enough total sample for player-level workload baselines."""
    if min_outings <= 0 and min_bf <= 0 and min_ip <= 0:
        return df
    if pitcher_col not in df.columns:
        raise ValueError(f"Cannot filter pitcher sample; missing column: {pitcher_col}")

    grouped = df.groupby(pitcher_col)
    summary = grouped.size().rename("outing_count").to_frame()
    if "BF" in df.columns:
        summary["bf_total"] = grouped["BF"].sum(min_count=1).fillna(0)
    else:
        summary["bf_total"] = 0
    if "IP" in df.columns:
        summary["ip_total"] = grouped["IP"].sum(min_count=1).fillna(0)
    elif "outs_recorded" in df.columns:
        summary["ip_total"] = grouped["outs_recorded"].sum(min_count=1).fillna(0) / 3.0
    else:
        summary["ip_total"] = 0

    keep = summary.index[
        (summary["outing_count"] >= int(min_outings))
        & (summary["bf_total"] >= int(min_bf))
        & (summary["ip_total"] >= float(min_ip))
    ]
    return df.loc[df[pitcher_col].isin(keep)].copy()


# %%
def load_or_prepare_data(
    input_path: str | Path | None,
    demo: bool,
    intermediate_dir: str | Path | None = None,
    min_pitcher_outings: int = 0,
    min_pitcher_bf: int = 0,
    min_pitcher_ip: float = 0,
) -> pd.DataFrame:
    if demo:
        raw = make_demo_outings()
    elif input_path is None:
        raise ValueError("Provide --input data/outings.parquet or use --demo.")
    else:
        path = Path(input_path)
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif path.suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")

    raw = filter_pitcher_sample(raw, min_pitcher_outings, min_pitcher_bf, min_pitcher_ip)

    if intermediate_dir is not None:
        save_dataframe_artifacts(raw, intermediate_dir, "01_input_outings")

    if "target_y" in raw.columns:
        labeled = raw
    else:
        features = prepare_features(raw)
        if intermediate_dir is not None:
            save_dataframe_artifacts(features, intermediate_dir, "02_features")
            feature_profile = {
                "rows": int(len(features)),
                "date_min": _json_ready(features["game_date"].min()) if "game_date" in features else None,
                "date_max": _json_ready(features["game_date"].max()) if "game_date" in features else None,
                "pitchers": int(features["pitcher"].nunique()) if "pitcher" in features else None,
                "feature_columns": list(features.columns),
            }
            (Path(intermediate_dir) / "02_features_profile.json").write_text(
                json.dumps(feature_profile, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        labeled = create_labels(features)

    labeled = labeled.loc[pd.to_numeric(labeled["target_y"], errors="coerce").notna()].copy()
    if intermediate_dir is not None:
        save_dataframe_artifacts(labeled, intermediate_dir, "03_labeled_nonnull")
    return labeled


# %%
def run_comparison(
    df: pd.DataFrame,
    run_id: str,
    output_dir: Path,
    intermediate_dir: str | Path | None = None,
    split_strategy: str = "rolling-origin",
    monthly_train_fraction: float = 0.80,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    if intermediate_dir is not None:
        save_dataframe_artifacts(df, intermediate_dir, "03_labeled")

    if split_strategy == "monthly":
        split = monthly_train_test_split(df, train_fraction=monthly_train_fraction)
    elif split_strategy == "rolling-origin":
        split = rolling_origin_split(df)
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    if intermediate_dir is not None:
        save_dataframe_artifacts(split.train_df, intermediate_dir, "04_train_split")
        save_dataframe_artifacts(split.validation_df, intermediate_dir, "05_validation_embargo_split")
        save_dataframe_artifacts(split.test_df, intermediate_dir, "06_test_split")

    modules = discover_model_modules(model_names=model_names)
    if not modules:
        raise RuntimeError("No models/model_*.py files found.")

    results = []
    for module in modules:
        if not hasattr(module, "run"):
            continue
        result = module.run({}, split.train_df, split.test_df)
        results.append(result)
        log_experiment(result, run_id, len(split.train_df), len(split.test_df))

    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        save_prediction_artifact(split.test_df, result, output_dir)

    metrics = pd.DataFrame(
        [
            {
                "model_name": result["model_name"],
                "task": result.get("task", "classification"),
                "target_col": result.get("target_col", "residual_class"),
                **result["metrics"],
                "git_commit": result.get("git_commit", ""),
            }
            for result in results
        ]
    )
    if "macro_f1" in metrics.columns and metrics["macro_f1"].notna().any():
        metrics = metrics.sort_values("macro_f1", ascending=False, na_position="last")
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)

    report = {
        "run_id": run_id,
        "n_train": len(split.train_df),
        "n_validation_embargo": len(split.validation_df),
        "n_test": len(split.test_df),
        "test_start": _json_ready(split.test_start),
        "split_strategy": split.strategy,
        "split_metadata": split.metadata or {},
        "caveat": (
            "Correlation, not causation: manager usage decisions and selection bias can "
            "affect both workload and performance."
        ),
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Run all model_*.py methods on one time split.")
    parser.add_argument("--input", type=Path, default=None, help="Outing-level parquet/csv data.")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic data.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated classification model stems to run. "
            "Default: classification_residual_tertile_xgboost."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["rolling-origin", "monthly"],
        default="rolling-origin",
        help="Train/test split strategy.",
    )
    parser.add_argument(
        "--monthly-train-fraction",
        type=float,
        default=0.80,
        help="Train fraction used by --split monthly.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Write input/features/labels/splits under output-dir/intermediate.",
    )
    parser.add_argument(
        "--min-pitcher-outings",
        type=int,
        default=0,
        help="Keep only pitchers with at least this many rows before feature preparation.",
    )
    parser.add_argument(
        "--min-pitcher-bf",
        type=int,
        default=0,
        help="Keep only pitchers with at least this many total batters faced before feature preparation.",
    )
    parser.add_argument(
        "--min-pitcher-ip",
        type=float,
        default=0,
        help="Keep only pitchers with at least this many total innings pitched before feature preparation.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("experiments") / "runs" / run_id
    intermediate_dir = output_dir / "intermediate" if args.save_intermediate else None
    df = load_or_prepare_data(
        args.input,
        args.demo,
        intermediate_dir,
        args.min_pitcher_outings,
        args.min_pitcher_bf,
        args.min_pitcher_ip,
    )
    model_names = [name.strip() for name in args.models.split(",") if name.strip()] if args.models else None
    metrics = run_comparison(
        df,
        run_id,
        output_dir,
        intermediate_dir,
        args.split,
        args.monthly_train_fraction,
        model_names,
    )
    print(metrics.to_string(index=False))
    print(f"Artifacts: {output_dir}")
    if intermediate_dir is not None:
        print(f"Intermediate artifacts: {intermediate_dir}")


# %%
if __name__ == "__main__":
    main()
