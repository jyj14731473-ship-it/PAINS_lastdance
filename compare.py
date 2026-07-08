# %%
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from lib.data_prep import prepare_features
from lib.demo_data import make_demo_outings
from lib.evaluate import (
    acwr_residual_summary,
    log_experiment,
    save_acwr_boxplot,
    save_metric_bar_chart,
)
from lib.labeling import create_labels
from lib.split import rolling_origin_split


# %%
def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# %%
def discover_model_modules(models_dir: str | Path = "models"):
    paths = sorted(Path(models_dir).glob("model_*.py"))
    return [_load_module(path) for path in paths]


# %%
def load_or_prepare_data(input_path: str | Path | None, demo: bool) -> pd.DataFrame:
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

    if "target_y" in raw.columns:
        return raw
    features = prepare_features(raw)
    return create_labels(features)


# %%
def run_comparison(df: pd.DataFrame, run_id: str, output_dir: Path) -> pd.DataFrame:
    split = rolling_origin_split(df)
    modules = discover_model_modules()
    if not modules:
        raise RuntimeError("No models/model_*.py files found.")

    results = []
    acwr_summaries = {}
    for module in modules:
        if not hasattr(module, "run"):
            continue
        result = module.run({}, split.train_df, split.test_df)
        results.append(result)
        log_experiment(result, run_id, len(split.train_df), len(split.test_df))
        acwr_summaries[result["model_name"]] = acwr_residual_summary(split.test_df, result["predictions"])

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [
            {
                "model_name": result["model_name"],
                **result["metrics"],
                "git_commit": result.get("git_commit", ""),
            }
            for result in results
        ]
    ).sort_values("rmse")
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    save_metric_bar_chart(results, output_dir / "metrics_bar.png")
    save_acwr_boxplot(acwr_summaries, output_dir / "acwr_residuals.png")

    report = {
        "run_id": run_id,
        "n_train": len(split.train_df),
        "n_validation_embargo": len(split.validation_df),
        "n_test": len(split.test_df),
        "test_start": split.test_start.isoformat(),
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
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("experiments") / "runs" / run_id
    df = load_or_prepare_data(args.input, args.demo)
    metrics = run_comparison(df, run_id, output_dir)
    print(metrics.to_string(index=False))
    print(f"Artifacts: {output_dir}")


# %%
if __name__ == "__main__":
    main()
