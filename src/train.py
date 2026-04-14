from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from src.config import load_settings
from src.data_pipeline import load_dataset, prepare_dataset
from src.modeling import fit_and_select_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train severity classification models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    raw_df = load_dataset(settings.paths.sample_data)
    prepared = prepare_dataset(
        raw_df,
        feature_columns=settings.train.feature_columns,
        target_column=settings.train.target_column,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=settings.train.test_size,
        random_state=settings.train.random_seed,
        stratify=prepared.y,
    )

    best, all_metrics = fit_and_select_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_seed=settings.train.random_seed,
        metric_key=settings.train.model_selection_metric,
    )

    model_payload: dict[str, Any] = {
        "model_name": best.name,
        "pipeline": best.pipeline,
        "feature_columns": settings.train.feature_columns,
        "feature_defaults": prepared.X.median(numeric_only=True).to_dict(),
        "class_labels": settings.app.class_labels,
        "target_column": settings.train.target_column,
    }

    settings.paths.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.metrics_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, settings.paths.model_artifact)

    metrics_payload = {
        "selected_model": best.name,
        "selected_model_metrics": best.metrics,
        "all_model_metrics": all_metrics,
        "rows_total": int(len(raw_df)),
        "rows_used_for_training": int(len(prepared.X)),
        "rows_removed_invalid_target": prepared.removed_invalid_target,
        "feature_columns": settings.train.feature_columns,
    }
    with settings.paths.metrics_artifact.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {settings.paths.model_artifact}")
    print(f"Metrics saved to: {settings.paths.metrics_artifact}")
    print(f"Selected model: {best.name}")
    print(f"Selection metric ({settings.train.model_selection_metric}): {best.metrics[settings.train.model_selection_metric]:.4f}")


if __name__ == "__main__":
    main()

