from __future__ import annotations

import argparse
import json

import joblib

from src.config import load_settings
from src.data_pipeline import load_dataset, prepare_dataset
from src.modeling import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on dataset.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None, help="Optional model path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    model_path = args.model or str(settings.paths.model_artifact)

    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    feature_columns = payload["feature_columns"]

    df = load_dataset(settings.paths.sample_data, settings.paths.sample_data_fallback)
    prepared = prepare_dataset(
        df=df,
        feature_columns=feature_columns,
        target_column=payload.get("target_column", settings.train.target_column),
        severity_code_map=payload.get("severity_code_map", settings.train.severity_code_map),
    )

    pred = pipeline.predict(prepared.X)
    metrics = evaluate_predictions(prepared.y.values, pred)
    output = {
        "model_name": payload["model_name"],
        "metrics": metrics,
        "rows_evaluated": int(len(prepared.X)),
    }

    settings.paths.evaluation_artifact.parent.mkdir(parents=True, exist_ok=True)
    with settings.paths.evaluation_artifact.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Evaluation saved to: {settings.paths.evaluation_artifact}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
