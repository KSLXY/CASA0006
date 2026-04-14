from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import pandas as pd

from src.config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-row severity prediction.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None, help="Optional model path override.")
    parser.add_argument("--input-json", type=str, default=None, help="JSON string input.")
    parser.add_argument("--input-file", type=str, default=None, help="JSON file input path.")
    return parser.parse_args()


def _load_input_data(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.input_json:
        return json.loads(args.input_json)
    raise ValueError("Provide --input-json or --input-file.")


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    model_path = args.model or str(settings.paths.model_artifact)

    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    feature_columns: list[str] = payload["feature_columns"]
    defaults: dict[str, float] = payload.get("feature_defaults", {})
    class_labels: list[str] = payload.get("class_labels", ["Class 1", "Class 2", "Class 3"])

    user_input = _load_input_data(args)
    row = {col: user_input.get(col, defaults.get(col, 0.0)) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)

    pred_class_idx = int(pipeline.predict(X)[0])
    probabilities = pipeline.predict_proba(X)[0].tolist()
    result = {
        "predicted_class_index": pred_class_idx,
        "predicted_class_label": class_labels[pred_class_idx],
        "predicted_accident_severity": pred_class_idx + 1,
        "probabilities": probabilities,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

