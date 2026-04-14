from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]


def build_models(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_seed),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def fit_and_select_model(
    X_train,
    y_train,
    X_test,
    y_test,
    random_seed: int,
    metric_key: str = "f1_macro",
) -> tuple[ModelResult, list[dict[str, Any]]]:
    all_metrics: list[dict[str, Any]] = []
    best_result: ModelResult | None = None
    best_score = -1.0

    models = build_models(random_seed=random_seed)
    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, pred)
        metrics["model_name"] = model_name
        all_metrics.append(metrics)
        score = float(metrics.get(metric_key, 0.0))
        if score > best_score:
            best_score = score
            best_result = ModelResult(name=model_name, pipeline=pipe, metrics=metrics)

    if best_result is None:
        raise RuntimeError("No model was trained.")

    return best_result, all_metrics

