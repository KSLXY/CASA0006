from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]


def build_models(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_seed),
    }


def build_param_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 3.0],
        },
        "random_forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 3],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [150, 300],
            "model__max_depth": [None, 8],
        },
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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
    enable_hyperparameter_search: bool = True,
) -> tuple[ModelResult, list[dict[str, Any]], list[dict[str, Any]]]:
    all_metrics: list[dict[str, Any]] = []
    search_records: list[dict[str, Any]] = []
    best_result: ModelResult | None = None
    best_score = -1.0

    models = build_models(random_seed=random_seed)
    param_grids = build_param_grids()
    min_count = int(np.bincount(np.asarray(y_train, dtype=int)).min())
    n_splits = max(2, min(5, min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for model_name, model in models.items():
        base_pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        if enable_hyperparameter_search:
            X_grid = X_train
            y_grid = y_train
            max_grid_rows = 120000
            if len(X_train) > max_grid_rows:
                X_grid, _, y_grid, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=max_grid_rows,
                    random_state=random_seed,
                    stratify=y_train,
                )
            grid = GridSearchCV(
                estimator=base_pipe,
                param_grid=param_grids.get(model_name, {}),
                scoring="f1_macro",
                cv=cv,
                n_jobs=1,
                refit=True,
                return_train_score=True,
            )
            grid.fit(X_grid, y_grid)
            best_params = grid.best_params_
            cv_best = float(grid.best_score_)
            top_trials: list[dict[str, Any]] = []
            mean_scores = np.asarray(grid.cv_results_["mean_test_score"], dtype=float)
            params_list = grid.cv_results_["params"]
            order = np.argsort(-mean_scores)[:5]
            for idx in order:
                top_trials.append(
                    {
                        "rank": int(len(top_trials) + 1),
                        "mean_test_f1_macro": float(mean_scores[idx]),
                        "params": params_list[idx],
                    }
                )
            model_to_fit = models[model_name]
            model_params = {k.replace("model__", ""): v for k, v in best_params.items() if k.startswith("model__")}
            if model_params:
                model_to_fit.set_params(**model_params)
            pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", model_to_fit)])
            pipe.fit(X_train, y_train)
        else:
            pipe = base_pipe.fit(X_train, y_train)
            best_params = {}
            cv_best = float("nan")
            top_trials = []

        pred = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, pred)
        metrics["model_name"] = model_name
        metrics["best_params"] = best_params
        metrics["cv_best_f1_macro"] = cv_best
        all_metrics.append(metrics)
        score = float(metrics.get(metric_key, 0.0))
        if score > best_score:
            best_score = score
            best_result = ModelResult(name=model_name, pipeline=pipe, metrics=metrics)

        search_records.append(
            {
                "model_name": model_name,
                "cv_splits": n_splits,
                "search_enabled": bool(enable_hyperparameter_search),
                "best_params": best_params,
                "best_score_f1_macro": cv_best,
                "fit_rows_for_search": int(min(len(X_train), 120000)) if enable_hyperparameter_search else int(len(X_train)),
                "top_trials": top_trials,
            }
        )

    if best_result is None:
        raise RuntimeError("No model was trained.")

    return best_result, all_metrics, search_records
