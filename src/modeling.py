from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]


def stringify_categories(x):
    return pd.DataFrame(x).where(pd.notna(x), "missing").astype(str)


def build_models(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting_balanced": HistGradientBoostingClassifier(
            random_state=random_seed,
            class_weight="balanced",
            max_iter=180,
            learning_rate=0.08,
        ),
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
        "hist_gradient_boosting_balanced": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [150, 300],
            "model__max_depth": [None, 8],
        },
    }


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("to_string", FunctionTransformer(stringify_categories, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(model: Any, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def get_transformed_feature_names(pipeline: Pipeline, fallback_features: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return fallback_features
    return [str(c) for c in preprocessor.get_feature_names_out()]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    fatal_idx: int = 0,
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    true_fatal = (y_true == fatal_idx).astype(int)
    pred_fatal = (y_pred == fatal_idx).astype(int)
    fatal_precision, fatal_recall, fatal_f1, _ = precision_recall_fscore_support(
        true_fatal,
        pred_fatal,
        average="binary",
        zero_division=0,
    )
    fatal_pr_auc = 0.0
    if y_proba is not None and true_fatal.sum() > 0:
        fatal_pr_auc = float(average_precision_score(true_fatal, y_proba[:, fatal_idx]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "fatal_precision": float(fatal_precision),
        "fatal_recall": float(fatal_recall),
        "fatal_f1": float(fatal_f1),
        "fatal_pr_auc": fatal_pr_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def fit_and_select_model(
    X_train,
    y_train,
    X_test,
    y_test,
    random_seed: int,
    numeric_features: list[str],
    categorical_features: list[str],
    metric_key: str = "f1_macro",
    enable_hyperparameter_search: bool = True,
    fatal_idx: int = 0,
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
        base_pipe = make_pipeline(model, numeric_features=numeric_features, categorical_features=categorical_features)
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
            pipe = make_pipeline(model_to_fit, numeric_features=numeric_features, categorical_features=categorical_features)
            pipe.fit(X_train, y_train)
        else:
            pipe = base_pipe.fit(X_train, y_train)
            best_params = {}
            cv_best = float("nan")
            top_trials = []

        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
        metrics = evaluate_predictions(y_test, pred, y_proba=proba, fatal_idx=fatal_idx)
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
