from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_settings
from src.data_pipeline import load_dataset, prepare_dataset
from src.modeling import build_models, evaluate_predictions, fit_and_select_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train severity classification models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _build_model_comparison(all_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "model_name": m["model_name"],
                "accuracy": float(m["accuracy"]),
                "f1_macro": float(m["f1_macro"]),
            }
            for m in all_metrics
        ]
    )
    return df.sort_values("f1_macro", ascending=False).reset_index(drop=True)


def _save_model_comparison_figure(model_compare_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(model_compare_df))
    width = 0.36
    ax.bar(x - width / 2, model_compare_df["accuracy"], width=width, label="Accuracy")
    ax.bar(x + width / 2, model_compare_df["f1_macro"], width=width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(model_compare_df["model_name"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (Sample Demo)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix_figure(confusion_matrix: list[list[int]], labels: list[str], output_path: Path) -> None:
    cm = np.array(confusion_matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Confusion Matrix (Selected Model)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_feature_importance_figure(
    pipeline,
    feature_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    importance = None
    method = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        method = "tree_importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
        method = "avg_abs_coef"

    if importance is None:
        df = pd.DataFrame({"feature": feature_columns, "importance": np.zeros(len(feature_columns))})
    else:
        df = pd.DataFrame({"feature": feature_columns, "importance": importance})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    top = df.head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["feature"][::-1], top["importance"][::-1])
    ax.set_title(f"Feature Influence ({method or 'not available'})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return df


def _fatal_recall(y_true: np.ndarray, y_pred: np.ndarray, fatal_class_index: int = 2) -> float:
    true_bin = (y_true == fatal_class_index).astype(int)
    pred_bin = (y_pred == fatal_class_index).astype(int)
    return float(recall_score(true_bin, pred_bin, zero_division=0))


def _run_cv_reliability(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
    class_labels: list[str],
) -> dict[str, Any]:
    min_count = int(y.value_counts().min())
    n_splits = max(2, min(5, min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    models = build_models(random_seed=random_seed)
    rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        fold_scores = {"accuracy": [], "f1_macro": [], "fatal_recall": []}
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_predictions(y_test.values, y_pred)
            fold_scores["accuracy"].append(float(metrics["accuracy"]))
            fold_scores["f1_macro"].append(float(metrics["f1_macro"]))
            fold_scores["fatal_recall"].append(_fatal_recall(y_test.values, y_pred))

        rows.append(
            {
                "model_name": model_name,
                "cv_splits": n_splits,
                "accuracy_mean": float(np.mean(fold_scores["accuracy"])),
                "accuracy_std": float(np.std(fold_scores["accuracy"])),
                "f1_macro_mean": float(np.mean(fold_scores["f1_macro"])),
                "f1_macro_std": float(np.std(fold_scores["f1_macro"])),
                "fatal_recall_mean": float(np.mean(fold_scores["fatal_recall"])),
                "fatal_recall_std": float(np.std(fold_scores["fatal_recall"])),
            }
        )

    rows = sorted(rows, key=lambda r: r["f1_macro_mean"], reverse=True)
    return {
        "class_labels": class_labels,
        "rows": rows,
        "selected_model_by_cv": rows[0]["model_name"] if rows else None,
    }


def _run_time_holdout_reliability(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    selected_model_name: str,
    random_seed: int,
) -> dict[str, Any]:
    models = build_models(random_seed=random_seed)
    if selected_model_name not in models:
        return {"available": False, "reason": f"Selected model {selected_model_name} not found."}

    valid_date_mask = dates.notna()
    if valid_date_mask.sum() < max(10, int(0.4 * len(dates))):
        return {
            "available": False,
            "reason": "Not enough valid date values for time-based holdout.",
        }

    Xd = X.loc[valid_date_mask].copy()
    yd = y.loc[valid_date_mask].copy()
    dd = dates.loc[valid_date_mask].sort_values()
    ordered_idx = dd.index
    Xd = Xd.loc[ordered_idx]
    yd = yd.loc[ordered_idx]
    split_idx = int(0.8 * len(Xd))
    if split_idx <= 0 or split_idx >= len(Xd):
        return {"available": False, "reason": "Invalid split index for time holdout."}

    X_train, X_test = Xd.iloc[:split_idx], Xd.iloc[split_idx:]
    y_train, y_test = yd.iloc[:split_idx], yd.iloc[split_idx:]
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", models[selected_model_name]),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test.values, y_pred)
    return {
        "available": True,
        "split_date_start": str(dd.iloc[split_idx].date()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "fatal_recall": _fatal_recall(y_test.values, y_pred),
        },
    }


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
    best_pred = best.pipeline.predict(X_test)

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

    artifacts_dir = settings.paths.model_artifact.parent
    reports_dir = Path("reports/figures").resolve()
    model_compare_path = artifacts_dir / "model_compare.csv"
    error_cases_path = artifacts_dir / "error_cases.csv"
    metrics_cv_path = artifacts_dir / "metrics_cv.json"
    model_comparison_fig_path = reports_dir / "model_comparison.png"
    confusion_matrix_fig_path = reports_dir / "confusion_matrix.png"
    feature_importance_fig_path = reports_dir / "feature_importance.png"

    model_compare_df = _build_model_comparison(all_metrics)
    model_compare_df.to_csv(model_compare_path, index=False)
    _save_model_comparison_figure(model_compare_df, model_comparison_fig_path)
    _save_confusion_matrix_figure(best.metrics["confusion_matrix"], settings.app.class_labels, confusion_matrix_fig_path)
    feature_importance_df = _save_feature_importance_figure(best.pipeline, settings.train.feature_columns, feature_importance_fig_path)

    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(best_pred)
    error_mask = y_test_arr != y_pred_arr
    error_df = X_test.copy().reset_index(drop=True)
    error_df["actual_class_index"] = y_test_arr
    error_df["predicted_class_index"] = y_pred_arr
    error_df["actual_class_label"] = error_df["actual_class_index"].map(lambda i: settings.app.class_labels[int(i)])
    error_df["predicted_class_label"] = error_df["predicted_class_index"].map(lambda i: settings.app.class_labels[int(i)])
    error_df = error_df[error_mask]
    error_df.to_csv(error_cases_path, index=False)

    cv_results = _run_cv_reliability(
        X=prepared.X,
        y=prepared.y,
        random_seed=settings.train.random_seed,
        class_labels=settings.app.class_labels,
    )
    time_holdout = _run_time_holdout_reliability(
        X=prepared.X,
        y=prepared.y,
        dates=prepared.dates,
        selected_model_name=best.name,
        random_seed=settings.train.random_seed,
    )
    with metrics_cv_path.open("w", encoding="utf-8") as f:
        json.dump({"cv": cv_results, "time_holdout": time_holdout}, f, ensure_ascii=False, indent=2)

    observations: list[str]
    if error_df.empty:
        observations = [
            "No error cases on the current sample split; this likely reflects small sample size rather than perfect generalization.",
            "Class distribution in the sample is limited, so metrics can look optimistic.",
            "Cross-validation and larger data are required before any policy-level interpretation.",
        ]
    else:
        top_error_features = error_df[settings.train.feature_columns].std(numeric_only=True).sort_values(ascending=False).head(3).index.tolist()
        observations = [
            f"Error cases are concentrated in {len(error_df)} rows on the sample split.",
            f"Most variable features among errors: {', '.join(top_error_features)}.",
            "Fatal vs serious boundary remains the key risk area and needs more examples.",
        ]

    metrics_payload = {
        "selected_model": best.name,
        "selected_model_metrics": best.metrics,
        "all_model_metrics": all_metrics,
        "rows_total": int(len(raw_df)),
        "rows_used_for_training": int(len(prepared.X)),
        "rows_removed_invalid_target": prepared.removed_invalid_target,
        "missing_rate_by_feature": prepared.missing_rate_by_feature,
        "feature_columns": settings.train.feature_columns,
        "performance_note": "sample demo performance",
        "model_compare_csv": str(model_compare_path),
        "metrics_cv_json": str(metrics_cv_path),
        "error_cases_csv": str(error_cases_path),
        "report_figures": {
            "model_comparison": str(model_comparison_fig_path),
            "confusion_matrix": str(confusion_matrix_fig_path),
            "feature_importance": str(feature_importance_fig_path),
        },
        "error_analysis_observations": observations,
        "error_case_count": int(len(error_df)),
        "fatal_recall_on_test_split": _fatal_recall(y_test_arr, y_pred_arr),
    }
    with settings.paths.metrics_artifact.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    feature_importance_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    print(f"Model saved to: {settings.paths.model_artifact}")
    print(f"Metrics saved to: {settings.paths.metrics_artifact}")
    print(f"Selected model: {best.name}")
    print(f"Selection metric ({settings.train.model_selection_metric}): {best.metrics[settings.train.model_selection_metric]:.4f}")


if __name__ == "__main__":
    main()
