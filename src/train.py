from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_settings
from src.data_pipeline import build_missingness_by_time, load_dataset, prepare_dataset
from src.modeling import build_models, evaluate_predictions, fit_and_select_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train severity classification models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _get_feature_columns(settings) -> list[str]:
    mode = settings.train.feature_set_mode
    if mode == "pre_event":
        return settings.train.pre_event_feature_columns
    if mode == "post_event":
        return settings.train.post_event_feature_columns
    return settings.train.feature_columns


def _enforce_pre_event_policy(mode: str, feature_columns: list[str]) -> None:
    if mode != "pre_event":
        return
    forbidden = {"number_of_casualties", "casualty_severity", "accident_severity", "did_police_officer_attend_scene_of_accident"}
    present = sorted([f for f in feature_columns if f in forbidden])
    if present:
        raise RuntimeError(f"pre_event mode cannot include post-event features: {present}")


def _build_model_comparison(all_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "model_name": m["model_name"],
                "accuracy": float(m["accuracy"]),
                "f1_macro": float(m["f1_macro"]),
                "balanced_accuracy": float(m.get("balanced_accuracy", 0.0)),
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
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix_figure(confusion_matrix: list[list[int]], labels: list[str], output_path: Path) -> None:
    cm = np.asarray(confusion_matrix)
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


def _save_feature_importance_figure(pipeline, feature_columns: list[str], output_path: Path) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    importance = None
    method = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        method = "tree_importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importance = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
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


def _fatal_recall(y_true: np.ndarray, y_pred: np.ndarray, fatal_class_index: int = 0) -> float:
    true_bin = (y_true == fatal_class_index).astype(int)
    pred_bin = (y_pred == fatal_class_index).astype(int)
    return float(recall_score(true_bin, pred_bin, zero_division=0))


def _run_cv_reliability(X: pd.DataFrame, y: pd.Series, random_seed: int, class_labels: list[str], fatal_idx: int) -> dict[str, Any]:
    max_rows = 120000
    if len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=random_seed, stratify=y)
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
            fold_scores["fatal_recall"].append(_fatal_recall(y_test.values, y_pred, fatal_class_index=fatal_idx))
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
    return {"class_labels": class_labels, "rows": rows, "selected_model_by_cv": rows[0]["model_name"] if rows else None}


def _run_time_holdout_reliability(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    selected_model_name: str,
    selected_best_params: dict[str, Any],
    random_seed: int,
    fatal_idx: int,
) -> dict[str, Any]:
    models = build_models(random_seed=random_seed)
    if selected_model_name not in models:
        return {"available": False, "reason": f"Selected model {selected_model_name} not found."}
    valid_date_mask = dates.notna()
    if valid_date_mask.sum() < max(10, int(0.4 * len(dates))):
        return {"available": False, "reason": "Not enough valid date values for time-based holdout."}

    Xd = X.loc[valid_date_mask].copy()
    yd = y.loc[valid_date_mask].copy()
    dd = dates.loc[valid_date_mask].sort_values()
    ordered_idx = dd.index
    Xd = Xd.loc[ordered_idx]
    yd = yd.loc[ordered_idx]
    split_idx = int(0.8 * len(Xd))
    if split_idx <= 0 or split_idx >= len(Xd):
        return {"available": False, "reason": "Invalid split index for time holdout."}

    base_model = models[selected_model_name]
    model_params = {k.replace("model__", ""): v for k, v in selected_best_params.items() if k.startswith("model__")}
    if model_params:
        base_model.set_params(**model_params)

    X_train, X_test = Xd.iloc[:split_idx], Xd.iloc[split_idx:]
    y_train, y_test = yd.iloc[:split_idx], yd.iloc[split_idx:]
    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", base_model)])
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
            "fatal_recall": _fatal_recall(y_test.values, y_pred, fatal_class_index=fatal_idx),
        },
    }


def _build_data_quality_report(raw_df: pd.DataFrame, prepared, dataset_used: Path, target_column: str) -> dict[str, Any]:
    target_raw = pd.to_numeric(raw_df.get(target_column), errors="coerce")
    target_valid = target_raw.isin([1, 2, 3])
    target_distribution = prepared.y.value_counts().sort_index().to_dict()
    date_values = pd.to_datetime(raw_df.get("date"), errors="coerce")
    spatial = prepared.spatial_keys.fillna("unknown").astype(str)
    return {
        "dataset_used": str(dataset_used),
        "rows_raw": int(len(raw_df)),
        "rows_valid_target": int(target_valid.sum()),
        "rows_invalid_target": int((~target_valid).sum()),
        "target_distribution_after_filter": {str(int(k) + 1): int(v) for k, v in target_distribution.items()},
        "missing_rate_by_feature": prepared.missing_rate_by_feature,
        "date_coverage": {
            "min": str(date_values.min().date()) if date_values.notna().any() else None,
            "max": str(date_values.max().date()) if date_values.notna().any() else None,
            "unique_days": int(date_values.nunique(dropna=True)),
        },
        "spatial_coverage": {
            "unique_spatial_keys": int(spatial.nunique(dropna=True)),
            "unknown_spatial_key_ratio": float((spatial == "unknown").mean()),
        },
    }


def _build_leakage_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_columns: list[str],
    dates_train: pd.Series,
    dates_test: pd.Series,
) -> dict[str, Any]:
    suspicious = {"number_of_casualties", "casualty_severity", "accident_severity", "did_police_officer_attend_scene_of_accident"}
    present_suspicious = sorted([f for f in feature_columns if f in suspicious])
    overlap = 0
    if not X_train.empty and not X_test.empty:
        train_hash = pd.util.hash_pandas_object(X_train, index=False).astype(str)
        test_hash = pd.util.hash_pandas_object(X_test, index=False).astype(str)
        overlap = int(pd.Series(test_hash).isin(set(train_hash)).sum())
    temporal_overlap = None
    if dates_train.notna().any() and dates_test.notna().any():
        temporal_overlap = bool(pd.to_datetime(dates_train).max() >= pd.to_datetime(dates_test).min())
    risk = "low"
    if present_suspicious or overlap > 0:
        risk = "medium"
    if present_suspicious and overlap > 0:
        risk = "high"
    return {
        "risk_level": risk,
        "checklist": {
            "suspicious_feature_present": bool(present_suspicious),
            "row_overlap_train_test": overlap > 0,
            "temporal_overlap_possible": temporal_overlap,
        },
        "suspicious_features_present": present_suspicious,
        "duplicate_row_overlap_count": overlap,
        "notes": [
            "Random split can mix adjacent periods. Use time-based holdout for deployment realism.",
            "Presence of post-event variables should be justified in README to avoid leakage concerns.",
        ],
    }


def _build_threshold_report(y_true: np.ndarray, fatal_prob: np.ndarray, fatal_idx: int) -> pd.DataFrame:
    true_bin = (y_true == fatal_idx).astype(int)
    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(0.05, 0.95, 19):
        pred_bin = (fatal_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_bin, pred_bin, average="binary", zero_division=0)
        tp = int(((true_bin == 1) & (pred_bin == 1)).sum())
        fp = int(((true_bin == 0) & (pred_bin == 1)).sum())
        fn = int(((true_bin == 1) & (pred_bin == 0)).sum())
        tn = int(((true_bin == 0) & (pred_bin == 0)).sum())
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def _build_calibration_report(y_true: np.ndarray, fatal_prob: np.ndarray, fatal_idx: int) -> dict[str, Any]:
    true_bin = (y_true == fatal_idx).astype(int)
    table = pd.DataFrame({"y": true_bin, "p": fatal_prob})
    if table["p"].nunique() > 1:
        table["bin"] = pd.qcut(table["p"], q=min(10, table["p"].nunique()), duplicates="drop")
    else:
        table["bin"] = "single_bin"
    grouped = table.groupby("bin", observed=False)
    bins = []
    ece = 0.0
    for key, g in grouped:
        mean_pred = float(g["p"].mean())
        obs_rate = float(g["y"].mean())
        count = int(len(g))
        bins.append(
            {
                "bin": str(key),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_fatal_rate": obs_rate,
                "abs_gap": abs(mean_pred - obs_rate),
            }
        )
        ece += abs(mean_pred - obs_rate) * (count / max(1, len(table)))
    return {
        "fatal_class_index": fatal_idx,
        "sample_count": int(len(table)),
        "brier_score": float(brier_score_loss(true_bin, fatal_prob)),
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


def _build_ablation_leakage(raw_df: pd.DataFrame, settings) -> pd.DataFrame:
    outputs: list[dict[str, Any]] = []
    for mode_name, cols in [("pre_event", settings.train.pre_event_feature_columns), ("post_event", settings.train.post_event_feature_columns)]:
        prepared = prepare_dataset(
            raw_df,
            feature_columns=cols,
            target_column=settings.train.target_column,
            severity_code_map=settings.train.severity_code_map,
        )
        X, y = prepared.X, prepared.y
        if len(X) > 120000:
            X, _, y, _ = train_test_split(X, y, train_size=120000, random_state=settings.train.random_seed, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.train.test_size,
            random_state=settings.train.random_seed,
            stratify=y if y.value_counts().min() >= 2 else None,
        )
        best, _, _ = fit_and_select_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_seed=settings.train.random_seed,
            metric_key=settings.train.model_selection_metric,
            enable_hyperparameter_search=False,
        )
        y_pred = best.pipeline.predict(X_test)
        outputs.append(
            {
                "feature_set_mode": mode_name,
                "rows": int(len(X)),
                "selected_model": best.name,
                "accuracy": float(best.metrics["accuracy"]),
                "f1_macro": float(best.metrics["f1_macro"]),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "feature_count": int(len(cols)),
            }
        )
    return pd.DataFrame(outputs)


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    dataset_used = settings.paths.sample_data
    if not dataset_used.exists() and settings.paths.sample_data_fallback is not None and settings.paths.sample_data_fallback.exists():
        dataset_used = settings.paths.sample_data_fallback
    raw_df = load_dataset(settings.paths.sample_data, settings.paths.sample_data_fallback)

    feature_columns = _get_feature_columns(settings)
    _enforce_pre_event_policy(settings.train.feature_set_mode, feature_columns)
    prepared = prepare_dataset(
        raw_df,
        feature_columns=feature_columns,
        target_column=settings.train.target_column,
        severity_code_map=settings.train.severity_code_map,
    )
    date_series = prepared.dates

    fatal_idx = [c.lower() for c in settings.app.class_labels].index("fatal") if "fatal" in [c.lower() for c in settings.app.class_labels] else 0

    stratify_value = prepared.y if prepared.y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        prepared.X,
        prepared.y,
        date_series,
        test_size=settings.train.test_size,
        random_state=settings.train.random_seed,
        stratify=stratify_value,
    )

    best, all_metrics, search_records = fit_and_select_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_seed=settings.train.random_seed,
        metric_key=settings.train.model_selection_metric,
        enable_hyperparameter_search=settings.train.enable_hyperparameter_search,
    )
    best_pred = best.pipeline.predict(X_test)

    model_payload: dict[str, Any] = {
        "model_name": best.name,
        "pipeline": best.pipeline,
        "feature_columns": feature_columns,
        "feature_defaults": prepared.X.median(numeric_only=True).to_dict(),
        "class_labels": settings.app.class_labels,
        "target_column": settings.train.target_column,
        "severity_code_map": settings.train.severity_code_map,
        "label_mapping_version": settings.train.label_mapping_version,
        "feature_set_mode": settings.train.feature_set_mode,
    }

    for output_path in [
        settings.paths.model_artifact,
        settings.paths.metrics_artifact,
        settings.paths.data_quality_artifact,
        settings.paths.leakage_check_artifact,
        settings.paths.threshold_artifact,
        settings.paths.calibration_artifact,
        settings.paths.search_artifact,
        settings.paths.ablation_leakage_artifact,
        settings.paths.missingness_by_time_artifact,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
    feature_importance_df = _save_feature_importance_figure(best.pipeline, feature_columns, feature_importance_fig_path)

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

    proba = best.pipeline.predict_proba(X_test)
    fatal_prob = np.asarray(proba[:, fatal_idx], dtype=float)
    threshold_df = _build_threshold_report(y_test_arr, fatal_prob, fatal_idx=fatal_idx)
    threshold_df.to_csv(settings.paths.threshold_artifact, index=False)
    calibration_report = _build_calibration_report(y_test_arr, fatal_prob, fatal_idx=fatal_idx)
    settings.paths.calibration_artifact.write_text(json.dumps(_to_jsonable(calibration_report), ensure_ascii=False, indent=2), encoding="utf-8")

    cv_results = _run_cv_reliability(prepared.X, prepared.y, random_seed=settings.train.random_seed, class_labels=settings.app.class_labels, fatal_idx=fatal_idx)
    time_holdout = _run_time_holdout_reliability(
        X=prepared.X,
        y=prepared.y,
        dates=prepared.dates,
        selected_model_name=best.name,
        selected_best_params=best.metrics.get("best_params", {}),
        random_seed=settings.train.random_seed,
        fatal_idx=fatal_idx,
    )
    metrics_cv_path.write_text(json.dumps(_to_jsonable({"cv": cv_results, "time_holdout": time_holdout}), ensure_ascii=False, indent=2), encoding="utf-8")

    if error_df.empty:
        observations = [
            "No error cases on this split, likely due to limited sample coverage.",
            "Current metrics should be treated as baseline evidence rather than final performance.",
            "Larger data and stricter time-based evaluation are required before strong claims.",
        ]
    else:
        top_error_features = error_df[feature_columns].std(numeric_only=True).sort_values(ascending=False).head(3).index.tolist()
        observations = [
            f"Error cases are concentrated in {len(error_df)} rows on the current split.",
            f"Most variable features among errors are {', '.join(top_error_features)}.",
            "Fatal vs serious boundary remains the highest-risk confusion zone.",
        ]

    quality_report = _build_data_quality_report(raw_df=raw_df, prepared=prepared, dataset_used=dataset_used, target_column=settings.train.target_column)
    settings.paths.data_quality_artifact.write_text(json.dumps(_to_jsonable(quality_report), ensure_ascii=False, indent=2), encoding="utf-8")

    leakage_report = _build_leakage_report(
        X_train=X_train,
        X_test=X_test,
        feature_columns=feature_columns,
        dates_train=d_train,
        dates_test=d_test,
    )
    settings.paths.leakage_check_artifact.write_text(json.dumps(_to_jsonable(leakage_report), ensure_ascii=False, indent=2), encoding="utf-8")
    settings.paths.search_artifact.write_text(json.dumps(_to_jsonable(search_records), ensure_ascii=False, indent=2), encoding="utf-8")

    missingness_by_time = build_missingness_by_time(
        raw_df,
        features=["cloud_cover", "sunshine", "global_radiation", "max_temp", "mean_temp", "min_temp", "precipitation", "pressure"],
    )
    missingness_by_time.to_csv(settings.paths.missingness_by_time_artifact, index=False)
    ablation_df = _build_ablation_leakage(raw_df, settings)
    ablation_df.to_csv(settings.paths.ablation_leakage_artifact, index=False)

    class_report = best.metrics.get("classification_report", {})
    fatal_key = str(fatal_idx)
    fatal_metrics = class_report.get(fatal_key, {})
    class_distribution = prepared.y.value_counts().sort_index().to_dict()
    overall_bal_acc = float(balanced_accuracy_score(y_test_arr, y_pred_arr))
    fatal_true_bin = (y_test_arr == fatal_idx).astype(int)
    fatal_pr_auc = float(average_precision_score(fatal_true_bin, fatal_prob)) if fatal_true_bin.sum() > 0 else 0.0

    metrics_payload = {
        "selected_model": best.name,
        "selected_model_metrics": best.metrics,
        "all_model_metrics": all_metrics,
        "rows_total": int(len(raw_df)),
        "rows_used_for_training": int(len(prepared.X)),
        "rows_removed_invalid_target": prepared.removed_invalid_target,
        "missing_rate_by_feature": prepared.missing_rate_by_feature,
        "feature_columns": feature_columns,
        "feature_set_mode": settings.train.feature_set_mode,
        "label_mapping_version": settings.train.label_mapping_version,
        "data_version_tag": f"{Path(dataset_used).name}_{len(raw_df)}rows",
        "performance_note": "baseline performance on current processed dataset",
        "dataset_used": str(dataset_used),
        "model_compare_csv": str(model_compare_path),
        "metrics_cv_json": str(metrics_cv_path),
        "error_cases_csv": str(error_cases_path),
        "data_quality_report_json": str(settings.paths.data_quality_artifact),
        "leakage_check_report_json": str(settings.paths.leakage_check_artifact),
        "threshold_report_csv": str(settings.paths.threshold_artifact),
        "calibration_report_json": str(settings.paths.calibration_artifact),
        "hyperparameter_search_json": str(settings.paths.search_artifact),
        "ablation_leakage_csv": str(settings.paths.ablation_leakage_artifact),
        "missingness_by_time_csv": str(settings.paths.missingness_by_time_artifact),
        "report_figures": {
            "model_comparison": str(model_comparison_fig_path),
            "confusion_matrix": str(confusion_matrix_fig_path),
            "feature_importance": str(feature_importance_fig_path),
        },
        "overall_metrics_on_test_split": {
            "accuracy": float(best.metrics["accuracy"]),
            "f1_macro": float(best.metrics["f1_macro"]),
            "balanced_accuracy": overall_bal_acc,
        },
        "safety_metrics_on_test_split": {
            "fatal_precision": float(fatal_metrics.get("precision", 0.0)),
            "fatal_recall": float(fatal_metrics.get("recall", 0.0)),
            "fatal_f1": float(fatal_metrics.get("f1-score", 0.0)),
            "fatal_pr_auc": fatal_pr_auc,
            "fatal_support": int(fatal_metrics.get("support", 0)),
        },
        "probability_metrics_on_test_split": {
            "fatal_brier": float(calibration_report.get("brier_score", 0.0)),
            "fatal_ece": float(calibration_report.get("expected_calibration_error", 0.0)),
        },
        "class_distribution": {str(int(k) + 1): int(v) for k, v in class_distribution.items()},
        "fatal_metrics_on_test_split": {
            "precision": float(fatal_metrics.get("precision", 0.0)),
            "recall": float(fatal_metrics.get("recall", 0.0)),
            "f1": float(fatal_metrics.get("f1-score", 0.0)),
            "support": int(fatal_metrics.get("support", 0)),
        },
        "error_analysis_observations": observations,
        "error_case_count": int(len(error_df)),
        "fatal_recall_on_test_split": _fatal_recall(y_test_arr, y_pred_arr, fatal_class_index=fatal_idx),
    }
    settings.paths.metrics_artifact.write_text(json.dumps(_to_jsonable(metrics_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    feature_importance_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    print(f"Model saved to: {settings.paths.model_artifact}")
    print(f"Metrics saved to: {settings.paths.metrics_artifact}")
    print(f"Selected model: {best.name}")
    print(f"Selection metric ({settings.train.model_selection_metric}): {best.metrics[settings.train.model_selection_metric]:.4f}")


if __name__ == "__main__":
    main()
