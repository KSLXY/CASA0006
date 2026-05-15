from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    master_data: Path
    model_artifact: Path
    metrics_artifact: Path
    evaluation_artifact: Path
    data_quality_artifact: Path
    leakage_check_artifact: Path
    threshold_artifact: Path
    calibration_artifact: Path
    search_artifact: Path
    ablation_leakage_artifact: Path
    missingness_by_time_artifact: Path
    report_figures_dir: Path


@dataclass
class TrainConfig:
    random_seed: int
    test_size: float
    min_training_rows: int
    target_column: str
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]
    external_weather_feature_columns: list[str]
    pre_event_feature_columns: list[str]
    post_event_feature_columns: list[str]
    feature_set_mode: str
    model_selection_metric: str
    safety_max_false_positive_rate: float
    permutation_importance_rows: int
    calibration_rows: int
    enable_hyperparameter_search: bool
    severity_code_map: dict[int, int]
    label_mapping_version: str


@dataclass
class AppConfig:
    project_name: str
    class_labels: list[str]


@dataclass
class Settings:
    paths: PathsConfig
    train: TrainConfig
    app: AppConfig


def _expand(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def load_settings(config_path: str | Path) -> Settings:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    paths_raw = raw["paths"]
    train_raw = raw["train"]
    app_raw = raw["app"]

    paths = PathsConfig(
        master_data=_expand(paths_raw["master_data"]),
        model_artifact=_expand(paths_raw["model_artifact"]),
        metrics_artifact=_expand(paths_raw["metrics_artifact"]),
        evaluation_artifact=_expand(paths_raw["evaluation_artifact"]),
        data_quality_artifact=_expand(paths_raw.get("data_quality_artifact", "artifacts/data_quality_report.json")),
        leakage_check_artifact=_expand(paths_raw.get("leakage_check_artifact", "artifacts/leakage_check_report.json")),
        threshold_artifact=_expand(paths_raw.get("threshold_artifact", "artifacts/threshold_report.csv")),
        calibration_artifact=_expand(paths_raw.get("calibration_artifact", "artifacts/calibration_report.json")),
        search_artifact=_expand(paths_raw.get("search_artifact", "artifacts/hyperparameter_search.json")),
        ablation_leakage_artifact=_expand(paths_raw.get("ablation_leakage_artifact", "artifacts/ablation_leakage.csv")),
        missingness_by_time_artifact=_expand(paths_raw.get("missingness_by_time_artifact", "artifacts/missingness_by_time.csv")),
        report_figures_dir=_expand(paths_raw.get("report_figures_dir", "reports/figures")),
    )
    feature_columns = [str(c) for c in train_raw["feature_columns"]]
    numeric_features = [str(c) for c in train_raw.get("numeric_feature_columns", feature_columns)]
    categorical_features = [str(c) for c in train_raw.get("categorical_feature_columns", [])]
    external_weather_features = [str(c) for c in train_raw.get("external_weather_feature_columns", [])]
    pre_event = [str(c) for c in train_raw.get("pre_event_feature_columns", feature_columns)]
    post_event = [str(c) for c in train_raw.get("post_event_feature_columns", feature_columns)]
    severity_code_map_raw = train_raw.get("severity_code_map", {1: 0, 2: 1, 3: 2})
    severity_code_map = {int(k): int(v) for k, v in severity_code_map_raw.items()}
    train = TrainConfig(
        random_seed=int(train_raw["random_seed"]),
        test_size=float(train_raw["test_size"]),
        min_training_rows=int(train_raw.get("min_training_rows", 100000)),
        target_column=str(train_raw["target_column"]),
        feature_columns=feature_columns,
        numeric_feature_columns=numeric_features,
        categorical_feature_columns=categorical_features,
        external_weather_feature_columns=external_weather_features,
        pre_event_feature_columns=pre_event,
        post_event_feature_columns=post_event,
        feature_set_mode=str(train_raw.get("feature_set_mode", "pre_event")),
        model_selection_metric=str(train_raw["model_selection_metric"]),
        safety_max_false_positive_rate=float(train_raw.get("safety_max_false_positive_rate", 0.2)),
        permutation_importance_rows=int(train_raw.get("permutation_importance_rows", 20000)),
        calibration_rows=int(train_raw.get("calibration_rows", 120000)),
        enable_hyperparameter_search=bool(train_raw.get("enable_hyperparameter_search", True)),
        severity_code_map=severity_code_map,
        label_mapping_version=str(train_raw.get("label_mapping_version", "stats19_default")),
    )
    app = AppConfig(
        project_name=str(app_raw["project_name"]),
        class_labels=[str(c) for c in app_raw["class_labels"]],
    )
    return Settings(paths=paths, train=train, app=app)
