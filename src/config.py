from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    sample_data: Path
    sample_data_fallback: Path | None
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


@dataclass
class TrainConfig:
    random_seed: int
    test_size: float
    target_column: str
    feature_columns: list[str]
    pre_event_feature_columns: list[str]
    post_event_feature_columns: list[str]
    feature_set_mode: str
    model_selection_metric: str
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

    sample_fallback = paths_raw.get("sample_data_fallback")
    paths = PathsConfig(
        sample_data=_expand(paths_raw["sample_data"]),
        sample_data_fallback=_expand(sample_fallback) if sample_fallback else None,
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
    )
    feature_columns = [str(c) for c in train_raw["feature_columns"]]
    pre_event = [str(c) for c in train_raw.get("pre_event_feature_columns", feature_columns)]
    post_event = [str(c) for c in train_raw.get("post_event_feature_columns", feature_columns)]
    severity_code_map_raw = train_raw.get("severity_code_map", {1: 0, 2: 1, 3: 2})
    severity_code_map = {int(k): int(v) for k, v in severity_code_map_raw.items()}
    train = TrainConfig(
        random_seed=int(train_raw["random_seed"]),
        test_size=float(train_raw["test_size"]),
        target_column=str(train_raw["target_column"]),
        feature_columns=feature_columns,
        pre_event_feature_columns=pre_event,
        post_event_feature_columns=post_event,
        feature_set_mode=str(train_raw.get("feature_set_mode", "pre_event")),
        model_selection_metric=str(train_raw["model_selection_metric"]),
        enable_hyperparameter_search=bool(train_raw.get("enable_hyperparameter_search", True)),
        severity_code_map=severity_code_map,
        label_mapping_version=str(train_raw.get("label_mapping_version", "stats19_default")),
    )
    app = AppConfig(
        project_name=str(app_raw["project_name"]),
        class_labels=[str(c) for c in app_raw["class_labels"]],
    )
    return Settings(paths=paths, train=train, app=app)
