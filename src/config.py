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


@dataclass
class TrainConfig:
    random_seed: int
    test_size: float
    target_column: str
    feature_columns: list[str]
    model_selection_metric: str
    enable_hyperparameter_search: bool


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
    )
    train = TrainConfig(
        random_seed=int(train_raw["random_seed"]),
        test_size=float(train_raw["test_size"]),
        target_column=str(train_raw["target_column"]),
        feature_columns=[str(c) for c in train_raw["feature_columns"]],
        model_selection_metric=str(train_raw["model_selection_metric"]),
        enable_hyperparameter_search=bool(train_raw.get("enable_hyperparameter_search", True)),
    )
    app = AppConfig(
        project_name=str(app_raw["project_name"]),
        class_labels=[str(c) for c in app_raw["class_labels"]],
    )
    return Settings(paths=paths, train=train, app=app)
