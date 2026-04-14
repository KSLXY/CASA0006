from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    sample_data: Path
    model_artifact: Path
    metrics_artifact: Path
    evaluation_artifact: Path


@dataclass
class TrainConfig:
    random_seed: int
    test_size: float
    target_column: str
    feature_columns: list[str]
    model_selection_metric: str


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
        sample_data=_expand(paths_raw["sample_data"]),
        model_artifact=_expand(paths_raw["model_artifact"]),
        metrics_artifact=_expand(paths_raw["metrics_artifact"]),
        evaluation_artifact=_expand(paths_raw["evaluation_artifact"]),
    )
    train = TrainConfig(
        random_seed=int(train_raw["random_seed"]),
        test_size=float(train_raw["test_size"]),
        target_column=str(train_raw["target_column"]),
        feature_columns=[str(c) for c in train_raw["feature_columns"]],
        model_selection_metric=str(train_raw["model_selection_metric"]),
    )
    app = AppConfig(
        project_name=str(app_raw["project_name"]),
        class_labels=[str(c) for c in app_raw["class_labels"]],
    )
    return Settings(paths=paths, train=train, app=app)

