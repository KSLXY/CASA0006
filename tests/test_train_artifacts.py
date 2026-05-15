import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from src.train import _validate_master_dataset_size, main


def _write_test_master(path: Path, rows: int = 60) -> None:
    records = []
    for i in range(rows):
        records.append(
            {
                "date": f"2024-01-{(i % 28) + 1:02d}",
                "time": f"{i % 24:02d}:00",
                "accident_severity": (i % 3) + 1,
                "number_of_vehicles": (i % 4) + 1,
                "number_of_casualties": (i % 2) + 1,
                "cloud_cover": float(i % 9),
                "sunshine": float((i * 2) % 12),
                "global_radiation": float(80 + i),
                "max_temp": float(15 + (i % 10)),
                "mean_temp": float(10 + (i % 8)),
                "min_temp": float(4 + (i % 6)),
                "precipitation": float((i % 5) / 10),
                "pressure": float(1000 + (i % 20)),
                "snow_depth": 0,
                "lsoa_of_accident_location": f"E010{i % 8:05d}",
                "road_type": [1, 2, 3, 6][i % 4],
                "speed_limit": [20, 30, 40, 60][i % 4],
                "junction_detail": [0, 13, 16][i % 3],
                "junction_control": [2, 4, -1][i % 3],
                "light_conditions": [1, 4, 6][i % 3],
                "weather_conditions": [1, 2, 8][i % 3],
                "road_surface_conditions": [1, 2, 4][i % 3],
                "urban_or_rural_area": [1, 2][i % 2],
            }
        )
    pd.DataFrame(records).to_parquet(path, index=False)


def _write_test_config(tmp_path: Path, master_path: Path) -> Path:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["master_data"] = str(master_path)
    cfg["train"]["min_training_rows"] = 10
    artifact_dir = tmp_path / "artifacts"
    cfg["paths"]["model_artifact"] = str(artifact_dir / "model.joblib")
    cfg["paths"]["metrics_artifact"] = str(artifact_dir / "metrics.json")
    cfg["paths"]["evaluation_artifact"] = str(artifact_dir / "evaluation.json")
    cfg["paths"]["data_quality_artifact"] = str(artifact_dir / "data_quality_report.json")
    cfg["paths"]["leakage_check_artifact"] = str(artifact_dir / "leakage_check_report.json")
    cfg["paths"]["threshold_artifact"] = str(artifact_dir / "threshold_report.csv")
    cfg["paths"]["calibration_artifact"] = str(artifact_dir / "calibration_report.json")
    cfg["paths"]["search_artifact"] = str(artifact_dir / "hyperparameter_search.json")
    cfg["paths"]["ablation_leakage_artifact"] = str(artifact_dir / "ablation_leakage.csv")
    cfg["paths"]["missingness_by_time_artifact"] = str(artifact_dir / "missingness_by_time.csv")
    cfg["paths"]["report_figures_dir"] = str(tmp_path / "figures")
    tmp_cfg = tmp_path / "test_config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return tmp_cfg


def test_default_config_uses_only_master_data():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    assert paths["master_data"] == "data/processed/processed_master.parquet"
    forbidden_keys = ["sample" + "_data", "sample" + "_data_fallback"]
    for key in forbidden_keys:
        assert key not in paths
    forbidden_path = "data/" + "sample/" + "merged_" + "sample.csv"
    assert forbidden_path not in Path("configs/default.yaml").read_text(encoding="utf-8")


def test_master_dataset_size_guard_rejects_small_data():
    with pytest.raises(RuntimeError, match="Master dataset was not loaded correctly"):
        _validate_master_dataset_size(row_count=36, min_training_rows=100000)


def test_training_missing_master_fails_without_fallback(monkeypatch, tmp_path):
    missing_master = tmp_path / "missing_master.parquet"
    tmp_cfg = _write_test_config(tmp_path, missing_master)

    monkeypatch.setattr("sys.argv", ["python", "--config", str(tmp_cfg)])
    with pytest.raises(FileNotFoundError, match="Master dataset not found"):
        main()


def test_training_generates_master_artifacts(monkeypatch, tmp_path):
    master_path = tmp_path / "processed_master.parquet"
    _write_test_master(master_path)
    tmp_cfg = _write_test_config(tmp_path, master_path)

    monkeypatch.setattr("sys.argv", ["python", "--config", str(tmp_cfg)])
    main()

    model_path = tmp_path / "artifacts" / "model.joblib"
    metrics_path = tmp_path / "artifacts" / "metrics.json"
    assert model_path.exists()
    assert metrics_path.exists()

    payload = joblib.load(model_path)
    assert "pipeline" in payload
    assert payload["model_name"] in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting_balanced",
    }
    assert "numeric_feature_columns" in payload
    assert "categorical_feature_columns" in payload
    assert "road_type" in payload["categorical_feature_columns"]
    unknown_category_row = pd.DataFrame([{col: payload["feature_defaults"].get(col, 0) for col in payload["feature_columns"]}])
    unknown_category_row["road_type"] = "unseen_test_category"
    assert len(payload["pipeline"].predict(unknown_category_row)) == 1

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["dataset_used"].endswith("processed_master.parquet")
    assert metrics["rows_total"] == 60
    assert "selected_model_metrics" in metrics
    assert "f1_macro" in metrics["selected_model_metrics"]
    assert "label_mapping_version" in metrics
    assert "feature_set_mode" in metrics
    assert "overall_metrics_on_test_split" in metrics
    assert "safety_metrics_on_test_split" in metrics
    assert "probability_metrics_on_test_split" in metrics
    assert "fatal_screening_model" in metrics
    assert "safety_threshold" in metrics

    compare_path = tmp_path / "artifacts" / "model_compare.csv"
    compare_df = pd.read_csv(compare_path)
    assert "balanced_accuracy" in compare_df.columns
    assert "fatal_recall" in compare_df.columns
    selected = metrics["selected_model"]
    selected_from_compare = compare_df.sort_values("f1_macro", ascending=False).iloc[0]["model_name"]
    assert selected == selected_from_compare
