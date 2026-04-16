import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.train import main


def test_training_generates_artifacts(monkeypatch, tmp_path):
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["sample_data"] = "data/sample/merged_sample.csv"
    cfg["paths"]["sample_data_fallback"] = "data/sample/merged_sample.csv"
    tmp_cfg = tmp_path / "test_config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["python", "--config", str(tmp_cfg)],
    )
    main()

    model_path = Path("artifacts/model.joblib")
    metrics_path = Path("artifacts/metrics.json")
    assert model_path.exists()
    assert metrics_path.exists()

    payload = joblib.load(model_path)
    assert "pipeline" in payload
    assert payload["model_name"] in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "selected_model_metrics" in metrics
    assert "f1_macro" in metrics["selected_model_metrics"]
    assert "label_mapping_version" in metrics
    assert "feature_set_mode" in metrics
    assert "overall_metrics_on_test_split" in metrics
    assert "safety_metrics_on_test_split" in metrics
    assert "probability_metrics_on_test_split" in metrics

    compare_path = Path("artifacts/model_compare.csv")
    compare_df = pd.read_csv(compare_path)
    assert "balanced_accuracy" in compare_df.columns
    selected = metrics["selected_model"]
    selected_from_compare = compare_df.sort_values("f1_macro", ascending=False).iloc[0]["model_name"]
    assert selected == selected_from_compare
