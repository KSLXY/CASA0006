import json
from pathlib import Path

import joblib

from src.train import main


def test_training_generates_artifacts(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["python", "--config", "configs/default.yaml"],
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

