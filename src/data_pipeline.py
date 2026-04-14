from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series
    removed_invalid_target: int


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def _ensure_snow_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "snow" not in out.columns:
        if "snow_depth" in out.columns:
            out["snow"] = (pd.to_numeric(out["snow_depth"], errors="coerce").fillna(0) > 0).astype(int)
        else:
            out["snow"] = 0
    return out


def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "accident_severity",
) -> PreparedDataset:
    work = _ensure_snow_feature(df)

    missing_features = [c for c in feature_columns if c not in work.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if target_column not in work.columns:
        raise ValueError(f"Missing target column: {target_column}")

    valid = work.copy()
    valid[target_column] = pd.to_numeric(valid[target_column], errors="coerce")
    valid = valid[valid[target_column].isin([1, 2, 3])]
    removed_invalid_target = int(len(work) - len(valid))

    X = valid[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    y = valid[target_column].astype(int) - 1
    y.name = "severity_class"

    return PreparedDataset(X=X, y=y, removed_invalid_target=removed_invalid_target)

