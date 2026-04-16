from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series
    dates: pd.Series
    spatial_keys: pd.Series
    removed_invalid_target: int
    missing_rate_by_feature: dict[str, float]


def load_dataset(data_path: str | Path, fallback_path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(data_path).expanduser().resolve()
    chosen = data_path
    if not chosen.exists():
        if fallback_path is None:
            raise FileNotFoundError(f"Dataset not found: {chosen}")
        fallback = Path(fallback_path).expanduser().resolve()
        if not fallback.exists():
            raise FileNotFoundError(f"Dataset not found: {chosen}; fallback not found: {fallback}")
        chosen = fallback

    suffix = chosen.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(chosen)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(chosen)
    raise ValueError(f"Unsupported dataset format: {chosen}")


def _ensure_snow_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "snow" not in out.columns:
        if "snow_depth" in out.columns:
            out["snow"] = (pd.to_numeric(out["snow_depth"], errors="coerce").fillna(0) > 0).astype(int)
        else:
            out["snow"] = 0
    return out


def enrich_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["day_of_week"] = out["date"].dt.dayofweek
        out["month"] = out["date"].dt.month
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
        out["season"] = out["month"].map(
            {
                12: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 1,
                5: 1,
                6: 2,
                7: 2,
                8: 2,
                9: 3,
                10: 3,
                11: 3,
            }
        )
    else:
        out["day_of_week"] = 0
        out["month"] = 1
        out["is_weekend"] = 0
        out["season"] = 0

    if "time" in out.columns:
        dt = pd.to_datetime(out["time"], format="%H:%M", errors="coerce")
        out["hour"] = dt.dt.hour.fillna(12).astype(int)
    else:
        out["hour"] = 12
    out["hour_peak"] = out["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    return out


def enrich_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    precipitation = pd.to_numeric(out.get("precipitation", 0), errors="coerce").fillna(0)
    hour_peak = pd.to_numeric(out.get("hour_peak", 0), errors="coerce").fillna(0)
    cloud_cover = pd.to_numeric(out.get("cloud_cover", 0), errors="coerce").fillna(0)
    sunshine = pd.to_numeric(out.get("sunshine", 0), errors="coerce").fillna(0)
    out["precipitation_peak_interaction"] = precipitation * hour_peak
    out["low_visibility_proxy"] = np.where((cloud_cover >= 7) & (sunshine <= 1), 1, 0)
    return out


def add_spatial_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lsoa_of_accident_location" in out.columns:
        out["spatial_key"] = out["lsoa_of_accident_location"].astype(str)
    elif {"location_easting_osgr", "location_northing_osgr"}.issubset(out.columns):
        e = pd.to_numeric(out["location_easting_osgr"], errors="coerce").fillna(-1).astype(int)
        n = pd.to_numeric(out["location_northing_osgr"], errors="coerce").fillna(-1).astype(int)
        out["spatial_key"] = (e // 1000).astype(str) + "_" + (n // 1000).astype(str)
    else:
        out["spatial_key"] = "unknown"
    return out


def prepare_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "accident_severity",
    severity_code_map: dict[int, int] | None = None,
) -> PreparedDataset:
    work = _ensure_snow_feature(df)
    work = enrich_temporal_features(work)
    work = enrich_interaction_features(work)
    work = add_spatial_key(work)

    missing_features = [c for c in feature_columns if c not in work.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if target_column not in work.columns:
        raise ValueError(f"Missing target column: {target_column}")

    valid = work.copy()
    valid[target_column] = pd.to_numeric(valid[target_column], errors="coerce")
    valid = valid[valid[target_column].isin([1, 2, 3])]
    removed_invalid_target = int(len(work) - len(valid))

    dates = pd.to_datetime(valid.get("date", pd.NaT), errors="coerce")
    spatial_keys = valid.get("spatial_key", pd.Series(["unknown"] * len(valid), index=valid.index)).astype(str)
    X = valid[feature_columns].apply(pd.to_numeric, errors="coerce")
    missing_rate_by_feature = X.isna().mean().fillna(0).to_dict()
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    mapping = severity_code_map or {1: 0, 2: 1, 3: 2}
    y = valid[target_column].astype(int).map(mapping)
    if y.isna().any():
        missing_codes = sorted(valid.loc[y.isna(), target_column].astype(int).unique().tolist())
        raise ValueError(f"Target codes not covered by severity_code_map: {missing_codes}")
    y = y.astype(int)
    y.name = "severity_class"

    return PreparedDataset(
        X=X,
        y=y,
        dates=dates,
        spatial_keys=spatial_keys,
        removed_invalid_target=removed_invalid_target,
        missing_rate_by_feature=missing_rate_by_feature,
    )


def build_missingness_by_time(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = enrich_temporal_features(df.copy())
    if "date" not in work.columns:
        return pd.DataFrame(columns=["year_month", "feature", "missing_rate", "rows"])
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work[work["date"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["year_month", "feature", "missing_rate", "rows"])
    work["year_month"] = work["date"].dt.to_period("M").astype(str)
    rows: list[dict[str, float | int | str]] = []
    for ym, part in work.groupby("year_month"):
        row_count = int(len(part))
        for feature in features:
            series = pd.to_numeric(part.get(feature), errors="coerce")
            rows.append(
                {
                    "year_month": ym,
                    "feature": feature,
                    "missing_rate": float(series.isna().mean()),
                    "rows": row_count,
                }
            )
    return pd.DataFrame(rows)
