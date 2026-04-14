from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import add_spatial_key, enrich_interaction_features, enrich_temporal_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed master table from fetched datasets.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_bank_holidays(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    england = data.get("england-and-wales", {})
    events = england.get("events", [])
    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame(columns=["date", "is_bank_holiday", "bank_holiday_title"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_bank_holiday"] = 1
    df["bank_holiday_title"] = df["title"].fillna("unknown")
    return df[["date", "is_bank_holiday", "bank_holiday_title"]]


def build_master(collision: pd.DataFrame, weather: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    collision["date"] = pd.to_datetime(collision["date"], errors="coerce")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    merged = pd.merge(collision, weather, on="date", how="left")
    merged = pd.merge(merged, holidays, on="date", how="left")
    merged["is_bank_holiday"] = merged["is_bank_holiday"].fillna(0).astype(int)
    merged["bank_holiday_title"] = merged["bank_holiday_title"].fillna("none")

    merged = add_spatial_key(merged)
    merged = enrich_temporal_features(merged)
    merged = enrich_interaction_features(merged)

    for col in ["road_class", "maxspeed", "junction_density", "no2", "pm25", "imd_decile"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged


def build_metadata(df: pd.DataFrame, cfg: dict) -> dict:
    missing = df.isna().mean().sort_values(ascending=False).to_dict()
    return {
        "built_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "city": cfg["project"]["city"],
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_rate_by_column": missing,
        "target_valid_ratio": float(pd.to_numeric(df.get("accident_severity"), errors="coerce").isin([1, 2, 3]).mean()),
        "join_keys": ["date", "spatial_key"],
        "source_notes": {
            "road_class/maxspeed/junction_density": "Placeholder columns for OSM/Geofabrik enrichment.",
            "no2/pm25": "Placeholder columns for London Datastore or TfL AirQuality enrichment.",
            "imd_decile": "Placeholder column for ONS/GOV deprivation enrichment.",
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = cfg["paths"]

    collision_path = Path(paths["collision_file"]).resolve()
    weather_path = Path(paths["weather_file"]).resolve()
    holiday_path = Path(paths["bank_holidays_file"]).resolve()
    out_path = Path(paths["processed_master"]).resolve()
    metadata_path = Path(paths["metadata_file"]).resolve()

    if not collision_path.exists():
        raise FileNotFoundError(f"Collision data not found: {collision_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather data not found: {weather_path}")
    if not holiday_path.exists():
        raise FileNotFoundError(f"Bank holidays data not found: {holiday_path}")

    collision = pd.read_csv(collision_path, low_memory=False)
    weather = pd.read_csv(weather_path, low_memory=False)
    holidays = load_bank_holidays(holiday_path)

    master = build_master(collision, weather, holidays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    master.to_parquet(out_path, index=False)
    metadata = build_metadata(master, cfg)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved processed master: {out_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Rows: {len(master)}")


if __name__ == "__main__":
    main()
