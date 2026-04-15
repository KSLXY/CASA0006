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


def aggregate_vehicle(vehicle: pd.DataFrame) -> pd.DataFrame:
    if vehicle.empty:
        return pd.DataFrame(columns=["accident_index", "vehicle_record_count"])
    grouped = vehicle.groupby("accident_index", dropna=False).size().rename("vehicle_record_count").reset_index()
    if "number_of_vehicles" in vehicle.columns:
        num = vehicle.groupby("accident_index", dropna=False)["number_of_vehicles"].max().reset_index()
        grouped = grouped.merge(num, on="accident_index", how="left")
    return grouped


def aggregate_casualty(casualty: pd.DataFrame) -> pd.DataFrame:
    if casualty.empty:
        return pd.DataFrame(columns=["accident_index", "casualty_record_count"])
    grouped = casualty.groupby("accident_index", dropna=False).size().rename("casualty_record_count").reset_index()
    return grouped


def build_master(collision: pd.DataFrame, vehicle: pd.DataFrame, casualty: pd.DataFrame, weather: pd.DataFrame, holidays: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    collision = collision.copy()
    collision["date"] = pd.to_datetime(collision["date"], errors="coerce")
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    vehicle_agg = aggregate_vehicle(vehicle)
    casualty_agg = aggregate_casualty(casualty)

    merged = collision.merge(vehicle_agg, on="accident_index", how="left")
    merged = merged.merge(casualty_agg, on="accident_index", how="left")

    vehicle_match_rate = float(merged["vehicle_record_count"].notna().mean()) if "vehicle_record_count" in merged.columns else 0.0
    casualty_match_rate = float(merged["casualty_record_count"].notna().mean()) if "casualty_record_count" in merged.columns else 0.0

    merged = merged.merge(weather, on="date", how="left", suffixes=("", "_weather"))
    weather_match_rate = float(merged["mean_temp"].notna().mean()) if "mean_temp" in merged.columns else 0.0

    merged = merged.merge(holidays, on="date", how="left")
    merged["is_bank_holiday"] = merged["is_bank_holiday"].fillna(0).astype(int)
    merged["bank_holiday_title"] = merged["bank_holiday_title"].fillna("none")

    if "number_of_vehicles" in merged.columns:
        merged["number_of_vehicles"] = pd.to_numeric(merged["number_of_vehicles"], errors="coerce")
        merged["number_of_vehicles"] = merged["number_of_vehicles"].fillna(merged.get("vehicle_record_count"))
    else:
        merged["number_of_vehicles"] = merged.get("vehicle_record_count", 0)

    if "number_of_casualties" in merged.columns:
        merged["number_of_casualties"] = pd.to_numeric(merged["number_of_casualties"], errors="coerce")
        merged["number_of_casualties"] = merged["number_of_casualties"].fillna(merged.get("casualty_record_count"))
    else:
        merged["number_of_casualties"] = merged.get("casualty_record_count", 0)

    merged["number_of_vehicles"] = pd.to_numeric(merged["number_of_vehicles"], errors="coerce").fillna(0)
    merged["number_of_casualties"] = pd.to_numeric(merged["number_of_casualties"], errors="coerce").fillna(0)

    merged = add_spatial_key(merged)
    merged = enrich_temporal_features(merged)
    merged = enrich_interaction_features(merged)

    for col in ["road_class", "maxspeed", "junction_density", "no2", "pm25", "imd_decile"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    join_stats = {
        "vehicle_match_rate": vehicle_match_rate,
        "casualty_match_rate": casualty_match_rate,
        "weather_match_rate": weather_match_rate,
    }
    return merged, join_stats


def build_metadata(df: pd.DataFrame, cfg: dict, join_stats: dict) -> dict:
    target_col = "accident_severity"
    target = pd.to_numeric(df.get(target_col), errors="coerce")
    missing = df.isna().mean().sort_values(ascending=False).to_dict()
    return {
        "built_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "city": cfg["project"]["city"],
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_rate_by_column": missing,
        "target_valid_ratio": float(target.isin([1, 2, 3]).mean()) if len(target) else 0.0,
        "target_invalid_count": int((~target.isin([1, 2, 3])).sum()) if len(target) else 0,
        "join_keys": ["accident_index", "date", "spatial_key"],
        "join_match_rates": join_stats,
        "source_notes": {
            "collision_vehicle_casualty": "Core STATS19 tables merged by accident_index.",
            "weather": "Merged by date with London weather daily records.",
            "road_class/maxspeed/junction_density": "Placeholder columns for OSM/Geofabrik enrichment.",
            "no2/pm25": "Placeholder columns for London Datastore or TfL AirQuality enrichment.",
            "imd_decile": "Placeholder column for IMD enrichment.",
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = cfg["paths"]

    collision_path = Path(paths["collision_file"]).resolve()
    vehicle_path = Path(paths["vehicle_file"]).resolve()
    casualty_path = Path(paths["casualty_file"]).resolve()
    weather_path = Path(paths["weather_file"]).resolve()
    holiday_path = Path(paths["bank_holidays_file"]).resolve()
    out_path = Path(paths["processed_master"]).resolve()
    metadata_path = Path(paths["metadata_file"]).resolve()

    required = {
        "collision": collision_path,
        "vehicle": vehicle_path,
        "casualty": casualty_path,
        "weather": weather_path,
        "bank_holidays": holiday_path,
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        msg = ", ".join([f"{k}={v}" for k, v in missing.items()])
        raise FileNotFoundError(f"Missing required inputs: {msg}")

    collision = pd.read_csv(collision_path, low_memory=False)
    vehicle = pd.read_csv(vehicle_path, low_memory=False)
    casualty = pd.read_csv(casualty_path, low_memory=False)
    weather = pd.read_csv(weather_path, low_memory=False)
    holidays = load_bank_holidays(holiday_path)

    master, join_stats = build_master(collision, vehicle, casualty, weather, holidays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    master.to_parquet(out_path, index=False)
    metadata = build_metadata(master, cfg, join_stats=join_stats)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved processed master: {out_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Rows: {len(master)}")


if __name__ == "__main__":
    main()
