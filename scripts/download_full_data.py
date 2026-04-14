from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROAD_COLLISION_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and merge full weather + collision data.")
    parser.add_argument("--output", type=str, default="data/raw/merged_full.csv")
    parser.add_argument("--start-year", type=int, default=2010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required. Install with `pip install kagglehub`.") from exc

    weather_dir = kagglehub.dataset_download("emmanuelfwerr/london-weather-data")
    weather_file = Path(weather_dir) / "london_weather.csv"
    weather = pd.read_csv(weather_file)
    weather["date"] = pd.to_datetime(weather["date"], format="%Y%m%d", errors="coerce")
    weather = weather[weather["date"].dt.year >= args.start_year]

    collision = pd.read_csv(ROAD_COLLISION_URL, low_memory=False)
    collision["date"] = pd.to_datetime(collision["date"], errors="coerce")
    collision = collision[collision["date"].dt.year >= args.start_year]

    merged = pd.merge(collision, weather, on="date", how="inner")
    merged.to_csv(output_path, index=False)

    print(f"Merged full dataset saved: {output_path}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()

