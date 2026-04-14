from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create placeholder file for London air quality enrichment.")
    parser.add_argument("--output", type=str, default="data/raw/air_quality_placeholder.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "placeholder",
        "todo": [
            "Fetch London Datastore/TfL air quality feeds (NO2/PM).",
            "Aggregate to daily level and map to spatial_key.",
            "Join into processed master table."
        ],
        "fields_expected": ["date", "spatial_key", "no2", "pm25"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated air-quality placeholder: {output}")


if __name__ == "__main__":
    main()

