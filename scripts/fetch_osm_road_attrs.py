from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create placeholder file for OSM road attributes enrichment.")
    parser.add_argument("--output", type=str, default="data/raw/osm_road_attrs_placeholder.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "placeholder",
        "todo": [
            "Download UK/London extract from Geofabrik.",
            "Derive road_class and maxspeed by spatial join.",
            "Compute junction_density on grid/LSOA."
        ],
        "fields_expected": ["spatial_key", "road_class", "maxspeed", "junction_density"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated OSM placeholder: {output}")


if __name__ == "__main__":
    main()

