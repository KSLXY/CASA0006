from __future__ import annotations

import argparse
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch open datasets for London severity project.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--from", dest="from_date", type=str, default=None)
    parser.add_argument("--to", dest="to_date", type=str, default=None)
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Disable SSL certificate verification when downloading HTTPS sources.",
    )
    parser.add_argument(
        "--skip-dft",
        action="store_true",
        help="Skip DfT collision/vehicle/casualty downloads (use existing local files).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not download files that already exist locally.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> tuple[Path, Path, Path]:
    raw_dir = Path(cfg["paths"]["raw_dir"]).resolve()
    interim_dir = Path(cfg["paths"]["interim_dir"]).resolve()
    processed_dir = Path(cfg["paths"]["processed_dir"]).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, interim_dir, processed_dir


def _read_csv_from_url(url: str, verify_ssl: bool) -> pd.DataFrame:
    response = requests.get(url, timeout=120, verify=verify_ssl)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), low_memory=False)


def fetch_dft_collision(cfg: dict, from_date: str, to_date: str, verify_ssl: bool, skip_existing: bool = False) -> Path:
    output = Path(cfg["paths"]["collision_file"]).resolve()
    if skip_existing and output.exists():
        print(f"Using existing DfT collision file: {output}")
        return output
    url = cfg["sources"]["dft_collision_url"]
    print(f"Downloading DfT collision data from: {url}")
    df = _read_csv_from_url(url, verify_ssl=verify_ssl)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    mask = (df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))
    df = df.loc[mask].copy()
    df.to_csv(output, index=False)
    print(f"Saved DfT collision: {output} (rows={len(df)})")
    return output


def fetch_dft_vehicle(cfg: dict, verify_ssl: bool, skip_existing: bool = False) -> Path:
    output = Path(cfg["paths"]["vehicle_file"]).resolve()
    if skip_existing and output.exists():
        print(f"Using existing DfT vehicle file: {output}")
        return output
    url = cfg["sources"]["dft_vehicle_url"]
    print(f"Downloading DfT vehicle data from: {url}")
    df = _read_csv_from_url(url, verify_ssl=verify_ssl)
    df.to_csv(output, index=False)
    print(f"Saved DfT vehicle: {output} (rows={len(df)})")
    return output


def fetch_dft_casualty(cfg: dict, verify_ssl: bool, skip_existing: bool = False) -> Path:
    output = Path(cfg["paths"]["casualty_file"]).resolve()
    if skip_existing and output.exists():
        print(f"Using existing DfT casualty file: {output}")
        return output
    url = cfg["sources"]["dft_casualty_url"]
    print(f"Downloading DfT casualty data from: {url}")
    df = _read_csv_from_url(url, verify_ssl=verify_ssl)
    df.to_csv(output, index=False)
    print(f"Saved DfT casualty: {output} (rows={len(df)})")
    return output


def fetch_weather(cfg: dict, from_date: str, to_date: str, skip_existing: bool = False) -> Path:
    output = Path(cfg["paths"]["weather_file"]).resolve()
    if skip_existing and output.exists():
        print(f"Using existing weather data: {output}")
        return output
    dataset_name = cfg["sources"]["kaggle_weather_dataset"]
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required. Install with `pip install kagglehub`.") from exc

    weather_dir = kagglehub.dataset_download(dataset_name)
    weather_file = Path(weather_dir) / "london_weather.csv"
    df = pd.read_csv(weather_file)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    mask = (df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))
    df = df.loc[mask].copy()
    df.to_csv(output, index=False)
    print(f"Saved weather data: {output} (rows={len(df)})")
    return output


def fetch_bank_holidays(cfg: dict, verify_ssl: bool, skip_existing: bool = False) -> Path:
    output = Path(cfg["paths"]["bank_holidays_file"]).resolve()
    if skip_existing and output.exists():
        print(f"Using existing bank holidays JSON: {output}")
        return output
    url = cfg["sources"]["gov_bank_holidays_url"]
    print(f"Downloading bank holidays from: {url}")
    response = requests.get(url, timeout=30, verify=verify_ssl)
    response.raise_for_status()
    data = response.json()
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved bank holidays JSON: {output}")
    return output


def write_fetch_metadata(cfg: dict, from_date: str, to_date: str, outputs: dict[str, str]) -> None:
    metadata = {
        "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "from_date": from_date,
        "to_date": to_date,
        "sources": cfg["sources"],
        "outputs": outputs,
    }
    out_file = Path(cfg["paths"]["processed_dir"]).resolve() / "fetch_metadata.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved fetch metadata: {out_file}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    _ensure_dirs(cfg)
    from_date = args.from_date or cfg["project"]["from_date"]
    to_date = args.to_date or cfg["project"]["to_date"]
    verify_ssl = not args.allow_insecure_ssl

    outputs: dict[str, str] = {}
    if args.skip_dft:
        outputs["collision_file"] = str(Path(cfg["paths"]["collision_file"]).resolve())
        outputs["vehicle_file"] = str(Path(cfg["paths"]["vehicle_file"]).resolve())
        outputs["casualty_file"] = str(Path(cfg["paths"]["casualty_file"]).resolve())
        print("Skipping DfT downloads and using local files.")
    else:
        outputs["collision_file"] = str(fetch_dft_collision(cfg, from_date, to_date, verify_ssl=verify_ssl, skip_existing=args.skip_existing))
        outputs["vehicle_file"] = str(fetch_dft_vehicle(cfg, verify_ssl=verify_ssl, skip_existing=args.skip_existing))
        outputs["casualty_file"] = str(fetch_dft_casualty(cfg, verify_ssl=verify_ssl, skip_existing=args.skip_existing))
    outputs["weather_file"] = str(fetch_weather(cfg, from_date, to_date, skip_existing=args.skip_existing))
    outputs["bank_holidays_file"] = str(fetch_bank_holidays(cfg, verify_ssl=verify_ssl, skip_existing=args.skip_existing))
    write_fetch_metadata(cfg, from_date, to_date, outputs)


if __name__ == "__main__":
    main()
