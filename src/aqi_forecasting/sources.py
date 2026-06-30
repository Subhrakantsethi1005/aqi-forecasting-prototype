"""Dataset fetchers.

Two sources are supported:

* ``gdrive`` — download the original notebook dataset from Google Drive (keeps the
  previous behaviour of ``scripts/download_dataset.py``).
* ``openaq`` — pull **real** measurements from the OpenAQ v3 API and reshape them
  into this project's CSV schema. This is the recommended way to get genuine
  high-AQI data (e.g. Delhi winter) so the model is no longer blind to hazardous
  air, which the synthetic demo dataset never contains.

Networked OpenAQ calls require a free API key (https://docs.openaq.org/). Pass it
with ``--api-key`` or the ``OPENAQ_API_KEY`` environment variable.

Unit caveat: CPCB AQI expects CO in mg/m3 and gaseous pollutants in ug/m3.
OpenAQ reports each sensor's native unit, which varies by provider. Validate and
convert units before training (see MODEL_CARD.md). The raw fetched units are
left untouched here on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OPENAQ_BASE = "https://api.openaq.org/v3"

# OpenAQ parameter name -> this project's column name.
PARAM_MAP = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
    "temperature": "Temperature",
    "relativehumidity": "Humidity",
    "humidity": "Humidity",
    "windspeed": "WindSpeed",
}

POLLUTANT_PARAMS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# --- Unit handling -----------------------------------------------------------
# CPCB AQI breakpoints assume particulates and gases in ug/m3 and CO in mg/m3.
# OpenAQ reports each sensor's native unit (ug/m3, mg/m3, ppm, ppb), so values
# must be converted before the AQI math, or the sub-indices will be nonsense.
MOLAR_VOLUME = 24.45  # litres/mol at 25 C, 1 atm
MOLECULAR_WEIGHT = {"no2": 46.0055, "so2": 64.066, "o3": 48.00, "co": 28.01}
CPCB_UNIT = {"pm25": "ug/m3", "pm10": "ug/m3", "no2": "ug/m3", "so2": "ug/m3", "o3": "ug/m3", "co": "mg/m3"}


def normalize_unit(unit: str | None) -> str | None:
    """Normalise unit strings like 'µg/m³' / 'ppm' to a canonical lowercase form."""
    if unit is None:
        return None
    cleaned = str(unit).strip().lower().replace("µ", "u").replace("μ", "u").replace("³", "3")
    return cleaned.replace(" ", "") or None


def to_cpcb_units(value, parameter: str, unit: str | None):
    """Convert a single reading to the unit CPCB AQI expects for that pollutant.

    Unknown pollutants or unrecognised units are passed through unchanged (and
    surfaced by ``validate_units``), so this never silently zeroes data.
    """
    if value is None or pd.isna(value):
        return value
    param = str(parameter).lower()
    target = CPCB_UNIT.get(param)
    source = normalize_unit(unit)
    if target is None or source is None or source == target:
        return value

    if param in ("pm25", "pm10"):
        return value * 1000.0 if source == "mg/m3" else value

    weight = MOLECULAR_WEIGHT.get(param)
    if weight is None:
        return value
    if source == "ppb":
        ugm3 = value * weight / MOLAR_VOLUME
    elif source == "ppm":
        ugm3 = value * 1000.0 * weight / MOLAR_VOLUME
    elif source == "mg/m3":
        ugm3 = value * 1000.0
    elif source == "ug/m3":
        ugm3 = value
    else:
        return value  # unrecognised unit: leave as-is, flagged by validate_units
    return ugm3 / 1000.0 if target == "mg/m3" else ugm3


def validate_units(records: list[dict]) -> list[str]:
    """Return human-readable warnings about units that could not be converted."""
    seen: dict[str, set] = {}
    for rec in records:
        param = str(rec.get("parameter", "")).lower()
        if param in CPCB_UNIT:
            seen.setdefault(param, set()).add(normalize_unit(rec.get("unit")))
    warnings = []
    known = {"ug/m3", "mg/m3", "ppm", "ppb", None}
    for param, units in sorted(seen.items()):
        unknown = units - known
        if unknown:
            warnings.append(f"{param}: unrecognised unit(s) {sorted(str(u) for u in unknown)} left unconverted.")
        if None in units:
            warnings.append(f"{param}: some readings had no unit; assumed already in {CPCB_UNIT[param]}.")
    return warnings


# ---------------------------------------------------------------------------
# Pure transform (unit-tested; no network)
# ---------------------------------------------------------------------------
def records_to_wide(records: list[dict]) -> pd.DataFrame:
    """Pivot flat OpenAQ measurement records into the wide project schema.

    Each record is ``{"datetime", "station_id", "lat", "lon", "parameter",
    "value"}``. The output has one row per (station, hour) with pollutant/weather
    columns, sorted by station then time. Columns that are entirely missing are
    dropped so the training pipeline simply ignores absent variables rather than
    discarding every row.
    """
    columns = ["Datetime", "station_id", "lat", "lon", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"]
    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    df["Datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["column"] = df["parameter"].str.lower().map(PARAM_MAP)
    df = df.dropna(subset=["column"])
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "unit" in df.columns:
        df["value"] = [to_cpcb_units(v, p, u) for v, p, u in zip(df["value"], df["parameter"], df["unit"])]
    wide = df.pivot_table(
        index=["Datetime", "station_id", "lat", "lon"],
        columns="column",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    wide = wide.sort_values(["station_id", "Datetime"]).reset_index(drop=True)
    wide = wide.dropna(axis=1, how="all")
    ordered = [c for c in columns if c in wide.columns]
    return wide[ordered]


# ---------------------------------------------------------------------------
# OpenAQ v3 networked fetch
# ---------------------------------------------------------------------------
def _get(path: str, api_key: str, params: dict | None = None, timeout: int = 60) -> dict:
    url = f"{OPENAQ_BASE}/{path.lstrip('/')}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    request = urllib.request.Request(url, headers={"X-API-Key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"OpenAQ request failed ({exc.code}) for {url}: {detail}") from exc


def discover_locations(api_key: str, iso: str | None = None, bbox: str | None = None, limit: int = 50) -> list[dict]:
    """Return simplified locations: ``{id, name, lat, lon, sensors:[{id, parameter}]}``."""
    params: dict = {"limit": limit, "page": 1}
    if iso:
        params["iso"] = iso
    if bbox:
        params["bbox"] = bbox
    payload = _get("locations", api_key, params)
    locations = []
    for loc in payload.get("results", []):
        coords = loc.get("coordinates") or {}
        sensors = [
            {
                "id": s["id"],
                "parameter": (s.get("parameter") or {}).get("name", "").lower(),
                "unit": (s.get("parameter") or {}).get("units"),
            }
            for s in loc.get("sensors", [])
        ]
        locations.append(
            {
                "id": loc["id"],
                "name": str(loc.get("name") or loc["id"]),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "sensors": sensors,
            }
        )
    return locations


def fetch_sensor_hours(api_key: str, sensor_id: int, date_from: str | None, date_to: str | None, max_pages: int = 50, sleep: float = 0.2) -> list[dict]:
    """Fetch hourly-aggregated values for one sensor, following pagination."""
    rows = []
    for page in range(1, max_pages + 1):
        params: dict = {"limit": 1000, "page": page}
        if date_from:
            params["datetime_from"] = date_from
        if date_to:
            params["datetime_to"] = date_to
        payload = _get(f"sensors/{sensor_id}/hours", api_key, params)
        results = payload.get("results", [])
        if not results:
            break
        rows.extend(results)
        if len(results) < params["limit"]:
            break
        time.sleep(sleep)
    return rows


def fetch_openaq_records(
    api_key: str,
    iso: str | None = None,
    bbox: str | None = None,
    location_ids: list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    parameters: list[str] | None = None,
    limit_locations: int = 25,
) -> list[dict]:
    parameters = [p.lower() for p in (parameters or POLLUTANT_PARAMS)]
    locations = discover_locations(api_key, iso=iso, bbox=bbox, limit=limit_locations)
    if location_ids:
        wanted = set(location_ids)
        locations = [loc for loc in locations if loc["id"] in wanted]

    records: list[dict] = []
    for loc in locations:
        for sensor in loc["sensors"]:
            if sensor["parameter"] not in parameters:
                continue
            for row in fetch_sensor_hours(api_key, sensor["id"], date_from, date_to):
                period = (row.get("period") or {}).get("datetimeFrom") or {}
                records.append(
                    {
                        "datetime": period.get("utc"),
                        "station_id": loc["name"],
                        "lat": loc["lat"],
                        "lon": loc["lon"],
                        "parameter": sensor["parameter"],
                        "unit": sensor.get("unit"),
                        "value": row.get("value"),
                    }
                )
    return records


def fetch_openaq(api_key: str, output: str | Path, **kwargs) -> pd.DataFrame:
    records = fetch_openaq_records(api_key, **kwargs)
    frame = records_to_wide(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


# ---------------------------------------------------------------------------
# Google Drive (unchanged behaviour)
# ---------------------------------------------------------------------------
def download_gdrive(sources_path: str | Path, output: str | Path) -> Path:
    sources = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    file_id = sources["primary"]["google_drive_file_id"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown first: pip install gdown") from exc
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output), quiet=False)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the AQI dataset from Google Drive or OpenAQ v3.")
    parser.add_argument("--source", default="gdrive", choices=["gdrive", "openaq"], help="Where to fetch data from.")
    parser.add_argument("--output", default="data/aqi_dataset.csv", help="Where to save the CSV.")
    parser.add_argument("--sources", default="data/dataset_sources.json", help="(gdrive) dataset source metadata.")
    parser.add_argument("--api-key", default=os.getenv("OPENAQ_API_KEY"), help="(openaq) API key, or set OPENAQ_API_KEY.")
    parser.add_argument("--iso", default=None, help="(openaq) ISO country code, e.g. IN for India.")
    parser.add_argument("--bbox", default=None, help="(openaq) minLon,minLat,maxLon,maxLat bounding box.")
    parser.add_argument("--location-ids", default=None, help="(openaq) comma-separated OpenAQ location ids.")
    parser.add_argument("--date-from", default=None, help="(openaq) ISO start datetime, e.g. 2024-11-01.")
    parser.add_argument("--date-to", default=None, help="(openaq) ISO end datetime, e.g. 2024-12-01.")
    parser.add_argument("--limit-locations", type=int, default=25, help="(openaq) max locations to pull.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.source == "gdrive":
        output = download_gdrive(args.sources, args.output)
        print(f"Downloaded dataset to {output}")
        return

    if not args.api_key:
        raise SystemExit("OpenAQ needs an API key. Pass --api-key or set OPENAQ_API_KEY.")
    location_ids = [int(x) for x in args.location_ids.split(",")] if args.location_ids else None
    records = fetch_openaq_records(
        args.api_key,
        iso=args.iso,
        bbox=args.bbox,
        location_ids=location_ids,
        date_from=args.date_from,
        date_to=args.date_to,
        limit_locations=args.limit_locations,
    )
    for warning in validate_units(records):
        print(f"[unit-check] {warning}")
    frame = records_to_wide(records)  # converts each reading to CPCB-expected units
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    high = int((pd.to_numeric(frame.get("PM2.5"), errors="coerce") > 90).sum()) if "PM2.5" in frame else 0
    print(f"Wrote {len(frame)} rows to {output} ({frame.get('station_id', pd.Series(dtype=str)).nunique()} stations).")
    print(f"Rows with PM2.5 > 90 ug/m3 (Poor+ territory): {high}")
    if frame.empty:
        print("No records returned. Check the API key, country/bbox, and date range.")


if __name__ == "__main__":
    main()
