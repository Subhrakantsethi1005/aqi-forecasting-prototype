"""Generate a synthetic but usable AQI dataset for the demo / Quick Start.

Produces multiple stations of autocorrelated hourly readings (slow drift + a
daily cycle) so lag features, time features, walk-forward validation, and the
classifier all have real signal to work with — unlike pure random noise. Writes
to ``data/aqi_dataset.csv`` by default, which is the path every example command
in the README expects.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = {
    "ST1": {"lat": 28.61, "lon": 77.21, "distance_km": 0.5, "base": 55.0},
    "ST2": {"lat": 28.70, "lon": 77.10, "distance_km": 1.2, "base": 45.0},
    "ST3": {"lat": 28.55, "lon": 77.25, "distance_km": 2.0, "base": 35.0},
}


def _station_frame(name: str, meta: dict, hours: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=hours, freq="h")
    hour = idx.hour.to_numpy()
    diurnal = 15.0 * np.sin((hour - 6) / 24.0 * 2 * np.pi)   # daily rise/fall
    drift = np.cumsum(rng.normal(0, 1.5, hours))             # slow random walk
    pm25 = np.clip(meta["base"] + diurnal + drift + rng.normal(0, 5, hours), 5, 180)
    pm10 = np.clip(pm25 * 1.6 + rng.normal(0, 8, hours), 8, 300)
    return pd.DataFrame(
        {
            "Datetime": [d.strftime("%Y-%m-%d %H:%M") for d in idx],
            "station_id": name,
            "distance_km": meta["distance_km"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "PM2.5": pm25.round(2),
            "PM10": pm10.round(2),
            "NO2": np.clip(rng.normal(30, 8, hours), 2, 90).round(2),
            "SO2": np.clip(rng.normal(12, 4, hours), 1, 40).round(2),
            "CO": np.clip(rng.normal(0.7, 0.2, hours), 0.1, 2.5).round(3),
            "O3": np.clip(rng.normal(28, 8, hours), 5, 90).round(2),
            "Temperature": np.clip(22 + 6 * np.sin((hour - 9) / 24.0 * 2 * np.pi) + rng.normal(0, 2, hours), 5, 45).round(2),
            "Humidity": np.clip(rng.normal(60, 12, hours), 15, 98).round(2),
            "WindSpeed": np.clip(rng.normal(3, 1.5, hours), 0, 15).round(2),
        }
    )


def generate_sample_data(output_path: str = "data/aqi_dataset.csv", hours_per_station: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = [_station_frame(name, meta, hours_per_station, rng) for name, meta in STATIONS.items()]
    df = pd.concat(frames, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample data generated at: {output_path} ({len(df)} rows, {len(STATIONS)} stations)")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic AQI demo dataset.")
    parser.add_argument("--output", default="data/aqi_dataset.csv", help="Where to save the CSV.")
    parser.add_argument("--hours", type=int, default=1500, help="Hours of data per station.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_sample_data(args.output, args.hours)
