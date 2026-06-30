from __future__ import annotations

import argparse
import json

from .aqi import add_cpcb_aqi
from .baselines import evaluate_baselines
from .data import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate persistence and seasonal-naive AQI baselines.")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--target-col", default="AQI_CPCB", help="Target column.")
    parser.add_argument("--station-col", default="station_id", help="Station column for station-aware baselines.")
    parser.add_argument("--season-length", type=int, default=24, help="Season length, e.g. 24 for previous day on hourly data.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = load_dataset(args.data)
    if args.target_col not in df.columns:
        df = add_cpcb_aqi(df, target_col=args.target_col)
    result = evaluate_baselines(df, args.target_col, args.station_col, args.season_length)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
