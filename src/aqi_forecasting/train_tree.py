from __future__ import annotations

import argparse
import json

from .tree_model import train_tree_from_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tabular AQI model with lag features.")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--output-dir", default="models/aqi_tree", help="Directory for model artifacts.")
    parser.add_argument("--model", default="sklearn", choices=["sklearn", "xgboost", "lightgbm"], help="Tree model backend.")
    parser.add_argument("--target-col", default="AQI_CPCB", help="Target column to forecast.")
    parser.add_argument("--horizon-hours", type=int, default=1, help="Future AQI horizon in hours.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_tree_from_csv(args.data, args.output_dir, args.model, args.target_col, args.horizon_hours)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
