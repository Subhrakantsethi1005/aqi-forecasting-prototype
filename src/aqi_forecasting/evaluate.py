from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aqi import add_cpcb_aqi
from .config import ModelConfig
from .data import load_dataset
from .features import add_lag_features, add_time_features, clean_numeric_columns, make_sequences
from .model import evaluate_regression, load_artifacts


def evaluate_from_csv(data_path: str | Path, model_dir: str | Path = "models/aqi_lstm") -> dict:
    model, scaler_X, scaler_y, metadata = load_artifacts(model_dir)
    df = load_dataset(data_path)
    target_col = metadata.get("target_col", "AQI_CPCB")
    if target_col not in df.columns:
        df = add_cpcb_aqi(df, target_col=target_col)

    config = ModelConfig(target_col=target_col, seq_len=int(metadata["seq_len"]))
    feature_cols = metadata["feature_cols"]
    base_cols = [c for c in config.feature_cols if c in df.columns]
    work = clean_numeric_columns(df, base_cols + [target_col])
    work = add_time_features(work)
    work = add_lag_features(work, base_cols, group_col=config.station_col)
    work = work.dropna(subset=feature_cols + [target_col])

    split_idx = int(len(work) * (1 - config.test_size))
    X_scaled = scaler_X.transform(work[feature_cols])
    y_scaled = scaler_y.transform(work[[target_col]])
    X_all, y_all = make_sequences(X_scaled, y_scaled, config.seq_len)
    test_start = max(0, split_idx - config.seq_len)
    X_test = X_all[test_start:]
    y_test = y_all[test_start:]

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    return evaluate_regression(y_true, y_pred)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained AQI model on a CSV dataset.")
    parser.add_argument("--data", required=True, help="Path to evaluation CSV.")
    parser.add_argument("--model-dir", default="models/aqi_lstm", help="Directory containing trained artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(evaluate_from_csv(args.data, args.model_dir), indent=2))


if __name__ == "__main__":
    main()
