from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aqi import add_cpcb_aqi
from .baselines import evaluate_baselines
from .categories import add_aqi_category
from .config import ModelConfig
from .data import load_dataset
from .features import prepare_training_data
from .metrics import metric_slices
from .model import build_lstm, evaluate_regression, save_artifacts, train_lstm


def train_from_csv(
    data_path: str | Path,
    output_dir: str | Path = "models/aqi_lstm",
    target_col: str = "AQI_CPCB",
    seq_len: int = 24,
    epochs: int = 60,
    batch_size: int = 64,
) -> dict:
    df = load_dataset(data_path)
    if target_col not in df.columns:
        df = add_cpcb_aqi(df, target_col=target_col)
    if "AQI_Category" not in df.columns:
        df = add_aqi_category(df, target_col, "AQI_Category")

    config = ModelConfig(target_col=target_col, seq_len=seq_len)
    X_train, y_train, X_test, y_test, train_df, test_df, scaler_X, scaler_y, metadata = prepare_training_data(df, config)

    model = build_lstm(seq_len=metadata["seq_len"], n_features=len(metadata["feature_cols"]))
    train_lstm(model, X_train, y_train, epochs=epochs, batch_size=batch_size)

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    metrics = evaluate_regression(y_true, y_pred)
    metrics["train_rows"] = len(train_df)
    metrics["test_rows"] = len(test_df)
    baselines = evaluate_baselines(df, target_col=target_col, station_col=config.station_col)

    eval_frame = test_df.copy().iloc[-len(y_pred) :].copy()
    eval_frame["actual"] = y_true.reshape(-1)
    eval_frame["predicted"] = y_pred.reshape(-1)
    sliced_metrics = metric_slices(eval_frame, station_col=config.station_col)

    save_artifacts(
        model,
        scaler_X,
        scaler_y,
        metadata,
        output_dir,
        {"lstm": metrics, "baselines": baselines, "slices": sliced_metrics},
    )
    return {"metrics": metrics, "baselines": baselines, "sliced_metrics": sliced_metrics, "metadata": metadata, "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AQI LSTM forecasting model.")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--output-dir", default="models/aqi_lstm", help="Directory for model artifacts.")
    parser.add_argument("--target-col", default="AQI_CPCB", help="Target column to forecast.")
    parser.add_argument("--seq-len", type=int, default=24, help="Number of historical time steps per prediction.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_from_csv(
        data_path=args.data,
        output_dir=args.output_dir,
        target_col=args.target_col,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
