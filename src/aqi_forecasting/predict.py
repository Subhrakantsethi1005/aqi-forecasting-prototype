from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd

from .categories import aqi_category
from .data import normalize_columns
from .features import add_lag_features, clean_numeric_columns
from .model import load_artifacts
from .tree_model import _add_forecast_time_features, _add_station_dummies


class AQIPredictor:
    def __init__(self, model_dir: str | Path):
        self.model, self.scaler_X, self.scaler_y, self.metadata = load_artifacts(model_dir)
        self.feature_cols = self.metadata["feature_cols"]
        self.seq_len = int(self.metadata["seq_len"])

    def _prepare_sequence(self, rows: pd.DataFrame) -> np.ndarray:
        rows = normalize_columns(rows.copy())
        missing = [c for c in self.feature_cols if c not in rows.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        rows = rows[self.feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
        if rows.isna().any().any():
            raise ValueError("Input contains non-numeric or missing values that could not be filled.")
        if len(rows) < self.seq_len:
            pad = pd.concat([rows.iloc[[0]]] * (self.seq_len - len(rows)), ignore_index=True)
            rows = pd.concat([pad, rows], ignore_index=True)
        rows = rows.tail(self.seq_len)
        scaled = self.scaler_X.transform(rows[self.feature_cols])
        return scaled.reshape(1, self.seq_len, len(self.feature_cols))

    def predict(self, rows: pd.DataFrame) -> float:
        sequence = self._prepare_sequence(rows)
        pred_scaled = self.model.predict(sequence, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)
        return float(pred.reshape(-1)[0])

    def predict_with_category(self, rows: pd.DataFrame) -> dict:
        value = self.predict(rows)
        return {"predicted_aqi": value, "category": aqi_category(value)}


class TreeAQIPredictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "tree_model.joblib")
        self.metadata = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
        self.feature_cols = self.metadata["feature_cols"]
        self.target_col = self.metadata.get("target_col", "AQI_CPCB")
        self.station_col = self.metadata.get("station_col") or "station_id"
        self.forecast_horizon_hours = int(self.metadata.get("forecast_horizon_hours", 0))
        self.base_feature_cols = self.metadata.get(
            "base_feature_cols",
            ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"],
        )
        self.location_cols = self.metadata.get("location_cols", [])
        self.station_dummy_cols = self.metadata.get("station_dummy_cols", [])

    def _prepare_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = normalize_columns(rows.copy())
        if "Datetime" in rows.columns:
            rows["Datetime"] = pd.to_datetime(rows["Datetime"], errors="coerce", dayfirst=True)
            rows = rows.dropna(subset=["Datetime"]).sort_values("Datetime").set_index("Datetime")

        base_cols = [c for c in self.base_feature_cols if c in rows.columns]
        numeric_cols = sorted(set(base_cols + [c for c in self.location_cols if c in rows.columns]))
        rows = clean_numeric_columns(rows, numeric_cols)
        rows = _add_forecast_time_features(rows, self.forecast_horizon_hours)
        rows = add_lag_features(rows, base_cols, group_col=self.station_col)
        rows, _ = _add_station_dummies(rows, self.station_col)
        for col in self.station_dummy_cols:
            if col not in rows.columns:
                rows[col] = 0
        rows = rows.ffill().bfill()

        missing = [c for c in self.feature_cols if c not in rows.columns]
        if missing:
            raise ValueError(f"Missing required features after preprocessing: {missing}")
        if rows[self.feature_cols].isna().any().any():
            raise ValueError("Input does not have enough history to build lag/rolling features. Upload at least 25 recent rows per station.")
        return rows[self.feature_cols].tail(1)

    def predict(self, rows: pd.DataFrame) -> float:
        features = self._prepare_frame(rows)
        return float(self.model.predict(features)[0])

    def predict_with_category(self, rows: pd.DataFrame) -> dict:
        value = self.predict(rows)
        return {
            "predicted_aqi": value,
            "category": aqi_category(value),
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "prediction_kind": self.metadata.get("prediction_kind", "same_timestamp_estimate"),
        }


def load_predictor(model_dir: str | Path):
    model_dir = Path(model_dir)
    if (model_dir / "tree_model.joblib").exists():
        return TreeAQIPredictor(model_dir)
    return AQIPredictor(model_dir)


def predict_from_csv(input_csv: str | Path, model_dir: str | Path) -> dict:
    predictor = load_predictor(model_dir)
    rows = pd.read_csv(input_csv)
    prediction = predictor.predict_with_category(rows)
    prediction["model_dir"] = str(model_dir)
    return prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict AQI from a CSV containing recent feature rows.")
    parser.add_argument("--input", required=True, help="CSV containing at least the model feature columns.")
    parser.add_argument("--model-dir", default="models/aqi_tree", help="Directory containing trained artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(predict_from_csv(args.input, args.model_dir), indent=2))


if __name__ == "__main__":
    main()
