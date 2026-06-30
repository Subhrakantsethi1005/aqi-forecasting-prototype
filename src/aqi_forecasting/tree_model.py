from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import pandas as pd

from .aqi import add_cpcb_aqi
from .config import ModelConfig
from .data import load_dataset
from .features import add_lag_features, clean_numeric_columns, select_feature_columns
from .metrics import metric_slices
from .model import evaluate_regression


def _make_estimator(model_name: str, random_seed: int):
    model_name = model_name.lower()
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_seed,
            )
        except ImportError as exc:
            raise ImportError("Install xgboost first: pip install xgboost") from exc
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_seed,
            )
        except ImportError as exc:
            raise ImportError("Install lightgbm first: pip install lightgbm") from exc

    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=random_seed)


def _add_forecast_time_features(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        future_index = out.index + pd.to_timedelta(horizon_hours, unit="h")
        out["hour"] = future_index.hour
        out["day_of_week"] = future_index.dayofweek
        out["month"] = future_index.month
        out["is_weekend"] = pd.Series(future_index.dayofweek, index=out.index).isin([5, 6]).astype(int)
    return out


def _add_station_dummies(df: pd.DataFrame, station_col: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    if station_col not in out.columns:
        return out, []
    dummies = pd.get_dummies(out[station_col].astype(str), prefix=station_col, dtype=int)
    out = pd.concat([out, dummies], axis=1)
    return out, dummies.columns.tolist()


def build_tabular_features(df: pd.DataFrame, config: ModelConfig, forecast_horizon_hours: int = 1):
    """Clean, engineer lag/time/station features, and attach the future-AQI target.

    Returns the fully engineered frame plus the column groups, without splitting.
    Both the single-split trainer and the walk-forward evaluator share this so the
    feature engineering stays identical across the two paths.
    """
    base_cols = [c for c in config.feature_cols if c in df.columns]
    location_cols = [c for c in ["lat", "lon", "distance_km"] if c in df.columns]
    work = clean_numeric_columns(df, base_cols + location_cols + [config.target_col])
    work = _add_forecast_time_features(work, forecast_horizon_hours)
    work = add_lag_features(work, base_cols, group_col=config.station_col)
    work, station_dummy_cols = _add_station_dummies(work, config.station_col)

    forecast_target_col = f"{config.target_col}_t_plus_{forecast_horizon_hours}h"
    if config.station_col in work.columns:
        work[forecast_target_col] = work.groupby(config.station_col, sort=False)[config.target_col].shift(-forecast_horizon_hours)
    else:
        work[forecast_target_col] = work[config.target_col].shift(-forecast_horizon_hours)

    feature_cols = sorted(set(select_feature_columns(work, config) + location_cols + station_dummy_cols))
    extra_cols = [config.station_col] if config.station_col in work.columns else []
    work = work.dropna(subset=feature_cols + [forecast_target_col])
    return work, feature_cols, extra_cols, forecast_target_col, base_cols, location_cols, station_dummy_cols


def prepare_tabular_data(df: pd.DataFrame, config: ModelConfig, forecast_horizon_hours: int = 1):
    work, feature_cols, extra_cols, forecast_target_col, base_cols, location_cols, station_dummy_cols = build_tabular_features(
        df, config, forecast_horizon_hours
    )
    split_idx = int(len(work) * (1 - config.test_size))
    train_df = work.iloc[:split_idx].copy()
    test_df = work.iloc[split_idx:].copy()
    return train_df, test_df, feature_cols, extra_cols, forecast_target_col, base_cols, location_cols, station_dummy_cols


def train_tree_from_csv(
    data_path: str | Path,
    output_dir: str | Path = "models/aqi_tree",
    model_name: str = "sklearn",
    target_col: str = "AQI_CPCB",
    forecast_horizon_hours: int = 1,
) -> dict:
    df = load_dataset(data_path)
    if target_col not in df.columns:
        df = add_cpcb_aqi(df, target_col=target_col)

    config = ModelConfig(target_col=target_col)
    train_df, test_df, feature_cols, extra_cols, forecast_target_col, base_cols, location_cols, station_dummy_cols = prepare_tabular_data(
        df,
        config,
        forecast_horizon_hours=forecast_horizon_hours,
    )
    estimator = _make_estimator(model_name, config.random_seed)
    estimator.fit(train_df[feature_cols], train_df[forecast_target_col])
    predictions = estimator.predict(test_df[feature_cols])

    metrics = evaluate_regression(test_df[forecast_target_col], predictions)
    eval_frame = test_df[[forecast_target_col] + extra_cols].copy()
    eval_frame["actual"] = test_df[forecast_target_col].values
    eval_frame["predicted"] = predictions
    sliced_metrics = metric_slices(eval_frame)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output_dir / "tree_model.joblib")
    metadata = {
        "model_name": model_name,
        "prediction_kind": "future_forecast",
        "forecast_horizon_hours": forecast_horizon_hours,
        "target_col": target_col,
        "forecast_target_col": forecast_target_col,
        "feature_cols": feature_cols,
        "base_feature_cols": base_cols,
        "location_cols": location_cols,
        "station_col": config.station_col if config.station_col in df.columns else None,
        "station_dummy_cols": station_dummy_cols,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"overall": metrics, "slices": sliced_metrics}, indent=2), encoding="utf-8")
    return {"metrics": metrics, "sliced_metrics": sliced_metrics, "metadata": metadata, "output_dir": str(output_dir)}
