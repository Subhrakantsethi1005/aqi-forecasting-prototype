from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import ModelConfig


def clean_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().any():
            out[col] = values.clip(values.quantile(0.001), values.quantile(0.999))
        else:
            out[col] = values
    out[columns] = out[columns].interpolate(limit=6).ffill().bfill()
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out["hour"] = out.index.hour
        out["day_of_week"] = out.index.dayofweek
        out["month"] = out.index.month
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 24),
    rolling_windows: tuple[int, ...] = (24,),
    group_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if group_col and group_col in out.columns:
            grouped = out.groupby(group_col, sort=False)[col]
            for lag in lags:
                out[f"{col}_lag_{lag}"] = grouped.shift(lag)
            for window in rolling_windows:
                out[f"{col}_roll_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        else:
            for lag in lags:
                out[f"{col}_lag_{lag}"] = out[col].shift(lag)
            for window in rolling_windows:
                out[f"{col}_roll_{window}"] = out[col].shift(1).rolling(window=window, min_periods=1).mean()
    return out


def select_feature_columns(df: pd.DataFrame, config: ModelConfig) -> list[str]:
    base = [c for c in config.feature_cols if c in df.columns]
    engineered = [c for c in df.columns if "_lag_" in c or "_roll_" in c or c in ["hour", "day_of_week", "month", "is_weekend"]]
    cols = sorted(set(base + engineered))
    if config.target_col in cols:
        cols.remove(config.target_col)
    return cols


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.asarray(X_seq), np.asarray(y_seq)


def prepare_training_data(df: pd.DataFrame, config: ModelConfig):
    """Clean, engineer features, split chronologically, scale, and create LSTM windows."""
    required = [c for c in config.feature_cols if c in df.columns]
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' is missing.")
    if not required:
        raise ValueError("No configured feature columns were found in the dataset.")

    work = clean_numeric_columns(df, required + [config.target_col])
    work = add_time_features(work)
    work = add_lag_features(work, required, group_col=config.station_col)
    feature_cols = select_feature_columns(work, config)
    work = work.dropna(subset=feature_cols + [config.target_col])

    split_idx = int(len(work) * (1 - config.test_size))
    if split_idx <= config.seq_len or len(work) - split_idx <= 0:
        raise ValueError("Not enough rows after preprocessing for the configured split and sequence length.")

    train_df = work.iloc[:split_idx].copy()
    test_df = work.iloc[split_idx:].copy()

    scaler_X = MinMaxScaler().fit(train_df[feature_cols])
    scaler_y = MinMaxScaler().fit(train_df[[config.target_col]])
    X_scaled = scaler_X.transform(work[feature_cols])
    y_scaled = scaler_y.transform(work[[config.target_col]])

    X_all, y_all = make_sequences(X_scaled, y_scaled, config.seq_len)
    train_seq_count = split_idx - config.seq_len
    X_train, y_train = X_all[:train_seq_count], y_all[:train_seq_count]
    X_test, y_test = X_all[train_seq_count:], y_all[train_seq_count:]

    metadata = {
        "target_col": config.target_col,
        "feature_cols": feature_cols,
        "seq_len": config.seq_len,
        "station_col": config.station_col if config.station_col in work.columns else None,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    return X_train, y_train, X_test, y_test, train_df, test_df, scaler_X, scaler_y, metadata


def save_feature_schema(metadata: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
