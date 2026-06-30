from __future__ import annotations

import numpy as np
import pandas as pd

from .model import evaluate_regression


def persistence_forecast(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Predict the next value as the latest observed value."""
    return series.shift(horizon)


def seasonal_naive_forecast(series: pd.Series, season_length: int = 24) -> pd.Series:
    """Predict using the value from the same hour in the previous season."""
    return series.shift(season_length)


def evaluate_baselines(df: pd.DataFrame, target_col: str = "AQI_CPCB", station_col: str = "station_id", season_length: int = 24) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing.")

    work = df.copy().sort_index()
    if station_col in work.columns:
        grouped = work.groupby(station_col, sort=False)[target_col]
        work["persistence_pred"] = grouped.shift(1)
        work["seasonal_naive_pred"] = grouped.shift(season_length)
    else:
        work["persistence_pred"] = persistence_forecast(work[target_col])
        work["seasonal_naive_pred"] = seasonal_naive_forecast(work[target_col], season_length=season_length)

    results = {}
    for name, col in [("persistence", "persistence_pred"), ("seasonal_naive", "seasonal_naive_pred")]:
        rows = work[[target_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if rows.empty:
            results[name] = None
        else:
            results[name] = evaluate_regression(rows[target_col], rows[col])
    return results
