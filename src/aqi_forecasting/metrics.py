from __future__ import annotations

import pandas as pd

from .categories import aqi_category
from .model import evaluate_regression


def metric_slices(frame: pd.DataFrame, y_col: str = "actual", pred_col: str = "predicted", station_col: str = "station_id") -> dict:
    required = {y_col, pred_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for metric slicing: {sorted(missing)}")

    out = {"overall": evaluate_regression(frame[y_col], frame[pred_col])}

    if station_col in frame.columns:
        out["by_station"] = {
            str(station): evaluate_regression(group[y_col], group[pred_col])
            for station, group in frame.groupby(station_col)
            if len(group) >= 2
        }

    if isinstance(frame.index, pd.DatetimeIndex):
        out["by_month"] = {
            str(month): evaluate_regression(group[y_col], group[pred_col])
            for month, group in frame.groupby(frame.index.to_period("M").astype(str))
            if len(group) >= 2
        }

    high = frame[frame[y_col] >= 200]
    out["high_aqi"] = None if len(high) < 2 else evaluate_regression(high[y_col], high[pred_col])

    category_frame = frame.copy()
    category_frame["aqi_category"] = category_frame[y_col].apply(aqi_category)
    out["by_category"] = {
        category: evaluate_regression(group[y_col], group[pred_col])
        for category, group in category_frame.groupby("aqi_category")
        if len(group) >= 2
    }
    return out
