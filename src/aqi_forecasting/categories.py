from __future__ import annotations

import pandas as pd


AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


def aqi_category(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 0:
        return "Good"
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def add_aqi_category(df: pd.DataFrame, aqi_col: str = "AQI_CPCB", category_col: str = "AQI_Category") -> pd.DataFrame:
    out = df.copy()
    if aqi_col not in out.columns:
        raise ValueError(f"Cannot create AQI categories because '{aqi_col}' is missing.")
    out[category_col] = out[aqi_col].apply(aqi_category)
    return out
