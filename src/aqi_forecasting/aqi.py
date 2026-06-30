from __future__ import annotations

import numpy as np
import pandas as pd


CPCB_BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 10000, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 10000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 100000, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 1000, 401, 500)],
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 100000, 401, 500)],
}


def calculate_subindex(value: float, breakpoints: list[tuple[float, float, float, float]]) -> float:
    if pd.isna(value):
        return np.nan
    value = float(value)
    ordered = sorted(breakpoints, key=lambda bp: bp[0])
    for low, high, index_low, index_high in ordered:
        if low <= value <= high:
            return ((index_high - index_low) / (high - low)) * (value - low) + index_low

    # CPCB breakpoint tables are usually written as rounded ranges, e.g. 0-30,
    # 31-60. Sensor data is decimal, so values like 30.5 should not fall
    # through to 500. Interpolate across the nearest surrounding band edges.
    for left, right in zip(ordered, ordered[1:]):
        _, left_high, _, left_index_high = left
        right_low, _, right_index_low, _ = right
        if left_high < value < right_low:
            return np.interp(value, [left_high, right_low], [left_index_high, right_index_low]).item()

    if value < ordered[0][0]:
        return ordered[0][2]
    return ordered[-1][3]


def add_cpcb_aqi(df: pd.DataFrame, pollutant_cols: list[str] | None = None, target_col: str = "AQI_CPCB") -> pd.DataFrame:
    """Add CPCB pollutant sub-indices and AQI target using the max-subindex rule."""
    out = df.copy()
    pollutant_cols = pollutant_cols or [c for c in CPCB_BREAKPOINTS if c in out.columns]
    subindex_cols = []

    for col in pollutant_cols:
        if col not in CPCB_BREAKPOINTS:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        sub_col = f"{col}_subindex"
        out[sub_col] = numeric.apply(lambda x: calculate_subindex(x, CPCB_BREAKPOINTS[col]))
        subindex_cols.append(sub_col)

    if subindex_cols:
        out[target_col] = out[subindex_cols].max(axis=1)
    return out
