from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "pm25": "PM2.5",
    "pm2_5": "PM2.5",
    "pm2.5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
    "ozone": "O3",
    "temperature": "Temperature",
    "temp": "Temperature",
    "humidity": "Humidity",
    "windspeed": "WindSpeed",
    "wind_speed": "WindSpeed",
    "aqi_cpcb": "AQI_CPCB",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        key_compact = key.replace("_", "")
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
        elif key_compact in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key_compact]
    return df.rename(columns=renamed)


def find_datetime_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if any(token in str(c).lower() for token in ["date", "time", "datetime"])]
    if candidates:
        return candidates[0]
    return df.columns[0]


def load_dataset(path: str | Path, datetime_col: str | None = None) -> pd.DataFrame:
    """Load a CSV and return a dataframe sorted by DatetimeIndex."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)
    dt_col = find_datetime_column(df, datetime_col)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
    df.index.name = "Datetime"
    return df
