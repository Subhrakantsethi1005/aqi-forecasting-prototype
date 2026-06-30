from pathlib import Path

from aqi_forecasting.data import load_dataset, normalize_columns


def test_normalize_columns_maps_common_names():
    import pandas as pd

    df = pd.DataFrame({"pm25": [1], "wind_speed": [2], "humidity": [3]})
    out = normalize_columns(df)
    assert {"PM2.5", "WindSpeed", "Humidity"}.issubset(out.columns)


def test_load_dataset_uses_datetime_index():
    path = Path("data/sample_aqi_dataset.csv")
    df = load_dataset(path)
    assert df.index.name == "Datetime"
    assert "PM2.5" in df.columns
    assert df.index.is_monotonic_increasing
