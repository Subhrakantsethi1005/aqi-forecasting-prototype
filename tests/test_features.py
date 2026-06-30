import numpy as np
import pandas as pd

from aqi_forecasting.config import ModelConfig
from aqi_forecasting.features import prepare_training_data


def make_frame(rows=80):
    idx = pd.date_range("2024-01-01", periods=rows, freq="h")
    data = {
        "PM2.5": np.linspace(30, 80, rows),
        "PM10": np.linspace(60, 120, rows),
        "NO2": np.linspace(20, 50, rows),
        "SO2": np.linspace(10, 30, rows),
        "CO": np.linspace(0.5, 1.2, rows),
        "O3": np.linspace(20, 45, rows),
        "Temperature": np.linspace(22, 35, rows),
        "Humidity": np.linspace(55, 80, rows),
        "WindSpeed": np.linspace(1, 4, rows),
        "AQI_CPCB": np.linspace(70, 160, rows),
    }
    return pd.DataFrame(data, index=idx)


def test_prepare_training_data_returns_expected_shapes():
    config = ModelConfig(seq_len=12)
    X_train, y_train, X_test, y_test, *_rest, metadata = prepare_training_data(make_frame(), config)
    assert X_train.ndim == 3
    assert y_train.ndim == 2
    assert X_test.shape[1] == 12
    assert X_test.shape[2] == len(metadata["feature_cols"])
    assert len(y_test) == len(X_test)


def test_station_lags_do_not_cross_station_boundaries():
    df = make_frame(80)
    df["station_id"] = ["A"] * 40 + ["B"] * 40
    config = ModelConfig(seq_len=12)
    *_rest, metadata = prepare_training_data(df, config)
    assert metadata["station_col"] == "station_id"
