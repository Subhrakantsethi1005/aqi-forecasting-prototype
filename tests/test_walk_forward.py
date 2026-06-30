import numpy as np
import pandas as pd

from aqi_forecasting.config import ModelConfig
from aqi_forecasting.walk_forward import _expanding_windows, walk_forward_evaluate


def test_expanding_windows_are_chronological_and_growing():
    timestamps = np.arange(60)
    windows = _expanding_windows(timestamps, n_splits=5)
    assert len(windows) == 5
    prev_train = 0
    for train_ts, test_ts in windows:
        # train set grows each fold and the test block is strictly in the future
        assert len(train_ts) >= prev_train
        assert test_ts[0] >= train_ts[-1]
        prev_train = len(train_ts)
    # the final fold absorbs any remainder up to the end of the timeline
    assert windows[-1][1][-1] == timestamps[-1]


def _synthetic_frame(periods: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    rng = np.random.default_rng(0)
    rows = []
    for station in ["A", "B"]:
        base = rng.normal(60, 10, periods).cumsum() / periods + 40
        frame = pd.DataFrame(
            {
                "station_id": station,
                "PM2.5": np.abs(base + rng.normal(0, 3, periods)),
                "PM10": np.abs(base * 1.5 + rng.normal(0, 5, periods)),
                "NO2": np.abs(rng.normal(30, 5, periods)),
                "SO2": np.abs(rng.normal(15, 3, periods)),
                "CO": np.abs(rng.normal(0.6, 0.1, periods)),
                "O3": np.abs(rng.normal(25, 5, periods)),
                "AQI_CPCB": np.abs(base + rng.normal(0, 2, periods)),
            },
            index=idx,
        )
        rows.append(frame)
    out = pd.concat(rows).sort_index()
    out.index.name = "Datetime"
    return out


def test_walk_forward_evaluate_reports_folds_and_aggregate():
    df = _synthetic_frame()
    result = walk_forward_evaluate(df, ModelConfig(), n_splits=3)
    assert result["aggregate"]["n_folds"] == len(result["folds"]) == 3
    for fold in result["folds"]:
        assert fold["test_rows"] > 0
        assert "mae" in fold["metrics"] and "r2" in fold["metrics"]
        # test block must start at or after the training data ends (no leakage)
        assert fold["test_start"] >= fold["train_end"]
    assert result["aggregate"]["mae"]["mean"] >= 0
    assert set(result["aggregate"]["by_station_r2"]) == {"A", "B"}
