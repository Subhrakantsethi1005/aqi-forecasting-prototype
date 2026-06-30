from dataclasses import dataclass, field


DEFAULT_FEATURE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
    "WindSpeed",
]


@dataclass(frozen=True)
class ModelConfig:
    target_col: str = "AQI_CPCB"
    datetime_col: str | None = None
    station_col: str = "station_id"
    seq_len: int = 24
    test_size: float = 0.2
    feature_cols: list[str] = field(default_factory=lambda: DEFAULT_FEATURE_COLUMNS.copy())
    random_seed: int = 42
