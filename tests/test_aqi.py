import pandas as pd

from aqi_forecasting.aqi import add_cpcb_aqi, calculate_subindex


def test_calculate_subindex_interpolates_cpcb_range():
    value = calculate_subindex(45, [(31, 60, 51, 100)])
    assert round(value, 2) == 74.66


def test_calculate_subindex_handles_decimal_breakpoint_gaps():
    value = calculate_subindex(30.5, [(0, 30, 0, 50), (31, 60, 51, 100)])
    assert 50 <= value <= 51


def test_add_cpcb_aqi_adds_target_from_pollutants():
    df = pd.DataFrame(
        {
            "PM2.5": [45.0],
            "PM10": [80.0],
            "NO2": [30.0],
            "SO2": [20.0],
            "CO": [0.8],
            "O3": [25.0],
        }
    )
    out = add_cpcb_aqi(df)
    assert "AQI_CPCB" in out.columns
    assert out["AQI_CPCB"].iloc[0] > 0
