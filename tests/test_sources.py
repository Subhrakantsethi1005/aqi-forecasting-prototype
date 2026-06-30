import numpy as np
import pandas as pd
import pytest

from aqi_forecasting.sources import records_to_wide, to_cpcb_units, validate_units


def _records():
    return [
        {"datetime": "2024-11-01T00:00:00Z", "station_id": "Delhi-A", "lat": 28.6, "lon": 77.2, "parameter": "pm25", "value": 250.0},
        {"datetime": "2024-11-01T00:00:00Z", "station_id": "Delhi-A", "lat": 28.6, "lon": 77.2, "parameter": "pm10", "value": 410.0},
        {"datetime": "2024-11-01T00:00:00Z", "station_id": "Delhi-A", "lat": 28.6, "lon": 77.2, "parameter": "no2", "value": 55.0},
        {"datetime": "2024-11-01T01:00:00Z", "station_id": "Delhi-A", "lat": 28.6, "lon": 77.2, "parameter": "pm25", "value": 300.0},
        {"datetime": "2024-11-01T00:00:00Z", "station_id": "Delhi-B", "lat": 28.7, "lon": 77.1, "parameter": "pm25", "value": 180.0},
    ]


def test_records_to_wide_pivots_pollutants_per_station_hour():
    wide = records_to_wide(_records())
    assert list(wide.columns[:4]) == ["Datetime", "station_id", "lat", "lon"]
    # one row per (station, hour): Delhi-A has 2 hours, Delhi-B has 1
    assert len(wide) == 3
    assert wide["station_id"].nunique() == 2
    first = wide.iloc[0]
    assert first["PM2.5"] == 250.0 and first["PM10"] == 410.0 and first["NO2"] == 55.0
    # sorted by station then time
    assert wide["Datetime"].is_monotonic_increasing or wide.groupby("station_id")["Datetime"].apply(lambda s: s.is_monotonic_increasing).all()


def test_records_to_wide_drops_all_missing_columns():
    wide = records_to_wide(_records())
    # no weather records were supplied, so those columns must not appear as all-NaN
    for missing in ["Temperature", "Humidity", "WindSpeed", "SO2", "CO", "O3"]:
        assert missing not in wide.columns


def test_records_to_wide_handles_empty_input():
    wide = records_to_wide([])
    assert wide.empty
    assert "PM2.5" in wide.columns


def test_records_to_wide_preserves_high_pollution_values():
    wide = records_to_wide(_records())
    # the whole point of pulling real data: hazardous-air values survive intact
    assert wide["PM2.5"].max() == 300.0


def test_to_cpcb_units_passthrough_when_already_correct():
    assert to_cpcb_units(250.0, "pm25", "µg/m³") == 250.0
    assert to_cpcb_units(1.5, "co", "mg/m³") == 1.5


def test_to_cpcb_units_particulate_mg_to_ug():
    assert to_cpcb_units(0.25, "pm25", "mg/m3") == pytest.approx(250.0)


def test_to_cpcb_units_co_ppm_to_mgm3():
    # 1 ppm CO -> ~1.146 mg/m3 at 25C, 1 atm
    assert to_cpcb_units(1.0, "co", "ppm") == pytest.approx(28.01 / 24.45, rel=1e-3)


def test_to_cpcb_units_no2_ppb_to_ugm3():
    # 50 ppb NO2 -> ~94.1 ug/m3
    assert to_cpcb_units(50.0, "no2", "ppb") == pytest.approx(50.0 * 46.0055 / 24.45, rel=1e-3)


def test_to_cpcb_units_unknown_unit_passes_through():
    assert to_cpcb_units(42.0, "o3", "weird-unit") == 42.0
    assert pd.isna(to_cpcb_units(np.nan, "o3", "ppb"))


def test_records_to_wide_applies_unit_conversion():
    records = [
        {"datetime": "2024-11-01T00:00:00Z", "station_id": "X", "lat": 1.0, "lon": 2.0,
         "parameter": "co", "unit": "ppm", "value": 1.0},
    ]
    wide = records_to_wide(records)
    assert wide["CO"].iloc[0] == pytest.approx(28.01 / 24.45, rel=1e-3)


def test_validate_units_flags_unknown_and_missing():
    records = [
        {"parameter": "co", "unit": "ppm"},
        {"parameter": "no2", "unit": None},
        {"parameter": "o3", "unit": "weird"},
    ]
    warnings = validate_units(records)
    assert any("o3" in w and "unrecognised" in w for w in warnings)
    assert any("no2" in w and "no unit" in w for w in warnings)
