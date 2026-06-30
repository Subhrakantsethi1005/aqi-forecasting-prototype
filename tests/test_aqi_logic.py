import pytest
import pandas as pd
from aqi_forecasting.aqi import add_cpcb_aqi

def test_aqi_calculation_simple():
    """Test AQI calculation with known values."""
    # Example values for PM2.5 and PM10
    data = {
        "PM2.5": [30.0],
        "PM10": [60.0],
        "NO2": [20.0],
        "SO2": [10.0],
        "CO": [0.5],
        "O3": [40.0]
    }
    df = pd.DataFrame(data)
    df_with_aqi = add_cpcb_aqi(df)
    
    assert "AQI_CPCB" in df_with_aqi.columns
    assert not df_with_aqi["AQI_CPCB"].isna().any()
    assert df_with_aqi["AQI_CPCB"].iloc[0] > 0
    assert df_with_aqi["AQI_CPCB"].iloc[0] <= 500

def test_aqi_category_logic():
    """Test if AQI values fall within expected ranges."""
    # PM2.5 of 250 should be 'Very Poor' or 'Severe'
    data = {
        "PM2.5": [250.0],
        "PM10": [400.0]
    }
    df = pd.DataFrame(data)
    df_with_aqi = add_cpcb_aqi(df)
    
    # AQI for 250 PM2.5 is usually > 300
    assert df_with_aqi["AQI_CPCB"].iloc[0] > 300
