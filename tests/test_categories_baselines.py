import pandas as pd

from aqi_forecasting.baselines import evaluate_baselines
from aqi_forecasting.categories import add_aqi_category, aqi_category


def test_aqi_category_labels():
    assert aqi_category(35) == "Good"
    assert aqi_category(125) == "Moderate"
    assert aqi_category(200.5) == "Poor"
    assert aqi_category(450) == "Severe"


def test_add_aqi_category_column():
    df = pd.DataFrame({"AQI_CPCB": [42, 180, 330]})
    out = add_aqi_category(df)
    assert out["AQI_Category"].tolist() == ["Good", "Moderate", "Very Poor"]


def test_station_aware_baselines_return_metrics():
    idx = pd.date_range("2024-01-01", periods=30, freq="h").tolist() * 2
    df = pd.DataFrame(
        {
            "station_id": ["A"] * 30 + ["B"] * 30,
            "AQI_CPCB": list(range(30)) + list(range(40, 70)),
        },
        index=idx,
    )
    result = evaluate_baselines(df, season_length=24)
    assert result["persistence"]["rmse"] >= 0
    assert result["seasonal_naive"]["rmse"] >= 0
