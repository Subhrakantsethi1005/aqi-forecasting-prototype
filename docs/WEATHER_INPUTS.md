# Weather Forecast Inputs

The model already supports weather-style features:

```text
Temperature
Humidity
WindSpeed
```

For true future AQI forecasting, the prediction CSV should contain recent pollutant readings plus weather values for the forecast period when available.

## Recommended Columns

```text
Datetime
station_id
PM2.5
PM10
NO2
SO2
CO
O3
Temperature
Humidity
WindSpeed
```

## Practical Approach

1. Train the model on historical pollutant and weather observations.
2. At prediction time, use the latest measured pollutant values.
3. Add weather forecast values for the upcoming horizon.
4. Keep units consistent with training data.

## Future Enhancement

Add a weather connector script that fetches forecast data from a weather API and writes a `recent_readings.csv` file with the same schema used by the predictor.
