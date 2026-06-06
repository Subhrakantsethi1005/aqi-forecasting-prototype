# Model Card

## Intended Use

This model forecasts short-term AQI from recent pollutant and weather readings. It is intended for learning, experimentation, dashboards, and prototype alerting workflows.

It should not be used as the only source for public health decisions without validation against trusted monitoring systems.

## Inputs

Default input features:

```text
PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, WindSpeed
```

The training pipeline also creates time, lag, and rolling features.

## Output

One numeric AQI forecast.

## Current Limitations

- The full dataset is not included in this repository.
- The original notebook showed duplicate timestamps, likely from multiple stations.
- More station-aware validation is needed before claiming real-world performance.
- LSTM performance should be compared with simpler baselines and tree models.
- Data quality, sensor calibration, and missing-value handling can strongly affect forecasts.

## Responsible Use

Use this model as a forecasting aid. Always display uncertainty, data freshness, and the source of input measurements when building a user-facing application.
