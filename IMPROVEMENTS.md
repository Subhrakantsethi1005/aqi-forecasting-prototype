# Improvement Roadmap

## Model Quality

1. Add walk-forward validation instead of one train/test split.
2. Train separate models per station or add station embeddings.
3. Add prediction intervals with quantile regression or MC dropout.
4. Add multi-step forecasting for 6-hour, 12-hour, and 24-hour horizons.
5. Add direct AQI category classification alongside numeric regression.
6. Add ensemble selection across LSTM, tree models, and baselines.

## Data

1. Keep raw data in `data/raw/` and processed data in `data/processed/`.
2. Add `station_id`, latitude, longitude, and source metadata.
3. Add weather forecast features when predicting future AQI.
4. Validate pollutant units before training.
5. Add a data dictionary and dataset license.

## Engineering

1. Add model versioning.
2. Add experiment tracking with MLflow or Weights & Biases.
3. Add a release process for trained artifacts.
4. Add input schema validation with clearer API examples.
5. Add Docker Compose for API plus dashboard.
6. Add scheduled model retraining.

## Product Ideas

1. Add alert rules for upcoming high AQI.
2. Show top contributing pollutants.
3. Add station comparison maps.
4. Add downloadable forecast reports.
5. Add public demo screenshots to the README.
