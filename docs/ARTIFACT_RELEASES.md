# Publishing Model Artifacts

Do not commit large model files directly to Git.

Ignored artifact examples:

```text
models/aqi_lstm/lstm_model.keras
models/aqi_lstm/scaler_X.joblib
models/aqi_lstm/scaler_y.joblib
models/aqi_lstm/feature_schema.json
models/aqi_lstm/metrics.json
```

## Recommended Release Flow

1. Train the model locally:

```bash
python scripts/train.py --data data/aqi_dataset.csv --output-dir models/aqi_lstm
```

2. Zip the artifact folder:

```bash
Compress-Archive -Path models/aqi_lstm -DestinationPath aqi_lstm_artifacts.zip
```

3. Create a GitHub Release.

4. Upload `aqi_lstm_artifacts.zip` to that release.

5. Add the release link to `README.md`.

## Why

This keeps the repository lightweight while still allowing users to download a ready-to-use trained model.
