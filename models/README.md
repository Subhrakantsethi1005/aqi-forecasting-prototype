# Models

Trained artifacts are intentionally ignored by Git.

After training, this directory should contain a model folder such as:

```text
models/aqi_lstm/
  lstm_model.keras
  scaler_X.joblib
  scaler_y.joblib
  feature_schema.json
  metrics.json
```

For public releases, upload large artifacts to GitHub Releases or another model hosting service and link them from the README.
