from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_lstm(seq_len: int, n_features: int, units: int = 128, dropout: float = 0.3, dense_units: int = 64):
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, n_features)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(model, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 60, batch_size: int = 64):
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    return model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_artifacts(model, scaler_X, scaler_y, metadata: dict, output_dir: str | Path, metrics: dict | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "lstm_model.keras")
    joblib.dump(scaler_X, output_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, output_dir / "scaler_y.joblib")
    (output_dir / "feature_schema.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if metrics is not None:
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_artifacts(model_dir: str | Path):
    import tensorflow as tf

    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / "lstm_model.keras", compile=False)
    scaler_X = joblib.load(model_dir / "scaler_X.joblib")
    scaler_y = joblib.load(model_dir / "scaler_y.joblib")
    metadata = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    return model, scaler_X, scaler_y, metadata
