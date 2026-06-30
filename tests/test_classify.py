import numpy as np
import pandas as pd

from aqi_forecasting.classify import _binary_metrics, train_classifier_from_csv


def test_binary_metrics_reports_auc_when_both_classes_present():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])  # perfectly separable
    out = _binary_metrics(y_true, proba)
    assert out["roc_auc"] == 1.0
    assert out["pr_auc"] == 1.0
    assert out["positives"] == 2 and out["n"] == 4


def test_binary_metrics_auc_none_for_single_class():
    out = _binary_metrics(np.array([0, 0, 0]), np.array([0.2, 0.3, 0.4]))
    assert out["roc_auc"] is None
    assert out["pr_auc"] is None


def _synthetic_csv(tmp_path, periods=400):
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    rng = np.random.default_rng(0)
    rows = []
    for station in ["A", "B"]:
        # PM2.5 spread wide so the derived AQI straddles the 100 threshold in every
        # slice (otherwise a chronological holdout can end up single-class).
        pm25 = rng.uniform(10, 110, periods)
        rows.append(
            pd.DataFrame(
                {
                    "Datetime": idx,
                    "station_id": station,
                    "PM2.5": pm25,
                    "PM10": pm25 * 1.4,
                    "NO2": np.abs(rng.normal(30, 5, periods)),
                }
            )
        )
    out = pd.concat(rows)
    path = tmp_path / "synthetic.csv"
    out.to_csv(path, index=False)
    return path


def test_train_classifier_produces_auc_and_artifacts(tmp_path):
    data = _synthetic_csv(tmp_path)
    result = train_classifier_from_csv(
        data, output_dir=tmp_path / "clf", threshold=100.0, n_splits=3
    )
    assert (tmp_path / "clf" / "classifier.joblib").exists()
    assert (tmp_path / "clf" / "metrics.json").exists()
    # a real AUC number was produced and is a valid probability-rank score
    auc = result["holdout"]["roc_auc"]
    assert auc is not None and 0.0 <= auc <= 1.0
    assert result["metadata"]["threshold"] == 100.0
