"""Binary AQI risk classifier — the path that produces an AUC.

Regression answers "what will the AQI be?"; this answers "will next-hour air be
unhealthy (AQI above a threshold)?" as a yes/no problem, which is what ROC-AUC
and PR-AUC measure. It reuses the exact same feature pipeline as the regressor so
the two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from .aqi import add_cpcb_aqi
from .config import ModelConfig
from .data import load_dataset
from .tree_model import build_tabular_features
from .walk_forward import _expanding_windows, _mean_std

DEFAULT_THRESHOLD = 100.0  # CPCB: AQI > 100 is "Moderate" or worse.


def _make_classifier(model_name: str, random_seed: int):
    model_name = model_name.lower()
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_seed,
            )
        except ImportError as exc:
            raise ImportError("Install xgboost first: pip install xgboost") from exc
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=random_seed)
        except ImportError as exc:
            raise ImportError("Install lightgbm first: pip install lightgbm") from exc

    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=random_seed)


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray, decision_threshold: float = 0.5) -> dict:
    """ROC-AUC / PR-AUC plus point metrics at a given decision threshold.

    AUC is undefined when the test block has only one class, so it is reported as
    None in that case rather than crashing.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= decision_threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return {
        "decision_threshold": float(decision_threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else None,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positives": int(y_true.sum()),
        "n": int(len(y_true)),
        "confusion_matrix": cm,
    }


def _fit(clf, X, y, balance: bool):
    """Fit with optional balanced sample weights (rare class counts more)."""
    sample_weight = compute_sample_weight("balanced", y) if balance else None
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the probability cutoff that maximises F1 on a validation slice.

    Sweeps candidate cutoffs (capped at 200 quantiles for speed) so we trade some
    precision for the recall the default 0.5 cutoff throws away on rare events.
    """
    y_true = np.asarray(y_true).astype(int)
    candidates = np.unique(proba)
    if len(candidates) > 200:
        candidates = np.quantile(proba, np.linspace(0.0, 1.0, 200))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        f1 = f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def _prepare(df: pd.DataFrame, config: ModelConfig, threshold: float, horizon: int):
    work, feature_cols, extra_cols, forecast_target_col, *_ = build_tabular_features(df, config, horizon)
    label = (work[forecast_target_col] > threshold).astype(int)
    return work, feature_cols, forecast_target_col, label


def walk_forward_auc(df, config, threshold, model_name, horizon, n_splits, balance=True, decision_threshold=0.5) -> dict:
    work, feature_cols, _, label = _prepare(df, config, threshold, horizon)
    timestamps = np.array(sorted(work.index.unique()))
    label = label.to_numpy()  # positional alignment; timestamps repeat across stations
    aucs, pr_aucs, recalls = [], [], []
    for train_ts, test_ts in _expanding_windows(timestamps, n_splits):
        train_mask = work.index.isin(train_ts)
        test_mask = work.index.isin(test_ts)
        train, test = work[train_mask], work[test_mask]
        y_train, y_test = label[train_mask], label[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = _fit(_make_classifier(model_name, config.random_seed), train[feature_cols], y_train, balance)
        proba = clf.predict_proba(test[feature_cols])[:, 1]
        aucs.append(float(roc_auc_score(y_test, proba)))
        pr_aucs.append(float(average_precision_score(y_test, proba)))
        recalls.append(float(recall_score(y_test, (proba >= decision_threshold).astype(int), zero_division=0)))
    return {
        "n_folds": len(aucs),
        "roc_auc": _mean_std(aucs),
        "pr_auc": _mean_std(pr_aucs),
        "recall_at_tuned_threshold": _mean_std(recalls),
    }


def train_classifier_from_csv(
    data_path: str | Path,
    output_dir: str | Path = "models/aqi_classifier",
    model_name: str = "sklearn",
    target_col: str = "AQI_CPCB",
    threshold: float = DEFAULT_THRESHOLD,
    horizon_hours: int = 1,
    n_splits: int = 5,
    balance: bool = True,
    tune_threshold: bool = True,
) -> dict:
    df = load_dataset(data_path)
    if target_col not in df.columns:
        df = add_cpcb_aqi(df, target_col=target_col)
    config = ModelConfig(target_col=target_col)

    work, feature_cols, forecast_target_col, label = _prepare(df, config, threshold, horizon_hours)
    split_idx = int(len(work) * (1 - config.test_size))
    train_df, test_df = work.iloc[:split_idx], work.iloc[split_idx:]
    y_train, y_test = label.iloc[:split_idx], label.iloc[split_idx:]
    if y_train.nunique() < 2:
        raise ValueError(f"Training set has only one class at threshold {threshold}; lower it or add data.")

    # Pick the decision threshold on a chronological validation tail of the
    # training data — never the test set — so the choice does not leak.
    decision_threshold = 0.5
    val_cut = int(len(train_df) * 0.8)
    fit_df, val_df = train_df.iloc[:val_cut], train_df.iloc[val_cut:]
    y_fit, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]
    if tune_threshold and y_fit.nunique() > 1 and y_val.nunique() > 1:
        tuner = _fit(_make_classifier(model_name, config.random_seed), fit_df[feature_cols], y_fit, balance)
        proba_val = tuner.predict_proba(val_df[feature_cols])[:, 1]
        decision_threshold = _best_threshold(y_val.to_numpy(), proba_val)

    clf = _fit(_make_classifier(model_name, config.random_seed), train_df[feature_cols], y_train, balance)
    proba = clf.predict_proba(test_df[feature_cols])[:, 1]
    holdout_default = _binary_metrics(y_test.to_numpy(), proba, 0.5)
    holdout_tuned = _binary_metrics(y_test.to_numpy(), proba, decision_threshold)
    cv = walk_forward_auc(df, config, threshold, model_name, horizon_hours, n_splits, balance, decision_threshold)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / "classifier.joblib")
    metadata = {
        "task": "binary_classification",
        "model_name": model_name,
        "target_col": target_col,
        "threshold": threshold,
        "positive_label": f"{target_col} > {threshold} (Moderate or worse)",
        "forecast_horizon_hours": horizon_hours,
        "balanced_class_weights": balance,
        "decision_threshold": decision_threshold,
        "feature_cols": feature_cols,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "holdout": holdout_default,
        "holdout_tuned": holdout_tuned,
        "decision_threshold": decision_threshold,
        "walk_forward": cv,
        "metadata": metadata,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a binary 'unhealthy air' AQI classifier and report AUC.")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--output-dir", default="models/aqi_classifier", help="Directory for model artifacts.")
    parser.add_argument("--model", default="sklearn", choices=["sklearn", "xgboost", "lightgbm"], help="Classifier backend.")
    parser.add_argument("--target-col", default="AQI_CPCB", help="AQI column.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="AQI above this counts as 'unhealthy'.")
    parser.add_argument("--horizon-hours", type=int, default=1, help="Future AQI horizon in hours.")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward folds for the AUC estimate.")
    parser.add_argument("--no-balance", action="store_true", help="Disable balanced class weighting.")
    parser.add_argument("--no-tune-threshold", action="store_true", help="Keep the default 0.5 decision threshold.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_classifier_from_csv(
        args.data,
        args.output_dir,
        args.model,
        args.target_col,
        args.threshold,
        args.horizon_hours,
        args.splits,
        balance=not args.no_balance,
        tune_threshold=not args.no_tune_threshold,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
