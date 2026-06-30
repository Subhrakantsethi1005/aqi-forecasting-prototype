from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd

from .aqi import add_cpcb_aqi
from .config import ModelConfig
from .data import load_dataset
from .metrics import metric_slices
from .model import evaluate_regression
from .tree_model import _make_estimator, build_tabular_features


def _expanding_windows(timestamps: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_ts, test_ts) timestamp blocks for an expanding-window scheme.

    The timeline is cut into ``n_splits + 1`` equal blocks. Fold ``k`` trains on
    everything up to block ``k`` and tests on block ``k + 1``, so training data
    grows each fold and the test block is always strictly in the future. Splits
    happen on whole timestamps, never mid-timestamp, so stations that share a
    timestamp always land in the same fold.
    """
    n = len(timestamps)
    block = n // (n_splits + 1)
    if block == 0:
        raise ValueError(f"Not enough distinct timestamps ({n}) for {n_splits} folds.")
    windows = []
    for k in range(1, n_splits + 1):
        train_end = block * k
        test_end = n if k == n_splits else block * (k + 1)
        train_ts = timestamps[:train_end]
        test_ts = timestamps[train_end:test_end]
        if len(test_ts) == 0:
            continue
        windows.append((train_ts, test_ts))
    return windows


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def walk_forward_evaluate(
    df: pd.DataFrame,
    config: ModelConfig,
    model_name: str = "sklearn",
    forecast_horizon_hours: int = 1,
    n_splits: int = 5,
) -> dict:
    """Run expanding-window walk-forward validation for the tabular forecaster.

    Trains a fresh model on each expanding window and evaluates it on the next
    unseen block. Reports per-fold metrics, per-station metrics, and the mean and
    standard deviation across folds — a more honest picture of forecasting skill
    than a single train/test split.
    """
    work, feature_cols, extra_cols, forecast_target_col, *_ = build_tabular_features(
        df, config, forecast_horizon_hours
    )
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("Walk-forward validation needs a DatetimeIndex; load data with load_dataset().")

    timestamps = np.array(sorted(work.index.unique()))
    windows = _expanding_windows(timestamps, n_splits)
    station_col = extra_cols[0] if extra_cols else None

    folds = []
    for i, (train_ts, test_ts) in enumerate(windows, start=1):
        train_df = work[work.index.isin(train_ts)]
        test_df = work[work.index.isin(test_ts)]
        if len(train_df) < 2 or len(test_df) < 2:
            continue

        estimator = _make_estimator(model_name, config.random_seed)
        estimator.fit(train_df[feature_cols], train_df[forecast_target_col])
        predictions = estimator.predict(test_df[feature_cols])

        eval_frame = pd.DataFrame(
            {"actual": test_df[forecast_target_col].to_numpy(), "predicted": predictions},
            index=test_df.index,
        )
        if station_col:
            eval_frame[station_col] = test_df[station_col].to_numpy()

        folds.append(
            {
                "fold": i,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_end": pd.Timestamp(train_ts[-1]).isoformat(),
                "test_start": pd.Timestamp(test_ts[0]).isoformat(),
                "test_end": pd.Timestamp(test_ts[-1]).isoformat(),
                "metrics": evaluate_regression(eval_frame["actual"], eval_frame["predicted"]),
                "slices": metric_slices(eval_frame, station_col=station_col or "station_id"),
            }
        )

    if not folds:
        raise ValueError("Walk-forward validation produced no usable folds; check data size and n_splits.")

    aggregate = {
        "n_folds": len(folds),
        "mae": _mean_std([f["metrics"]["mae"] for f in folds]),
        "rmse": _mean_std([f["metrics"]["rmse"] for f in folds]),
        "r2": _mean_std([f["metrics"]["r2"] for f in folds]),
    }

    # Average per-station R2 across folds, which is the honest within-station skill.
    per_station: dict[str, list[float]] = {}
    for fold in folds:
        for station, station_metrics in fold["slices"].get("by_station", {}).items():
            per_station.setdefault(station, []).append(station_metrics["r2"])
    aggregate["by_station_r2"] = {station: _mean_std(values) for station, values in sorted(per_station.items())}

    return {
        "scheme": "expanding_window",
        "model_name": model_name,
        "forecast_horizon_hours": forecast_horizon_hours,
        "n_splits": n_splits,
        "feature_count": len(feature_cols),
        "aggregate": aggregate,
        "folds": folds,
    }


def walk_forward_from_csv(
    data_path: str | Path,
    model_name: str = "sklearn",
    target_col: str = "AQI_CPCB",
    forecast_horizon_hours: int = 1,
    n_splits: int = 5,
    output_path: str | Path | None = None,
) -> dict:
    df = load_dataset(data_path)
    if target_col not in df.columns:
        df = add_cpcb_aqi(df, target_col=target_col)
    config = ModelConfig(target_col=target_col)
    result = walk_forward_evaluate(df, config, model_name, forecast_horizon_hours, n_splits)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward (expanding-window) validation for the AQI tree forecaster.")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument("--model", default="sklearn", choices=["sklearn", "xgboost", "lightgbm"], help="Tree model backend.")
    parser.add_argument("--target-col", default="AQI_CPCB", help="Target column to forecast.")
    parser.add_argument("--horizon-hours", type=int, default=1, help="Future AQI horizon in hours.")
    parser.add_argument("--splits", type=int, default=5, help="Number of expanding-window folds.")
    parser.add_argument("--output", default=None, help="Optional path to write the full JSON report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = walk_forward_from_csv(
        args.data, args.model, args.target_col, args.horizon_hours, args.splits, args.output
    )
    # Print the aggregate summary; the per-fold detail goes to --output if requested.
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
