"""Leakage-aware time-series feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import (
    DELTA_SIGNALS,
    EXCLUDED_FEATURE_COLUMNS,
    LAGS,
    LAG_SIGNALS,
    ROLLING_SIGNALS,
    ROLLING_WINDOWS,
)


@dataclass(frozen=True)
class FeatureDataset:
    """Feature matrix, target, and identifying metadata."""

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def _engineer_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(["node", "timestamp"]).reset_index(drop=True).copy()
    grouped = data.groupby("node", sort=False, observed=True)
    derived: dict[str, pd.Series] = {}

    for column in LAG_SIGNALS:
        for lag in LAGS:
            derived[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    for column in ROLLING_SIGNALS:
        for window in ROLLING_WINDOWS:
            derived[f"{column}_roll_mean_{window}"] = grouped[column].transform(
                lambda series, size=window: series.rolling(
                    size, min_periods=1
                ).mean()
            )
            derived[f"{column}_roll_std_{window}"] = grouped[column].transform(
                lambda series, size=window: series.rolling(
                    size, min_periods=2
                ).std()
            )

    for column in DELTA_SIGNALS:
        derived[f"{column}_delta_1"] = data[column] - derived[f"{column}_lag_1"]
        derived[f"{column}_delta_5"] = data[column] - derived[f"{column}_lag_5"]

    minute_of_day = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    derived["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    derived["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    derived_frame = pd.DataFrame(derived, index=data.index)
    return pd.concat([data, derived_frame], axis=1)


def _to_feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in rows.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and column != "node"
        and pd.api.types.is_numeric_dtype(rows[column])
    ]
    matrix = rows[numeric_columns].reset_index(drop=True)
    node_dummies = pd.get_dummies(
        rows["node"].astype("string"), prefix="node", dtype=float
    ).reset_index(drop=True)
    return pd.concat([matrix, node_dummies], axis=1)


def build_supervised(
    frame: pd.DataFrame,
    target_column: str,
    horizon_minutes: int = 5,
) -> FeatureDataset:
    """Create current/past features and an exactly future target per node."""

    if horizon_minutes < 1:
        raise ValueError("horizon_minutes must be at least 1")
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")

    rows = _engineer_rows(frame)
    grouped = rows.groupby("node", sort=False, observed=True)
    rows["target"] = grouped[target_column].shift(-horizon_minutes)
    rows["target_timestamp"] = grouped["timestamp"].shift(-horizon_minutes)

    expected_delta = pd.Timedelta(minutes=horizon_minutes)
    exact_horizon = rows["target_timestamp"] - rows["timestamp"] == expected_delta
    rows = rows.loc[exact_horizon & rows["target"].notna()].copy()

    # Ten minutes of complete thermal/power history are required. Remaining
    # sensor gaps are handled by the model pipeline's train-fitted imputer.
    rows = rows.dropna(
        subset=[
            "cpu_temp_max_lag_10",
            "cpu_temp_mean_lag_10",
            "total_power_lag_10",
        ]
    )

    X = _to_feature_matrix(rows)
    y = rows["target"].astype(float).reset_index(drop=True)
    meta = rows[
        ["node", "timestamp", "target_timestamp", target_column]
    ].reset_index(drop=True)
    return FeatureDataset(X=X, y=y, meta=meta)


def build_inference_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features for forecasting from the latest available telemetry."""

    rows = _engineer_rows(frame)
    rows = rows.dropna(
        subset=[
            "cpu_temp_max_lag_10",
            "cpu_temp_mean_lag_10",
            "total_power_lag_10",
        ]
    ).copy()
    X = _to_feature_matrix(rows)
    meta = rows[["node", "timestamp"]].reset_index(drop=True)
    return X, meta


def align_feature_columns(
    matrix: pd.DataFrame, expected_columns: Sequence[str]
) -> pd.DataFrame:
    """Align inference features to training columns.

    Missing node one-hot columns are safe and are filled with zero. Missing
    telemetry-derived columns indicate incompatible input and raise an error.
    """

    expected = list(expected_columns)
    missing = [column for column in expected if column not in matrix.columns]
    unsafe_missing = [column for column in missing if not column.startswith("node_")]
    if unsafe_missing:
        raise ValueError(
            "Input cannot produce required telemetry features: "
            f"{unsafe_missing}"
        )

    aligned = matrix.copy()
    for column in missing:
        aligned[column] = 0.0
    return aligned.reindex(columns=expected)
