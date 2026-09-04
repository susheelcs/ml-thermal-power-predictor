from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series


def load_and_prepare(
    path: str,
    target: str = "temperature",
    timestamp: str = "timestamp",
    horizon: int = 10,
) -> PreparedData:
    """Load telemetry and create a future-target regression dataset.

    The target at row t is the temperature at t + horizon samples.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    df = pd.read_csv(path)
    missing = {target, timestamp} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")
    df = df.dropna(subset=[timestamp, target]).sort_values(timestamp).reset_index(drop=True)

    numeric = df.select_dtypes(include="number").copy()
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' must be numeric")

    # Avoid leakage: current/future target is never included as a feature.
    y = numeric[target].shift(-horizon).rename(f"future_{target}")
    X = numeric.drop(columns=[target])

    # Add a small amount of target history. These are available at prediction time.
    for lag in (1, 2, 3, 5):
        X[f"{target}_lag_{lag}"] = numeric[target].shift(lag)

    valid = X.notna().all(axis=1) & y.notna()
    return PreparedData(
        X=X.loc[valid].reset_index(drop=True),
        y=y.loc[valid].reset_index(drop=True),
        timestamps=df.loc[valid, timestamp].reset_index(drop=True),
    )
