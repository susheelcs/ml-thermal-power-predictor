"""Evaluation metrics for forecasting and event-proxy assessment."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    tolerances: Iterable[float] = (),
) -> dict[str, float]:
    y_true = np.asarray(list(actual), dtype=float)
    y_pred = np.asarray(list(predicted), dtype=float)
    absolute_error = np.abs(y_true - y_pred)
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "median_absolute_error": float(np.median(absolute_error)),
    }
    for tolerance in tolerances:
        result[f"within_{tolerance:g}"] = float(
            np.mean(absolute_error <= tolerance)
        )
    return result


def event_proxy_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
    threshold: float,
) -> dict[str, object]:
    y_true = np.asarray(list(actual), dtype=float) >= threshold
    y_pred = np.asarray(list(predicted), dtype=float) >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rows": int(y_true.sum()),
    }
