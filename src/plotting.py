"""Plotting utilities used by the reproducible experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    meta: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
    title: str,
    ylabel: str,
    output: Path,
) -> str:
    plot_frame = pd.DataFrame(
        {
            "node": meta["node"],
            "timestamp": meta["target_timestamp"],
            "actual": actual,
            "predicted": predicted,
        }
    )
    representative_node = (
        plot_frame.groupby("node", observed=True)["actual"]
        .var()
        .sort_values(ascending=False)
        .index[0]
    )
    selected = plot_frame.loc[
        plot_frame["node"] == representative_node
    ].sort_values("timestamp")

    plt.figure(figsize=(10, 5))
    plt.plot(selected["timestamp"], selected["actual"], label="Actual")
    plt.plot(selected["timestamp"], selected["predicted"], label="Predicted")
    plt.title(f"{title} — node {representative_node}")
    plt.xlabel("Target timestamp (UTC)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return str(representative_node)


def plot_residuals(
    actual: pd.Series,
    predicted: np.ndarray,
    title: str,
    unit: str,
    output: Path,
) -> None:
    residuals = np.asarray(actual) - np.asarray(predicted)
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=35)
    plt.title(f"{title} residual distribution")
    plt.xlabel(f"Actual - predicted ({unit})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_validation_comparison(
    validation_models: dict[str, dict[str, Any]],
    task_name: str,
    output: Path,
) -> None:
    names = list(validation_models)
    values = [validation_models[name]["rmse"] for name in names]
    plt.figure(figsize=(9, 5))
    plt.bar(names, values)
    plt.title(f"{task_name} validation RMSE")
    plt.ylabel("RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def feature_importance_frame(
    pipeline: Any, feature_columns: list[str]
) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).reshape(-1)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
    else:
        return pd.DataFrame(columns=["feature", "importance", "absolute_importance"])

    if len(values) != len(feature_columns):
        return pd.DataFrame(columns=["feature", "importance", "absolute_importance"])
    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "absolute_importance": np.abs(values),
        }
    )
    return result.sort_values("absolute_importance", ascending=False).reset_index(
        drop=True
    )


def plot_feature_importance(
    importance: pd.DataFrame, title: str, output: Path, top_n: int = 15
) -> None:
    if importance.empty:
        return
    selected = importance.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(selected["feature"], selected["absolute_importance"])
    plt.title(title)
    plt.xlabel("Absolute coefficient / feature importance")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
