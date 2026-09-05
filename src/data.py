"""Dataset loading, validation, and quality reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import REQUIRED_COLUMNS


def load_telemetry(path: str | Path) -> pd.DataFrame:
    """Load the compact M100 telemetry dataset.

    Supported formats are Parquet, CSV, and gzip-compressed CSV. The result is
    sorted by node and timestamp, with UTC timestamps and string node IDs.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Telemetry file does not exist: {source}")

    lower_name = source.name.lower()
    if lower_name.endswith(".parquet"):
        frame = pd.read_parquet(source)
    elif lower_name.endswith(".csv") or lower_name.endswith(".csv.gz"):
        frame = pd.read_csv(source)
    else:
        raise ValueError(
            "Unsupported telemetry format. Use .parquet, .csv, or .csv.gz."
        )

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["node"] = frame["node"].astype("string")

    if frame["timestamp"].isna().any():
        invalid = int(frame["timestamp"].isna().sum())
        raise ValueError(f"Telemetry contains {invalid} invalid timestamps")
    if frame["node"].isna().any():
        invalid = int(frame["node"].isna().sum())
        raise ValueError(f"Telemetry contains {invalid} missing node IDs")

    duplicate_count = int(frame.duplicated(["node", "timestamp"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Telemetry contains {duplicate_count} duplicate node/timestamp rows"
        )

    return frame.sort_values(["node", "timestamp"]).reset_index(drop=True)


def quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable data-quality information."""

    sorted_frame = frame.sort_values(["node", "timestamp"])
    gaps = (
        sorted_frame.groupby("node", observed=True)["timestamp"]
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    numeric = frame.select_dtypes(include="number")
    ranges: dict[str, dict[str, float | None]] = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        ranges[column] = {
            "minimum": float(values.min()) if not values.empty else None,
            "maximum": float(values.max()) if not values.empty else None,
            "mean": float(values.mean()) if not values.empty else None,
        }

    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "nodes": int(frame["node"].nunique()),
        "unique_timestamps": int(frame["timestamp"].nunique()),
        "timestamp_min": frame["timestamp"].min().isoformat(),
        "timestamp_max": frame["timestamp"].max().isoformat(),
        "duplicates_node_timestamp": int(
            frame.duplicated(["node", "timestamp"]).sum()
        ),
        "rows_per_node": {
            str(key): int(value)
            for key, value in frame.groupby("node", observed=True).size().items()
        },
        "sampling_gap_seconds": {
            "minimum": float(gaps.min()) if not gaps.empty else None,
            "median": float(gaps.median()) if not gaps.empty else None,
            "maximum": float(gaps.max()) if not gaps.empty else None,
        },
        "missing_percent": {
            column: round(float(frame[column].isna().mean() * 100), 3)
            for column in frame.columns
        },
        "numeric_ranges": ranges,
    }
