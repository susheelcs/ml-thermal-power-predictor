from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import load_telemetry, quality_report


def test_load_csv_and_quality_report(
    tmp_path: Path, synthetic_telemetry: pd.DataFrame
) -> None:
    path = tmp_path / "telemetry.csv.gz"
    synthetic_telemetry.to_csv(path, index=False, compression="gzip")
    loaded = load_telemetry(path)
    report = quality_report(loaded)
    assert len(loaded) == 80
    assert report["nodes"] == 2
    assert report["sampling_gap_seconds"]["median"] == 60.0


def test_duplicate_node_timestamp_is_rejected(
    tmp_path: Path, synthetic_telemetry: pd.DataFrame
) -> None:
    duplicate = pd.concat(
        [synthetic_telemetry, synthetic_telemetry.iloc[[0]]], ignore_index=True
    )
    path = tmp_path / "duplicate.csv"
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_telemetry(path)
