from pathlib import Path

import pandas as pd

from src.preprocess import load_and_prepare


def test_future_target_and_lags(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="min"),
            "temperature": range(10),
            "power": range(10, 20),
            "frequency": [3.0] * 10,
        }
    ).to_csv(path, index=False)

    data = load_and_prepare(str(path), horizon=2)
    assert "temperature_lag_1" in data.X.columns
    # The first usable row is timestamp 5 because the longest feature lag is 5;
    # with a two-sample forecast horizon its target is timestamp 7.
    assert data.y.iloc[0] == 7
