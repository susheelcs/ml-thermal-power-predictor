from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    align_feature_columns,
    build_inference_features,
    build_supervised,
)


def test_future_target_and_node_safe_lags(
    synthetic_telemetry: pd.DataFrame,
) -> None:
    dataset = build_supervised(
        synthetic_telemetry, target_column="cpu_temp_max", horizon_minutes=5
    )
    assert len(dataset.y) == 50  # 25 usable rows per node after lag/target loss.
    assert (dataset.meta["target_timestamp"] - dataset.meta["timestamp"]).eq(
        pd.Timedelta(minutes=5)
    ).all()

    first_node_row = dataset.meta[dataset.meta["node"] == "101"].index[0]
    assert dataset.X.loc[first_node_row, "cpu_temp_max_lag_10"] == pytest.approx(
        45.0
    )
    assert dataset.y.loc[first_node_row] == pytest.approx(46.5)


def test_inference_features_and_alignment(
    synthetic_telemetry: pd.DataFrame,
) -> None:
    matrix, meta = build_inference_features(synthetic_telemetry)
    assert len(matrix) == 60
    assert len(meta) == 60

    aligned = align_feature_columns(
        matrix.loc[meta["node"] == "101"].reset_index(drop=True),
        list(matrix.columns),
    )
    assert list(aligned.columns) == list(matrix.columns)
    assert (aligned["node_202"] == 0).all()
