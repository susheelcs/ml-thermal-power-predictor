from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from src.data import load_telemetry
from src.features import align_feature_columns, build_supervised
from src.metrics import regression_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/m100-thermal-subset.csv.gz"


def test_included_processed_dataset() -> None:
    frame = load_telemetry(DATA)
    assert len(frame) == 10_068
    assert frame["node"].nunique() == 12
    rows_per_node = frame.groupby("node", observed=True).size()
    assert rows_per_node.nunique() == 1
    assert rows_per_node.iloc[0] == 839


@pytest.mark.parametrize(
    ("task", "target"),
    [
        ("temperature", "cpu_temp_max"),
        ("power", "total_power"),
    ],
)
def test_saved_model_reproduces_reported_metrics(
    task: str, target: str
) -> None:
    artifact = joblib.load(ROOT / f"models/{task}_5min.joblib")
    frame = load_telemetry(DATA)
    dataset = build_supervised(frame, target, 5)
    matrix = align_feature_columns(dataset.X, artifact["feature_columns"])
    mask = dataset.meta["target_timestamp"] > artifact["validation_target_end"]

    prediction = artifact["model"].predict(matrix.loc[mask])
    persistence = dataset.meta.loc[mask, target].to_numpy()
    result = regression_metrics(dataset.y.loc[mask], prediction)
    baseline = regression_metrics(dataset.y.loc[mask], persistence)

    report = json.loads((ROOT / "reports/metrics.json").read_text())
    reported = report["experiments"][task]["test_models"]

    # CSV/Parquet round trips can change floating-point values at the
    # sub-micro level, so compare reproducibility with a practical tolerance.
    assert result["mae"] == pytest.approx(
        reported[artifact["selected_model"]]["mae"], abs=1e-6
    )
    assert result["rmse"] == pytest.approx(
        reported[artifact["selected_model"]]["rmse"], abs=1e-6
    )
    assert result["r2"] == pytest.approx(
        reported[artifact["selected_model"]]["r2"], abs=1e-8
    )

    # The selected learned model must improve on persistence for the primary
    # validation objective when evaluated on the held-out test period.
    assert result["rmse"] < baseline["rmse"]
