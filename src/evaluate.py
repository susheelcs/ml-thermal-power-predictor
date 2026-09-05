"""Evaluate a saved model artifact on its chronological held-out period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .constants import DEFAULT_DATA_PATH, TARGETS
from .data import load_telemetry
from .features import align_feature_columns, build_supervised
from .metrics import event_proxy_metrics, regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved forecast model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--task", choices=list(TARGETS), default="temperature")
    parser.add_argument("--model", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TARGETS[args.task]
    model_path = args.model or Path("models") / spec.model_filename
    artifact = joblib.load(model_path)

    frame = load_telemetry(args.input)
    dataset = build_supervised(
        frame,
        artifact["target_column"],
        int(artifact["horizon_minutes"]),
    )
    matrix = align_feature_columns(dataset.X, artifact["feature_columns"])
    validation_end = pd.Timestamp(artifact["validation_target_end"])
    test_mask = dataset.meta["target_timestamp"] > validation_end

    X_test = matrix.loc[test_mask]
    y_test = dataset.y.loc[test_mask]
    meta_test = dataset.meta.loc[test_mask]
    predicted = artifact["model"].predict(X_test)
    persistence = meta_test[artifact["target_column"]].to_numpy()

    output = {
        "model": str(model_path),
        "task": artifact["task"],
        "selected_model": artifact["selected_model"],
        "test_rows": int(len(y_test)),
        "model_metrics": regression_metrics(
            y_test, predicted, spec.tolerances
        ),
        "persistence_metrics": regression_metrics(
            y_test, persistence, spec.tolerances
        ),
    }
    threshold = artifact.get("high_temperature_threshold")
    if threshold is not None:
        output["high_temperature_proxy"] = event_proxy_metrics(
            y_test, predicted, float(threshold)
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
