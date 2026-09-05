"""Generate five-minute-ahead forecasts from recent telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .constants import DEFAULT_DATA_PATH, TARGETS
from .data import load_telemetry
from .features import (
    align_feature_columns,
    build_inference_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future telemetry values.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--task", choices=list(TARGETS), default="temperature")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--node", help="Optional node ID to predict")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Predict every usable row instead of only the latest row per node.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TARGETS[args.task]
    model_path = args.model or Path("models") / spec.model_filename
    artifact = joblib.load(model_path)

    frame = load_telemetry(args.input)
    matrix, meta = build_inference_features(frame)
    matrix = align_feature_columns(matrix, artifact["feature_columns"])

    if args.node is not None:
        mask = meta["node"].astype(str) == str(args.node)
        matrix = matrix.loc[mask].reset_index(drop=True)
        meta = meta.loc[mask].reset_index(drop=True)
        if meta.empty:
            raise ValueError(f"No usable rows found for node {args.node}")

    if not args.all_rows:
        latest_indices = meta.groupby("node", observed=True)["timestamp"].idxmax()
        matrix = matrix.loc[latest_indices].reset_index(drop=True)
        meta = meta.loc[latest_indices].reset_index(drop=True)

    predicted = artifact["model"].predict(matrix)
    output = meta.copy()
    output["forecast_timestamp"] = output["timestamp"] + pd.Timedelta(
        minutes=int(artifact["horizon_minutes"])
    )
    output[f"predicted_{artifact['target_column']}"] = predicted
    output["unit"] = artifact["unit"]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Saved {len(output)} predictions to {args.output}")
    else:
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
