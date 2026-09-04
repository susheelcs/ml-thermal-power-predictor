from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocess import load_and_prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a thermal prediction baseline.")
    parser.add_argument("--input", required=True, help="Path to telemetry CSV")
    parser.add_argument("--target", default="temperature")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--model-out", default="models/random_forest.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_ratio < 0.5:
        raise ValueError("test-ratio must be between 0 and 0.5")

    data = load_and_prepare(
        args.input,
        target=args.target,
        timestamp=args.timestamp,
        horizon=args.horizon,
    )

    split = int(len(data.X) * (1 - args.test_ratio))
    if split <= 0 or split >= len(data.X):
        raise ValueError("Dataset is too small for the requested test ratio")

    # Time-aware split: earliest observations train, latest observations test.
    X_train, X_test = data.X.iloc[:split], data.X.iloc[split:]
    y_train, y_test = data.y.iloc[:split], data.y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, pred) ** 0.5
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": rmse,
        "r2": r2_score(y_test, pred),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "horizon_samples": args.horizon,
    }

    print(pd.Series(metrics).to_string())

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": list(X_train.columns),
            "target": args.target,
            "timestamp": args.timestamp,
            "horizon": args.horizon,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
