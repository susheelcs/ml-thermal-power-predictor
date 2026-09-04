from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .preprocess import load_and_prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained thermal model and save a plot.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/random_forest.joblib")
    parser.add_argument("--target", default="temperature")
    parser.add_argument("--timestamp", default="timestamp")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    data = load_and_prepare(
        args.input,
        target=args.target,
        timestamp=args.timestamp,
        horizon=bundle["horizon"],
    )

    split = int(len(data.X) * 0.8)
    pred = bundle["model"].predict(data.X.iloc[split:])

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    plot_path = out / "actual_vs_predicted.png"

    plt.figure(figsize=(10, 5))
    plt.plot(data.timestamps.iloc[split:], data.y.iloc[split:], label="Actual")
    plt.plot(data.timestamps.iloc[split:], pred, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Actual vs Predicted Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
