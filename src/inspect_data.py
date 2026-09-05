"""Inspect the compact telemetry data and print a quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import DEFAULT_DATA_PATH
from .data import load_telemetry, quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect M100 telemetry data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = quality_report(load_telemetry(args.input))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
