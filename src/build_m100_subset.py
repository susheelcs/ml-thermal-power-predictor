"""Create a small, aligned Marconi100 telemetry subset for the first experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


METRICS = {
    "temperature_c": ("ipmi_pub", "p0_core0_temp"),
    "socket1_temperature_c": ("ipmi_pub", "p1_core0_temp"),
    "socket0_power_w": ("ipmi_pub", "p0_power"),
    "socket1_power_w": ("ipmi_pub", "p1_power"),
    "cpu_user_pct": ("ganglia_pub", "cpu_user"),
    "cpu_idle_pct": ("ganglia_pub", "cpu_idle"),
    "load_1m": ("ganglia_pub", "load_one"),
}


def resolve_raw_root(raw_root: Path) -> Path:
    """Resolve raw dataset directory, navigating into year_month= subfolder if needed."""
    if (raw_root / "plugin=ipmi_pub").is_dir():
        return raw_root
    year_months = list(raw_root.glob("year_month=*"))
    if year_months and (year_months[0] / "plugin=ipmi_pub").is_dir():
        return year_months[0]
    return raw_root


def read_metric(raw_root: Path, plugin: str, metric: str, node: str) -> pd.Series:
    """Read one metric for one node, resampled as one-minute means."""
    path = raw_root / f"plugin={plugin}" / f"metric={metric}" / "a_0.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Metric file not found: {path}")
    table = pq.read_table(path, filters=[("node", "=", str(node))])
    frame = table.to_pandas()[["timestamp", "value"]]
    if frame.empty:
        raise ValueError(f"No observations for node {node} in {plugin}/{metric}")
    series = frame.set_index("timestamp")["value"].sort_index().resample("1min").mean()
    # Normalize DatetimeIndex unit to nanoseconds to avoid pandas 2.2 DatetimeIndex
    # intersection bug on millisecond units with frequency.
    if hasattr(series.index, "as_unit"):
        series.index = series.index.as_unit("ns")
    return series


def build_subset(raw_root: Path, node: str, start: str, end: str) -> pd.DataFrame:
    """Build a complete, minute-level feature table for the requested UTC window."""
    raw_root = resolve_raw_root(raw_root)
    series = {
        name: read_metric(raw_root, plugin, metric, node)
        for name, (plugin, metric) in METRICS.items()
    }
    frame = pd.DataFrame(series).loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    if frame.empty:
        raise ValueError("No complete rows in the requested time window")
    frame["total_cpu_power_w"] = frame["socket0_power_w"] + frame["socket1_power_w"]
    frame.insert(0, "node", str(node))
    return frame.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small M100 thermal-model CSV subset.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/21-03"),
        help="Extracted 21-03 or year_month=21-03 directory (default: data/raw/21-03)",
    )
    parser.add_argument("--node", default="582")
    parser.add_argument("--start", default="2021-03-01T20:18:00Z")
    parser.add_argument("--end", default="2021-03-02T10:29:00Z")
    parser.add_argument("--output", default="data/processed/m100_node582_first_window.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = build_subset(args.raw_root, args.node, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)
    print(f"Wrote {len(subset)} rows for node {args.node} to {args.output}")


if __name__ == "__main__":
    main()

