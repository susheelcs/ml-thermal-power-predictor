#!/usr/bin/env python3
"""Build a compact, aligned M100 thermal telemetry dataset.

This script is intended to run locally against the extracted March 2021
M100 ExaData partition. It selects nodes with overlapping telemetry,
aggregates selected metrics to one-minute intervals, and writes a compact
Parquet file suitable for upload and ML experimentation.
"""

import argparse
import json
import math
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


# Representative CPU core sensors selected from the inspection report.
# These files have among the strongest row coverage for both sockets.
CORE_TEMP_METRICS = [
    "p0_core10_temp",
    "p0_core11_temp",
    "p0_core20_temp",
    "p0_core21_temp",
    "p1_core10_temp",
    "p1_core11_temp",
    "p1_core14_temp",
    "p1_core15_temp",
]

SINGLE_METRICS = {
    # IPMI sensor telemetry
    "ambient": ("ipmi_pub", "ambient", "mean"),
    "p0_power": ("ipmi_pub", "p0_power", "mean"),
    "p1_power": ("ipmi_pub", "p1_power", "mean"),
    "total_power": ("ipmi_pub", "total_power", "mean"),
    "p0_vdd_temp": ("ipmi_pub", "p0_vdd_temp", "mean"),
    "p1_vdd_temp": ("ipmi_pub", "p1_vdd_temp", "mean"),
    # Ganglia operating-system telemetry
    "cpu_user": ("ganglia_pub", "cpu_user", "mean"),
    "cpu_system": ("ganglia_pub", "cpu_system", "mean"),
    "cpu_idle": ("ganglia_pub", "cpu_idle", "mean"),
    "cpu_wio": ("ganglia_pub", "cpu_wio", "mean"),
    "cpu_speed": ("ganglia_pub", "cpu_speed", "mean"),
    "load_one": ("ganglia_pub", "load_one", "mean"),
    "load_five": ("ganglia_pub", "load_five", "mean"),
    "load_fifteen": ("ganglia_pub", "load_fifteen", "mean"),
}

# Use one target sensor, power, and utilization metric to choose nodes that
# have overlapping data before reading every selected metric.
NODE_SELECTION_METRICS = [
    ("ipmi_pub", "p0_core20_temp"),
    ("ipmi_pub", "p1_core10_temp"),
    ("ipmi_pub", "total_power"),
    ("ganglia_pub", "cpu_user"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact M100 thermal telemetry subset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "Downloads" / "year_month=21-03",
        help="Extracted M100 month directory containing plugin=* folders.",
    )
    parser.add_argument(
        "--start",
        default="2021-03-03T00:00:00Z",
        help="Inclusive UTC start time (default: 2021-03-03).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to include (default: 7).",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=12,
        help="Number of high-coverage nodes to retain (default: 12).",
    )
    parser.add_argument(
        "--nodes",
        default="",
        help="Optional comma-separated node IDs; bypasses automatic selection.",
    )
    parser.add_argument(
        "--frequency",
        default="1min",
        help="Aggregation interval accepted by pandas (default: 1min).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Parquet rows per streaming batch (default: 250000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Downloads" / "m100-selected",
        help="Directory for the compact dataset and metadata.",
    )
    return parser.parse_args()


def metric_path(root: Path, plugin: str, metric: str) -> Path:
    path = root / f"plugin={plugin}" / f"metric={metric}" / "a_0.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Required metric file was not found: {path}")
    return path


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def stream_filtered_batches(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int,
    nodes: Optional[Sequence[str]] = None,
) -> Iterable[pd.DataFrame]:
    """Yield timestamp/value/node batches filtered by time and optional nodes."""
    parquet_file = pq.ParquetFile(path)
    allowed_nodes = set(str(node) for node in nodes) if nodes else None

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["timestamp", "value", "node"],
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue

        frame["timestamp"] = normalize_timestamp_series(frame["timestamp"])
        frame["node"] = frame["node"].astype(str)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

        mask = (
            frame["timestamp"].notna()
            & frame["value"].notna()
            & (frame["timestamp"] >= start)
            & (frame["timestamp"] < end)
        )

        if allowed_nodes is not None:
            mask &= frame["node"].isin(allowed_nodes)

        filtered = frame.loc[mask, ["timestamp", "value", "node"]]
        if not filtered.empty:
            yield filtered


def count_rows_by_node(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int,
) -> Counter:
    counts: Counter = Counter()
    for frame in stream_filtered_batches(path, start, end, batch_size):
        counts.update(frame["node"].value_counts().to_dict())
    return counts


def choose_nodes(
    root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int,
    node_count: int,
) -> Tuple[List[str], pd.DataFrame]:
    count_tables: List[pd.Series] = []

    print("Selecting nodes with overlapping temperature, power, and CPU data...")
    for plugin, metric in NODE_SELECTION_METRICS:
        path = metric_path(root, plugin, metric)
        print(f"  scanning {plugin}/{metric}")
        counts = count_rows_by_node(path, start, end, batch_size)
        series = pd.Series(counts, name=metric, dtype="int64")
        count_tables.append(series)

    if not count_tables:
        raise RuntimeError("No node-selection metrics were available.")

    coverage = pd.concat(count_tables, axis=1).fillna(0).astype("int64")
    coverage["minimum_rows"] = coverage.min(axis=1)
    coverage["total_rows"] = coverage.drop(
        columns=["minimum_rows"], errors="ignore"
    ).sum(axis=1)

    coverage = coverage.loc[coverage["minimum_rows"] > 0]
    coverage = coverage.sort_values(
        ["minimum_rows", "total_rows"], ascending=False
    )

    selected = coverage.head(node_count).index.astype(str).tolist()
    if not selected:
        raise RuntimeError(
            "No nodes had overlapping data for the selected period. "
            "Try a different --start date or a longer --days value."
        )

    print(f"Selected {len(selected)} nodes: {', '.join(selected)}")
    return selected, coverage


def aggregate_metric(
    path: Path,
    column_name: str,
    aggregation: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    nodes: Sequence[str],
    frequency: str,
    batch_size: int,
) -> pd.DataFrame:
    """Aggregate one metric by node and time interval without loading it all."""
    pieces: List[pd.DataFrame] = []

    for frame in stream_filtered_batches(
        path=path,
        start=start,
        end=end,
        batch_size=batch_size,
        nodes=nodes,
    ):
        frame = frame.copy()
        frame["timestamp"] = frame["timestamp"].dt.floor(frequency)
        grouped = (
            frame.groupby(["node", "timestamp"], as_index=False)["value"]
            .agg(aggregation)
            .rename(columns={"value": column_name})
        )
        pieces.append(grouped)

    if not pieces:
        return pd.DataFrame(columns=["node", "timestamp", column_name])

    combined = pd.concat(pieces, ignore_index=True)
    # A one-minute group can cross a Parquet batch boundary, so aggregate again.
    combined = (
        combined.groupby(["node", "timestamp"], as_index=False)[column_name]
        .agg(aggregation)
        .sort_values(["node", "timestamp"])
    )
    return combined


def merge_metric_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        raise RuntimeError("No telemetry rows were found for the selected period.")

    merged = non_empty[0]
    for frame in non_empty[1:]:
        merged = merged.merge(frame, on=["node", "timestamp"], how="outer")
    return merged.sort_values(["node", "timestamp"]).reset_index(drop=True)


def build_core_temperature_features(
    root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    nodes: Sequence[str],
    frequency: str,
    batch_size: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metric in CORE_TEMP_METRICS:
        print(f"  aggregating ipmi_pub/{metric}")
        frames.append(
            aggregate_metric(
                path=metric_path(root, "ipmi_pub", metric),
                column_name=metric,
                aggregation="mean",
                start=start,
                end=end,
                nodes=nodes,
                frequency=frequency,
                batch_size=batch_size,
            )
        )

    merged = merge_metric_frames(frames)
    temperature_columns = [
        metric for metric in CORE_TEMP_METRICS if metric in merged.columns
    ]
    merged["cpu_temp_mean"] = merged[temperature_columns].mean(axis=1)
    merged["cpu_temp_max"] = merged[temperature_columns].max(axis=1)
    merged["cpu_temp_sensor_count"] = merged[temperature_columns].notna().sum(axis=1)

    return merged[
        [
            "node",
            "timestamp",
            "cpu_temp_mean",
            "cpu_temp_max",
            "cpu_temp_sensor_count",
        ]
    ]


def add_complete_time_grid(
    data: pd.DataFrame,
    nodes: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=start,
        end=end,
        freq=frequency,
        inclusive="left",
        tz="UTC",
    )
    grid = pd.MultiIndex.from_product(
        [list(nodes), timestamps], names=["node", "timestamp"]
    ).to_frame(index=False)
    return grid.merge(data, on=["node", "timestamp"], how="left")


def add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.sort_values(["node", "timestamp"]).copy()

    if {"cpu_user", "cpu_system"}.issubset(result.columns):
        result["cpu_busy"] = result[["cpu_user", "cpu_system"]].sum(
            axis=1, min_count=1
        )

    if {"p0_power", "p1_power"}.issubset(result.columns):
        result["cpu_socket_power"] = result[["p0_power", "p1_power"]].sum(
            axis=1, min_count=1
        )

    result["hour_utc"] = result["timestamp"].dt.hour.astype("int16")
    result["day_of_week_utc"] = result["timestamp"].dt.dayofweek.astype("int8")

    return result


def summarize_data(data: pd.DataFrame) -> Dict[str, object]:
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    ranges: Dict[str, Dict[str, Optional[float]]] = {}

    for column in numeric_columns:
        series = pd.to_numeric(data[column], errors="coerce")
        non_null = series.dropna()
        ranges[column] = {
            "minimum": float(non_null.min()) if not non_null.empty else None,
            "maximum": float(non_null.max()) if not non_null.empty else None,
            "mean": float(non_null.mean()) if not non_null.empty else None,
        }

    missing_percent = {
        column: round(float(data[column].isna().mean() * 100.0), 3)
        for column in data.columns
    }

    return {
        "rows": int(len(data)),
        "columns": data.columns.tolist(),
        "nodes": sorted(data["node"].dropna().astype(str).unique().tolist()),
        "timestamp_min": (
            data["timestamp"].min().isoformat() if not data.empty else None
        ),
        "timestamp_max": (
            data["timestamp"].max().isoformat() if not data.empty else None
        ),
        "missing_percent": missing_percent,
        "numeric_ranges": ranges,
    }


def write_data_dictionary(path: Path) -> None:
    text = """# M100 selected telemetry dataset

This compact dataset was generated from the March 2021 M100 ExaData
partition. Raw source files are not included.

## Main columns

- `node`: anonymized M100 compute-node identifier.
- `timestamp`: UTC timestamp aggregated to one-minute intervals.
- `cpu_temp_mean`: mean of eight representative CPU-core temperature sensors.
- `cpu_temp_max`: maximum of those representative CPU-core sensors.
- `cpu_temp_sensor_count`: number of representative sensors available.
- `ambient`: node ambient temperature sensor.
- `p0_power`, `p1_power`: processor-socket power sensors.
- `total_power`: total node power sensor.
- `p0_vdd_temp`, `p1_vdd_temp`: socket power-delivery temperature sensors.
- `cpu_user`, `cpu_system`, `cpu_idle`, `cpu_wio`: Ganglia CPU percentages.
- `cpu_speed`: Ganglia CPU speed value. It may be nominal/static on some nodes.
- `load_one`, `load_five`, `load_fifteen`: operating-system load averages.
- `cpu_busy`: derived as `cpu_user + cpu_system`.
- `cpu_socket_power`: derived as `p0_power + p1_power`.
- `hour_utc`, `day_of_week_utc`: calendar features.

The final ML pipeline will create lag, rolling-window, and future-temperature
target columns after chronological splitting rules are defined.
"""
    path.write_text(text, encoding="utf-8")


def create_zip(output_dir: Path) -> Path:
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(output_dir.parent))

    return zip_path


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    start = pd.Timestamp(args.start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    end = start + pd.Timedelta(days=args.days)

    if args.days <= 0:
        raise ValueError("--days must be greater than zero.")
    if args.node_count <= 0:
        raise ValueError("--node-count must be greater than zero.")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if args.nodes.strip():
        selected_nodes = [
            item.strip() for item in args.nodes.split(",") if item.strip()
        ]
        coverage = pd.DataFrame(index=selected_nodes)
        print(f"Using explicitly selected nodes: {', '.join(selected_nodes)}")
    else:
        selected_nodes, coverage = choose_nodes(
            root=root,
            start=start,
            end=end,
            batch_size=args.batch_size,
            node_count=args.node_count,
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if not coverage.empty:
        coverage.to_csv(output_dir / "node-coverage.csv", index_label="node")

    print("Building representative CPU temperature features...")
    frames: List[pd.DataFrame] = [
        build_core_temperature_features(
            root=root,
            start=start,
            end=end,
            nodes=selected_nodes,
            frequency=args.frequency,
            batch_size=args.batch_size,
        )
    ]

    print("Aggregating power, ambient, CPU, and load metrics...")
    for column_name, (plugin, metric, aggregation) in SINGLE_METRICS.items():
        print(f"  aggregating {plugin}/{metric}")
        frames.append(
            aggregate_metric(
                path=metric_path(root, plugin, metric),
                column_name=column_name,
                aggregation=aggregation,
                start=start,
                end=end,
                nodes=selected_nodes,
                frequency=args.frequency,
                batch_size=args.batch_size,
            )
        )

    merged = merge_metric_frames(frames)
    merged = add_complete_time_grid(
        data=merged,
        nodes=selected_nodes,
        start=start,
        end=end,
        frequency=args.frequency,
    )
    merged = add_derived_columns(merged)

    # Keep rows with a usable temperature target. Other metric gaps are retained
    # and documented so the training pipeline can handle them explicitly.
    merged = merged.loc[merged["cpu_temp_max"].notna()].copy()
    merged = merged.sort_values(["node", "timestamp"]).reset_index(drop=True)

    if merged.empty:
        raise RuntimeError(
            "The selected period produced no temperature rows. "
            "Try a different --start date or --days value."
        )

    parquet_path = output_dir / "m100-thermal-subset.parquet"
    csv_path = output_dir / "m100-thermal-subset.csv.gz"
    merged.to_parquet(parquet_path, index=False, compression="snappy")
    merged.to_csv(csv_path, index=False, compression="gzip")

    summary = summarize_data(merged)
    summary.update(
        {
            "source_partition": root.name,
            "requested_start": start.isoformat(),
            "requested_end_exclusive": end.isoformat(),
            "frequency": args.frequency,
            "selected_core_temperature_metrics": CORE_TEMP_METRICS,
            "selected_single_metrics": {
                key: {"plugin": value[0], "metric": value[1]}
                for key, value in SINGLE_METRICS.items()
            },
            "selection_method": (
                "explicit --nodes" if args.nodes.strip() else "automatic overlap coverage"
            ),
        }
    )

    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_data_dictionary(output_dir / "README.md")

    zip_path = create_zip(output_dir)

    print()
    print("Subset creation completed successfully.")
    print(f"Rows written: {len(merged):,}")
    print(f"Nodes: {len(selected_nodes)}")
    print(f"Parquet: {parquet_path}")
    print(f"Compressed CSV: {csv_path}")
    print(f"Upload this ZIP: {zip_path}")
    print(f"ZIP size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
