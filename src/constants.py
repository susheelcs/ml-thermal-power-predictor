"""Shared configuration for feature engineering and prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_PATH = Path("data/processed/m100-thermal-subset.csv.gz")
DEFAULT_HORIZON_MINUTES = 5
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class TargetSpec:
    """Definition of one forecasting task."""

    key: str
    column: str
    model_filename: str
    display_name: str
    unit: str
    tolerances: tuple[float, ...]


TARGETS: dict[str, TargetSpec] = {
    "temperature": TargetSpec(
        key="temperature",
        column="cpu_temp_max",
        model_filename="temperature_5min.joblib",
        display_name="maximum CPU-core temperature",
        unit="degrees Celsius",
        tolerances=(1.0, 2.0),
    ),
    "power": TargetSpec(
        key="power",
        column="total_power",
        model_filename="power_5min.joblib",
        display_name="total node power",
        unit="watts",
        tolerances=(25.0, 50.0, 100.0),
    ),
}

REQUIRED_COLUMNS = {
    "node",
    "timestamp",
    "cpu_temp_mean",
    "cpu_temp_max",
    "cpu_temp_sensor_count",
    "ambient",
    "p0_power",
    "p1_power",
    "total_power",
    "p0_vdd_temp",
    "p1_vdd_temp",
    "cpu_user",
    "cpu_system",
    "cpu_idle",
    "cpu_wio",
    "cpu_speed",
    "load_one",
    "load_five",
    "load_fifteen",
    "cpu_busy",
    "cpu_socket_power",
    "hour_utc",
    "day_of_week_utc",
}

LAG_SIGNALS = (
    "cpu_temp_max",
    "cpu_temp_mean",
    "ambient",
    "total_power",
    "cpu_socket_power",
    "p0_power",
    "p1_power",
    "p0_vdd_temp",
    "p1_vdd_temp",
    "cpu_busy",
    "cpu_idle",
    "load_one",
    "load_five",
    "load_fifteen",
)
LAGS = (1, 2, 5, 10)

ROLLING_SIGNALS = (
    "cpu_temp_max",
    "cpu_temp_mean",
    "total_power",
    "cpu_socket_power",
    "cpu_busy",
    "load_one",
)
ROLLING_WINDOWS = (3, 5, 10)

DELTA_SIGNALS = (
    "cpu_temp_max",
    "cpu_temp_mean",
    "total_power",
    "cpu_socket_power",
    "cpu_busy",
    "load_one",
)

EXCLUDED_FEATURE_COLUMNS = {
    "node",
    "timestamp",
    "target",
    "target_timestamp",
    "hour_utc",
    "day_of_week_utc",
}
