from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_telemetry() -> pd.DataFrame:
    rows = []
    for node_index, node in enumerate(("101", "202")):
        for minute, timestamp in enumerate(
            pd.date_range("2021-01-01", periods=40, freq="min", tz="UTC")
        ):
            temperature = 45.0 + node_index * 5 + minute * 0.1
            p0_power = 80.0 + minute
            p1_power = 75.0 + minute * 0.8
            rows.append(
                {
                    "node": node,
                    "timestamp": timestamp,
                    "cpu_temp_mean": temperature - 1.0,
                    "cpu_temp_max": temperature,
                    "cpu_temp_sensor_count": 8.0,
                    "ambient": 22.0,
                    "p0_power": p0_power,
                    "p1_power": p1_power,
                    "total_power": 400.0 + minute * 2,
                    "p0_vdd_temp": 35.0 + minute * 0.02,
                    "p1_vdd_temp": 34.0 + minute * 0.02,
                    "cpu_user": 30.0 + minute * 0.1,
                    "cpu_system": 1.0,
                    "cpu_idle": 69.0 - minute * 0.1,
                    "cpu_wio": 0.0,
                    "cpu_speed": 3800.0,
                    "load_one": 20.0 + minute * 0.2,
                    "load_five": 19.0 + minute * 0.2,
                    "load_fifteen": 18.0 + minute * 0.2,
                    "cpu_busy": 31.0 + minute * 0.1,
                    "cpu_socket_power": p0_power + p1_power,
                    "hour_utc": timestamp.hour,
                    "day_of_week_utc": timestamp.dayofweek,
                }
            )
    return pd.DataFrame(rows)
