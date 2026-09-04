import pandas as pd

from src.build_m100_subset import build_subset


def test_build_subset_adds_total_power(monkeypatch, tmp_path) -> None:
    index = pd.date_range("2021-03-01", periods=3, freq="min", tz="UTC")

    def fake_read_metric(_root, _plugin, metric, _node):
        values = {
            "p0_core0_temp": [50, 51, 52], "p1_core0_temp": [49, 50, 51],
            "p0_power": [100, 101, 102], "p1_power": [90, 91, 92],
            "cpu_user": [50, 51, 52], "cpu_idle": [49, 48, 47], "load_one": [10, 11, 12],
        }
        return pd.Series(values[metric], index=index)

    monkeypatch.setattr("src.build_m100_subset.read_metric", fake_read_metric)
    result = build_subset(tmp_path, "42", "2021-03-01T00:00:00Z", "2021-03-01T00:02:00Z")
    assert result["node"].eq("42").all()
    assert result["total_cpu_power_w"].tolist() == [190, 192, 194]
