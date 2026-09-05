from __future__ import annotations

import pytest

from src.metrics import event_proxy_metrics, regression_metrics


def test_regression_metrics() -> None:
    result = regression_metrics([1, 2, 3], [1, 3, 2], tolerances=(1,))
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx((2 / 3) ** 0.5)
    assert result["within_1"] == 1.0


def test_event_proxy_metrics() -> None:
    result = event_proxy_metrics([1, 5, 6, 2], [2, 6, 4, 1], threshold=5)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)
