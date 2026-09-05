from __future__ import annotations

import pandas as pd

from src.features import build_supervised
from src.splitting import chronological_split


def test_chronological_split_has_no_target_overlap(
    synthetic_telemetry: pd.DataFrame,
) -> None:
    dataset = build_supervised(synthetic_telemetry, "cpu_temp_max", 5)
    split = chronological_split(dataset)
    assert split.train.meta["target_timestamp"].max() <= split.train_target_end
    assert split.validation.meta["target_timestamp"].min() > split.train_target_end
    assert (
        split.test.meta["target_timestamp"].min()
        > split.validation_target_end
    )
    assert len(split.train.y) + len(split.validation.y) + len(split.test.y) == len(
        dataset.y
    )
