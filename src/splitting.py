"""Chronological train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import FeatureDataset


@dataclass(frozen=True)
class DataPart:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


@dataclass(frozen=True)
class ChronologicalSplit:
    train: DataPart
    validation: DataPart
    test: DataPart
    train_target_end: pd.Timestamp
    validation_target_end: pd.Timestamp


def chronological_split(
    dataset: FeatureDataset,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> ChronologicalSplit:
    """Split by target timestamp so future labels never leak backward."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")

    timestamps = pd.Index(dataset.meta["target_timestamp"].sort_values().unique())
    if len(timestamps) < 3:
        raise ValueError("At least three unique target timestamps are required")

    train_position = max(0, int(len(timestamps) * train_ratio) - 1)
    validation_position = max(
        train_position + 1,
        int(len(timestamps) * (train_ratio + validation_ratio)) - 1,
    )
    validation_position = min(validation_position, len(timestamps) - 2)

    train_end = pd.Timestamp(timestamps[train_position])
    validation_end = pd.Timestamp(timestamps[validation_position])
    train_mask = dataset.meta["target_timestamp"] <= train_end
    validation_mask = (
        (dataset.meta["target_timestamp"] > train_end)
        & (dataset.meta["target_timestamp"] <= validation_end)
    )
    test_mask = dataset.meta["target_timestamp"] > validation_end

    def take(mask: pd.Series) -> DataPart:
        return DataPart(
            X=dataset.X.loc[mask].reset_index(drop=True),
            y=dataset.y.loc[mask].reset_index(drop=True),
            meta=dataset.meta.loc[mask].reset_index(drop=True),
        )

    split = ChronologicalSplit(
        train=take(train_mask),
        validation=take(validation_mask),
        test=take(test_mask),
        train_target_end=train_end,
        validation_target_end=validation_end,
    )
    if min(len(split.train.y), len(split.validation.y), len(split.test.y)) == 0:
        raise ValueError("Chronological split produced an empty partition")
    return split
