"""Candidate forecasting models and artifact helpers."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


@dataclass(frozen=True)
class Candidate:
    name: str
    pipeline: Pipeline
    parameters: dict[str, Any]


def candidate_models(random_state: int = 42) -> list[Candidate]:
    """Small validation-guided comparison designed to run in under a minute."""

    return [
        Candidate(
            name="ridge",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=100.0)),
                ]
            ),
            parameters={"alpha": 100.0},
        ),
        Candidate(
            name="random_forest",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=120,
                            max_depth=14,
                            min_samples_leaf=2,
                            max_features=0.7,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            parameters={
                "n_estimators": 120,
                "max_depth": 14,
                "min_samples_leaf": 2,
                "max_features": 0.7,
            },
        ),
        Candidate(
            name="extra_trees",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=120,
                            min_samples_leaf=2,
                            max_features=0.7,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            parameters={
                "n_estimators": 120,
                "max_depth": None,
                "min_samples_leaf": 2,
                "max_features": 0.7,
            },
        ),
        Candidate(
            name="hist_gradient_boosting",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.05,
                            max_iter=250,
                            max_leaf_nodes=31,
                            l2_regularization=1.0,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            parameters={
                "learning_rate": 0.05,
                "max_iter": 250,
                "max_leaf_nodes": 31,
                "l2_regularization": 1.0,
            },
        ),
        Candidate(
            name="xgboost",
            pipeline=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=400,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=4,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            parameters={
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
    ]


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }
