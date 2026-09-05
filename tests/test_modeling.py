from __future__ import annotations

from src.modeling import candidate_models


def test_expected_candidate_models_are_available() -> None:
    names = {candidate.name for candidate in candidate_models()}
    assert names == {
        "ridge",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
    }
