"""Train, select, evaluate, and persist thermal/power forecasting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_HORIZON_MINUTES,
    DEFAULT_RANDOM_STATE,
    TARGETS,
    TargetSpec,
)
from .data import load_telemetry, quality_report
from .features import build_supervised
from .metrics import event_proxy_metrics, regression_metrics
from .modeling import candidate_models, software_versions
from .plotting import (
    feature_importance_frame,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    plot_validation_comparison,
)
from .splitting import chronological_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train five-minute M100 temperature and power forecasts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--task", choices=["temperature", "power", "all"], default="all"
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON_MINUTES
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def _task_output_name(spec: TargetSpec, horizon: int) -> str:
    return f"{spec.key}_{horizon}min"


def run_task(
    frame: pd.DataFrame,
    spec: TargetSpec,
    horizon: int,
    models_dir: Path,
    reports_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    dataset = build_supervised(frame, spec.column, horizon)
    split = chronological_split(dataset)

    validation_models: dict[str, dict[str, Any]] = {}
    test_models: dict[str, dict[str, Any]] = {}

    validation_persistence = split.validation.meta[spec.column].to_numpy()
    test_persistence = split.test.meta[spec.column].to_numpy()
    validation_models["persistence"] = regression_metrics(
        split.validation.y, validation_persistence, spec.tolerances
    )
    test_models["persistence"] = regression_metrics(
        split.test.y, test_persistence, spec.tolerances
    )

    selected = None
    selected_validation_rmse = float("inf")
    for candidate in candidate_models(random_state):
        candidate.pipeline.fit(split.train.X, split.train.y)
        prediction = candidate.pipeline.predict(split.validation.X)
        result = regression_metrics(
            split.validation.y, prediction, spec.tolerances
        )
        result["parameters"] = candidate.parameters
        validation_models[candidate.name] = result
        if result["rmse"] < selected_validation_rmse:
            selected_validation_rmse = result["rmse"]
            selected = candidate

    if selected is None:
        raise RuntimeError("No candidate model was available")

    train_validation_X = pd.concat(
        [split.train.X, split.validation.X], ignore_index=True
    )
    train_validation_y = pd.concat(
        [split.train.y, split.validation.y], ignore_index=True
    )
    selected.pipeline.fit(train_validation_X, train_validation_y)
    test_prediction = selected.pipeline.predict(split.test.X)
    selected_test_metrics = regression_metrics(
        split.test.y, test_prediction, spec.tolerances
    )
    test_models[selected.name] = selected_test_metrics

    task_name = _task_output_name(spec, horizon)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    high_temperature_proxy = None
    if spec.key == "temperature":
        threshold = float(train_validation_y.quantile(0.90))
        high_temperature_proxy = event_proxy_metrics(
            split.test.y, test_prediction, threshold
        )
        high_temperature_proxy["definition"] = (
            "Target at or above the 90th percentile of train+validation "
            "targets; this is an analytical proxy, not a hardware safety limit."
        )

    artifact = {
        "model": selected.pipeline,
        "feature_columns": list(dataset.X.columns),
        "task": spec.key,
        "target_column": spec.column,
        "display_name": spec.display_name,
        "unit": spec.unit,
        "horizon_minutes": horizon,
        "selected_model": selected.name,
        "selected_parameters": selected.parameters,
        "selection_metric": "validation_rmse",
        "train_target_end": split.train_target_end.isoformat(),
        "validation_target_end": split.validation_target_end.isoformat(),
        "high_temperature_threshold": (
            high_temperature_proxy["threshold"]
            if high_temperature_proxy is not None
            else None
        ),
        "software_versions": software_versions(),
    }
    model_path = models_dir / f"{task_name}.joblib"
    joblib.dump(artifact, model_path)

    prediction_frame = split.test.meta.copy()
    prediction_frame["actual"] = split.test.y
    prediction_frame["predicted"] = test_prediction
    prediction_frame["persistence"] = test_persistence
    prediction_frame["absolute_error"] = (
        prediction_frame["actual"] - prediction_frame["predicted"]
    ).abs()
    prediction_frame.to_csv(
        reports_dir / f"{task_name}_test_predictions.csv", index=False
    )

    representative_node = plot_actual_vs_predicted(
        split.test.meta,
        split.test.y,
        test_prediction,
        title=f"{horizon}-minute {spec.display_name} forecast",
        ylabel=spec.unit,
        output=figures_dir / f"{task_name}_actual_vs_predicted.png",
    )
    plot_residuals(
        split.test.y,
        test_prediction,
        title=f"{horizon}-minute {spec.display_name} forecast",
        unit=spec.unit,
        output=figures_dir / f"{task_name}_residuals.png",
    )
    plot_validation_comparison(
        validation_models,
        task_name=spec.display_name.title(),
        output=figures_dir / f"{task_name}_validation_models.png",
    )

    importance = feature_importance_frame(
        selected.pipeline, list(dataset.X.columns)
    )
    importance.to_csv(
        reports_dir / f"{task_name}_feature_importance.csv", index=False
    )
    plot_feature_importance(
        importance,
        title=f"{spec.display_name.title()} — top model signals",
        output=figures_dir / f"{task_name}_feature_importance.png",
    )

    return {
        "task": spec.key,
        "target": spec.column,
        "horizon_minutes": horizon,
        "unit": spec.unit,
        "rows": int(len(dataset.X)),
        "features": int(len(dataset.X.columns)),
        "split": {
            "train_rows": int(len(split.train.y)),
            "validation_rows": int(len(split.validation.y)),
            "test_rows": int(len(split.test.y)),
            "train_target_end": split.train_target_end.isoformat(),
            "validation_target_end": split.validation_target_end.isoformat(),
            "test_target_end": split.test.meta["target_timestamp"].max().isoformat(),
        },
        "selection_metric": "validation_rmse",
        "validation_models": validation_models,
        "selected_model": selected.name,
        "selected_parameters": selected.parameters,
        "test_models": test_models,
        "high_temperature_proxy": high_temperature_proxy,
        "representative_plot_node": representative_node,
        "model_path": str(model_path),
    }


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1")

    frame = load_telemetry(args.input)
    selected_tasks = list(TARGETS) if args.task == "all" else [args.task]
    results: dict[str, Any] = {
        "dataset": str(args.input),
        "data_quality": quality_report(frame),
        "software_versions": software_versions(),
        "experiments": {},
    }

    for task in selected_tasks:
        print(f"Training {task} forecast...")
        results["experiments"][task] = run_task(
            frame=frame,
            spec=TARGETS[task],
            horizon=args.horizon,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            random_state=args.random_state,
        )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for task, experiment in results["experiments"].items():
        for model_name, values in experiment["validation_models"].items():
            rows.append(
                {
                    "task": task,
                    "partition": "validation",
                    "model": model_name,
                    **{
                        key: value
                        for key, value in values.items()
                        if key != "parameters"
                    },
                }
            )
        for model_name, values in experiment["test_models"].items():
            rows.append(
                {
                    "task": task,
                    "partition": "test",
                    "model": model_name,
                    **values,
                }
            )
    pd.DataFrame(rows).to_csv(
        args.reports_dir / "model_comparison.csv", index=False
    )

    print(f"Saved metrics to {metrics_path}")
    for task, experiment in results["experiments"].items():
        selected = experiment["selected_model"]
        test = experiment["test_models"][selected]
        print(
            f"{task}: {selected}; MAE={test['mae']:.3f}, "
            f"RMSE={test['rmse']:.3f}, R2={test['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
