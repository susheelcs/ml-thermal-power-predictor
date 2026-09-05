# Experiment report

## Objective

Forecast maximum representative CPU-core temperature and total node power
five minutes into the future using current and historical node telemetry.

## Data

- Source: M100 ExaData, March 2021 partition
- Processed rows: 10,068
- Nodes: 12
- Timestamp range: 2021-03-03T00:00:00+00:00 to 2021-03-03T13:58:00+00:00
- Sampling interval: one minute
- Duplicate node/timestamp rows: 0

The supervised feature table contains 9,888 rows and 137
features after requiring ten minutes of history and a valid five-minute future
target.

## Split

| Partition | Rows | Target-time boundary |
|---|---:|---|
| Train | 6,912 | through 2021-03-03T09:50:00+00:00 |
| Validation | 1,488 | through 2021-03-03T11:54:00+00:00 |
| Test | 1,488 | through 2021-03-03T13:58:00+00:00 |

Splits use target timestamps. Rows are not randomly shuffled.

## Validation model comparison

### Temperature

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Persistence | 1.203 | 2.045 | 0.956 |
| Ridge | 1.272 | 1.977 | 0.959 |
| Random Forest | 1.365 | 2.562 | 0.931 |
| Extra Trees | 1.546 | 3.245 | 0.889 |
| Hist Gradient Boosting | 1.335 | 2.582 | 0.930 |
| Xgboost | 1.712 | 3.845 | 0.845 |

### Power

| Model | MAE (W) | RMSE (W) | R² |
|---|---:|---:|---:|
| Persistence | 78.347 | 119.512 | 0.664 |
| Ridge | 73.017 | 106.019 | 0.736 |
| Random Forest | 79.077 | 122.188 | 0.649 |
| Extra Trees | 77.301 | 117.301 | 0.677 |
| Hist Gradient Boosting | 79.646 | 119.493 | 0.665 |
| Xgboost | 79.998 | 123.820 | 0.640 |


The learned model with the lowest validation RMSE was Ridge for both targets.

## Held-out test results

| Task | Model | MAE | RMSE | R² | RMSE reduction vs persistence |
|---|---|---:|---:|---:|---:|
| Temperature | Ridge | 1.186 °C | 2.022 °C | 0.921 | 11.0% |
| Power | Ridge | 57.428 W | 81.164 W | 0.816 | 21.7% |

### Temperature tolerance

- within ±1 °C: 60.48%
- within ±2 °C: 84.01%

### Power tolerance

- within ±25 W: 36.76%
- within ±50 W: 56.05%
- within ±100 W: 81.38%

## High-temperature proxy

Threshold: 69.333 °C, defined using the 90th percentile of
training plus validation targets.

| Precision | Recall | F1 | Accuracy | Positive test rows |
|---:|---:|---:|---:|---:|
| 0.788 | 0.558 | 0.653 | 0.907 | 233 |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[1220, 35], [103, 130]]`.

This is a statistical high-temperature proxy, not a physical thermal safety
limit.

## Interpretation

Temperature is strongly autocorrelated, and recent temperature history is the
main predictive signal. Power history and power-delivery temperatures also
contribute. For total node power, rolling node-power history is dominant.
Standardized Ridge coefficients are available in the feature-importance CSVs.

The model tracks gradual changes well but underestimates some abrupt thermal
transitions. That is expected for a five-minute horizon built from a short
sample and motivates broader training data and uncertainty estimation.

## Reproducibility

```bash
python -m src.train --task all
python -m src.evaluate --task temperature
python -m src.evaluate --task power
python -m pytest
```

The complete machine-readable output is in `reports/metrics.json`.
