# Resume wording

## Recommended concise bullet

> Built a time-series machine-learning pipeline on public M100 HPC telemetry
to forecast CPU temperature and node power five minutes ahead across 12
compute nodes; achieved **1.19 °C temperature MAE (R²=0.921)**
and **57.4 W power MAE (R²=0.816)** on a chronological
held-out test set.

## Impact-oriented alternative

> Engineered lag, rolling-window, utilization, and power features from 10,068
M100 telemetry records and compared five ML regressors against persistence;
reduced test RMSE by **11.0% for temperature** and
**21.7% for node power** using validation-selected Ridge
models.

Do not describe the project as production-ready or claim unseen-node
performance; the experiment uses 12 nodes over approximately 14 hours.
