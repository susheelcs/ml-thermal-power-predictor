### ML Thermal & Power Predictor

Built a reproducible time-series ML pipeline using public M100 supercomputer
telemetry to forecast maximum CPU temperature and total node power five minutes
ahead.

- **Temperature:** 1.19 °C MAE, 2.02 °C RMSE, R² 0.921
- **Power:** 57.4 W MAE, 81.2 W RMSE, R² 0.816
- Used chronological train/validation/test splits, persistence baselines,
  lag/rolling features, automated tests, and reproducible CLI workflows.

[View the project](https://github.com/susheelcs/ml-thermal-power-predictor)
