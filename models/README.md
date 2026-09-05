# Trained models

The checked-in `.joblib` artifacts are the selected Ridge pipelines trained on
the combined training and validation periods:

- `temperature_5min.joblib`
- `power_5min.joblib`

Each artifact contains the fitted imputer, scaler, Ridge model, expected
feature columns, forecast horizon, split boundaries, and software versions.
Because Python model serialization is version-sensitive, reinstall the
versions allowed by `requirements.txt` or retrain locally if loading fails.
