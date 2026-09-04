# ML Thermal & Power Predictor

Machine learning based thermal prediction using real HPC system telemetry.

## Goal

Predict near-future CPU/system temperature from workload, power, frequency, and recent thermal history, then estimate thermal risk.

## Project status

The M100 March 2021 schema has been validated and a reproducible first subset builder is included. The initial model predicts the future `p0_core0_temp` reading for one Marconi100 node from current thermal, power, and utilization telemetry. See `data/README.md` for the exact raw metrics and selected window, and [`reports/first_experiment.md`](reports/first_experiment.md) for results and limitations.

## First experiment result

Using node `582`, 852 consecutive one-minute observations, and a chronological 80/20 split, the Random Forest baseline predicts CPU-core temperature 10 minutes ahead with:

| Metric | Result |
| --- | ---: |
| MAE | 1.263 °C |
| RMSE | 1.702 °C |
| R² | 0.160 |

This is a small proof-of-pipeline experiment, not a production thermal-control model. The next meaningful improvement is evaluating more continuous node windows and comparing against persistence, XGBoost, and time-aware validation folds.

## Data

This project is designed for the Marconi100 (M100) ExaData dataset. See `data/README.md` for attribution and preparation notes.

## Pipeline

```text
M100 Parquet metrics
     |
     v
One-minute alignment -> feature engineering -> time-aware train/test split
     |
     v
Random Forest baseline
     |
     +--> MAE / RMSE / R2
     |
     +--> Actual vs predicted temperature
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The processed dataset for the first experiment is included at `data/processed/m100_node582_first_window.csv`. To train and evaluate the baseline immediately:

```bash
python -m src.train --input data/processed/m100_node582_first_window.csv --target temperature_c --horizon 10
python -m src.evaluate --input data/processed/m100_node582_first_window.csv --target temperature_c
```

To rebuild the dataset from raw M100 Parquet telemetry (e.g. from an extracted `21-03` download folder or `data/raw/21-03`):

```bash
python -m src.build_m100_subset --raw-root /path/to/21-03
```

`--raw-root` defaults to `data/raw/21-03` and accepts either the directory containing `year_month=21-03` or the `year_month=21-03` directory directly.

## Tests

Run the test suite with pytest:

```bash
pytest
```

## Planned milestones

- [x] Validate M100 ExaData subset and exact columns
- [x] Establish Random Forest baseline
- [x] Add actual-vs-predicted evaluation plots
- [x] Add automated unit tests (`pytest`)
- [ ] Add XGBoost comparison
- [ ] Add thermal-risk classification
- [ ] Add inference CLI
- [ ] Add multi-node / multi-core evaluation
- [ ] Add lightweight dashboard

## Attribution

M100 ExaData: Marconi100 supercomputer telemetry dataset. See `data/README.md` and the dataset publication for full attribution and license details.
