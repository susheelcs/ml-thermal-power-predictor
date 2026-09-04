# ML Thermal & Power Predictor

Machine learning based thermal prediction using real HPC system telemetry.

## Goal

Predict near-future CPU/system temperature from workload, power, frequency, and recent thermal history, then estimate thermal risk.

## Project status

The M100 March 2021 schema has been validated and a reproducible first subset builder is included. The initial model predicts the future `p0_core0_temp` reading for one Marconi100 node from current thermal, power, and utilization telemetry. See `data/README.md` for the exact raw metrics and selected window.

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

Build the documented March 2021 subset from the locally extracted data, then train the baseline. Replace the raw-data path with your own location.

```bash
python -m src.build_m100_subset --raw-root /path/to/year_month=21-03
python -m src.train --input data/processed/m100_node582_first_window.csv --target temperature_c --horizon 10
```

The input CSV should contain a timestamp column plus numeric telemetry features. The training script creates a future-temperature target by shifting the temperature column by the requested number of samples.

## Planned milestones

- [x] Validate M100 ExaData subset and exact columns
- [ ] Establish Random Forest baseline
- [ ] Add XGBoost comparison
- [ ] Add thermal-risk classification
- [ ] Add actual-vs-predicted plots
- [ ] Add inference CLI
- [ ] Add tests and reproducible experiment configuration
- [ ] Add lightweight dashboard

## Attribution

M100 ExaData: Marconi100 supercomputer telemetry dataset. See `data/README.md` and the dataset publication for full attribution and license details.
