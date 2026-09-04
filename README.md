# ML Thermal & Power Predictor

Machine learning based thermal prediction using real HPC system telemetry.

## Goal

Predict near-future CPU/system temperature from workload, power, frequency, and recent thermal history, then estimate thermal risk.

## Project status

Initial project scaffold. The first milestone uses a CSV telemetry file and a Random Forest baseline. Dataset-specific column mapping will be added after selecting a subset of the M100 ExaData dataset.

## Data

This project is designed for the Marconi100 (M100) ExaData dataset. See `data/README.md` for attribution and preparation notes.

## Pipeline

```text
Telemetry CSV
     |
     v
Preprocessing -> feature engineering -> time-aware train/test split
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

Place a prepared telemetry CSV at `data/telemetry.csv` and run:

```bash
python -m src.train --input data/telemetry.csv --target temperature --horizon 10
```

The input CSV should contain a timestamp column plus numeric telemetry features. The training script creates a future-temperature target by shifting the temperature column by the requested number of samples.

## Planned milestones

- [ ] Validate M100 ExaData subset and exact columns
- [ ] Establish Random Forest baseline
- [ ] Add XGBoost comparison
- [ ] Add thermal-risk classification
- [ ] Add actual-vs-predicted plots
- [ ] Add inference CLI
- [ ] Add tests and reproducible experiment configuration
- [ ] Add lightweight dashboard

## Attribution

M100 ExaData: Marconi100 supercomputer telemetry dataset. See `data/README.md` and the dataset publication for full attribution and license details.
