# Processed telemetry files

This directory contains the compact dataset used by the checked-in models and
reported experiments.

- `m100-thermal-subset.csv.gz`: default, portable training input.
- `m100-thermal-subset.parquet`: equivalent columnar representation generated
  by the subset-preparation workflow.
- `metadata.json`: selected metrics, ranges, and missingness.
- `node-coverage.csv`: source-node coverage used during subset selection.
- `SHA256SUMS.txt`: integrity checksums for the processed data artifacts.

The source `data/raw/` directory is intentionally empty except for its README.
The original March 2021 archive is roughly 602 MB and should not be committed
to this repository. The small processed subset required to reproduce the ML
experiment is included here.

The subset contains derived data from M100 ExaData. See `../README.md` and
`../../DATA_LICENSE.md` for citation, transformations, and license information.
