# Data and attribution

## Included processed subset

`data/processed/` contains a compact, derived subset of the public M100
ExaData telemetry:

- 10,068 one-minute records
- 12 anonymized compute nodes
- 839 continuous minutes per node
- 2021-03-03 00:00 through 13:58 UTC
- CPU-core temperature, ambient temperature, power, utilization, and load

Both Parquet and gzip-compressed CSV versions are included. They represent the
same rows. The project defaults to the compressed CSV for portability.

## Original dataset

Borghesi, A., Di Santi, C., Molan, M. et al. *M100 ExaData: a data collection
campaign on the CINECA's Marconi100 Tier-0 supercomputer.* Scientific Data 10,
288 (2023). DOI: `10.1038/s41597-023-02174-3`.

March 2021 source archive: Zenodo record `10.5281/zenodo.7589131`, file
`21-03.tar`.

The original dataset is licensed under Creative Commons Attribution 4.0. The
processed subset is redistributed with attribution and an indication that it
was filtered, aggregated to one-minute intervals, and reduced to selected
metrics/nodes. Project code is separately licensed under MIT.

## Rebuild the subset

```bash
python scripts/prepare_m100_subset.py \
  --root data/raw/year_month=21-03 \
  --start "2021-03-03T00:00:00Z" \
  --days 7 \
  --node-count 12 \
  --output-dir data/processed/rebuilt
```

The source interval requested was seven days, but the selected overlapping
telemetry in the generated subset contains 839 continuous one-minute samples
per retained node.
