# Data license and attribution

The Python source code in this repository is licensed under the MIT License.
The included files under `data/processed/` are a filtered and transformed
subset of **M100 ExaData** and are not relicensed under MIT.

The source dataset is made available under **Creative Commons Attribution 4.0
International (CC BY 4.0)**. Appropriate credit must be preserved when the
derived subset is reused.

## Source

Borghesi, A., Di Santi, C., Molan, M. et al. *M100 ExaData: a data collection
campaign on the CINECA's Marconi100 Tier-0 supercomputer.* Scientific Data 10,
288 (2023). https://doi.org/10.1038/s41597-023-02174-3

March 2021 archive: https://doi.org/10.5281/zenodo.7589131

## Changes made in this repository

The included data was:

1. restricted to selected anonymized compute nodes;
2. restricted to selected IPMI and Ganglia temperature, power, utilization,
   and load metrics;
3. aggregated to one-minute intervals;
4. aligned by node and UTC timestamp;
5. augmented with derived mean/max CPU temperature and socket-power columns;
6. exported as Parquet and gzip-compressed CSV.
