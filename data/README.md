# Data

The planned source dataset is **M100 ExaData**, a production telemetry dataset from the Marconi100 supercomputer.

Dataset publication:
https://doi.org/10.1038/s41597-023-02174-3

Do not commit the full raw dataset to this repository. Store downloaded/raw data locally and commit only small samples or scripts/configuration needed to reproduce the preparation steps.

## First experiment: March 2021

The first experiment uses the extracted `year_month=21-03` directory from `21-03.tar`; the full archive is not stored in Git. The raw files are `ipmi_pub` metrics `p0_core0_temp`, `p1_core0_temp`, `p0_power`, and `p1_power`, plus `ganglia_pub` metrics `cpu_user`, `cpu_idle`, and `load_one`.

Each selected Parquet file has exactly three fields: `timestamp` (UTC), `value`, and `node` (a string node identifier). `ganglia_pub/metric=cpu_speed` is the available CPU-frequency field, but it is fixed at 3800 MHz in the inspected records, so it is documented rather than used as a first-model feature.

The default builder selects node `582`, the continuous interval `2021-03-01 20:18` through `2021-03-02 10:29 UTC`, averages readings to one-minute means, and retains 852 complete observations. Its generated, Git-ignored CSV has `timestamp`, `node`, `temperature_c`, `socket1_temperature_c`, `socket0_power_w`, `socket1_power_w`, `cpu_user_pct`, `cpu_idle_pct`, `load_1m`, and `total_cpu_power_w` columns.
