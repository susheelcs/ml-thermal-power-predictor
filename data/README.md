# Data

The planned source dataset is **M100 ExaData**, a production telemetry dataset from the Marconi100 supercomputer.

Dataset publication:
https://doi.org/10.1038/s41597-023-02174-3

Do not commit the full raw dataset to this repository. Store downloaded/raw data locally and commit only small samples or scripts/configuration needed to reproduce the preparation steps.

## March 2021 dataset (`21-03`)

The Marconi100 raw telemetry archive `21-03.tar` extracts to `year_month=21-03` containing 338 Parquet files across 5 metric plugins (~574 MB uncompressed / ~601 MB on disk).

### Available metric plugins and telemetry:

| Plugin | Metrics | Telemetry scope |
| --- | ---: | --- |
| `ipmi_pub` | 104 | CPU per-core temperatures (`p0_core0_temp`..`p0_core23_temp`, `p1_core0_temp`..`p1_core23_temp`), DIMM temperatures (`dimm0_temp`..`dimm15_temp`), GPU core & memory temperatures (`gpu0`..`gpu4`), node ambient temperature (`ambient`), VRM/VDD temperatures, socket and component power (`p0_power`, `p1_power`, `p0_io_power`, `p0_mem_power`), power supply inputs/voltages (`ps0_input_power`, `ps1_input_power`, `total_power`), and fan speeds (`fan0_0`..`fan3_1`). |
| `ganglia_pub` | 33 | CPU utilization (`cpu_user`, `cpu_idle`, `cpu_system`, `cpu_wio`), system load (`load_one`, `load_five`, `load_fifteen`), memory (`mem_total`, `mem_free`, `mem_cached`), network throughput (`bytes_in`, `bytes_out`), and CPU frequency (`cpu_speed`). |
| `logics_pub` | 36 | Electrical facility telemetry including currents (`Corrente_L1`..`L3`), power factor (`Fattore_di_potenza`), frequency, energy, and data center infrastructure efficiency (`Dcie`). |
| `schneider_pub` | 164 | Cooling plant and chiller subsystem metrics (chiller status, valve positions, pump running hours, inverter states, and system alarms). |
| `nagios_pub` | 1 | Node health monitoring status (`state`). |

Each raw Parquet file has exactly three fields: `timestamp` (UTC), `value` (numeric telemetry reading), and `node` (node identifier string). In `ganglia_pub/metric=cpu_speed`, the frequency reading is fixed at 3800 MHz across the inspected period and is documented rather than used as a varying feature.

### First experiment subset

The initial baseline models CPU-core thermal behavior on node `582` over the continuous window `2021-03-01 20:18` through `2021-03-02 10:29 UTC`, averaged into 1-minute intervals (852 complete observations).

Selected raw metrics:
- Target & thermal history: `ipmi_pub/p0_core0_temp` (`temperature_c`)
- Cross-socket thermal context: `ipmi_pub/p1_core0_temp` (`socket1_temperature_c`)
- Socket power: `ipmi_pub/p0_power` (`socket0_power_w`), `ipmi_pub/p1_power` (`socket1_power_w`), and their sum (`total_cpu_power_w`)
- CPU utilization & load: `ganglia_pub/cpu_user`, `ganglia_pub/cpu_idle`, `ganglia_pub/load_one`

### Storage and reproducibility

- **Raw dataset**: Keep the extracted `year_month=21-03` locally (e.g. in `data/raw/21-03/` or any external folder like `C:\Users\Admin\Downloads\21-03`). Raw data files are Git-ignored to prevent repository bloat.
- **Processed dataset**: The lightweight (108 KB) processed CSV for the first experiment is committed at `data/processed/m100_node582_first_window.csv` to allow immediate model training and evaluation without needing the full 600 MB raw download.
- **Rebuilding**: To rebuild the processed CSV from the raw dataset:
  ```bash
  python -m src.build_m100_subset --raw-root /path/to/21-03
  ```
  `--raw-root` defaults to `data/raw/21-03` and accepts either the directory containing `year_month=21-03` or the `year_month=21-03` directory directly.

