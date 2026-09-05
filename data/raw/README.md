# Raw M100 data

The original `21-03.tar` archive is intentionally not committed. Download it
from Zenodo record `10.5281/zenodo.7589131` and extract it locally when you
want to rebuild the compact subset.

Expected local layout:

```text
data/raw/year_month=21-03/
├── plugin=ganglia_pub/
├── plugin=ipmi_pub/
├── plugin=logics_pub/
├── plugin=nagios_pub/
└── plugin=schneider_pub/
```

To recreate the processed subset, use `scripts/prepare_m100_subset.py` and
follow `data/README.md`.
