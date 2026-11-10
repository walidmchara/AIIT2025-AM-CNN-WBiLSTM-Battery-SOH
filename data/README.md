# NASA PCoE battery-aging records

The dataset directory contains discharge-cycle aggregates for the two NASA battery groups used in the AIIT 2025 study. Each CSV includes measurements of voltage, current, temperature, discharge duration and cell capacity. The `data/processed/manifest.json` file records record counts and SHA-256 fingerprints of the eight included CSV files.

| Experimental group | Model-development cells | Held-out test cell | Ambient temperature |
| --- | --- | --- | --- |
| Group I | B0005, B0006, B0007 | B0018 | 24 °C |
| Group II | B0029, B0031, B0032 | B0030 | 43 °C |

The paper occasionally uses the `A` prefix for the battery IDs; the corresponding NASA files use `B` identifiers. The original MATLAB records provide richer within-cycle traces. The CSV data are per-discharge-cycle summaries and should not be confused with the complete source waveform files.

SOH is calculated from a cell's capacity relative to its **first observed discharge**. Capacity is used only to construct the target; the predictor windows contain earlier cycles' electrical and thermal measurements. Both preprocessing statistics and validation partitions are derived exclusively from the development batteries. The held-out battery remains independent until final evaluation.

Source: [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The conference article is identified by [DOI 10.1109/AIIT63112.2025.11082918](https://doi.org/10.1109/AIIT63112.2025.11082918).
