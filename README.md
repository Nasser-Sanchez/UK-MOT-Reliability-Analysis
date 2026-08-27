# UK Car Analyser

A data pipeline and exploratory analysis project for UK MOT (Ministry of Transport) test data.

## Overview

This project downloads, parses, and combines anonymised MOT test results from [DfT Open Data](https://data.dft.gov.uk/anonymised-mot-test/test_data/) and [DVSA Early Data Hub](https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/) (2005–2025) into queryable parquet files. The goal is to build a reusable dataset for vehicle reliability analysis, survival modelling, and predictive modelling of MOT pass/fail outcomes.

## What's here

| Component | Description |
|---|---|
| `src/fetch_mot_data.py` | Downloads and extracts yearly MOT result files from DfT/DVSA sources |
| `src/write_mot_data_to_parquet.py` | Converts raw TXT/CSV files into per-year parquet files via DuckDB |
| `src/combine_mot_data.py` | Filters and merges all yearly parquet files into a single combined dataset |
| `src/uk_car_analyser/` | Python package (entry point: `uk-car-analyser`) |

## Data

- **Source**: DfT (2005–2023) and DVSA EDH (2024–2025) — anonymised vehicle-level MOT test records
- **Fields**: `vehicle_id`, `first_use_date`, `test_date`, `test_type`, `test_result`, `test_mileage`, `make`, `model`, `fuel_type`, `cylinder_capacity`
- **Storage**: Parquet files in `data/` (excluded from git)
- **Combined output**: `data/mot_data_combined.parquet`

## Setup

```bash
# Install dependencies
uv sync

# Fetch raw data (downloads ~years of MOT results)
uv run src/fetch_mot_data.py

# Convert to parquet
uv run src/write_mot_data_to_parquet.py

# Combine into a single dataset
uv run src/combine_mot_data.py
```

**Prerequisites**: 7-Zip must be installed and on PATH (used by `fetch_mot_data.py` for zip extraction).

## Planned work

- [ ] Survival analysis — time-to-MOT-failure by vehicle cohort
- [ ] **Used car listings extension** — scrape UK used car data to predict remaining vehicle life (miles/years to failure) based on make, model, year, and current mileage
- [ ] **Cost-benefit analysis** — compare used car price to new/historical MSRP, and evaluate cost per remaining miles/years vs buying new
- [ ] Interactive visualisation dashboard (Plotly)

## Tech stack

Python · DuckDB · Polars · scikit-learn · LightGBM · XGBoost · Matplotlib · Plotly · Seaborn
