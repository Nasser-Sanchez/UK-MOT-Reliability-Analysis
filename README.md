# UK Car Analyser

A data pipeline and exploratory analysis project for UK MOT (Ministry of Transport) test data.

## Overview

This project downloads, parses, and combines MOT test results from the [DVSA MOT History API](https://documentation.history.mot.api.gov.uk/) (trade tier) into queryable parquet files. The goal is to build a reusable dataset for vehicle reliability analysis, survival modelling, and predictive modelling of MOT pass/fail outcomes.

## Planned work

- [ ] Survival analysis — time-to-MOT-failure by vehicle cohort
- [ ] **Used car listings extension** — scrape UK used car data to predict remaining vehicle life (miles/years to failure) based on make, model, year, and current mileage
- [ ] **Cost-benefit analysis** — compare used car price to new/historical MSRP, and evaluate cost per remaining miles/years vs buying new
- [ ] Interactive visualisation dashboard (Plotly)

## Data source

| Component | Description |
|---|---|
| `src/fetch_mot_bulk.py` | Downloads the weekly bulk file from the MOT History API |
| `src/fetch_mot_delta.py` | Downloads daily delta files (re-authenticates for fresh S3 URLs) |
| `src/process_mot_api.py` | Extracts `.json.gz` from bulk/delta zips, flattens `motTests` arrays, writes parquet |

## Data

- **Source**: DVSA MOT History API (trade tier) — bulk + daily delta files
- **Fields**: `registration`, `firstUsedDate`, `registrationDate`, `manufactureDate`, `make`, `model`, `fuelType`, `engineSize`, `test_completedDate`, `test_testResult`, `test_odometerValue`, `test_odometerUnit`, `test_motTestNumber`, `test_dataSource`, `test_expiryDate`, `test_registrationAtTimeOfTest`, `defects`
- **Storage**: Parquet files in `data/mot_api_parquet/` (excluded from git)
- **Combined output**: `data/mot_data_combined.parquet` (legacy, from DfT/DVSA open data)

## Setup

```bash
# Install dependencies
uv sync

# Set environment variables (from DVSA registration email)
$env:MOT_CLIENT_ID="your-client-id"
$env:MOT_CLIENT_SECRET="your-client-secret"
$env:MOT_API_KEY="your-api-key"
$env:MOT_TOKEN_URL="https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/token"

# Initial bulk download (53 GB, runs once)
uv run src/fetch_mot_bulk.py

# Process bulk + delta files into parquet
uv run src/process_mot_api.py

# Daily update (run each morning)
uv run src/fetch_mot_delta.py
uv run src/process_mot_api.py
```

## Tech stack

Python · DuckDB · Polars · scikit-learn · LightGBM · XGBoost · Matplotlib · Plotly · Seaborn
