# UK Car Analyser

A data pipeline and Bayesian survival model for predicting terminal mileage of UK vehicles using MOT test data.

## Overview

This project downloads, parses, and combines MOT test results from the [DVSA MOT History API](https://documentation.history.mot.api.gov.uk/) (trade tier) into queryable parquet files. A Bayesian Weibull survival model estimates terminal mileage per vehicle, and a Streamlit frontend provides a registration-based lookup interface.

## Architecture

```
MOT API (bulk + delta)
    |
    v
[fetch_mot_api_bulk.py]  [fetch_mot_api_delta.py]   -- download raw JSON.gz
    |                        |
    v                        v
[src/process_mot_bulk.py]  [src/process_mot_delta.py] -- flatten NDJSON, write parquet batches
    |                        |
    +----------+-------------+
               |
               v
    [combine_mot_bulk.py]  -- DuckDB: deduplicate, filter, produce mot_data_combined.parquet
               |
               v
    [prep_mot_data_surv.py] -- survival preprocessing: event labels, mileage estimation, engine buckets
               |
               v
    [normalise_mileage.py]  -- compute global + make-level mileage stats -> mileage_stats.json
               |
               v
    [run_bayesian_batches.py] -- orchestrator: batches data, calls Bayesian model
               |
               v
    [bayesian_model.py]     -- PyMC Weibull Censored model, warm-start across batches
               |
               v
    [predict_bayes_weib.py] -- apply posterior samples to mot_last_test.parquet -> terminal_predictions/
               |
               v
    [app/streamlit_app.py]  -- v1.0 frontend: registration lookup via DuckDB
```

## Data

| Component | Description |
|---|---|
| `data/mot_api_bulk/` | Raw bulk download zips from MOT History API |
| `data/mot_api_delta/` | Daily delta files from MOT History API |
| `data/mot_api_parquet/` | Flattened parquet batches (bulk + delta) |
| `data/mot_data_combined.parquet` | Deduplicated, filtered combined dataset |
| `data/mot_last_test.parquet` | Survival-preprocessed data (one row per vehicle) |
| `data/mileage_stats.json` | Global and per-make mileage statistics (prior anchors) |
| `data/model_state.json` | Posterior means/stds from Bayesian model |
| `data/model_trace_latest.nc` | ArviZ posterior trace (NetCDF) |
| `data/terminal_predictions/` | Per-vehicle terminal mileage predictions (250+ parquet files) |
| `data/processed_registrations.csv` | Registrations already processed by the Bayesian model |

## Model

**Bayesian Weibull Censored Model** (PyMC):

- **Likelihood**: Weibull survival model with censored observations (passed MOT tests are right-censored)
- **Parameters**:
  - `mu_global`: global log-mean terminal mileage (~125k miles)
  - `sigma_global`: global log-scale parameter
  - `beta_advisory`, `beta_dangerous`: defect count coefficients (log-mileage impact)
  - `make_model_effect`: per make/model offset, anchored by make-level statistics
  - `engine_effect`: per engine-size-bucket offset
  - `fuel_effect`: per fuel-type offset
- **Inference**: MCMC (NUTS via nutpie), 1000 draws + 1000 tune per batch
- **Warm-start**: Each batch's posterior means/stds become the prior for the next batch. `batch_number` tracks progress (currently 1).

**Prediction**: Posterior samples are applied to `mot_last_test.parquet` to produce per-vehicle terminal mileage (median, mean, P75) and remaining life estimates.

## Caveats

### Event definition

The model treats the **last MOT test** as a proxy for vehicle failure. This is imperfect:

- **True failures**: If a car truly failed an MOT, the owner may not have brought it back to a test centre at all. The last test is therefore a **left-censored** event -- the car may have failed earlier, and we only see the last test before it disappeared from the data.
- **Abandonment**: A car may have been scrapped, exported, or kept off-road without failing an MOT. The last test is then a **right-censored** observation that happens to be the final one in our data, not a failure event.
- **Repair decisions**: Some owners may have chosen not to repair expensive faults rather than bring the car back. In this case, the last test is partially informative -- the MOT faults were too costly to fix, but the car may have continued driving.

### Expired-pass proxy

When a car's last MOT test was a **pass** but has since expired (no subsequent test), we assume the car drove for an additional year after that pass and then failed, which is why it was never brought back to an MOT. This is a simplifying assumption -- it is not uncommon for people to drive (illegally) without a valid MOT. The model treats this as a failure event, which may overestimate terminal mileage for some vehicles.

### Model limitations

- **No hierarchical pooling**: Each make/model effect is independently estimated with its own prior (centred on the previous batch's result). Rare make/model combos with few observations have poorly estimated offsets that are just as influential as well-supported ones.
- **Sequential warm-start fragility**: The prior for batch N+1 is the posterior mean/std from batch N. If batch N had a skewed sample (e.g., over-representing Land Rovers), the prior for batch N+1 is biased.
- **Single batch**: With `batch_number: 1`, the model has only seen one batch. The priors for batch 2 will be the posteriors from batch 1, which may not have converged well if the batch was small or the data was noisy.

## Setup

```bash
# Install dependencies
uv sync

# Set environment variables (from DVSA registration email)
$env:MOT_CLIENT_ID="your-client-id"
$env:MOT_CLIENT_SECRET="your-client-secret"
$env:MOT_API_KEY="your-api-key"
$env:MOT_TOKEN_URL="https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/token"
```

## Usage

### 1. Fetch data

```bash
# Initial bulk download (53 GB, runs once)
uv run src/fetch_mot_api_bulk.py

# Daily delta update
uv run src/fetch_mot_api_delta.py
```

### 2. Process and combine

```bash
# Flatten bulk + delta into parquet
uv run src/process_mot_bulk.py
uv run src/process_mot_delta.py

# Combine, deduplicate, filter
uv run src/combine_mot_bulk.py

# Survival preprocessing
uv run src/prep_mot_data_surv.py

# Compute mileage statistics
uv run src/normalise_mileage.py
```

### 3. Train the Bayesian model

```bash
# Run batches (each batch updates posterior state)
uv run src/run_bayesian_batches.py --batch_size 50000 --num_batches 10
```

### 4. Generate predictions

```bash
uv run src/predict_bayes_weib.py --batch-size 10000000
```

### 5. Run the frontend

```bash
pip install streamlit duckdb
streamlit run app/streamlit_app.py
```

## Source files

| File | Purpose |
|---|---|
| `src/fetch_mot_api_bulk.py` | Downloads bulk zip from MOT History API |
| `src/fetch_mot_api_delta.py` | Downloads daily delta files |
| `src/process_mot_bulk.py` | Extracts bulk zip, flattens NDJSON, writes parquet batches |
| `src/process_mot_delta.py` | Extracts delta zips, preserves modification flags |
| `src/combine_mot_bulk.py` | DuckDB pipeline: deduplicate, filter, produce combined parquet |
| `src/prep_mot_data_surv.py` | Survival preprocessing: event labels, mileage estimation, engine buckets |
| `src/normalise_mileage.py` | Compute global and per-make mileage statistics |
| `src/encode_cat_vars.py` | Category ID mappings for make, model, fuel type, engine size |
| `src/bayesian_model.py` | PyMC Weibull Censored model, sequential warm-start |
| `src/run_bayesian_batches.py` | Orchestrator: batches data, runs model, saves state |
| `src/predict_bayes_weib.py` | Apply posterior to predict terminal mileage per vehicle |
| `app/streamlit_app.py` | v1.0 frontend: registration lookup |

## Planned work

- [ ] Add confidence intervals (P25/P90) to prediction output
- [ ] Batch registration upload for bulk lookups
- [ ] Interactive visualisation dashboard (Plotly)
- [ ] **Used car value calculator** -- user inputs price, current mileage, and registration; app calculates mean/median/P75 remaining miles per pound and ranks against similar vehicles
- [ ] **Used car listing ranking** -- scrape UK used car listings and rank them by mean/median/P75 remaining miles per pound
- [ ] **Delta application pipeline** -- apply daily delta files to the combined dataset with a scheduled job (e.g., GitHub Actions or cron)
- [ ] **Hierarchical pooling** -- add shared hyperpriors for make/model effects to shrink rare combos toward the global mean

## Tech stack

Python 3.11 · DuckDB · PyMC · nutpie (NUTS sampler) · ArviZ · Pandas · PyArrow · Streamlit · Matplotlib · Plotly · Seaborn
