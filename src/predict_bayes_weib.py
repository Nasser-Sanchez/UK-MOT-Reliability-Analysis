import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import arviz as az
import argparse
import os

# Argument Parsing
parser = argparse.ArgumentParser(description="Bayesian Weibull Prediction")
parser.add_argument("--batch-size", type=int, default=10_000_000, help="Rows per batch")
parser.add_argument("--num-batches", type=int, default=None, help="Max new batches to process in this run (None = all)")
args = parser.parse_args()

# Configuration 
INPUT_FILE  = "data/mot_last_test.parquet"
STATE_FILE  = "data/model_state.json"
TRACE_FILE  = "data/model_trace_latest.nc"
OUTPUT_DIR  = "data/terminal_predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model State
with open(STATE_FILE) as f:
    state = json.load(f)

make_model_means = state["make_model_means"]
engine_means     = state["engine_means"]
fuel_means       = state["fuel_means"]

# Load Posterior Trace
trace = az.from_netcdf(TRACE_FILE)
samples = trace.posterior.stack(sample=("chain", "draw"))
n_samples = len(samples.sample)

mu_global_s = samples["mu_global"].values
sigma_s     = samples["sigma_global"].values
beta_adv_s  = samples["beta_advisory"].values
beta_dan_s  = samples["beta_dangerous"].values

k_median = np.log(-np.log(0.5))
k_p75    = np.log(-np.log(0.75))

# Streaming setup
pf = pq.ParquetFile(INPUT_FILE)
batches_processed_this_run = 0

for i, batch in enumerate(pf.iter_batches(batch_size=args.batch_size)):
    
    # Stop if we've reached the processing limit for this specific run
    if args.num_batches is not None and batches_processed_this_run >= args.num_batches:
        print(f"Reached limit of {args.num_batches} batches for this run.")
        break
        
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{i:04d}.parquet")
    
    # Check FIRST before doing any math
    if os.path.exists(batch_file):
        print(f"Skipping batch {i+1} (already exists)...")
        continue

    print(f"Processing batch {i+1}...")
    chunk = batch.to_pandas()

    # 1. Prepare categorical lookups 
    combined_key = chunk["make"] + "_" + chunk["model"]

    # 2. Linear predictor
    make_eff   = combined_key.map(make_model_means).fillna(0).values
    engine_eff = chunk["engineSize_bucket"].map(engine_means).fillna(0).values
    fuel_eff   = chunk["fuelType"].map(fuel_means).fillna(0).values

    adv_count = chunk["defect_count_advisory"].values
    dan_count = chunk["defect_count_dangerous"].values
    current_mileage = chunk["mileage"].astype(float).values

    base_mu = make_eff + engine_eff + fuel_eff

    beta_adv_term = np.outer(beta_adv_s, adv_count)
    beta_dan_term = np.outer(beta_dan_s, dan_count)

    mu_s = (np.outer(mu_global_s, np.ones(len(chunk)))
            + base_mu + beta_adv_term + beta_dan_term)

    # 3. Terminal Mileage
    log_terminal_median = mu_s + np.outer(sigma_s, np.ones(len(chunk))) * k_median
    terminal_median_samples = np.exp(log_terminal_median)

    log_terminal_p75 = mu_s + np.outer(sigma_s, np.ones(len(chunk))) * k_p75
    terminal_p75_samples = np.exp(log_terminal_p75)

    log_terminal_mean_samples = mu_s + (np.outer(sigma_s**2, np.ones(len(chunk))) / 2)
    terminal_mean_samples = np.exp(log_terminal_mean_samples)

    terminal_median = np.median(terminal_median_samples, axis=0)
    terminal_mean   = np.mean(terminal_mean_samples, axis=0)
    terminal_p75    = np.percentile(terminal_p75_samples, 75, axis=0)

    # 4. Build output
    out_chunk = pd.DataFrame({
        "registration":         chunk["registration"],
        "make":                 chunk["make"],
        "model":                chunk["model"],
        "fuelType":             chunk["fuelType"],
        "engineSize_bucket":    chunk["engineSize_bucket"],
        "defect_count_advisory": chunk["defect_count_advisory"],
        "defect_count_dangerous": chunk["defect_count_dangerous"],
        "current_mileage":      current_mileage,
        "terminal_median":      terminal_median,
        "terminal_mean":        terminal_mean,
        "terminal_p75":         terminal_p75,
        "remaining_life_median": terminal_median - current_mileage,
        "remaining_life_mean":   terminal_mean - current_mileage,
        "remaining_life_p75":    terminal_p75 - current_mileage,
    })

    # 5. Write to file
    out_chunk.to_parquet(batch_file, index=False)
    batches_processed_this_run += 1

print(f"Saved results to {OUTPUT_DIR}/")