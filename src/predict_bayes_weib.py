import numpy as np
import pandas as pd
import json

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_parquet("data/mot_last_test.parquet")

# ── Load model state ───────────────────────────────────────────────────
with open("data/model_state.json") as f:
    state = json.load(f)

mu_global      = state["global_params"]["mu_global"]
sigma_global   = state["global_params"]["sigma_global"]
beta_advisory  = state["global_params"]["beta_advisory"]
beta_dangerous = state["global_params"]["beta_dangerous"]

make_means     = state["make_means"]
model_means    = state["model_means"]
engine_means   = state["engine_means"]
fuel_means     = state["fuel_means"]

# ── Look up effects for each row ───────────────────────────────────────
make_eff  = df["make"].map(make_means).fillna(0).values
model_eff = df["model"].map(model_means).fillna(0).values
engine_eff = df["engineSize_bucket"].map(engine_means).fillna(0).values
fuel_eff  = df["fuelType"].map(fuel_means).fillna(0).values

# ── Linear predictor (in log-space) ────────────────────────────────────
mu = (
    mu_global
    + make_eff
    + model_eff
    + engine_eff
    + fuel_eff
    + beta_advisory * df["defect_count_advisory"].values
    + beta_dangerous * df["defect_count_dangerous"].values
)

# ── Weibull quantiles in log-space ─────────────────────────────────────
# log(T_p) = mu + sigma * log(-log(1 - p))
log_median   = mu + sigma_global * np.log(-np.log(0.5))   # p = 0.5
log_p75      = mu + sigma_global * np.log(-np.log(0.75))  # p = 0.75

# ── Convert to mileage predictions ─────────────────────────────────────
# These are remaining mileage from last MOT
remaining_median = np.exp(log_median)
remaining_p75    = np.exp(log_p75)

# Terminal mileage = current mileage + remaining
terminal_median = df["mileage"].astype(float) + remaining_median
terminal_p75    = df["mileage"].astype(float) + remaining_p75

# ── Build output ───────────────────────────────────────────────────────
out = pd.DataFrame({
    "registration": df["registration"],
    "make": df["make"],
    "model": df["model"],
    "fuelType": df["fuelType"],
    "engineSize_bucket": df["engineSize_bucket"],
    "defect_count_advisory": df["defect_count_advisory"],
    "defect_count_dangerous": df["defect_count_dangerous"],
    "current_mileage": df["mileage"].astype(float),
    "remaining_mileage_median": remaining_median,
    "remaining_mileage_p75": remaining_p75,
    "terminal_mileage_median": terminal_median,
    "terminal_mileage_p75": terminal_p75,
})

out.to_parquet("data/terminal_predictions.parquet", index=False)
print(f"Saved {len(out)} rows to data/terminal_predictions.parquet")
print(out[["make", "model", "current_mileage", "terminal_mileage_median", "terminal_mileage_p75"]].describe())
