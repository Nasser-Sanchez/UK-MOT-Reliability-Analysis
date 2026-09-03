import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import json
import duckdb
from src.encode_cat_vars import compute_and_save_mappings, load_mappings, encode_dataframe

# encoding for categorical variables
mappings = compute_and_save_mappings()

STATS_PATH = "data/mileage_stats.json"
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
    
global_mean_mileage = float(stats['mean_mileage'])
global_std_mileage = float(stats['std_mileage'])

# Calculate priors
# We use the log of the global mean as the center for the prior
prior_mu = np.log(global_mean_mileage)
# Use the log of the global SD as the center for the shape parameter (sigma)
prior_sigma = np.log(global_std_mileage) 

#get last mot test data
con = duckdb.connect("uk_car_analyser.duckdb")
df = con.execute( """
WITH valid_makes AS (
    -- 1. Identify makes with sufficient data
    SELECT make
    FROM 'data/mot_last_test.parquet'
    GROUP BY make
    HAVING COUNT(DISTINCT registration) > 10000
)
-- 2. Select only rows belonging to those valid makes
SELECT
    make, model, fuelType, engineSize_bucket,
    defect_count_advisory, defect_count_dangerous,
    mileage_estimate, event_interval
FROM 'data/mot_last_test.parquet'
WHERE make IN (SELECT make FROM valid_makes)
LIMIT 50000
""").df()

# Preprocessing data
# code categorical variables for pymc 
df = encode_dataframe(df, mappings)

# Continuous 
df['log_mileage'] = np.log(df['mileage_estimate'])


make_ids = df['make_id'].values
model_ids = df['model_id'].values
engineSize_bucket_ids = df['engineSize_bucket_id'].values
fuelType_ids = df['fuelType_id'].values
defect_advisory = df['defect_count_advisory'].values
defect_dangerous = df['defect_count_dangerous'].values
log_mileage = df['log_mileage'].values
event_interval = df['event_interval'].values


# code for the bayesian hierarchical weibull survival model itself
y = np.log(df['mileage_estimate'].values)
event = np.asarray(event_interval)

with pm.Model(coords={
    "make_id": np.arange(df['make'].nunique()),
    "model_id": np.arange(df['model'].nunique()),
    "engineSize_bucket_id": np.arange(df['engineSize_bucket'].nunique()),
    "fuelType_id": np.arange(df['fuelType'].nunique())
}) as hierarchical_weibull:

    # Global intercept & scale parameter
    mu_global = pm.Normal('mu_global', mu=float(prior_mu), sigma=0.5)
    sigma_global = pm.HalfNormal('sigma_global', sigma=prior_sigma)

    # Hierarchical scale hyperpriors to pool group variations
    sigma_make = pm.HalfStudentT('sigma_make', nu=3, sigma=2.5)
    sigma_model = pm.HalfStudentT('sigma_model', nu=3, sigma=2.5)
    sigma_engine = pm.HalfStudentT('sigma_engine', nu=3, sigma=2.5)
    sigma_fuel = pm.HalfStudentT('sigma_fuel', nu=3, sigma=2.5)

    # Effects (Non-centered parameterization for stable sampling)
    make_raw = pm.Normal('make_raw', mu=0, sigma=1, dims='make_id')
    model_raw = pm.Normal('model_raw', mu=0, sigma=1, dims='model_id')
    engine_raw = pm.Normal('engine_raw', mu=0, sigma=1, dims='engineSize_bucket_id')
    fuel_raw = pm.Normal('fuel_raw', mu=0, sigma=1, dims='fuelType_id')

    make_effect = pm.Deterministic('make_effect', make_raw * sigma_make, dims='make_id')
    model_effect = pm.Deterministic('model_effect', model_raw * sigma_model, dims='model_id')
    engine_effect = pm.Deterministic('engine_effect', engine_raw * sigma_engine, dims='engineSize_bucket_id')
    fuel_effect = pm.Deterministic('fuel_effect', fuel_raw * sigma_fuel, dims='fuelType_id')
    
    beta_advisory = pm.Normal('beta_advisory', mu=0, sigma=1)
    beta_dangerous = pm.Normal('beta_dangerous', mu=0, sigma=1)

    # Linear Predictor
    mu = (
        mu_global 
        + make_effect[make_ids] 
        + model_effect[model_ids] 
        + engine_effect[engineSize_bucket_ids] 
        + fuel_effect[fuelType_ids]
        + beta_advisory * defect_advisory   
        + beta_dangerous * defect_dangerous     
    )

    # Standardized residual: z = (log(T) - mu) / sigma
    z = (y - mu) / sigma_global

    # Correct Log-Weibull Likelihood:
    # Event (uncensored): log(f(y)) = -log(sigma) + z - exp(z)
    # Censored:           log(S(y)) = -exp(z)
    log_lik_obs = -pm.math.log(sigma_global) + z - pm.math.exp(z)
    log_lik_cens = -pm.math.exp(z)

    log_lik = pm.math.switch(pm.math.eq(event, 1), log_lik_obs, log_lik_cens)

    # Sum across all observations
    trace = pm.sample(10000, tune=5000, target_accept=0.9, random_seed=42, return_inferencedata=True, nuts_sampler="nutpie")
    #approx = pm.fit(n=40000, method="advi", obj_optimizer=pm.adam(learning_rate=0.01))

