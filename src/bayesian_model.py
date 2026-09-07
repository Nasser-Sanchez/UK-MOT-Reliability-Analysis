"""
Streaming Hierarchical Weibull Model.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import matplotlib.pyplot as plt
import json
import logging
import duckdb
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import jax
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
import pytensor
pytensor.config.floatX = "float32"
import pymc.sampling.jax as pmjax
import pytensor.tensor as pt
from src.encode_cat_vars import encode_dataframe, load_mappings, compute_and_save_mappings  # Load mappings helper

# encoding for categorical variables
mappings = compute_and_save_mappings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/model_run.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_PATH = "data/model_state.json"
REGISTRATIONS_PATH = "data/processed_registrations.csv"
DIAGNOSTICS_PATH = "data/diagnostics.csv"
STATS_PATH = "data/mileage_stats.json"

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_state():
    """Load previous model state if it exists."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return None

def load_processed_registrations():
    """Load list of already processed registrations."""
    if os.path.exists(REGISTRATIONS_PATH):
        return set(pd.read_csv(REGISTRATIONS_PATH)['registration'].tolist())
    return set()

def save_state(state, diagnostics, trace):
    """Save model state, diagnostics, and full trace."""
    # 1. Save JSON State
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)
    
    # 2. Append Diagnostics
    df_diag = pd.DataFrame([diagnostics])
    if os.path.exists(DIAGNOSTICS_PATH):
        df_diag.to_csv(DIAGNOSTICS_PATH, mode='a', header=False, index=False)
    else:
        df_diag.to_csv(DIAGNOSTICS_PATH, mode='w', header=True, index=False)
        
    # 3. Save Full Trace for Prediction (Arviz NetCDF format)
    trace.to_netcdf("data/model_trace_latest.nc")


def get_priors_for_categories(categories, state, level):
    """
    Get prior means for a list of categories.
    If category exists in state, use posterior mean. Else use 0 (hyperprior).
    """
    if state and f"{level}_means" in state:
        means = state[f"{level}_means"]
        return [means.get(cat, 0.0) for cat in categories]
    return [0.0] * len(categories)

# ---------------------------------------------------------------------------
# Main Model Function
# ---------------------------------------------------------------------------

def run_streaming_batch(batch_df: pd.DataFrame, state=None):
    """
    Run the hierarchical Weibull model on a batch of data.
    
    Model structure:
      - Global mean mu_global
      - make_model effects: centred on Make-level stats (mean/std)
      - Engine + fuel effects: centred on mu_global (parallel)
      - Advisory + dangerous defect coefficients
    """
    logger.info(f"Processing batch of {len(batch_df)} cars.")
    
    # 1. Preprocessing
    batch_df = batch_df.copy()
    
    # Clip mileage to 1m to prevent log overflow and ensure z is well-behaved
    # log(1,000,000) ≈ 13.8, which keeps exp(z) safe without needing tanh hacks.
    batch_df['mileage_estimate'] = np.clip(batch_df['mileage_estimate'], 0, 1_000_000)
    batch_df['log_mileage'] = np.log(batch_df['mileage_estimate'])
    
    # Create combined make_model key (e.g., "Ford_Focus")
    # We use the 'make' column directly to ensure we have the correct anchor
    batch_df['make_model'] = batch_df['make'] + '_' + batch_df['model']
    
    # Create local mappings for this batch to ensure IDs are 0..n-1
    local_mappings = {}
    for col in ['make', 'model', 'make_model', 'fuelType', 'engineSize_bucket']:
        unique_vals = batch_df[col].unique()
        local_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

    # Encode using local mappings
    batch_df_encoded = batch_df.copy()
    for col, mapping in local_mappings.items():
        batch_df_encoded[f'{col}_id'] = batch_df[col].map(mapping).astype(int)
    
    # We need to preserve category names for prior lookup
    make_model_cats = batch_df['make_model'].unique()
    fuel_cats = batch_df['fuelType'].unique()
    engine_cats = batch_df['engineSize_bucket'].unique()
    
    # 2. Extract Data
    make_model_ids = batch_df_encoded['make_model_id'].values
    engine_ids = batch_df_encoded['engineSize_bucket_id'].values
    fuel_ids = batch_df_encoded['fuelType_id'].values
    
    y = batch_df['log_mileage'].values
    event = batch_df_encoded['event_interval'].values
    
    # 3. Define Priors
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    # Use global stats for the main hyperparameters
    global_stats = stats['global']
    prior_mu = np.log(float(global_stats['mean']))
    prior_sigma = max(np.log(float(global_stats['std'])), 0.1)

    with pm.Model(coords={
        "make_model_id": np.arange(batch_df_encoded['make_model'].nunique()),
        "engine_id": np.arange(batch_df_encoded['engineSize_bucket'].nunique()),
        "fuel_id": np.arange(batch_df_encoded['fuelType'].nunique())
    }) as model:
    
        # Global/Hyper priors
        if state:
            mu_global = pm.Normal('mu_global', mu=state['global_params']['mu_global'], 
                                sigma=state['global_params_std']['mu_global'] / 2)
            sigma_global = pm.HalfStudentT('sigma_global', nu=3, sigma=state['global_params']['sigma_global'])
            
            beta_advisory = pm.Normal('beta_advisory', mu=state['global_params'].get('beta_advisory', 0), 
                                    sigma=state['global_params_std'].get('beta_advisory', 1))
            beta_dangerous = pm.Normal('beta_dangerous', mu=state['global_params'].get('beta_dangerous', 0), 
                                    sigma=state['global_params_std'].get('beta_dangerous', 1))
        else:
            mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=0.5)
            sigma_global = pm.HalfNormal('sigma_global', sigma=prior_sigma) 
            
            beta_advisory = pm.Normal('beta_advisory', mu=0, sigma=1)
            beta_dangerous = pm.Normal('beta_dangerous', mu=0, sigma=1)
        
        # ------------------------------------------------------------------
        # make_model Effects: Anchored by Make-level stats
        # ------------------------------------------------------------------
        
        # 1. Extract the 'make' name from the 'make_model' key (e.g., "Ford" from "Ford_Focus")
        # We use the 'make' column directly from the table to ensure correct lookup
        make_to_make_map = dict(zip(batch_df['make_model'], batch_df['make']))
        make_names = [make_to_make_map[cat] for cat in make_model_cats]
        
        # 2. Build priors using the Make-level stats
        prior_means = []
        prior_stds = []
        
        for make_name in make_names:
            if make_name in stats:
                # Use specific make stats
                prior_means.append(stats[make_name]['mean'])
                prior_stds.append(stats[make_name]['std'])
            else:
                # Fallback to global if make is unknown
                prior_means.append(stats['global']['mean'])
                prior_stds.append(stats['global']['std'])

        # 3. Define the effect
        # The 'mu' anchors the specific model to its Make's average
        # The 'sigma' anchors the uncertainty to the Make's spread
        make_model_effect = pm.Normal(
            'make_model_effect', 
            mu=prior_means, 
            sigma=prior_stds, 
            dims='make_model_id'
        )
        
        # ------------------------------------------------------------------
        # Engine + fuel effects: centred on mu_global (parallel)
        # ------------------------------------------------------------------
        sigma_engine = pm.HalfNormal('sigma_engine', sigma=1.0)
        sigma_fuel = pm.HalfNormal('sigma_fuel', sigma=1.0)
        
        engine_raw = pm.Normal('engine_raw', mu=0, sigma=1, 
                               shape=batch_df_encoded['engineSize_bucket'].nunique())
        fuel_raw = pm.Normal('fuel_raw', mu=0, sigma=1, 
                             shape=batch_df_encoded['fuelType'].nunique())
        
        engine_effect = pm.Deterministic('engine_effect', sigma_engine * engine_raw, dims='engine_id')
        fuel_effect = pm.Deterministic('fuel_effect', sigma_fuel * fuel_raw, dims='fuel_id')
        
        # Linear Predictor
        mu = (
            mu_global 
            + make_model_effect[make_model_ids] 
            + engine_effect[engine_ids] 
            + fuel_effect[fuel_ids]
            + beta_advisory * batch_df_encoded['defect_count_advisory'].values
            + beta_dangerous * batch_df_encoded['defect_count_dangerous'].values
        )
        
        # z is now bounded because y is clipped to log(1m) ≈ 13.8
        # exp(z) will not overflow for reasonable sigma_global values.
        z = (y - mu) / sigma_global

        # Event (uncensored): log(f(y)) = -log(sigma) + z - exp(z)
        # Censored:           log(S(y)) = -exp(z)
        log_lik_obs = -pm.math.log(sigma_global) + z - pm.math.exp(z)
        log_lik_cens = -pm.math.exp(z)

        # Apply switch based on event observation
        log_lik = pm.math.switch(pm.math.eq(event, 1), log_lik_obs, log_lik_cens)

        # Attach the custom log-likelihood to the PyMC model graph
        pm.Potential('likelihood', log_lik)


        # 4. Sampling
        logger.info("Running MCMC Sampler...")
        
        trace = pm.sample(
            draws=2000,
            tune=1000,
            target_accept=0.9,
            random_seed=123,
            backend="jax"
        )

    # Diagnostics
    diagnostics = {
        'batch_size': len(batch_df)
    }

    # 6. Update State
    new_state = {
        'global_params': {
            v: trace.posterior[v].mean().item() for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'global_params_std': {
            v: trace.posterior[v].std().item() for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'hyper_params': {
            v: trace.posterior[v].mean().item() for v in ['sigma_engine', 'sigma_fuel']
        },
        'hyper_params_std': {
            v: trace.posterior[v].std().item() for v in ['sigma_engine', 'sigma_fuel']
        },
        # Store make_model effects directly (anchored by Make stats)
        'make_model_means': {cat: trace.posterior['make_model_effect'].sel(make_model_id=i).mean().item() 
                             for i, cat in enumerate(make_model_cats)},
        'make_model_means_std': {cat: trace.posterior['make_model_effect'].sel(make_model_id=i).std().item() 
                                 for i, cat in enumerate(make_model_cats)},
        'engine_means': {cat: trace.posterior['engine_effect'].sel(engine_id=i).mean().item() 
                         for i, cat in enumerate(engine_cats)},
        'fuel_means': {cat: trace.posterior['fuel_effect'].sel(fuel_id=i).mean().item() 
                       for i, cat in enumerate(fuel_cats)},
        'batch_number': (state.get('batch_number', 0) + 1) if state else 1
    }
    
    return trace, new_state, diagnostics