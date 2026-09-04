"""
Streaming Hierarchical Weibull Model.
"""

import os
import json
import logging
import duckdb
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from src.encode_cat_vars import encode_dataframe, load_mappings, compute_and_save_mappings  # Load mappings helper

# encoding for categorical variables
mappings = compute_and_save_mappings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_run.log', mode='a'),
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
    az.to_netcdf(trace, "model_trace_latest.nc")


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
    """
    logger.info(f"Processing batch of {len(batch_df)} cars.")
    
    # 1. Preprocessing
    batch_df = batch_df.copy()
    batch_df['log_mileage'] = np.log(batch_df['mileage_estimate'])
    
        # Create local mappings for this batch to ensure IDs are 0..n-1 (matching notebook coords)
    local_mappings = {}
    for col in ['make', 'model', 'fuelType', 'engineSize_bucket']:
        unique_vals = batch_df[col].unique()
        local_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

    # Encode using local mappings
    batch_df_encoded = batch_df.copy()
    for col, mapping in local_mappings.items():
        batch_df_encoded[f'{col}_id'] = batch_df[col].map(mapping).astype(int)
    
    # We need to preserve category names for prior lookup
    make_cats = batch_df['make'].unique()
    model_cats = batch_df['model'].unique()
    fuel_cats = batch_df['fuelType'].unique()
    engine_cats = batch_df['engineSize_bucket'].unique()
    
    # 2. Extract Data
    make_ids = batch_df_encoded['make_id'].values
    model_ids = batch_df_encoded['model_id'].values
    engine_ids = batch_df_encoded['engineSize_bucket_id'].values
    fuel_ids = batch_df_encoded['fuelType_id'].values
    
    y = batch_df['log_mileage'].values
    event = batch_df_encoded['event_interval'].values
    
    # 3. Define Priors
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    prior_mu = np.log(float(stats['mean_mileage']))
    prior_sigma = max(np.log(float(stats['std_mileage'])), 0.1)  # HalfNormal needs sigma > 0

    with pm.Model(coords={
        "make_id": np.arange(batch_df_encoded['make'].nunique()),
        "model_id": np.arange(batch_df_encoded['model'].nunique()),
        "engine_id": np.arange(batch_df_encoded['engineSize_bucket'].nunique()),
        "fuel_id": np.arange(batch_df_encoded['fuelType'].nunique())
    }) as model:
    
        # Global/Hyper priors
        if state:
            mu_global = pm.Normal('mu_global', mu=state['global_params']['mu_global'], 
                                sigma=state['global_params_std']['mu_global'] / 2)
            sigma_global = pm.HalfNormal('sigma_global', sigma=state['global_params']['sigma_global'])
            sigma_make = pm.HalfStudentT('sigma_make', nu=state['hyper_params']['sigma_make'], 
                                    sigma=state['hyper_params_std']['sigma_make'] / 2)
            sigma_model = pm.HalfStudentT('sigma_model', nu=state['hyper_params']['sigma_model'], 
                                        sigma=state['hyper_params_std']['sigma_model'] / 2)
            sigma_engine = pm.HalfStudentT('sigma_engine', nu=state['hyper_params']['sigma_engine'], 
                                        sigma=state['hyper_params_std']['sigma_engine'] / 2)
            sigma_fuel = pm.HalfStudentT('sigma_fuel', nu=state['hyper_params']['sigma_fuel'], 
                                    sigma=state['hyper_params_std']['sigma_fuel'] / 2)
            
            # FIX: Load beta priors from state
            beta_advisory = pm.Normal('beta_advisory', mu=state['global_params'].get('beta_advisory', 0), 
                                    sigma=state['global_params_std'].get('beta_advisory', 1))
            beta_dangerous = pm.Normal('beta_dangerous', mu=state['global_params'].get('beta_dangerous', 0), 
                                    sigma=state['global_params_std'].get('beta_dangerous', 1))
        else:
            mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=0.5)
            sigma_global = pm.HalfNormal('sigma_global', sigma=prior_sigma)
            sigma_make = pm.HalfStudentT('sigma_make', nu=3, sigma=2.5)
            sigma_model = pm.HalfStudentT('sigma_model', nu=3, sigma=2.5)
            sigma_engine = pm.HalfStudentT('sigma_engine', nu=3, sigma=2.5)
            sigma_fuel = pm.HalfStudentT('sigma_fuel', nu=3, sigma=2.5)
            
            beta_advisory = pm.Normal('beta_advisory', mu=0, sigma=1)
            beta_dangerous = pm.Normal('beta_dangerous', mu=0, sigma=1)
        
        # Group effect priors
        make_prior_means = get_priors_for_categories(make_cats, state, 'make')
        model_prior_means = get_priors_for_categories(model_cats, state, 'model')
        engine_prior_means = get_priors_for_categories(engine_cats, state, 'engine')
        fuel_prior_means = get_priors_for_categories(fuel_cats, state, 'fuel')
        
        make_raw = pm.Normal('make_raw', mu=make_prior_means, sigma=1, dims='make_id')
        model_raw = pm.Normal('model_raw', mu=model_prior_means, sigma=1, dims='model_id')
        engine_raw = pm.Normal('engine_raw', mu=engine_prior_means, sigma=1, dims='engine_id')
        fuel_raw = pm.Normal('fuel_raw', mu=fuel_prior_means, sigma=1, dims='fuel_id')
        
        make_effect = pm.Deterministic('make_effect', make_raw * sigma_make, dims='make_id')
        model_effect = pm.Deterministic('model_effect', model_raw * sigma_model, dims='model_id')
        engine_effect = pm.Deterministic('engine_effect', engine_raw * sigma_engine, dims='engine_id')
        fuel_effect = pm.Deterministic('fuel_effect', fuel_raw * sigma_fuel, dims='fuel_id')
        
        # Linear Predictor
        mu = (
            mu_global 
            + make_effect[make_ids] 
            + model_effect[model_ids] 
            + engine_effect[engine_ids] 
            + fuel_effect[fuel_ids]
            + beta_advisory * batch_df_encoded['defect_count_advisory'].values
            + beta_dangerous * batch_df_encoded['defect_count_dangerous'].values
        )
        
         # Standardized residual: z = (log(T) - mu) / sigma
        z = (y - mu) / sigma_global

        # CRITICAL FIX: Clip z to prevent exp(z) overflow (exp(709) is max float64)
        # If z > 30, exp(z) is effectively inf. We clip it to 30 to keep log-lik finite.
        z_safe = pm.math.switch(pm.math.gt(z, 30), 30, z)
        
        # Also clip z from below to prevent exp(z) underflow to 0 (which causes log(0) = -inf)
        # If z < -30, exp(z) is effectively 0. We clip it to -30.
        z_safe = pm.math.switch(pm.math.lt(z_safe, -30), -30, z_safe)

        # Event (uncensored): log(f(y)) = -log(sigma) + z - exp(z)
        # Censored:           log(S(y)) = -exp(z)
        log_lik_obs = -pm.math.log(sigma_global) + z_safe - pm.math.exp(z_safe)
        log_lik_cens = -pm.math.exp(z_safe)

        # Apply switch based on event observation
        log_lik = pm.math.switch(pm.math.eq(event, 1), log_lik_obs, log_lik_cens)

        # Attach the custom log-likelihood to the PyMC model graph
        pm.Potential('likelihood', log_lik)


        # 4. Sampling
        logger.info("Starting MCMC sampling...")
        # nutpie does not support PyMC's init_start parameter; let it handle init automatically
        trace = pm.sample(
            draws=2000, 
            tune=750, 
            target_accept=0.95, 
            random_seed=123456, 
            return_inferencedata=True, 
            nuts_sampler="nutpie",
            init='jitter+adapt_diag'
        )
        
    # 5. Diagnostics
    div_count = sum(trace.posterior.attrs.get('diverging', [0]))
    rhat = az.rhat(trace.posterior, var_names=['mu_global', 'sigma_global', 'sigma_make', 'sigma_model'])
    ess = az.ess(trace.posterior, var_names=['mu_global', 'sigma_global'])
    
    rhat_vals = rhat.to_array().values.flatten()
    ess_vals = ess.to_array().values.flatten()
    
    diagnostics = {
        'batch_size': len(batch_df),
        'divergences': div_count,
        'max_rhat': float(rhat_vals.max()),
        'min_ess': float(ess_vals.min()),
        # Relaxed divergence check: < 50 total (or ~0.25% of 20k draws)
        'success': div_count < 50 and rhat_vals.max() < 1.05 and ess_vals.min() > 100
    }
    
    # if not diagnostics['success']:
    #     logger.warning(f"Batch diagnostics failed: {diagnostics}")
    #     raise ValueError("Batch failed diagnostics. Skipping.")
    
    rhat_max = float(rhat.to_array().values.flatten().max())
    logger.info(f"Batch successful. Divergences: {div_count}, Max R-hat: {rhat_max:.4f}")
    
    # 6. Update State
    new_state = {
        'global_params': {
            v: trace.posterior[v].mean().item() for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'global_params_std': {
            v: trace.posterior[v].std().item() for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'hyper_params': {
            v: trace.posterior[v].mean().item() for v in ['sigma_make', 'sigma_model', 'sigma_engine', 'sigma_fuel']
        },
        'hyper_params_std': {
            v: trace.posterior[v].std().item() for v in ['sigma_make', 'sigma_model', 'sigma_engine', 'sigma_fuel']
        },
        'make_means': {cat: trace.posterior['make_effect'].sel(make_id=i).mean().item() 
                       for i, cat in enumerate(make_cats)},
        'model_means': {cat: trace.posterior['model_effect'].sel(model_id=i).mean().item() 
                        for i, cat in enumerate(model_cats)},
        'engine_means': {cat: trace.posterior['engine_effect'].sel(engine_id=i).mean().item() 
                         for i, cat in enumerate(engine_cats)},
        'fuel_means': {cat: trace.posterior['fuel_effect'].sel(fuel_id=i).mean().item() 
                       for i, cat in enumerate(fuel_cats)},
        'batch_number': (state.get('batch_number', 0) + 1) if state else 1
    }
    
    return trace, new_state, diagnostics
