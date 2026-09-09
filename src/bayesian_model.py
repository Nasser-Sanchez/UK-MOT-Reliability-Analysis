"""
Streaming Hierarchical Weibull Model.
"""

import os
import matplotlib.pyplot as plt
import json
import logging
import duckdb
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import jax
from pymc.variational.callbacks import CheckParametersConvergence
import pytensor.tensor as pt
from src.encode_cat_vars import encode_dataframe, load_mappings, compute_and_save_mappings

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

def save_state(state, trace):
    """Save model state and full trace."""
    # 1. Save JSON State
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)
        
    # 2. Save Full Trace for Prediction (Arviz NetCDF format)
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
      - Global mean mu_global with data-driven priors
      - Make effect (top-level)
      - Model effect (deviation from make mean, prevents cross-make name collisions)
      - Engine + fuel effects: flat priors
      - Advisory + dangerous defect coefficients
    """
    logger.info(f"Processing batch of {len(batch_df)} cars.")
    
    # 1. Preprocessing
    batch_df = batch_df.copy()
    
    # Clip mileage to prevent log overflow and ensure z is well-behaved
    batch_df['mileage_estimate'] = np.clip(batch_df['mileage_estimate'], 1.0, 1_000_000)
    
    # Create local mappings for this batch to ensure IDs are 0..n-1
    local_mappings = {}
    for col in ['make', 'model', 'fuelType', 'engineSize_bucket']:
        unique_vals = batch_df[col].unique()
        local_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

    batch_df_encoded = batch_df.copy()
    for col, mapping in local_mappings.items():
        batch_df_encoded[f'{col}_id'] = batch_df[col].map(mapping).astype(int)
    
    make_cats = batch_df['make'].unique()
    model_cats = batch_df['model'].unique()
    fuel_cats = batch_df['fuelType'].unique()
    engine_cats = batch_df['engineSize_bucket'].unique()
    
    make_ids = batch_df_encoded['make_id'].values
    model_ids = batch_df_encoded['model_id'].values
    engine_ids = batch_df_encoded['engineSize_bucket_id'].values
    fuel_ids = batch_df_encoded['fuelType_id'].values
    
    y = batch_df['mileage_estimate'].values.astype(np.float64)
    event = batch_df_encoded['event_interval'].values
    
    # 2. Load global stats for hyperpriors only
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    global_stats = stats['global']
    
    prior_mu_raw = global_stats['mean']
    prior_mu = np.log(prior_mu_raw) if prior_mu_raw > 500 else prior_mu_raw
    prior_sigma = global_stats['std'] if global_stats['std'] < 5 else 1.0

    # Build make-to-model mapping: which model IDs belong to which make
    make_to_models = {}
    for make in make_cats:
        make_to_models[make] = batch_df.loc[batch_df['make'] == make, 'model'].unique()
    
    with pm.Model(coords={
        "make_id": np.arange(len(make_cats)),
        "model_id": np.arange(len(model_cats)),
        "engine_id": np.arange(len(engine_cats)),
        "fuel_id": np.arange(len(fuel_cats))
    }) as model:
    
        # ------------------------------------------------------------------
        # Global/Hyper priors
        # ------------------------------------------------------------------
        if state:
            mu_global = pm.Normal('mu_global', mu=state['global_params']['mu_global'], 
                                sigma=state['global_params_std']['mu_global'] / 2,
                                initval=state['global_params']['mu_global'])
            sigma_global = pm.HalfStudentT('sigma_global', nu=3, sigma=state['global_params']['sigma_global'],
                                         initval=state['global_params']['sigma_global'])
            
            beta_advisory = pm.Normal('beta_advisory', mu=state['global_params'].get('beta_advisory', 0.0), 
                                    sigma=state['global_params_std'].get('beta_advisory', 1.0),
                                    initval=state['global_params'].get('beta_advisory', 0.0))
            beta_dangerous = pm.Normal('beta_dangerous', mu=state['global_params'].get('beta_dangerous', 0.0), 
                                    sigma=state['global_params_std'].get('beta_dangerous', 1.0),
                                    initval=state['global_params'].get('beta_dangerous', 0.0))
        else:
            mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=0.5, initval=prior_mu)
            sigma_global = pm.HalfNormal('sigma_global', sigma=prior_sigma, initval=1.0)
            beta_advisory = pm.Normal('beta_advisory', mu=0.0, sigma=0.5, initval=0.0)
            beta_dangerous = pm.Normal('beta_dangerous', mu=0.0, sigma=0.5, initval=0.0)
        
        # ------------------------------------------------------------------
        # Make effect (top-level)
        # ------------------------------------------------------------------
        if state:
            mu_make = np.array([state['make_means'].get(cat, 0.0) for cat in make_cats])
            sig_make = np.array([state['make_means_std'].get(cat, 1.0) for cat in make_cats])
        else:
            mu_make = np.zeros(len(make_cats))
            sig_make = np.ones(len(make_cats))
        
        make_effect = pm.Normal(
            'make_effect', 
            mu=mu_make, 
            sigma=sig_make, 
            dims='make_id',
            initval=mu_make
        )
        
        # ------------------------------------------------------------------
        # Model effect (deviation from make mean)
        # ------------------------------------------------------------------
        if state:
            mu_model = np.array([state['model_means'].get(cat, 0.0) for cat in model_cats])
            sig_model = np.array([state['model_means_std'].get(cat, 1.0) for cat in model_cats])
        else:
            mu_model = np.zeros(len(model_cats))
            sig_model = np.ones(len(model_cats))
        
        model_deviation = pm.Normal(
            'model_deviation', 
            mu=mu_model, 
            sigma=sig_model, 
            dims='model_id',
            initval=mu_model
        )
        
        # ------------------------------------------------------------------
        # Engine + Fuel effects (flat priors)
        # ------------------------------------------------------------------
        WIDE = 5.0
        
        mu_engine = np.zeros(len(engine_cats))
        sig_engine = np.full(len(engine_cats), WIDE)
        
        mu_fuel = np.zeros(len(fuel_cats))
        sig_fuel = np.full(len(fuel_cats), WIDE)
        
        engine_effect = pm.Normal(
            'engine_effect', 
            mu=mu_engine, 
            sigma=sig_engine, 
            dims='engine_id',
            initval=mu_engine
        )
        
        fuel_effect = pm.Normal(
            'fuel_effect', 
            mu=mu_fuel, 
            sigma=sig_fuel, 
            dims='fuel_id',
            initval=mu_fuel
        )
        
        # ------------------------------------------------------------------
        # Linear Predictor
        # ------------------------------------------------------------------
        # Model effect = make effect + (make_sigma * model deviation)
        # This ensures each model is centred on its make's mean
        # and scaled by the make's spread
        full_model_effect = (
            make_effect[make_ids] 
            + make_effect[make_ids]  # placeholder — see below
        )
        
        # Correct parameterisation:
        # model_effect[obs] = make_effect[make_id] + sig_make[make_id] * model_deviation[model_id]
        mu = (
            mu_global 
            + make_effect[make_ids]
            + sig_make[make_ids] * model_deviation[model_ids]
            + engine_effect[engine_ids] 
            + fuel_effect[fuel_ids]
            + beta_advisory * batch_df_encoded['defect_count_advisory'].values
            + beta_dangerous * batch_df_encoded['defect_count_dangerous'].values
        )
        
        alpha = 1.0 / sigma_global
        beta = pm.math.exp(mu)

        # Censoring: upper bound is max(observed mileage, 300000)
        upper_bounds = pm.math.switch(
            pm.math.eq(event, 1), 
            np.inf, 
            pt.maximum(y, 300000.0)
        )

        latent = pm.Weibull.dist(alpha=alpha, beta=beta)
        pm.Censored(
            'likelihood', 
            dist=latent, 
            lower=None, 
            upper=upper_bounds, 
            observed=y
        )

        trace = pm.sample(
            draws=1000,
            tune=1000,
            cores=4,
            random_seed=123,
            nuts_sampler="nutpie"
        )

    
    new_state = {
        'global_params': {
            v: trace.posterior[v].mean().item() 
            for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'global_params_std': {
            v: trace.posterior[v].std().item() 
            for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'make_means': {cat: trace.posterior['make_effect'].sel(make_id=i).mean().item()
                       for i, cat in enumerate(make_cats)},
        'make_means_std': {cat: trace.posterior['make_effect'].sel(make_id=i).std().item()
                          for i, cat in enumerate(make_cats)},
        'model_means': {cat: trace.posterior['model_deviation'].sel(model_id=i).mean().item()
                        for i, cat in enumerate(model_cats)},
        'model_means_std': {cat: trace.posterior['model_deviation'].sel(model_id=i).std().item()
                           for i, cat in enumerate(model_cats)},
        'engine_means': {cat: trace.posterior['engine_effect'].sel(engine_id=i).mean().item()
                         for i, cat in enumerate(engine_cats)},
        'engine_means_std': {cat: trace.posterior['engine_effect'].sel(engine_id=i).std().item()
                            for i, cat in enumerate(engine_cats)},
        'fuel_means': {cat: trace.posterior['fuel_effect'].sel(fuel_id=i).mean().item()
                       for i, cat in enumerate(fuel_cats)},
        'fuel_means_std': {cat: trace.posterior['fuel_effect'].sel(fuel_id=i).std().item()
                          for i, cat in enumerate(fuel_cats)},
        'make_to_models': {make: list(models) for make, models in make_to_models.items()},
        'batch_number': (state.get('batch_number', 0) + 1) if state else 1
    }
    
    return trace, new_state
