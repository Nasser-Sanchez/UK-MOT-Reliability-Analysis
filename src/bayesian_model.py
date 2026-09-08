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
      - Engine + fuel effects: flat priors (data-driven)
      - Advisory + dangerous defect coefficients
    """
    logger.info(f"Processing batch of {len(batch_df)} cars.")
    
    # 1. Preprocessing
    batch_df = batch_df.copy()
    
    # Clip mileage to prevent log overflow and ensure z is well-behaved
    batch_df['mileage_estimate'] = np.clip(batch_df['mileage_estimate'], 1.0, 1_000_000)
    
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
    
    y = batch_df['mileage_estimate'].values.astype(np.float64)
    event = batch_df_encoded['event_interval'].values
    
    # 3. Define Priors
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    # Use global stats for the main hyperparameters
    global_stats = stats['global']
    
    # Use log-mean and log-std directly from mileage_stats.json
    prior_mu_raw = global_stats['mean']
    prior_mu = np.log(prior_mu_raw) if prior_mu_raw > 500 else prior_mu_raw
    prior_sigma = global_stats['std'] if global_stats['std'] < 5 else 1.0

    with pm.Model(coords={
        "make_model_id": np.arange(batch_df_encoded['make_model'].nunique()),
        "engine_id": np.arange(batch_df_encoded['engineSize_bucket'].nunique()),
        "fuel_id": np.arange(batch_df_encoded['fuelType'].nunique())
    }) as model:
    
        # Global/Hyper priors
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
                                    
            mu_make_model = np.array([state['make_model_means'].get(cat, 0.0) for cat in make_model_cats])
            sig_make_model = np.array([state['make_model_means_std'].get(cat, 1.0) for cat in make_model_cats])
            
            mu_engine = np.array([state['engine_means'].get(cat, 0.0) for cat in engine_cats])
            sig_engine = np.array([state['engine_means_std'].get(cat, 1.0) for cat in engine_cats])
            
            mu_fuel = np.array([state['fuel_means'].get(cat, 0.0) for cat in fuel_cats])
            sig_fuel = np.array([state['fuel_means_std'].get(cat, 1.0) for cat in fuel_cats])
        else:
            mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=0.5, initval=prior_mu)
            sigma_global = pm.HalfNormal('sigma_global', sigma=prior_sigma, initval=1.0) 
            
            beta_advisory = pm.Normal('beta_advisory', mu=0.0, sigma=0.5, initval=0.0)
            beta_dangerous = pm.Normal('beta_dangerous', mu=0.0, sigma=0.5, initval=0.0)
            
            mu_make_model = np.zeros(len(make_model_cats))
            sig_make_model = np.ones(len(make_model_cats))
            mu_engine = np.zeros(len(engine_cats))
            sig_engine = np.ones(len(engine_cats))
            mu_fuel = np.zeros(len(fuel_cats))
            sig_fuel = np.ones(len(fuel_cats))
        
        # ------------------------------------------------------------------
        # make_model Effects: Anchored by Make-level stats
        # ------------------------------------------------------------------
        
        # 3. Define the effect
        # The 'mu' anchors the specific model to its Make's average
        # The 'sigma' anchors the uncertainty to the Make's spread
        make_model_effect = pm.Normal(
            'make_model_effect', 
            mu=mu_make_model, 
            sigma=sig_make_model, 
            dims='make_model_id',
            initval=mu_make_model
        )

        # Using wide flat priors allows the data to drive these effects.
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
        
        # Linear Predictor
        mu = (
            mu_global 
            + make_model_effect[make_model_ids] 
            + engine_effect[engine_ids] 
            + fuel_effect[fuel_ids]
            + beta_advisory * batch_df_encoded['defect_count_advisory'].values
            + beta_dangerous * batch_df_encoded['defect_count_dangerous'].values
        )
        
        alpha = 1.0 / sigma_global
        beta = pm.math.exp(mu)

        upper_bounds = pm.math.switch(pm.math.eq(event, 1), np.inf, y)

        latent = pm.Weibull.dist(alpha=alpha, beta=beta)
        pm.Censored(
            'likelihood', 
            dist=latent, 
            lower=None, 
            upper=upper_bounds, 
            observed=y
        )

        # 4. Sampling

        trace = pm.sample(
            draws=1000,
            tune=1000,
            cores=4,
            random_seed=123,
            nuts_sampler="nutpie"
        )
        
        # # ADVI inference 
        # logger.info("Running ADVI fit...")
        # fit = pm.fit(
        #     n=15000,
        #     method="advi",
        #     random_seed=123,
        #     callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")]
        # )
        
        # # Draw samples from the variational distribution
        # # 3. Draw samples from the learned flow
        # n_draws = 5000
        # logger.info(f"Drawing {n_draws} posterior samples from flow...")
        # trace = fit.sample(n_draws)

    # plt.figure(figsize=(10, 6))
    # plt.plot(fit.hist[-3000:], label='ELBO')
    # plt.title('ADVI ELBO History')
    # plt.xlabel('Iteration')
    # plt.ylabel('ELBO')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('data/elbo_history.png')
    # plt.close()
    # logger.info("ELBO history saved to 'elbo_history.png'")

    diagnostics = {
         'batch_size': len(batch_df)
    #     'elbo_final': float(fit.hist[-1]),
     }
    
    new_state = {
        'global_params': {
            v: trace.posterior[v].mean().item() 
            for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'global_params_std': {
            v: trace.posterior[v].std().item() 
            for v in ['mu_global', 'sigma_global', 'beta_advisory', 'beta_dangerous']
        },
        'make_model_means': {cat: trace.posterior['make_model_effect'].sel(make_model_id=i).mean().item()
                             for i, cat in enumerate(make_model_cats)},
        'make_model_means_std': {cat: trace.posterior['make_model_effect'].sel(make_model_id=i).std().item()
                                 for i, cat in enumerate(make_model_cats)},
        'engine_means': {cat: trace.posterior['engine_effect'].sel(engine_id=i).mean().item()
                         for i, cat in enumerate(engine_cats)},
        'engine_means_std': {cat: trace.posterior['engine_effect'].sel(engine_id=i).std().item()
                             for i, cat in enumerate(engine_cats)},
        'fuel_means': {cat: trace.posterior['fuel_effect'].sel(fuel_id=i).mean().item()
                       for i, cat in enumerate(fuel_cats)},
        'fuel_means_std': {cat: trace.posterior['fuel_effect'].sel(fuel_id=i).std().item()
                           for i, cat in enumerate(fuel_cats)},
        'batch_number': (state.get('batch_number', 0) + 1) if state else 1
    }
    
    return trace, new_state, diagnostics