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
        """
        logger.info(f"Processing batch of {len(batch_df)} cars.")
        
        # 1. Preprocessing
        batch_df = batch_df.copy()
        
        # Clip mileage
        batch_df['mileage_estimate'] = np.clip(batch_df['mileage_estimate'], 1.0, 1_000_000)
        
        # Create local mappings for this batch
        local_mappings = {}
        for col in ['make', 'model', 'fuelType', 'engineSize_bucket']:
            unique_vals = batch_df[col].unique()
            local_mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

        batch_df_encoded = batch_df.copy()
        for col, mapping in local_mappings.items():
            batch_df_encoded[f'{col}_id'] = batch_df[col].map(mapping).astype(int)
        
        # --- NEW: Create 'make_model' field for unique identification ---
        batch_df['make_model'] = batch_df['make'] + "_" + batch_df['model']
        unique_make_models = batch_df['make_model'].unique()
        
        make_cats = batch_df['make'].unique()
        model_cats = batch_df['make_model'].unique()
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
        
        prior_mu = global_stats['log_mean']
        prior_sigma = global_stats['sigma_global_prior'] 

        # --- UPDATED COORDS: Added make_model_id ---
        with pm.Model(coords={
            "make_id": np.arange(len(make_cats)),
            "model_id": np.arange(len(model_cats)),
            "make_model_id": np.arange(len(unique_make_models)),
            "engine_id": np.arange(len(engine_cats)),
            "fuel_id": np.arange(len(fuel_cats))
        }) as model:
        
            # ------------------------------------------------------------------
            # Global/Hyper priors (unchanged)
            # ------------------------------------------------------------------
            if state:
                mu_global = pm.Normal('mu_global', mu=state['global_params']['mu_global'], 
                                    sigma=state['global_params_std']['mu_global'] / 2,
                                    initval=state['global_params']['mu_global'])
                sigma_global = pm.HalfStudentT('sigma_global', nu=3, sigma=state['global_params']['sigma_global'],
                                             initval=state['global_params']['sigma_global'])

            else:
                mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=prior_sigma, initval=prior_mu)
                sigma_global = pm.HalfNormal('sigma_global', sigma=prior_sigma, initval=1.0)
                
            
            # ------------------------------------------------------------------
            # Make effect (top-level)
            # ------------------------------------------------------------------
            if state:
                mu_make = np.array([state['make_means'].get(cat, 0.0) for cat in make_cats])
                sig_make = np.array([state['make_means_std'].get(cat, 1.0) for cat in make_cats])
            else:
                mu_make = np.zeros(len(make_cats))
                sig_make = np.full(len(make_cats), 0.25)
            
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
                sig_model = np.full(len(model_cats), 0.25)
            
            model_deviation = pm.Normal(
                'model_deviation', 
                mu=mu_model, 
                sigma=sig_model, 
                dims='model_id',
                initval=mu_model
            )
            
            # ------------------------------------------------------------------
            # Engine + Fuel effects
            # ------------------------------------------------------------------
            WIDE = 0.1
            
            if state and 'engine_means' in state:
                mu_engine = np.array([state['engine_means'].get(cat, 0.0) for cat in engine_cats])
                sig_engine = np.array([max(0.1, state['engine_means_std'].get(cat, WIDE) / 2) for cat in engine_cats])
            else:
                mu_engine = np.zeros(len(engine_cats))
                sig_engine = np.full(len(engine_cats), WIDE)
                
            if state and 'fuel_means' in state:
                mu_fuel = np.array([state['fuel_means'].get(cat, 0.0) for cat in fuel_cats])
                sig_fuel = np.array([max(0.1, state['fuel_means_std'].get(cat, WIDE) / 2) for cat in fuel_cats])
            else:
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
                make_effect[make_ids] + model_deviation[model_ids]
            )
            
            mu = (
                mu_global 
                + full_model_effect
                + engine_effect[engine_ids] 
                + fuel_effect[fuel_ids]

            )
            
            alpha = 1.0 / sigma_global
            beta = pm.math.exp(mu)

            # Censoring: upper bound is max(observed mileage, p_90_mileage_make)
            p90 = batch_df['p_90_mileage_make'].values.astype(np.float64)
            upper_bounds = pm.math.switch(
                pm.math.eq(event, 1), 
                np.inf, 
                y
            )

            latent = pm.Weibull.dist(alpha=alpha, beta=beta)

            pm.Censored(
                'likelihood', 
                dist=latent, 
                lower=None, 
                upper=upper_bounds, 
                observed=y
            )

            # ------------------------------------------------------------------
        # Prior Predictive Check (Optional)
        # ------------------------------------------------------------------
        # 1. Define the variable to sample
            terminal_mileage = pm.Weibull('terminal_mileage', alpha=alpha, beta=beta)

            # 2. Prior Predictive Check Logic
            ppc_dir = "data/prior_predictive_checks"
            os.makedirs(ppc_dir, exist_ok=True)
            
            run_ppc = input("Run Prior Predictive Check? (y/n): ").strip().lower()
            proceed_mcmc = True

            if run_ppc == 'y':
                print("Running Prior Predictive Check...")
                # Explicitly request both mu_global and terminal_mileage
                prior_trace = pm.sample_prior_predictive(samples=1000, random_seed=123, var_names=['mu_global', 'terminal_mileage'])
                
                
                # Plot mu (from prior group)
                az.plot_posterior(prior_trace.prior['mu_global'], ref_val=0)
                plt.title("Prior Distribution of mu_global")
                plt.savefig(os.path.join(ppc_dir, "prior_mu_global.png"))
                plt.close()
                
                # Plot terminal mileage (from prior group because it's a stochastic variable)
                az.plot_posterior(prior_trace.prior['terminal_mileage'], ref_val=150000)
                plt.title("Prior Predictive Distribution of Terminal Mileage")
                plt.savefig(os.path.join(ppc_dir, "prior_terminal_mileage.png"))
                plt.close()
                
                print(f"Prior predictive checks saved to {ppc_dir}/")
                proceed_mcmc = input("Proceed with MCMC sampling? (y/n): ").strip().lower() == 'y'
            else:
                print("Skipping Prior Predictive Check.")

            y_pred = pm.Weibull('y_pred', alpha=alpha, beta=beta, shape=len(y))
            # 3. MCMC Sampling
            if proceed_mcmc:
                trace = pm.sample(
                    draws=1000,
                    tune=1000,
                    cores=4,
                    random_seed=123,
                    nuts_sampler="nutpie"
                )
            else:
                print("MCMC sampling cancelled by user.")
                return None, None

            
             # ------------------------------------------------------------------
            # Posterior Predictive Check (Optional)
            # ------------------------------------------------------------------
            run_ppc = input("Run Posterior Predictive Check? (y/n): ").strip().lower()
        
            if run_ppc == 'y':
                print("Running Posterior Predictive Check...")
                
                # Sample terminal mileage predictions
                ppc_trace = pm.sample_posterior_predictive(trace, var_names=['y_pred'], random_seed=123)
                
                
                ppc_dir = "data/posterior_predictive_checks"
                os.makedirs(ppc_dir, exist_ok=True)
                
                # Filter for cars that have actually failed (event=1)
                failed_mask = batch_df_encoded['event_interval'].values == 1
                
                # Get observed terminal mileage for failed cars
                observed_terminal = batch_df['mileage_estimate'].values[failed_mask]
                
                # Get mean predicted terminal mileage for ALL cars, then filter
                mean_pred_all = ppc_trace.posterior_predictive['y_pred'].mean(dim=['chain', 'draw']).values
                mean_pred_failed = mean_pred_all[failed_mask] # Apply same mask
                
                if len(observed_terminal) > 0:
                    # Plot 1: Observed vs Predicted for FAILED cars only
                    plt.figure(figsize=(8, 5))
                    plt.hist(observed_terminal, bins=50, alpha=0.5, label='Observed Terminal Mileage', color='blue')
                    plt.hist(mean_pred_failed, bins=50, alpha=0.5, label='Predicted Terminal Mileage', color='red')
                    
                    batch_num = state.get('batch_number', 0) + 1 if state else 1
                    plt.title(f"PPC: Observed vs Predicted (Failed Cars Only)")
                    plt.xlabel("Mileage")
                    plt.ylabel("Frequency")
                    plt.legend()
                    plt.savefig(os.path.join(ppc_dir, "ppc_failed_cars.png"))
                    plt.close()
                    
                    # Plot 2: Residuals for Failed Cars
                    residuals = observed_terminal - mean_pred_failed
                    
                    plt.figure(figsize=(8, 5))
                    plt.hist(residuals, bins=50, color='green', edgecolor='black')
                    plt.title("Residuals (Observed - Predicted) for Failed Cars")
                    plt.xlabel("Residual")
                    plt.ylabel("Frequency")
                    plt.axvline(0, color='black', linestyle='--')
                    plt.savefig(os.path.join(ppc_dir, "residuals_failed_cars.png"))
                    plt.close()
                    
                    print(f"Posterior predictive checks saved to {ppc_dir}/")
                else:
                    print("No failed cars found in this batch for PPC.")
            else:
                print("Skipping Posterior Predictive Check.")

        # ------------------------------------------------------------------
        # Save State (Updated keys to use make_model)
        # ------------------------------------------------------------------
        new_state = {
            'global_params': {
                v: trace.posterior[v].mean().item() 
                for v in ['mu_global', 'sigma_global']
            },
            'global_params_std': {
                v: trace.posterior[v].std().item() 
                for v in ['mu_global', 'sigma_global']
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
            'batch_number': (state.get('batch_number', 0) + 1) if state else 1
        }
        # Preserve parameters for categories not present in the current batch
        if state:
            for key in ['make_means', 'make_means_std', 'model_means', 'model_means_std', 
                        'engine_means', 'engine_means_std', 'fuel_means', 'fuel_means_std']:
                if key in state:
                    for cat in state[key]:
                        if cat not in new_state[key]:
                            new_state[key][cat] = state[key][cat]
        
        return trace, new_state