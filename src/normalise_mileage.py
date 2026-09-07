# src/normalise_mileage.py
import duckdb
import os
import json
import pandas as pd

MOT_DATA = "data/mot_last_test.parquet"
OUTPUT = "data/mileage_stats.json"


def compute_and_save_stats():
    print(f"Querying {MOT_DATA}...")
    
    # 1. Calculate Global Stats (fallback for unknown makes)
    global_stats = duckdb.query(f"""
        SELECT 
            AVG(mileage_estimate) as mean_mileage,
            STDDEV(mileage_estimate) as std_mileage
        FROM read_parquet('{MOT_DATA}')
        WHERE event = 1
    """).df()
    
    global_mean = float(global_stats['mean_mileage'].iloc[0])
    global_std = float(global_stats['std_mileage'].iloc[0])
    
    # 2. Calculate Stats by MAKE
    # This creates the prior anchors for the model effects
    make_stats = duckdb.query(f"""
        SELECT 
            make,
            AVG(mileage_estimate) as mean_mileage,
            STDDEV(mileage_estimate) as std_mileage
        FROM read_parquet('{MOT_DATA}')
        WHERE event = 1
        GROUP BY make
    """).df()
    
    # 3. Build the dictionary
    result = {}
    
    for _, row in make_stats.iterrows():
        make = row['make']
        m = row['mean_mileage']
        s = row['std_mileage']
        
        # Handle cases where a make has very few data points (STDDEV is NaN)
        if pd.isna(s):
            s = global_std  # Fallback to global std for rare makes
        
        result[make] = {
            'mean': float(m),
            'std': float(s)
        }
    
    # 4. Add Global Fallback
    result['global'] = {
        'mean': global_mean,
        'std': global_std
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    # Save to JSON
    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Stats saved to {OUTPUT}")
    print(f"Total makes processed: {len(result) - 1}")  # -1 for 'global'
    print(f"Global Mean: {global_mean:.2f}, Global StdDev: {global_std:.2f}")

if __name__ == "__main__":
    compute_and_save_stats()
