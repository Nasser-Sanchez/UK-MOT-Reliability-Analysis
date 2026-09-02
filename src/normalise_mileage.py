import duckdb
import os
import json

MOT_DATA = "data/mot_last_test.parquet"
OUTPUT = "data/mileage_stats.json"


def compute_and_save_stats():
    print(f"Querying {MOT_DATA}...")
    
    # Query the parquet file directly using DuckDB
    stats = duckdb.query(f"""
        SELECT 
            AVG(mileage_estimate) as mean_mileage,
            STDDEV(mileage_estimate) as std_mileage
        FROM read_parquet('{MOT_DATA}')
        WHERE event = 1
    """).df()
    
    mean_val = stats['mean_mileage'].iloc[0]
    std_val = stats['std_mileage'].iloc[0]
    
    result = {
        "mean_mileage": float(mean_val),
        "std_mileage": float(std_val)
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Stats saved to {OUTPUT}")
    print(f"Mean: {mean_val:.2f}, StdDev: {std_val:.2f}")

if __name__ == "__main__":
    compute_and_save_stats()