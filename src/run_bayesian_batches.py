"""
Orchestrator for streaming Bayesian model updates.

Usage:
    python src/run_batches.py --batch_size 50000 --num_batches 1

"""

import os
import sys
import argparse
import pandas as pd
import duckdb
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bayesian_model import run_streaming_batch, load_state, load_processed_registrations, save_state, STATE_PATH, REGISTRATIONS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(batch_size: int, num_batches: int):
    # 1. Initialize State
    state = load_state()
    processed_regs = load_processed_registrations()
    
    logger.info(f"Starting streaming. State: {'Loaded' if state else 'New'}, Processed: {len(processed_regs)}")
    
    # 2. Query Data
    con = duckdb.connect("uk_car_analyser.duckdb")
    
    base_query = """
    WITH valid_makes AS (
        SELECT make
        FROM 'data/mot_last_test.parquet'
        GROUP BY make
        HAVING COUNT(DISTINCT registration) > 10000
    )
    SELECT
        registration, make, model, fuelType, engineSize_bucket,
        defect_count_advisory, defect_count_dangerous,
        mileage_estimate, event_interval
    FROM 'data/mot_last_test.parquet'
    WHERE make IN (SELECT make FROM valid_makes)
    ORDER BY registration
    """
    
    # Get total available
    total_available = con.execute(f"SELECT COUNT(*) FROM ({base_query})").fetchone()[0]
    logger.info(f"Total available cars in data: {total_available}")
    
    # 3. Loop Batches
    cars_processed = 0
    batch_num = 0
    
    # Fetch all remaining IDs once to avoid re-querying
    all_regs = con.execute(f"SELECT registration FROM ({base_query})").fetchall()
    all_regs = [r[0] for r in all_regs]
    remaining_regs = [r for r in all_regs if r not in processed_regs]
    
    logger.info(f"Remaining cars to process: {len(remaining_regs)}")
    
    while batch_num < num_batches and len(remaining_regs) > 0:
        batch_num += 1
        logger.info(f"--- Batch {batch_num} ---")
        
        # Get next batch IDs
        batch_ids = remaining_regs[:batch_size]
        remaining_regs = remaining_regs[batch_size:] # Remove processed IDs from list
        
        # Query data for these IDs
        # Use a temp table for large batches to avoid SQL string limits
        if len(batch_ids) > 10000:
            # Create temp table for IDs
            temp_df = pd.DataFrame(batch_ids, columns=['registration'])
            con.execute("CREATE TEMP TABLE temp_ids AS SELECT * FROM temp_df")
            query = f"""
                SELECT * FROM 'data/mot_last_test.parquet'
                WHERE registration IN (SELECT registration FROM temp_ids)
            """
            batch_df = con.execute(query).df()
            con.execute("DROP TABLE temp_ids")
        else:
            batch_df = con.execute(f"""
                SELECT * FROM 'data/mot_last_test.parquet'
                WHERE registration IN ({','.join(["'"] + batch_ids + ["'"])})
            """).df()
        
        if len(batch_df) == 0:
            logger.info("Batch empty after filtering. Skipping.")
            continue
            
        # Run Model
        try:
            trace, new_state, diagnostics = run_streaming_batch(batch_df, state)
            
            # Update State & Registrations
            state = new_state
            processed_regs.update(batch_df['registration'].tolist())
            
            # Save State, Diagnostics, and Trace
            save_state(state, diagnostics, trace)
            
            # Save registrations
            pd.DataFrame({'registration': list(processed_regs)}).to_csv(REGISTRATIONS_PATH, index=False)
            
            logger.info(f"Batch {batch_num} complete. Total processed: {len(processed_regs)}")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            logger.info("Skipping batch and continuing.")
            # Update processed regs to avoid re-processing failed batch
            # processed_regs.update(batch_df['registration'].tolist())
            # pd.DataFrame({'registration': list(processed_regs)}).to_csv(REGISTRATIONS_PATH, index=False)
            
    con.close()
    logger.info("Streaming complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Bayesian model updates.")
    parser.add_argument('--batch_size', type=int, default=50000, help="Number of cars per batch.")
    parser.add_argument('--num_batches', type=int, default=1, help="Number of batches to process.")
    args = parser.parse_args()
    
    main(args.batch_size, args.num_batches)
