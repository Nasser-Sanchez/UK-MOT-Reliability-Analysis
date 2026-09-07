# src/encode_categories.py
import duckdb
import pandas as pd
import os

def compute_and_save_mappings(data_path="data/mot_last_test.parquet", output_dir="data"):
    """
    Uses DuckDB to find unique categories directly from the parquet file
    without loading the whole dataset into memory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Original columns
    cols = ['make', 'model', 'fuelType', 'engineSize_bucket']
    
    # Check if all mapping files already exist
    all_exist = all(
        os.path.exists(f"{output_dir}/{col}_mapping.csv") 
        for col in cols
    )
    
    # Check if make_model mapping exists
    make_model_exists = os.path.exists(f"{output_dir}/make_model_mapping.csv")
    
    if all_exist and make_model_exists:
        print("Mapping files already exist. Skipping computation.")
        return load_mappings(output_dir)
    
    print("Computing category mappings using SQL...")
    
    # Use DuckDB to get distinct values without loading full dataset
    query = f"""
    SELECT DISTINCT make, model, fuelType, engineSize_bucket
    FROM '{data_path}'
    """
    
    df = duckdb.query(query).df()
    
    mappings = {}
    
    # 1. Standard Mappings
    for col in cols:
        # Get unique values and sort them for consistency
        unique_vals = sorted(df[col].dropna().unique())
        
        # Create mapping: value -> index
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        mappings[col] = mapping
        
        # Save to CSV
        df_map = pd.DataFrame(list(mapping.items()), columns=['category', 'id'])
        df_map.to_csv(f"{output_dir}/{col}_mapping.csv", index=False)
        
    # 2. Combined make_model Mapping
    # Create a combined key to avoid ambiguity (e.g., "Focus" could be Ford or Mazda)
    df['make_model'] = df['make'].astype(str) + '_' + df['model'].astype(str)
    unique_make_models = sorted(df['make_model'].dropna().unique())
    
    make_model_mapping = {val: idx for idx, val in enumerate(unique_make_models)}
    mappings['make_model'] = make_model_mapping
    
    df_map_mm = pd.DataFrame(list(make_model_mapping.items()), columns=['category', 'id'])
    df_map_mm.to_csv(f"{output_dir}/make_model_mapping.csv", index=False)
    
    print(f"Mappings saved to {output_dir}/")
    return mappings

def load_mappings(output_dir="data"):
    """
    Loads mappings from CSV files.
    """
    # Include make_model in the list of columns to load
    cols = ['make', 'model', 'fuelType', 'engineSize_bucket', 'make_model']
    mappings = {}
    
    for col in cols:
        path = f"{output_dir}/{col}_mapping.csv"
        if os.path.exists(path):
            df_map = pd.read_csv(path)
            mapping = dict(zip(df_map['category'], df_map['id']))
            mappings[col] = mapping
        else:
            raise FileNotFoundError(f"Mapping file not found: {path}")
            
    return mappings

def encode_dataframe(df, mappings):
    """
    Applies mappings to a dataframe to create *_id columns.
    """
    df_encoded = df.copy()
    
    for col in mappings:
        new_col = f"{col}_id"
        # Map values to IDs, fillna with -1 for unknown categories
        df_encoded[new_col] = df[col].map(mappings[col]).fillna(-1).astype(int)
        
    return df_encoded
