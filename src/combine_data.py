import duckdb
import os
from pathlib import Path

# Define the base directory
base_dir = Path("data/mot_results")

if not base_dir.exists():
    print(f"Directory {base_dir} does not exist.")
else:
    # Iterate through each year folder (e.g., 2017, 2018, etc.)
    for year_folder in base_dir.iterdir():
        if year_folder.is_dir():
            print(f"\nProcessing year: {year_folder.name}")
            
            # Iterate through subdirectories within the year folder
            for sub_folder in year_folder.iterdir():
                if sub_folder.is_dir():
                    # Check if "test" is in the folder name
                    if "test" not in sub_folder.name.lower():
                        print(f"  Deleting folder: {sub_folder.name}")
                        # Use shutil.rmtree to delete the folder and its contents
                        import shutil
                        shutil.rmtree(sub_folder)
                    else:
                        print(f"  Keeping folder: {sub_folder.name}")


con = duckdb.connect('uk_car_analyser.duckdb')

con.execute("""
    COPY (
        WITH Initial_Clean AS (
            SELECT 
                vehicle_id,
                first_use_date,
                test_date,
                test_type,
                test_result,
                test_mileage,
                make,
                model,
                fuel_type,
                cylinder_capacity,
            FROM read_csv_auto(
                'data/mot_results/**/*.csv',
                union_by_name=true,
                filename=true
            )
            --WHERE (filename LIKE '%/test/%' OR filename LIKE '%/test_result_%')
                

        )
        SELECT * FROM Initial_Clean
    ) TO 'data/mot_data_2024.parquet' (FORMAT PARQUET);
""")

print("Parquet file created.")