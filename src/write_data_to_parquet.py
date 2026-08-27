import duckdb
import os
from pathlib import Path
import glob

# # Define the base directory
base_dir = Path("data/mot_results")

# if not base_dir.exists():
#     print(f"Directory {base_dir} does not exist.")
# else:
#     # Iterate through each year folder (e.g., 2017, 2018, etc.)
#     for year_folder in base_dir.iterdir():
#         if year_folder.is_dir():
#             print(f"\nProcessing year: {year_folder.name}")
            
#             # Iterate through subdirectories within the year folder
#             for sub_folder in year_folder.iterdir():
#                 if sub_folder.is_dir():
#                     # Check if "test" is in the folder name
#                     if "test" not in sub_folder.name.lower():
#                         print(f"  Deleting folder: {sub_folder.name}")
#                         # Use shutil.rmtree to delete the folder and its contents
#                         import shutil
#                         shutil.rmtree(sub_folder)
#                     else:
#                         print(f"  Keeping folder: {sub_folder.name}")


con = duckdb.connect("uk_car_analyser.duckdb")

# Get sorted list of year folders
year_folders = sorted(
    [d for d in base_dir.iterdir() if d.is_dir()],
    key=lambda p: p.name
)

for year_folder in year_folders:
    year = year_folder.name
    output_parquet = f"data/mot_data_{year}.parquet"

    # 1. Find files using Python's glob (handles Windows paths correctly)
    # Matches both .txt and .csv
    txt_files = glob.glob(str(year_folder / "**" / "*.txt"), recursive=True)
    csv_files = glob.glob(str(year_folder / "**" / "*.csv"), recursive=True)
    
    all_files = txt_files + csv_files

    if not all_files:
        print(f"\nNo files found for {year} — skipping.")
        continue

    # 2. Convert backslashes to forward slashes (DuckDB requirement on Windows)
    # This ensures paths like 'data\mot_results\2005\...' become 'data/mot_results/2005/...'
    duck_files = [f.replace("\\", "/") for f in all_files]

    print(f"\nBuilding {output_parquet} ({len(duck_files)} files)...")

    # 3. Execute query
    # We pass the list of files as a parameter to read_csv_auto.
    # DuckDB auto-detects delimiters (| or ,) and file types.
    sql = f"""
        COPY (
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
                cylinder_capacity
            FROM read_csv_auto(?,
            ignore_errors=true)
        ) TO '{output_parquet}' (FORMAT PARQUET);
    """

    con.execute(sql, [duck_files])
    print(f"  -> {output_parquet} written.")

print("\nDone.")