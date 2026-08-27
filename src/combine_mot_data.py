import duckdb

con = duckdb.connect("uk_car_analyser.duckdb")

query = """
    SELECT *
    FROM read_parquet('data/mot_data_*.parquet')
    WHERE first_use_date IS NOT NULL
"""

# Save the result
con.execute(f"COPY ({query}) TO 'data/mot_data_combined.parquet' (FORMAT PARQUET)")
print("Combined parquet created.")