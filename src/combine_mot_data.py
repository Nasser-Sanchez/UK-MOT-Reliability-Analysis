import duckdb

con = duckdb.connect("uk_car_analyser.duckdb")

query = """
    SELECT *
    FROM read_parquet('data/mot_data_*.parquet')
    WHERE first_use_date>='2000-01-01'
        AND test_mileage IS NOT NULL
        AND(

            (cylinder_capacity BETWEEN 799 AND 7000) 
            OR fuel_type='EL'    
        )
        AND model<>'UNCLASSIFIED'
        
"""

# Save the result
con.execute(f"COPY ({query}) TO 'data/mot_data_combined.parquet' (FORMAT PARQUET)")
print("Combined parquet created.")