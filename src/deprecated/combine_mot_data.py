import duckdb

con = duckdb.connect("uk_car_analyser.duckdb")

query = """
    SELECT *
    FROM read_parquet('data/mot_data_*.parquet')
    WHERE test_mileage IS NOT NULL
        AND(

            (cylinder_capacity BETWEEN  950 AND 7000) 
            OR fuel_type='EL'    
        )
        AND model<>'UNCLASSIFIED'
        AND first_use_date<=test_date
        
"""

# Save the result
con.execute(f"COPY ({query}) TO 'data/mot_data_combined.parquet' (FORMAT PARQUET)")
print("Combined parquet created.")