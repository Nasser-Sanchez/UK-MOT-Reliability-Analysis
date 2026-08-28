import duckdb
con = duckdb.connect("uk_car_analyser.duckdb")
con.execute("""
    COPY(
        WITH last_test_prep AS(
           SELECT vehicle_id, (DATE_DIFF('day', first_use_date, test_date))/365.25 AS t, 
           IF(test_result IN ('ABA', 'F'), 1, 0) AS event,
           IF(
               test_result NOT IN ('ABA', 'F') AND test_date<'2024-12-31',
               1,
               0
            ) AS interval_censored,
           make, model, 
           fuel_type, cylinder_capacity, test_mileage,
           first_use_date AS start_date, test_date AS end_date,
           ROW_NUMBER()OVER(PARTITION BY vehicle_id ORDER BY test_date DESC) AS rownum
           
           FROM read_parquet('data/mot_data_combined.parquet')
           WHERE first_use_date<test_date
           
        )
    
        SELECT * EXCLUDE(rownum) FROM last_test_prep WHERE rownum=1
    ) TO 'data/mot_last_test.parquet' (FORMAT PARQUET); 
""")