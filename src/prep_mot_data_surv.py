import duckdb
con = duckdb.connect("uk_car_analyser.duckdb")
con.execute("""
    COPY(
        WITH max_date AS(
            SELECT MAX(CAST(test_completedDate AS DATE)) AS last_mot_date
            FROM read_parquet('data/mot_data_combined.parquet')
        ),
        
        last_test_prep AS(
           SELECT registration, firstUsedDate, years, mileage, 
           event,
           IF(
               event<>1 AND DATE_ADD(test_completedDate, INTERVAL 13 MONTH)<m.last_mot_date,
               1,
               0
            ) AS interval_censored,
            IF(
                interval_censored=1,
                mileage+15000,
                mileage
            ) AS mileage_interval,
            IF(
                interval_censored=1,
                years+(13/12),
                years
            ) AS years_interval,
            
           make, model, 
           fuelType, engineSize,
           SUM(defect_count_advisory)OVER(PARTITION BY vehicle_id) AS defect_count_advisory,
           SUM(defect_count_dangerous)OVER(PARTITION BY vehicle_id) AS defect_count_dangerous,
           firstusedDate AS start_date, CAST(test_completedDate AS DATE) AS end_date, last_test,
           1 AS event_interval
           
           FROM read_parquet('data/mot_data_combined.parquet') d
           JOIN max_date m ON 1=1
           
           
        )
    
        SELECT * EXCLUDE(last_test) FROM last_test_prep WHERE last_test=1 AND mileage_interval<3000000 AND NOT isinf(mileage_interval)
    ) TO 'data/mot_last_test.parquet' (FORMAT PARQUET); 
""")