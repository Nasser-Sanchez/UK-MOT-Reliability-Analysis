import duckdb
con = duckdb.connect("uk_car_analyser.duckdb")
con.execute("""
    COPY(
        WITH max_date AS(
            SELECT MAX(CAST(test_completedDate AS DATE)) AS last_mot_date
            FROM read_parquet('data/mot_data_combined.parquet')
        ),
        valid_makes AS (
            SELECT make
            FROM read_parquet('data/mot_data_combined.parquet')
            GROUP BY make
            HAVING COUNT(*) > 20
        ),
        
        valid_models AS (
            SELECT model
            FROM read_parquet('data/mot_data_combined.parquet')
            GROUP BY model
            HAVING COUNT(*) > 5
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
                mileage+(mileage/years),
                mileage
            ) AS mileage_estimate,
            IF(
                interval_censored=1,
                years+1,
                years
            ) AS years_estimate,
            
           make, model, 
           fuelType, engineSize,
        CASE
            WHEN engineSize = 0 OR engineSize IS NULL THEN 'Electric'
            WHEN engineSize < 1000 THEN '<1000'
            WHEN engineSize > 7000 THEN '7000+'
            ELSE
                (FLOOR(engineSize / 500) * 500)::VARCHAR || '-' || 
                (FLOOR(engineSize / 500) * 500 + 499)::VARCHAR
        END AS engineSize_bucket,
           SUM(defect_count_advisory)OVER(PARTITION BY registration) AS defect_count_advisory,
           SUM(defect_count_dangerous)OVER(PARTITION BY registration) AS defect_count_dangerous,
           last_test,
           IF(interval_censored=1, 1, event) AS event_interval
           
           FROM read_parquet('data/mot_data_combined.parquet') d
           JOIN max_date m ON 1=1
           
           
        )
    
        SELECT registration, make, model, fuelType, engineSize, engineSize_bucket, defect_count_advisory, defect_count_dangerous,
        years, mileage, event, years_estimate, mileage_estimate, event_interval
        FROM last_test_prep 
        WHERE last_test=1 AND mileage_estimate<3000000 
        AND NOT isinf(mileage_estimate) 
        AND make IN (SELECT make FROM valid_makes)
        AND model IN (SELECT model FROM valid_models)

    ) TO 'data/mot_last_test.parquet' (FORMAT PARQUET); 
""")