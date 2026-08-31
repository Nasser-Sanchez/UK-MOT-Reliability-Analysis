import duckdb

con = duckdb.connect("uk_car_analyser.duckdb")

query = """
    WITH preprocess AS(
        SELECT source, 
        registration,
        CAST(firstUsedDate AS DATE) AS firstUsedDate, 
        CAST(manufactureDate AS DATE) AS manufactureDate, 
        CAST(test_completedDate AS TIMESTAMP) AS test_completedDate,
        make, model, 
        IF(fuelType = 'Electric' AND CAST(engineSize AS INT32)>0, 'Hybrid Electric (Clean)', fuelType) AS fuelType,
        CAST(IF(fuelType = 'Electric', '0', engineSize) AS INT32) AS engineSize, 
        (DATE_DIFF('day', CAST(firstUsedDate AS DATE), CAST(test_completedDate AS DATE)))/365.25 AS years,
        CASE
            WHEN test_testResult == 'PASSED' THEN 0
            WHEN test_testResult == 'FAILED' OR defect_count_fail>0 THEN 1
        ELSE NULL
        END AS event,
        CASE 
            WHEN test_odometerUnit = 'KM' THEN 0.621371*CAST(test_odometerValue AS INT32) 
            WHEN test_odometerUnit = 'MI' THEN CAST(test_odometerValue AS INT32)
            ELSE NULL
        END AS mileage,
        CAST(defect_count_fail AS INT32) AS defect_count_fail,
        CAST(defect_count_advisory AS INT32) AS defect_count_advisory,
        CAST(defect_count_dangerous AS INT32) AS defect_count_dangerous,
        DENSE_RANK()OVER(ORDER BY registration, manufactureDate) AS vehicle_id,
        ROW_NUMBER()OVER(PARTITION BY registration, manufactureDate ORDER BY test_completedDate DESC) AS last_test

            
                
        
        FROM read_parquet('data/mot_api_bulk_parquet/mot_bulk*.parquet')
    ),
    
    bad_groups AS(

    SELECT vehicle_id
    FROM preprocess
    WHERE NOT(mileage BETWEEN 1 AND 3000000
        AND mileage IS NOT NULL
        AND firstUsedDate IS NOT NULL
        AND registration IS NOT NULL
        AND engineSize IS NOT NULL
        AND fuelType IN ('Petrol','Diesel','Hybrid Electric (Clean)', 'Electric', 'Electric Diesel' )
        AND(

            (CAST(engineSize AS INT32) BETWEEN  999 AND 7000) 
            OR fuelType = 'Electric' 
        )
        AND CAST(firstUsedDate AS DATE)<=CAST(test_completedDate AS DATE)
        AND event IS NOT NULL
        AND manufactureDate>='1990-01-01'
        )
    )

    SELECT p.*
    FROM preprocess p
    LEFT JOIN bad_groups b USING(vehicle_id)
    WHERE b.vehicle_id IS NULL
    
"""

# Save the result
con.execute(f"COPY ({query}) TO 'data/mot_data_combined.parquet' (FORMAT PARQUET)")
print("Combined parquet created.")