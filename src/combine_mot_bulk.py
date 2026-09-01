import duckdb

con = duckdb.connect("uk_car_analyser.duckdb")
con.execute("SET preserve_insertion_order = false;")
con.execute("SET temp_directory = 'data/duckdb_temp';")
con.execute("SET threads = 8;")  # Match your physical cores

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
        

            
                
        
        FROM read_parquet('data/mot_api_parquet/mot_bulk*.parquet')
    ),
    
    valid_data AS(

    SELECT *,
    bool_and(
        mileage BETWEEN 1 AND 3000000
        AND mileage IS NOT NULL
        AND firstUsedDate IS NOT NULL
        AND registration IS NOT NULL
        AND engineSize IS NOT NULL
        AND fuelType IN ('Petrol','Diesel','Hybrid Electric (Clean)', 'Electric', 'Electric Diesel' )
        AND(

            (CAST(engineSize AS INT32) BETWEEN  999 AND 7000) 
            OR fuelType = 'Electric' 
        )
        AND firstUsedDate<=CAST(test_completedDate AS DATE)
        AND event IS NOT NULL
        AND manufactureDate>='1990-01-01'
    )OVER (PARTITION BY registration) AS is_valid_group
    FROM preprocess
    QUALIFY is_valid_group=TRUE
    )

    SELECT *,
    ROW_NUMBER() OVER (PARTITION BY registration ORDER BY test_completedDate DESC) AS last_test
    FROM valid_data    
"""

# Save the result
con.execute(f"COPY ({query}) TO 'data/mot_data_combined.parquet' (FORMAT PARQUET)")
print("Combined parquet created.")