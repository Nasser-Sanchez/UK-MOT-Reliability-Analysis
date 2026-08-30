import duckdb

con = duckdb.connect("uk_car_analyser.duckdb")

query = """
    WITH preprocess AS(
    SELECT source, 
    registration,
    CAST(firstUsedDate AS DATE) AS firstUsedDate, 
    CAST(registrationDate AS DATE) AS registrationDate, 
    make, model, fuelType, CAST(engineSize AS INT32) AS engineSize, 
    CAST(test_completedDate AS TIMESTAMP) AS test_completedDate,
    (DATE_DIFF('day', CAST(firstUsedDate AS DATE), CAST(test_completedDate AS DATE)))/365.25 AS years,
    CASE
        WHEN test_testResult == 'PASSED' THEN 0
        WHEN test_testResult == 'FAILED' OR defect_count_fail>0 THEN 1
    ELSE NULL
    END AS event,
    IF(test_odometerUnit = 'KM', 0.621371*CAST(test_odometerValue AS INT32), CAST(test_odometerValue AS INT32)) AS mileage,
    CAST(defect_count_fail AS INT1) AS defect_count_fail,
    CAST(defect_count_advisory AS INT1) AS defect_count_advisory,
    CAST(defect_count_dangerous AS INT1) AS defect_count_dangerous,

        COUNT(
            CASE WHEN CAST(test_odometerValue AS INT32) >0
            AND test_odometerUnit IN ('MI','KM')
                AND registration IS NOT NULL
                AND fuelType IN ('Petrol','Diesel','Hybrid Electric (Clean)', 'Electric', 'Electric Diesel' )
                AND(

                    (CAST(engineSize AS INT32) BETWEEN  999 AND 7000) 
                    OR fuelType = 'Electric' 
                )
                AND CAST(firstUsedDate AS DATE)<=CAST(test_completedDate AS DATE)
                AND test_testResult IN ('PASSED','FAILED')
            
        THEN 1
        ELSE 0
        END) OVER(PARTITION BY registration, registrationDate) AS filter
    FROM read_parquet('data/mot_api_parquet/mot_bulk*.parquet')
 
    )

    SELECT *, DENSE_RANK()OVER(ORDER BY registration, registrationDate) AS vehicle_id
    FROM preprocess
    --WHERE filter=0
"""

# Save the result
con.execute(f"COPY ({query}) TO 'data/mot_data_combined.parquet' (FORMAT PARQUET)")
print("Combined parquet created.")