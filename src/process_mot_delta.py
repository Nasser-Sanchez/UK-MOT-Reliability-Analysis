"""process_mot_delta.py — Extract MOT API delta zips, flatten NDJSON, write Parquet.

Streams .json.gz files from delta zips, explodes the nested motTests arrays
into flat rows, preserves the modification flag (CREATED/UPDATED/DELETED),
and writes parquet in batches.

Run this daily after fetching deltas. The output is meant to be merged into
the bulk parquet later using the modification flag.
"""

import os, sys, json, gzip, logging
from pathlib import Path
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mot_delta")

DELTA_DIR = Path("data/mot_api_delta")
OUT_DIR = Path("data/mot_api_parquet")

# Schema for the flattened output — includes modification flag
SCHEMA = pa.schema([
    ("source", pa.string()),
    ("registration", pa.string()),
    ("firstUsedDate", pa.string()),
    ("registrationDate", pa.string()),
    ("manufactureDate", pa.string()),
    ("make", pa.string()),
    ("model", pa.string()),
    ("fuelType", pa.string()),
    ("engineSize", pa.string()),
    ("test_completedDate", pa.string()),
    ("test_testResult", pa.string()),
    ("test_odometerValue", pa.string()),
    ("test_odometerUnit", pa.string()),
    ("test_motTestNumber", pa.string()),
    ("test_dataSource", pa.string()),
    ("test_expiryDate", pa.string()),
    ("test_registrationAtTimeOfTest", pa.string()),
    ("lastMotTestDate", pa.string()),
    ("defect_count_fail", pa.int32()),
    ("defect_count_advisory", pa.int32()),
    ("defect_count_dangerous", pa.int32()),
    ("modification", pa.string()),  # CREATED | UPDATED | DELETED
])


def flatten_vehicle(rec: dict, source: str) -> list[dict]:
    rows = []
    tests = rec.get("motTests") or []
    modification = rec.get("modification")  # Only present in delta
    
    # DELETED records have no motTests — still emit one row so we know to remove it
    if not tests and modification == "DELETED":
        rows.append({
            "source": source,
            "registration": rec.get("registration"),
            "firstUsedDate": None,
            "registrationDate": None,
            "manufactureDate": None,
            "make": None,
            "model": None,
            "fuelType": None,
            "engineSize": None,
            "test_completedDate": None,
            "test_testResult": None,
            "test_odometerValue": None,
            "test_odometerUnit": None,
            "test_motTestNumber": None,
            "test_dataSource": None,
            "test_expiryDate": None,
            "test_registrationAtTimeOfTest": None,
            "lastMotTestDate": None,
            "defect_count_fail": 0,
            "defect_count_advisory": 0,
            "defect_count_dangerous": 0,
            "modification": modification,
        })
        return rows
    
    for test in tests:
        if test.get("dataSource") == "dvla":
            continue
        defects = test.get("defects") or []
        fail_count = sum(1 for d in defects if d.get("type") in ("DANGEROUS", "MAJOR"))
        advisory_count = sum(1 for d in defects if d.get("type") in ("MINOR", "ADVISORY", "PRS"))
        dangerous_count = sum(1 for d in defects if d.get("type") == "DANGEROUS")
        rows.append({
            "source": source,
            "registration": rec.get("registration"),
            "firstUsedDate": rec.get("firstUsedDate"),
            "registrationDate": rec.get("registrationDate"),
            "manufactureDate": rec.get("manufactureDate"),
            "make": rec.get("make"),
            "model": rec.get("model"),
            "fuelType": rec.get("fuelType"),
            "engineSize": rec.get("engineSize"),
            "test_completedDate": test.get("completedDate"),
            "test_testResult": test.get("testResult"),
            "test_odometerValue": test.get("odometerValue"),
            "test_odometerUnit": test.get("odometerUnit"),
            "test_motTestNumber": test.get("motTestNumber"),
            "test_dataSource": test.get("dataSource"),
            "test_expiryDate": test.get("expiryDate"),
            "test_registrationAtTimeOfTest": test.get("registrationAtTimeOfTest"),
            "lastMotTestDate": test.get("lastMotTestDate"),
            "defect_count_fail": fail_count,
            "defect_count_advisory": advisory_count,
            "defect_count_dangerous": dangerous_count,
            "modification": modification,
        })
    return rows


def write_batch(rows: list[dict], idx: int):
    df = __import__("pandas").DataFrame(rows)
    
    # Convert all string-schema columns to string type, handling NaN → None
    string_cols = [f.name for f in SCHEMA if f.type == pa.string()]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    table = pa.Table.from_pandas(df, schema=SCHEMA)
    out_path = OUT_DIR / f"mot_delta_batch_{idx}.parquet"
    pq.write_table(table, out_path)
    log.info("  -> %s (%d rows)", out_path.name, len(rows))


def process_delta_zip(zip_path: Path, batch_rows: list, batch_size: int, parquet_idx: int):
    log.info("Processing delta %s ...", zip_path.name)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        gz_files = [f for f in z.namelist() if f.endswith('.json.gz')]
        
        if not gz_files:
            log.warning("No .json.gz files in %s", zip_path.name)
            return batch_rows, parquet_idx

        for gz_name in gz_files:
            log.info("  Reading %s ...", gz_name)
            with z.open(gz_name) as gz_file:
                with gzip.GzipFile(fileobj=gz_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        
                        batch_rows.extend(flatten_vehicle(rec, "delta"))
                        
                        if len(batch_rows) >= batch_size:
                            write_batch(batch_rows, parquet_idx)
                            parquet_idx += 1
                            batch_rows = []
    
    return batch_rows, parquet_idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    delta_zips = sorted(DELTA_DIR.glob("delta-*.zip"))
    
    if not delta_zips:
        log.error("No delta zip files found in %s", DELTA_DIR)
        return

    batch_rows = []
    batch_size = 5_000_000
    parquet_idx = len(list(OUT_DIR.glob("mot_delta_batch_*.parquet"))) + 1

    for zip_path in delta_zips:
        batch_rows, parquet_idx = process_delta_zip(zip_path, batch_rows, batch_size, parquet_idx)

    if batch_rows:
        write_batch(batch_rows, parquet_idx)

    log.info("\nDone. Parquet files in %s:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("mot_delta_batch_*.parquet")):
        log.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
