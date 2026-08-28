"""process_mot_api.py — Extract MOT API bulk/delta zips, flatten NDJSON, write Parquet.

Streams .json_gz files from the bulk zip, explodes the nested motTests arrays
into flat rows, and writes parquet in batches.
"""

import os, sys, json, gzip, logging
from pathlib import Path
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mot_api")

BULK_DIR = Path("data/mot_api_bulk")
DELTA_DIR = Path("data/mot_api_delta")
OUT_DIR = Path("data/mot_api_parquet")

# Schema for the flattened output
SCHEMA = pa.schema([
    ("source", pa.string()),  # 'bulk' or 'delta'
    ("registration", pa.string()),
    ("firstUsedDate", pa.string()),
    ("make", pa.string()),
    ("model", pa.string()),
    ("fuelType", pa.string()),
    ("engineSize", pa.string()),
    ("test_completedDate", pa.string()),
    ("test_testResult", pa.string()),
    ("test_odometerValue", pa.string()),
    ("test_odometerUnit", pa.string()),
    ("test_motTestNumber", pa.uint64()),
    ("test_dataSource", pa.string()),
    ("defects", pa.string()),  # JSON string of defect array
])


def flatten_vehicle(rec: dict, source: str) -> list[dict]:
    """One vehicle record -> list of flat rows (one per motTest)."""
    rows = []
    tests = rec.get("motTests") or []
    for test in tests:
        if test.get("dataSource") == "dvla":
            continue  # DVLA records have no test data
        defects = test.get("defects") or []
        defects_json = json.dumps(defects)
        
        rows.append({
            "source": source,
            "registration": rec.get("registration"),
            "firstUsedDate": rec.get("firstUsedDate"),
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
            "defects": defects_json,
        })
    return rows


def write_batch(rows: list[dict], idx: int):
    """Write a batch of rows to parquet."""
    table = pa.Table.from_pandas(
        __import__("pandas").DataFrame(rows),
        schema=SCHEMA,
    )
    out_path = OUT_DIR / f"mot_api_batch_{idx}.parquet"
    pq.write_table(table, out_path)
    log.info("  -> %s (%d rows)", out_path.name, len(rows))


def process_zip(zip_path: Path, source: str):
    """Process a single zip file."""
    log.info("Processing %s (source: %s) ...", zip_path.name, source)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        gz_files = [f for f in z.namelist() if f.endswith('.json_gz')]
    
    if not gz_files:
        log.warning("No .json_gz files in %s", zip_path.name)
        return

    for gz_name in gz_files:
        log.info("  Extracting & reading %s ...", gz_name)
        with z.open(gz_name) as gz_file:
            for line in gz_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                batch_rows.extend(flatten_vehicle(rec, source))
                
                if len(batch_rows) >= batch_size:
                    write_batch(batch_rows, parquet_idx)
                    parquet_idx += 1
                    batch_rows = []


def process_files():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find zips in both dirs
    bulk_zips = sorted(BULK_DIR.glob("bulk-*.zip"))
    delta_zips = sorted(DELTA_DIR.glob("delta-*.zip"))
    
    if not bulk_zips and not delta_zips:
        log.error("No zip files found in %s or %s", BULK_DIR, DELTA_DIR)
        return

    batch_rows = []
    batch_size = 500_000  # Adjust based on RAM
    parquet_idx = len(list(OUT_DIR.glob("*.parquet"))) + 1

    # Process bulk first
    for zip_path in bulk_zips:
        process_zip(zip_path, "bulk")
    
    # Process deltas
    for zip_path in delta_zips:
        process_zip(zip_path, "delta")

    # Flush remaining
    if batch_rows:
        write_batch(batch_rows, parquet_idx)

    log.info("\nDone. Parquet files in %s:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        log.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    process_files()