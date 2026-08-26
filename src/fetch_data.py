import zipfile
import os
import urllib.request
import urllib.parse
import shutil
import gzip

data_folder = "data"
mot_folder = "data/mot_results"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

if not os.path.exists(mot_folder):
    os.makedirs(mot_folder)

# DfT (data.gov.uk) — results files 2005–2023 
# Format switch: pre-2017 is .txt.gz, 2017+ is .zip
DFT_BASE = "https://data.dft.gov.uk/anonymised-mot-test/test_data/"

dft_results = {}
for year in range(2005, 2024):
    if year < 2017:
        dft_results[year] = f"test_result_{year}.txt.gz"
    else:
        dft_results[year] = f"dft_test_result_{year}.zip"

# DVSA Open Data — results files 2024–2025 
DVSA_BASE = "https://edh-dvsa-data-gov-uk-files-prod.s3.eu-west-1.amazonaws.com/"

dvsa_results = {
    2024: [
        "MOT+testing+data+results+(2024).zip",
        "dft_test_result_extracts_2024.zip",
    ],
    2025: [
        "dft_test_result_extracts_2025.zip",
    ],
}

def download_and_extract(url, filename, dest_dir, extract=True):
    filepath = f"{dest_dir}/{filename}"
    if os.path.exists(filepath):
        print(f"    Skipping (exists): {filename}")
        return filepath

    print(f"    Downloading {filename} ...")
    urllib.request.urlretrieve(url, str(filepath))

    if extract:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(dest_dir)
            print(f"     ✓ Extracted {len(zf.namelist())} files")
        elif filename.endswith(".gz"):
            out_txt = filename[:-3]
            out_path = f"{dest_dir}/{out_txt}"
            with gzip.open(filepath, "rb") as gz_in, open(out_path, "wb") as txt_out:
                shutil.copyfileobj(gz_in, txt_out)
            print(f"     ✓ Extracted {out_txt}")

    return filepath

# Download DfT results
print("DfT (data.gov.uk) — data_results 2005–2023")
print("=" * 60)
for year, filename in dft_results.items():
    url = DFT_BASE + filename
    download_and_extract(url, filename, mot_folder)

# Download DVSA results
print()
print("-" * 60, "\n")
print("DVSA Open Data — data_results 2024–2025")
print("=" * 60)
for year, filenames in dvsa_results.items():
    print(f"\n--- {year} ---")
    for filename in filenames:
        url = DVSA_BASE + filename
        download_and_extract(url, filename, mot_folder)