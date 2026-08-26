import zipfile
import os
import urllib.request
import urllib.parse
import shutil
import gzip
import pyzipper
import struct
import subprocess
import tarfile

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


def download_file(url, dest_path):
    """Download a file, skip if it already exists."""
    if os.path.exists(dest_path):
        print(f"Skipping (already exists): {os.path.basename(dest_path)}")
        return dest_path
    print(f" Downloading {os.path.basename(dest_path)} ...")
    urllib.request.urlretrieve(url, dest_path)
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"     → {size_mb:.1f} MB")
    return dest_path


def extract_gz(filepath, dest_dir):
    """Extract a .txt.gz file."""
    out_txt = os.path.splitext(os.path.basename(filepath))[0]  # strip .gz
    out_path = os.path.join(dest_dir, out_txt)
    with gzip.open(filepath, "rb") as gz_in, open(out_path, "wb") as txt_out:
        shutil.copyfileobj(gz_in, txt_out)
    print(f"     ✓ Extracted {out_txt}")


def extract_zip(filepath, dest_dir):
    """
    Extract any archive type using 7-Zip.
    Handles: zip, tar, bz2, gz, deflate64, split archives.
    """
    try:
        # 7z x -o<dest> -y
        # x = extract with full paths
        # -o = output directory
        # -y = overwrite without prompt
        result = subprocess.run(
            ["7z", "x", filepath, f"-o{dest_dir}", "-y"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Count extracted files
        file_count = len([f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))])
        print(f"     ✓ Extracted {file_count} files")
        
    except FileNotFoundError:
        print(f"     ✗ 7-Zip not found in PATH. Install 7-Zip or add it to PATH.")
    except subprocess.CalledProcessError as e:
        print(f"     ✗ Extraction failed: {e.stderr}")

def download_and_extract_year(year, filename, base_url, dest_dir, file_type="dft"):
    """Download and extract a single year's data."""
    # Create year-specific subfolder
    year_folder = os.path.join(dest_dir, str(year))
    os.makedirs(year_folder, exist_ok=True)

    # Download URL
    if file_type == "dft":
        url = base_url + filename
    else:
        url = base_url + filename

    # Download to year folder
    download_path = os.path.join(year_folder, filename)
    download_file(url, download_path)

    # Extract
    if filename.endswith(".zip"):
        extract_zip(download_path, year_folder)
    elif filename.endswith(".gz"):
        extract_gz(download_path, year_folder)

    return year_folder


# Download DfT results 
print("DfT (data.gov.uk) — data_results 2005–2023")
print("=" * 60)
for year, filename in dft_results.items():
    print(f"\n--- {year} ---")
    folder = download_and_extract_year(year, filename, DFT_BASE, mot_folder, "dft")
    file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    print(f"   Total files in {year}/: {file_count}")

# Download DVSA results
print("\n" + "-" * 60)
print("DVSA Open Data — data_results 2024–2025")
print("=" * 60)
for year, filenames in dvsa_results.items():
    print(f"\n--- {year} ---")
    for filename in filenames:
        year_folder = os.path.join(mot_folder, str(year))
        os.makedirs(year_folder, exist_ok=True)
        download_and_extract_year(year, filename, DVSA_BASE, mot_folder, "dvsa")

# Summary
print("\n" + "=" * 60)
print("Done! Directory structure:")
print("=" * 60)
for item in sorted(os.listdir(mot_folder)):
    full_path = os.path.join(mot_folder, item)
    if os.path.isdir(full_path):
        file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
        print(f"  {item}/  ({file_count} files)")
    else:
        size_mb = os.path.getsize(full_path) / 1e6
        print(f"  {item}  ({size_mb:.1f} MB)")