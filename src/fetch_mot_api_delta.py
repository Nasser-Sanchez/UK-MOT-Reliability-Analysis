import os, sys, requests
from pathlib import Path

# API auth
CLIENT_ID     = os.getenv("MOT_CLIENT_ID")
CLIENT_SECRET = os.getenv("MOT_CLIENT_SECRET")
API_KEY       = os.getenv("MOT_API_KEY")
TOKEN_URL     = os.getenv("MOT_TOKEN_URL")

DELTA_DIR = Path("data/mot_api_delta")


def get_access_token() -> str:
    if not TOKEN_URL or not CLIENT_ID or not CLIENT_SECRET:
        sys.exit("ERROR: Set MOT_CLIENT_ID, MOT_CLIENT_SECRET, MOT_TOKEN_URL as env vars.")
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type":    "client_credentials",
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope":         "https://tapi.dvsa.gov.uk/.default",
        },
        timeout=30,
    )
    resp.raise_for_status()
    print(f"  Token acquired (expires in ~{resp.json().get('expires_in', '?')}s)")
    return resp.json()["access_token"]


def request_files(token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-API-Key":     API_KEY,
        "Accept":        "application/json",
    }
    resp = requests.get(
        "https://history.mot.api.gov.uk/v1/trade/vehicles/bulk-download",
        headers=headers, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / 1e6
        print(f"Skip (exists, {size_mb:.1f} MB): {dest.name}")
        return dest
    print(f"  ↓ Downloading {dest.name} ...", end="", flush=True)
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  ↓ {pct:5.1f}% ({downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB)", end="", flush=True)
    print(f"\r Done ({dest.stat().st_size / 1e6:.1f} MB)  {dest}")
    return dest


def main():
    print("=" * 60)
    print("MOT History API — Delta fetcher")
    print("=" * 60)

    print("\n[1/2] Getting OAuth token …")
    token = get_access_token()

    print("\n[2/2] Requesting & downloading delta files…")
    files = request_files(token)
    delta_files = files.get("delta", [])
    if not delta_files:
        print("  No delta files available today.")
        return

    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    for item in delta_files:
        url = item["downloadUrl"]
        name = item["filename"].split("/")[-1]
        dest = DELTA_DIR / name
        download_file(url, dest)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Delta files -> {DELTA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()