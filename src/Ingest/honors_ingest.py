'''
from pathlib import Path
import json
import time
import pandas as pd
import requests

API_KEY = "YOUR_API_KEY"
API_BASE = "https://api.collegefootballdata.com"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

def _get_json(url: str, params: dict | None = None):
    resp = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30
    )

    resp.raise_for_status()
    return resp.json()
RAW_DIR = Path("data/raw/honors")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_awards(year: int) -> list[dict]:
    url = f"{API_BASE}/awards"
    params = {"year": year}
    return _get_json(url, params=params)

def get_or_fetch_awards(year: int) -> list[dict]:
    cache_path = RAW_DIR / f"honors_{year}.json"

    if cache_path.exists():
        print(f"[cache] {year}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[fetch] {year}")
    data = fetch_awards(year)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    time.sleep(2)
    return data

def main():
    all_rows = []

    for year in range(2019, 2026):
        rows = get_or_fetch_awards(year)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(df.head())

    out_path = Path("data/processed/honors.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved to {out_path}")
'''