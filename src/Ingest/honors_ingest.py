import os
import time
import requests
from pathlib import Path
import pandas as pd
import json


API_KEY = os.environ.get("CFBD_API_KEY")
print("https://api.collegefootballdata.com")
BASE_URL = "https://api.collegefootballdata.com"

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "combine"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def headers():
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY not set")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

def fetch_combine_year(year: int) -> list[dict]:
    url = f"{BASE_URL}/draft/combine"
    print(f"→ Fetching combine data for {year}", flush=True)

    r = requests.get(url, headers=headers(), params={"year": year}, timeout=30)

    print("← status:", r.status_code, flush=True)
    print("← content-type:", r.headers.get("Content-Type"), flush=True)
    print("← body preview:", r.text[:250].replace("\n", " "), flush=True)

    r.raise_for_status()

    # If it's not JSON, fail with a clear message
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/json" not in ct:
        raise ValueError("Combine endpoint did not return JSON (see body preview above).")

    data = r.json()
    print(f"Retrieved {len(data):,} combine rows", flush=True)
    return data

def save_raw(data: list[dict], year: int):
    path = RAW_DIR / f"combine_{year}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved raw → {path}", flush=True)


def normalize_combine(data: list[dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    df = pd.json_normalize(data)

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename common fields
    rename_map = {
        "playerid": "player_id",
        "year": "draft_year",
        "fortyyd": "forty",
        "vertical": "vertical",
        "broadjump": "broad_jump",
        "threecone": "three_cone",
        "shuttle": "shuttle",
        "benchpress": "bench"
    }

    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Convert numeric fields
    numeric_cols = [
        "height", "weight", "forty", "vertical",
        "broad_jump", "three_cone", "shuttle", "bench"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate by player + draft year
    if {"player_id", "draft_year"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["player_id", "draft_year"])

    print("Normalized shape:", df.shape, flush=True)
    return df


def save_processed(df: pd.DataFrame, year: int):
    if df.empty:
        print("No combine data to save")
        return

    path = PROCESSED_DIR / f"combine_{year}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved processed → {path}", flush=True)


if __name__ == "__main__":
    year = 2024
    raw = fetch_combine_year(year)
    save_raw(raw, year)

    df = normalize_combine(raw)
    save_processed(df, year)

    print("Combine ingest complete", flush=True)