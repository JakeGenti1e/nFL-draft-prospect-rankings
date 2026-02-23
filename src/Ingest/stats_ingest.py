import os
import requests
import json
from pathlib import Path
import pandas as pd
import time


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "stats"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = os.getenv("CFBD_API_BASE", "https://api.collegefootballdata.com")
API_KEY = os.getenv("CFBD_API_KEY") 

PLAYER_STATS_ENDPOINT = os.getenv("PLAYER_STATS_ENDPOINT", "/stats/player/season")


def _headers() -> dict:
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

DEFAULT_TIMEOUT = 30
REQUEST_SLEEP_SEC = 0.2 


def _get(url: str, params: dict) -> list | dict:
    """Generic GET with error handling. Returns parsed JSON."""
    r = requests.get(url, headers=_headers(), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()



def fetch_player_stats(year: int) -> list:
    url = f"{API_BASE}{PLAYER_STATS_ENDPOINT}"
    params = {"year": year}  # no page/per_page

    data = _get(url, params=params)
    rows = data.get("data", []) if isinstance(data, dict) else data

    raw_path = RAW_DIR / f"player_stats_{year}.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return rows


def normalize_stats(raw_data: list) -> pd.DataFrame:
    """
    Flatten JSON into a clean rectangular table.
    You should edit the 'keep_cols' mapping once you see your raw keys.
    """
    if not raw_data:
        return pd.DataFrame()

    df = pd.json_normalize(raw_data)

    # ---- EDIT THIS SECTION AFTER YOU PRINT df.columns ONCE ----
    # A common pattern: rename messy columns to stable snake_case names.
    rename_map = {
        "playerId": "player_id",
        "player": "player_name",
        "team": "school",
        "school": "school",
        "position": "position",
        "year": "season",
        # examples of stat fields (update to match your API)
        "stats.passing.yards": "pass_yards",
        "stats.passing.tds": "pass_tds",
        "stats.rushing.yards": "rush_yards",
        "stats.rushing.tds": "rush_tds",
        "stats.receiving.yards": "rec_yards",
        "stats.receiving.tds": "rec_tds",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Pick a minimal stable set if present (prevents wide messy tables)
    keep_cols = [c for c in [
        "player_id", "player_name", "school", "position", "season",
        "pass_yards", "pass_tds", "rush_yards", "rush_tds", "rec_yards", "rec_tds",
    ] if c in df.columns]

    if keep_cols:
        df = df[keep_cols]
    # ----------------------------------------------------------

    # Basic cleanup
    # Convert numeric columns safely
    for col in df.columns:
        if col.endswith(("yards", "tds")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing the join keys you’ll need later
    key_cols = [c for c in ["player_id", "season"] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols)

    # Remove duplicates
    if set(["player_id", "season"]).issubset(df.columns):
        df = df.drop_duplicates(subset=["player_id", "season","school"])

    return df.reset_index(drop=True)



def save_player_stats(df: pd.DataFrame, year: int):
    if df.empty:
        return

    out_path = PROCESSED_DIR / f"player_season_stats_{year}.parquet"
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    year = 2023
    raw = fetch_player_stats(year)
    df = normalize_stats(raw)
    save_player_stats(df, year)
  