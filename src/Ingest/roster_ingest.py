import os
import time
import json
import requests
from pathlib import Path
import pandas as pd

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "rosters"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# CFBD Config
# -----------------------
API_BASE = os.getenv("CFBD_API_BASE", "https://api.collegefootballdata.com")
API_KEY = os.getenv("CFBD_API_KEY")

DEFAULT_TIMEOUT = 30
SLEEP_SEC = 0.6 


def _headers() -> dict:
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY not found in environment variables")
    return {"Accept": "application/json", "Authorization": f"Bearer {API_KEY}"}


def _get_json(url: str, params: dict | None = None):
    """GET with small diagnostics + basic retry on 429."""
    params = params or {}

    for attempt in range(1, 7):
        r = requests.get(url, headers=_headers(), params=params, timeout=DEFAULT_TIMEOUT)

        # Rate limit handling
        if r.status_code == 429:
              retry_after = r.headers.get("Retry-After")
              if retry_after is not None:
                wait = float(retry_after)
              else:
                wait = min(2 ** attempt, 30)  # exponential backoff cap
                print(f"[429] rate limited, sleeping {wait:.1f}s (attempt {attempt}/6)", flush=True)
                time.sleep(wait)
                continue
              
        # Non-JSON debugging (helps if you hit HTML)
        ct = (r.headers.get("Content-Type") or "").lower()
        if not r.ok:
            print("STATUS:", r.status_code, "URL:", url, "PARAMS:", params, flush=True)
            print("CONTENT-TYPE:", r.headers.get("Content-Type"), flush=True)
            print("BODY PREVIEW:", r.text[:250].replace("\n", " "), flush=True)
            r.raise_for_status()

        if "application/json" not in ct:
            print("STATUS:", r.status_code, "URL:", url, "PARAMS:", params, flush=True)
            print("CONTENT-TYPE:", r.headers.get("Content-Type"), flush=True)
            print("BODY PREVIEW:", r.text[:250].replace("\n", " "), flush=True)
            raise ValueError("Response was not JSON (see preview above).")

        return r.json()

    raise RuntimeError("Failed after retries (429 rate limiting).")


def fetch_fbs_teams(year: int) -> list[str]:
    """
    Returns a list of FBS team names for a given year.
    Endpoint: /teams/fbs?year=YYYY
    """
    url = f"{API_BASE}/teams/fbs"
    data = _get_json(url, params={"year": year})

    # Postman docs show this returns a list of team objects; commonly includes "school"
    teams = []
    if isinstance(data, list):
        for t in data:
            # robust: try a few likely keys
            name = t.get("school") or t.get("team") or t.get("name")
            if name:
                teams.append(name)

    teams = sorted(set(teams))
    print(f"[OK] teams for {year}: {len(teams)}", flush=True)
    return teams


def fetch_roster(year: int, team: str) -> list[dict]:
    """
    Endpoint: /roster?team=<team>&year=<year>
    Returns list of players with measurables (height/weight), position, etc.
    """
    url = f"{API_BASE}/roster"
    params = {"year": year, "team": team}
    return _get_json(url, params=params)


def fetch_rosters_year(year: int) -> list[dict]:
    teams = fetch_fbs_teams(year)

    all_rows: list[dict] = []
    for i, team in enumerate(teams, start=1):
        print(f"→ roster {year} [{i}/{len(teams)}] {team}", flush=True)
        rows = fetch_roster(year, team)

        # add team/year as explicit columns (even if API includes them)
        for row in rows:
            row.setdefault("team", team)
            row.setdefault("season", year)

        all_rows.extend(rows)
        time.sleep(SLEEP_SEC)

    raw_path = RAW_DIR / f"rosters_{year}.json"
    raw_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"[OK] saved raw rosters: {len(all_rows):,} rows -> {raw_path}", flush=True)

    return all_rows


def normalize_rosters(raw_rows: list[dict]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame()

    df = pd.json_normalize(raw_rows)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # CFBD roster rows commonly include athleteId; keep both if present
    rename_map = {
        "athleteid": "athlete_id",
        "id": "athlete_id",            # sometimes it's just "id"
        "playerid": "athlete_id",      # fallback if different naming
        "firstname": "first_name",
        "lastname": "last_name",
        "name": "player_name",
        "position": "position",
        "height": "height",
        "weight": "weight",
        "team": "team",
        "season": "season",
        "year": "class_year",          # FR/SO/JR/SR sometimes
        "home_town": "hometown",
        "hometown": "hometown",
        "home_state": "home_state",
        "state": "home_state",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Build player_name if missing but first/last present
    if "player_name" not in df.columns and {"first_name", "last_name"}.issubset(df.columns):
        df["player_name"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).str.strip()

    # Convert numeric measurables
    for col in ["height", "weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep a clean, join-friendly set (but don't be too strict)
    keep = [c for c in [
        "athlete_id", "player_name", "first_name", "last_name",
        "team", "position", "season", "class_year",
        "height", "weight"
    ] if c in df.columns]
    df = df[keep]

    # Drop duplicates (same player/team/season)
    dedupe_keys = [c for c in ["athlete_id", "team", "season"] if c in df.columns]
    if dedupe_keys:
        df = df.drop_duplicates(subset=dedupe_keys)

    return df.reset_index(drop=True)


def save_rosters(df: pd.DataFrame, year: int):
    if df.empty:
        print(f"[WARN] no roster rows to save for {year}", flush=True)
        return

    out_path = PROCESSED_DIR / f"rosters_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] saved processed rosters: {len(df):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    year = 2018
    raw = fetch_rosters_year(year)
    df = normalize_rosters(raw)
    print("[INFO] df shape:", df.shape, flush=True)
    print("[INFO] cols:", df.columns.tolist(), flush=True)
    save_rosters(df, year)