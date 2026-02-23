import os
import time
import requests
from pathlib import Path
import pandas as pd

BASE_URL = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY")

BASE_DIR = Path(__file__).resolve().parents[2]
DRAFT_DIR = BASE_DIR / "data" / "raw" / "draft"
DRAFT_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV  = DRAFT_DIR / "draft_picks_1970_2021.csv"
OUT_CSV = DRAFT_DIR / "draft_picks_1970_2025.csv"

DEFAULT_TIMEOUT = 30
SLEEP_SEC = 0.2

base = pd.read_csv(IN_CSV)
base = snake_cols(base)

# Detect the name column in your base CSV
name_col = None
for c in ["player_name", "player", "name"]:
    if c in base.columns:
        name_col = c
        break
if not name_col:
    raise RuntimeError(f"No player name column found in base CSV. Columns: {base.columns.tolist()}")

print("Using base name column:", name_col, flush=True)


def headers() -> dict:
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY not found in environment variables")
    return {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


def fetch_draft_picks_year(year: int) -> list[dict]:
    """
    CFBD /draft/picks often returns a list for the year.
    We try pagination, but also stop if page repeats (prevents infinite loops).
    """
    all_rows: list[dict] = []
    page = 1
    max_pages = 50  # safety

    while page <= max_pages:
        r = requests.get(
            f"{BASE_URL}/draft/picks",
            headers=headers(),
            params={"year": year, "page": page},
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        # stop if endpoint ignores page and returns the same first page again
        if all_rows and data == all_rows[: len(data)]:
            break

        all_rows.extend(data)
        page += 1
        time.sleep(SLEEP_SEC)

    return all_rows


def snake_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def normalize_new_to_match_base(new_df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    Map CFBD columns into the naming your CSV uses.
    This is a best-guess mapping; if your CSV uses different names,
    you'll tweak rename_map after printing columns once.
    """
    new_df = snake_cols(new_df)

    # Common CFBD fields: year, round, pick, overall, team, school, position, player, playerId
    rename_map = {
        "year": "draft_year",
        "overall": "overall_pick",
        "team": "nfl_team",
        "school": "college",
        "player": "player_name",
        "playerid": "player_id",
    }

    # Only rename if the destination exists in your base CSV (prevents creating new columns)
    for src, dst in rename_map.items():
        if src in new_df.columns and dst in base_cols:
            new_df = new_df.rename(columns={src: dst})

    # If your base CSV uses "year" not "draft_year", handle that
    if "year" in base_cols and "draft_year" in new_df.columns and "year" not in new_df.columns:
        new_df = new_df.rename(columns={"draft_year": "year"})

    # Keep only the columns your base CSV already has (alignment!)
    keep = [c for c in base_cols if c in new_df.columns]
    new_df = new_df[keep]

    return new_df


def choose_dedupe_keys(cols: list[str]) -> list[str]:
    """
    Prefer stable draft uniqueness keys.
    """
    candidates = [
        ["draft_year", "overall_pick"],
        ["year", "overall_pick"],
        ["year", "overall"],
        ["draft_year", "overall"],
        ["draft_year", "round", "pick"],
        ["year", "round", "pick"],
    ]
    for keys in candidates:
        if all(k in cols for k in keys):
            return keys
    return []


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    base = pd.read_csv(IN_CSV)
    base = snake_cols(base)

    print("[INFO] base rows:", len(base))
    print("[INFO] base cols:", base.columns.tolist())

    # fetch new years
    frames = []
    for y in [2022, 2023, 2024, 2025]:
        rows = fetch_draft_picks_year(y)
        print(f"[OK] fetched {len(rows):,} rows for {y}")
        frames.append(pd.json_normalize(rows))

    new = pd.concat(frames, ignore_index=True)
    new = normalize_new_to_match_base(new, base.columns.tolist())
    # CFBD draft data usually uses 'player' for the name
    if "player" in new.columns and name_col != "player":
        new = new.rename(columns={"player": name_col})
    
    combined = pd.concat([base, new], ignore_index=True)

    dedupe_keys = choose_dedupe_keys(combined.columns.tolist())
    if dedupe_keys:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_keys)
        print(f"[INFO] deduped on {dedupe_keys}: {before} -> {len(combined)}")
    else:
        before = len(combined)
        combined = combined.drop_duplicates()
        print(f"[INFO] deduped on all-cols: {before} -> {len(combined)}")

    combined.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {len(combined):,} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()