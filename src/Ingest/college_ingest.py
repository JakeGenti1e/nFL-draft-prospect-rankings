import os
import requests
import json
from pathlib import Path


API_KEY = os.environ.get("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

RAW_DATA_DIR = Path("data/raw/college_stats")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_headers():
    if API_KEY is None:
        raise RuntimeError("CFBD_API_KEY not found in environment variables")

    return {
        "Authorization": f"Bearer {API_KEY}"
    }


def get_player_stats(year:int, team: str, category:str):
    print("API key loaded:", API_KEY is not None)

    url = BASE_URL + "/stats/player/season"
    params = {
        "year": year,
        "team": team,
        "category": category
    }

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=30
    )

    print("Status code:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        return

    data = response.json()
    print("Records returned:", len(data))
    print("First record keys:", list(data[0].keys()))
    return data



def save_raw_data(data,year, team, category):
    file_path = RAW_DATA_DIR / f"{year}_{team}_{category}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved:", file_path)


def run_college_ingest(years, teams, categories):
    for year in years:
        for team in teams:
            for category in categories:
                data = get_player_stats(year, team, category)
                save_raw_data(data, year, team, category)

if __name__ == "__main__":
    run_college_ingest(
        years=[2019],
        teams=["LSU"],
        categories=["passing"]
    )