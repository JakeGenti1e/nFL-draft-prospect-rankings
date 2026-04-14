import pandas as pd

def normalize_team_name(team: str) -> str:
    team = str(team).lower().strip()

    aliases = {
        "ole miss": "mississippi",
        "miami (oh)": "miami ohio",
        "miami-oh": "miami ohio",
        "ulm": "louisiana monroe",
        "louisiana-monroe": "louisiana monroe",
        "louisiana monroe": "louisiana monroe",
        "ull": "louisiana",
        "ul lafayette": "louisiana",
        "louisiana-lafayette": "louisiana",
        "utsa": "texas san antonio",
        "utep": "texas el paso",
        "uab": "alabama birmingham",
        "smu": "southern methodist",
        "tcu": "texas christian",
        "ucf": "central florida",
        "usf": "south florida",
        "unc": "north carolina",
        "nc state": "north carolina state",
        "ecu": "east carolina",
        "byu": "brigham young",
        "lsu": "louisiana state",
        "pitt": "pittsburgh",
        "southern miss": "southern mississippi",
        "hawai'i": "hawaii",
        "san josé state": "san jose state",
        "fau": "florida atlantic",
        "fiu": "florida international",
        "app state": "appalachian state",
        "k-state": "kansas state",
        "ok state": "oklahoma state",
        "washington st": "washington state",
        "washington state university": "washington state",
        "oregon st": "oregon state",
        "texas a&m": "texas a&m",
        "sam houston state": "sam houston",
        "jax state": "jacksonville state",
        "southern mississippi": "southern mississippi",
        "la tech": "louisiana tech",
        "miss st": "mississippi state",
        "ohio st": "ohio state",
        "san diego st": "san diego state",
        "boise st": "boise state",
        "fresno st": "fresno state",
        "utah st": "utah state",
        "kent st": "kent state",
        "ball st": "ball state",
        "bowling green state": "bowling green",
        "middle tennessee state": "middle tennessee",
    }

    return aliases.get(team, team)


def build_fbs_conference_map():
    # -------------------------
    # 2019–2022 alignment
    # -------------------------
    y2019_2022 = {
        # ACC
        "boston college": "ACC",
        "clemson": "ACC",
        "duke": "ACC",
        "florida state": "ACC",
        "georgia tech": "ACC",
        "louisville": "ACC",
        "miami": "ACC",
        "north carolina": "ACC",
        "north carolina state": "ACC",
        "pittsburgh": "ACC",
        "syracuse": "ACC",
        "virginia": "ACC",
        "virginia tech": "ACC",
        "wake forest": "ACC",

        # Big Ten
        "illinois": "Big Ten",
        "indiana": "Big Ten",
        "iowa": "Big Ten",
        "maryland": "Big Ten",
        "michigan": "Big Ten",
        "michigan state": "Big Ten",
        "minnesota": "Big Ten",
        "nebraska": "Big Ten",
        "northwestern": "Big Ten",
        "ohio state": "Big Ten",
        "penn state": "Big Ten",
        "purdue": "Big Ten",
        "rutgers": "Big Ten",
        "wisconsin": "Big Ten",

        # Big 12
        "baylor": "Big 12",
        "iowa state": "Big 12",
        "kansas": "Big 12",
        "kansas state": "Big 12",
        "oklahoma": "Big 12",
        "oklahoma state": "Big 12",
        "texas": "Big 12",
        "texas christian": "Big 12",
        "texas tech": "Big 12",
        "west virginia": "Big 12",

        # Pac-12
        "arizona": "Pac-12",
        "arizona state": "Pac-12",
        "california": "Pac-12",
        "colorado": "Pac-12",
        "oregon": "Pac-12",
        "oregon state": "Pac-12",
        "southern california": "Pac-12",
        "stanford": "Pac-12",
        "ucla": "Pac-12",
        "utah": "Pac-12",
        "washington": "Pac-12",
        "washington state": "Pac-12",

        # SEC
        "alabama": "SEC",
        "arkansas": "SEC",
        "auburn": "SEC",
        "florida": "SEC",
        "georgia": "SEC",
        "kentucky": "SEC",
        "louisiana state": "SEC",
        "mississippi": "SEC",
        "mississippi state": "SEC",
        "missouri": "SEC",
        "south carolina": "SEC",
        "tennessee": "SEC",
        "texas a&m": "SEC",
        "vanderbilt": "SEC",

        # AAC
        "cincinnati": "AAC",
        "east carolina": "AAC",
        "houston": "AAC",
        "memphis": "AAC",
        "navy": "AAC",
        "south florida": "AAC",
        "southern methodist": "AAC",
        "temple": "AAC",
        "tulane": "AAC",
        "tulsa": "AAC",
        "central florida": "AAC",

        # C-USA
        "charlotte": "C-USA",
        "florida international": "C-USA",
        "florida atlantic": "C-USA",
        "louisiana tech": "C-USA",
        "marshall": "C-USA",
        "middle tennessee": "C-USA",
        "north texas": "C-USA",
        "rice": "C-USA",
        "southern mississippi": "C-USA",
        "utep": "C-USA",
        "texas san antonio": "C-USA",
        "uab": "C-USA",
        "western kentucky": "C-USA",
        "old dominion": "C-USA",

        # MAC
        "akron": "MAC",
        "ball state": "MAC",
        "bowling green": "MAC",
        "buffalo": "MAC",
        "central michigan": "MAC",
        "eastern michigan": "MAC",
        "kent state": "MAC",
        "miami ohio": "MAC",
        "northern illinois": "MAC",
        "ohio": "MAC",
        "toledo": "MAC",
        "western michigan": "MAC",

        # Mountain West
        "air force": "MWC",
        "boise state": "MWC",
        "colorado state": "MWC",
        "fresno state": "MWC",
        "hawaii": "MWC",
        "nevada": "MWC",
        "new mexico": "MWC",
        "san diego state": "MWC",
        "san jose state": "MWC",
        "unlv": "MWC",
        "utah state": "MWC",
        "wyoming": "MWC",

        # Sun Belt
        "appalachian state": "Sun Belt",
        "arkansas state": "Sun Belt",
        "coastal carolina": "Sun Belt",
        "georgia southern": "Sun Belt",
        "georgia state": "Sun Belt",
        "louisiana": "Sun Belt",
        "louisiana monroe": "Sun Belt",
        "south alabama": "Sun Belt",
        "texas state": "Sun Belt",
        "troy": "Sun Belt",

        # Independents
        "army": "Independent",
        "brigham young": "Independent",
        "liberty": "Independent",
        "new mexico state": "Independent",
        "notre dame": "Notre Dame",
        "umass": "Independent",
        "connecticut": "Independent",
    }

    # -------------------------
    # 2023 alignment
    # -------------------------
    y2023 = dict(y2019_2022)

    # Big 12 adds four
    for team in ["brigham young", "cincinnati", "houston", "central florida"]:
        y2023[team] = "Big 12"

    # AAC loses 3, adds 6
    for team in [
        "charlotte", "florida atlantic", "north texas",
        "rice", "uab", "texas san antonio"
    ]:
        y2023[team] = "AAC"

    # C-USA reset
    for team in [
        "jacksonville state", "liberty", "new mexico state", "sam houston"
    ]:
        y2023[team] = "C-USA"

    # C-USA remaining core in 2023
    for team in [
        "florida international", "louisiana tech", "middle tennessee",
        "utep", "western kentucky"
    ]:
        y2023[team] = "C-USA"

    # Sun Belt additions already effectively in by this era
    for team in ["james madison", "marshall", "old dominion", "southern mississippi"]:
        y2023[team] = "Sun Belt"

    # No longer independents in 2023
    # BYU, Liberty, NMSU moved out
    # Army / Notre Dame / UMass / UConn remain independent-ish for football, ND custom
    y2023["army"] = "Independent"
    y2023["umass"] = "Independent"
    y2023["connecticut"] = "Independent"
    y2023["notre dame"] = "Notre Dame"

    # -------------------------
    # 2024–2025 alignment
    # -------------------------
    y2024_2025 = dict(y2023)

    # ACC adds SMU
    y2024_2025["southern methodist"] = "ACC"

    # Big Ten adds Pac-12 schools
    for team in ["oregon", "ucla", "southern california", "washington"]:
        y2024_2025[team] = "Big Ten"

    # SEC adds Texas, Oklahoma
    for team in ["texas", "oklahoma"]:
        y2024_2025[team] = "SEC"

    # Big 12 adds four former Pac-12 schools
    for team in ["arizona", "arizona state", "colorado", "utah"]:
        y2024_2025[team] = "Big 12"

    # AAC adds Army
    y2024_2025["army"] = "AAC"

    # C-USA adds Kennesaw State
    y2024_2025["kennesaw state"] = "C-USA"

    # Pac-12 becomes Pac-2 in your custom model
    y2024_2025["oregon state"] = "Pac-2"
    y2024_2025["washington state"] = "Pac-2"

    return {
        2019: y2019_2022,
        2020: y2019_2022,
        2021: y2019_2022,
        2022: y2019_2022,
        2023: y2023,
        2024: y2024_2025,
        2025: y2024_2025,
    }

FBS_CONFERENCE_MAP = build_fbs_conference_map()


def map_conference(team: str, season: int) -> str | None:
    season = int(season)
    team = normalize_team_name(team)

    season_map = FBS_CONFERENCE_MAP.get(season)
    if season_map is None:
        raise ValueError(f"Season {season} not supported. Use 2019–2025.")

    return season_map.get(team)