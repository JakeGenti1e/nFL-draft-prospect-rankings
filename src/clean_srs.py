import pandas as pd
import re

INPUT_FILE = "srs_2019_2025_clean.csv"
OUTPUT_FILE = "srs_2019_2025_clean_v2.csv"

df = pd.read_csv(INPUT_FILE)

# ---------- basic whitespace cleanup ----------
for col in ["School", "Conf"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s+\)", ")", regex=True)
        .str.replace(r"\(\s+", "(", regex=True)
        .str.strip()
    )

# ---------- normalize conference names ----------
conf_map = {
    "Big12": "Big 12",
    "Big Ten(East)": "Big Ten (East)",
    "Big Ten(West)": "Big Ten (West)",
    "SEC(East)": "SEC (East)",
    "SEC(West)": "SEC (West)",
    "ACC(Atlantic)": "ACC (Atlantic)",
    "ACC(Coastal)": "ACC (Coastal)",
    "Pac-12(North)": "Pac-12 (North)",
    "Pac-12(South)": "Pac-12 (South)",
    "Sun Belt(East)": "Sun Belt (East)",
    "Sun Belt(West)": "Sun Belt (West)",
    "MAC(East)": "MAC (East)",
    "MAC(West)": "MAC (West)",
    "MWC(Mountain)": "MWC (Mountain)",
    "MWC(West)": "MWC (West)",
    "American(East)": "American (East)",
    "American(West)": "American (West)",
    "CUSA(East)": "CUSA (East)",
    "CUSA(West)": "CUSA (West)",
    "Sun Belt East": "Sun Belt (East)",
    "Sun Belt West": "Sun Belt (West)",
}
df["Conf"] = df["Conf"].replace(conf_map)

# generic space before parentheses if missing
df["Conf"] = df["Conf"].str.replace(r"(?<!\s)\(", " (", regex=True)

# ---------- normalize school names ----------
school_map = {
    "Miami(FL)": "Miami (FL)",
    "Miami(OH)": "Miami (OH)",
    "Nevada-LasVegas": "Nevada-Las Vegas",
    "TexasChristian": "Texas Christian",
    "MiddleTennessee State": "Middle Tennessee State",
    "Middle TennesseeState": "Middle Tennessee State",
    "NorthCarolina State": "North Carolina State",
    "SouthFlorida": "South Florida",
    "EastCarolina": "East Carolina",
    "NorthTexas": "North Texas",
    "WesternKentucky": "Western Kentucky",
    "FloridaState": "Florida State",
    "OhioState": "Ohio State",
    "PennState": "Penn State",
    "IowaState": "Iowa State",
    "KansasState": "Kansas State",
    "OklahomaState": "Oklahoma State",
    "OregonState": "Oregon State",
    "WashingtonState": "Washington State",
    "BoiseState": "Boise State",
    "FresnoState": "Fresno State",
    "SanDiego State": "San Diego State",
    "SanJose State": "San Jose State",
    "UtahState": "Utah State",
    "ColoradoState": "Colorado State",
    "BallState": "Ball State",
    "KentState": "Kent State",
    "OhioState": "Ohio State",
    "BowlingGreen": "Bowling Green",
    "CentralMichigan": "Central Michigan",
    "EasternMichigan": "Eastern Michigan",
    "WesternMichigan": "Western Michigan",
    "NorthernIllinois": "Northern Illinois",
    "AppalachianState": "Appalachian State",
    "GeorgiaSouthern": "Georgia Southern",
    "GeorgiaState": "Georgia State",
    "ArkansasState": "Arkansas State",
    "TexasState": "Texas State",
    "LouisianaTech": "Louisiana Tech",
    "NewMexico": "New Mexico",
    "NewMexico State": "New Mexico State",
    "SouthernMississippi": "Southern Mississippi",
    "FloridaAtlantic": "Florida Atlantic",
    "FloridaInternational": "Florida International",
}
df["School"] = df["School"].replace(school_map)

# insert missing spaces in CamelCase school names as a fallback
def split_camel(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

df["School"] = df["School"].apply(split_camel).str.replace(r"\s+", " ", regex=True).str.strip()

# ---------- numeric cleanup ----------
numeric_cols = [
    "Rk", "season", "AP Rank", "W", "L",
    "OSRS", "DSRS", "SRS", "Off", "Def",
    "Pass_Off", "Pass_Def", "Rush_Off", "Rush_Def", "Tot_Off", "Tot_Def"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- remove obvious bad rows ----------
df = df.dropna(subset=["Rk", "season", "School", "Conf"])
df = df[df["season"].between(2019, 2025)]

# ---------- drop duplicates ----------
df = df.drop_duplicates(subset=["season", "Rk", "School"])

# ---------- sort ----------
df = df.sort_values(["season", "Rk"], ascending=[False, True]).reset_index(drop=True)
df["Conf"] = df["Conf"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved cleaned file to {OUTPUT_FILE}")
print(df.head(15))
print(df['season'].value_counts().sort_index())
print(df[['season', 'Rk', 'School', 'Conf']].head(25))