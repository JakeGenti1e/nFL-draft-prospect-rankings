import pandas as pd
from pathlib import Path

rosters_19 = pd.read_json("data/raw/rosters/rosters_2019.json")
rosters_20 = pd.read_json("data/raw/rosters/rosters_2020.json")
rosters_21 = pd.read_json("data/raw/rosters/rosters_2021.json")
rosters_22 = pd.read_json("data/raw/rosters/rosters_2022.json")
rosters_23 = pd.read_json("data/raw/rosters/rosters_2023.json")
rosters_24 = pd.read_json("data/raw/rosters/rosters_2024.json")
rosters_25 = pd.read_json("data/raw/rosters/rosters_2025.json")


rosters_19["name"] = (rosters_19["firstName"] + " " + rosters_19["lastName"]).str.lower().str.strip()
rosters_20["name"] = (rosters_20["firstName"] + " " + rosters_20["lastName"]).str.lower().str.strip()
rosters_21["name"] = (rosters_21["firstName"] + " " + rosters_21["lastName"]).str.lower().str.strip()
rosters_22["name"] = (rosters_22["firstName"] + " " + rosters_22["lastName"]).str.lower().str.strip()
rosters_23["name"] = (rosters_23["firstName"] + " " + rosters_23["lastName"]).str.lower().str.strip()
rosters_24["name"] = (rosters_24["firstName"] + " " + rosters_24["lastName"]).str.lower().str.strip()
rosters_25["name"] = (rosters_25["firstName"] + " " + rosters_25["lastName"]).str.lower().str.strip()

processed_rosters_19 = rosters_19[['name','height','weight','team','position', 'season']]
processed_rosters_20 = rosters_20[['name','height','weight','team','position', 'season']]
processed_rosters_21 = rosters_21[['name','height','weight','team','position', 'season']]
processed_rosters_22 = rosters_22[['name','height','weight','team','position', 'season']]
processed_rosters_23 = rosters_23[['name','height','weight','team','position', 'season']]
processed_rosters_24 = rosters_24[['name','height','weight','team','position', 'season']]
processed_rosters_25 = rosters_25[['name','height','weight','team','position', 'season']]

roster_years = [processed_rosters_19, processed_rosters_20, processed_rosters_21, processed_rosters_22, processed_rosters_23, processed_rosters_24, processed_rosters_25]
processed_roster = pd.concat(roster_years, ignore_index = True)

dfs = []
for year in range(2015, 2026):

    stats_raw = pd.read_json(f"data/raw/stats/player_stats_{year}.json")

    stats_raw["stat_name"] = stats_raw["category"] + "_" + stats_raw["statType"]

    stats_df = stats_raw.pivot_table(
        index=["season","playerId","player","position","team","conference"],
        columns="stat_name",
        values="stat",
        aggfunc="first"
    ).reset_index()

    dfs.append(stats_df)

all_stats = pd.concat(dfs, ignore_index=True) 
all_stats["name"] = all_stats["player"].str.lower().str.strip()
all_stats = all_stats.drop(columns=['player'])

processed_roster["team"] = processed_roster["team"].str.lower().str.strip()
all_stats["team"] = all_stats["team"].str.lower().str.strip()

player_data = pd.merge(
    processed_roster,
    all_stats,
    left_on= ["name","team", "season"],
    right_on= ["name","team", "season"],
    how="left"
)


all_americans_2010s = pd.read_csv("data/raw/honors/2010s_all_americans.csv")
all_americans_2020s = pd.read_csv("data/raw/honors/2020s_all_americans.csv")

all_americans_2020s["Unaminous"] = all_americans_2020s["Unaminous "].str.lower().str.strip()
total_all_americans = [all_americans_2010s, all_americans_2020s]
processed_all_americans = pd.concat(total_all_americans, ignore_index= True)
processed_all_americans.columns = processed_all_americans.columns.str.strip()

processed_all_americans["unanimous"] = (
    processed_all_americans.loc[:, "Unaminous"]
    .bfill(axis=1)
    .iloc[:, 0]
    .fillna(0)
)

processed_all_americans = processed_all_americans.loc[:, ~processed_all_americans.columns.duplicated()]
combine_data = pd.read_csv("data/raw/combine/Combine_data_repaired_v2.csv")
combine_data["draft_year"] = combine_data["Drafted"].str.extract(r"(\d{4})")
combine_data["draft_round"] = combine_data["Drafted"].str.extract(r"/ (\d+)(?:st|nd|rd|th) /")
combine_data["draft_pick"] = combine_data["Drafted"].str.extract(r"/ (\d+)(?:st|nd|rd|th) pick /")

combine_data["name"] = combine_data["Player"].str.lower().str.strip()

player_data["name"] = player_data["name"].str.lower().str.strip()
combine_data["player"] = combine_data["Player"].str.lower().str.strip()

player_data = pd.merge(
    player_data,
    combine_data,
    on="name",
    how="left"   
)


processed_all_americans["name"] = (
    processed_all_americans["Name"]
    .str.lower()
    .str.strip()
)

processed_all_americans.columns = processed_all_americans.columns.str.strip()
processed_all_americans["name"] = processed_all_americans["Name"].str.lower().str.strip()
processed_all_americans["season"] = processed_all_americans["Year"].astype(int)
processed_all_americans["is_all_american"] = 1

processed_all_americans = processed_all_americans.drop(columns=[
    "Year",
    "Name",
    "Stats",
    "Unaminous",
    "ID",
    "Unnamed: 6"
], errors="ignore")

print(processed_all_americans.columns)

aa_features = (
    processed_all_americans[["name", "season", "is_all_american"]]
    .drop_duplicates()
)

player_data = pd.merge(
    player_data,
    aa_features,
    on=["name", "season"],
    how="left"
)
player_data["is_all_american"] = player_data["is_all_american"].fillna(0).astype(int)
player_data = player_data.drop(columns=["player"])

draft_data = pd.read_csv("data/raw/draft/nfl_draft_pre_draft_only.csv")
draft_data = draft_data.drop(columns=["college_stats_label"])
draft_data = draft_data.drop(columns=["player_id"])
draft_data["name"] = draft_data["player"].str.lower().str.strip()
draft_data = draft_data.drop(columns=["player"])
draft_data["pro_team"] = draft_data["team"].str.lower().str.strip()
draft_data = draft_data.drop(columns=["team"])
draft_data["team"] = draft_data["college_university"].str.lower().str.strip()
draft_data = draft_data.drop(columns=["college_university"])
print(draft_data.head())

player_data = pd.merge(
    player_data,
    draft_data,
    on ="name",
    how ="left"
)
final_df = (
    player_data
    .sort_values("season")
    .groupby("name", as_index=False)
    .tail(1)
)

# --- COMBINE ---
combine_data["name"] = combine_data["Player"].str.lower().str.strip()

combine_data = combine_data[[
    col for col in [
        "name", "height", "weight", "40yd", "vertical",
        "bench", "broad_jump", "3cone", "shuttle"
    ] if col in combine_data.columns
]]

final_df = pd.merge(final_df, combine_data, on="name", how="left")

# --- DRAFT ---
draft_data["name"] = draft_data["player"].str.lower().str.strip()

draft_data = draft_data.rename(columns={
    "round": "draft_round",
    "pick": "draft_pick"
})

draft_data = draft_data[[
    col for col in ["name", "draft_round", "draft_pick"]
    if col in draft_data.columns
]]

final_df = pd.merge(final_df, draft_data, on="name", how="left")

final_df["drafted"] = final_df["draft_round"].notna().astype(int)

combine_data = combine_data.rename(columns={
    "height": "combine_height",
    "weight": "combine_weight",
    "40yd": "forty_time",
    "vertical": "vertical_jump",
    "bench": "bench_reps",
    "broad_jump": "broad_jump",
    "3cone": "three_cone",
    "shuttle": "shuttle_time"
})

final_df = pd.merge(final_df, combine_data, on="name", how="left")