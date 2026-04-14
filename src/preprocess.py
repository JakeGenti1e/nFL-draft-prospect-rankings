import pandas as pd
from pathlib import Path
from conference_mapping import build_fbs_conference_map
from conference_mapping import map_conference

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

processed_rosters_19 = rosters_19[['name','height','weight','team','position', 'season', 'id']]
processed_rosters_20 = rosters_20[['name','height','weight','team','position', 'season', 'id']]
processed_rosters_21 = rosters_21[['name','height','weight','team','position', 'season', 'id']]
processed_rosters_22 = rosters_22[['name','height','weight','team','position', 'season', 'id']]
processed_rosters_23 = rosters_23[['name','height','weight','team','position', 'season', 'id']]
processed_rosters_24 = rosters_24[['name','height','weight','team','position', 'season', 'id']]
processed_rosters_25 = rosters_25[['name','height','weight','team','position', 'season', 'id']]

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
all_stats = all_stats.drop(columns=['player', 'conference'])

processed_roster["team"] = processed_roster["team"].str.lower().str.strip()
all_stats["team"] = all_stats["team"].str.lower().str.strip()
all_stats.to_csv("data/raw/stats/all_stats.csv")
processed_roster["conference"] = processed_roster.apply(
    lambda row: map_conference(row["team"], row["season"]),
    axis=1
)
print(all_stats.duplicated(["playerId","season"]).sum())

processed_roster["playerId"] = processed_roster["id"]
processed_roster = processed_roster.drop(columns= ["id"])


player_data = pd.merge(
    processed_roster,
    all_stats,
    on= ["name","team","playerId", "season"],
    how="left"
)


#player_data.to_csv("data/processed/player_data.csv")

player_data["position"] = player_data["position_x"]
player_data["position"] = player_data["position"].fillna("none")
player_data["position"] = player_data["position"].replace("?","none")

player_data = player_data.drop(columns=["position_y", "position_x"])

all_americans_2010s = pd.read_csv("data/raw/honors/2010s_all_americans.csv")
all_americans_2020s = pd.read_csv("data/raw/honors/2020s_all_americans.csv")

#all_americans_2020s["Unaminous"] = all_americans_2020s["Unaminous "].str.lower().str.strip()
total_all_americans = [all_americans_2010s, all_americans_2020s]
processed_all_americans = pd.concat(total_all_americans, ignore_index= True)
processed_all_americans.columns = processed_all_americans.columns.str.strip()
'''
processed_all_americans["unanimous"] = (
    processed_all_americans.loc[:, "Unaminous"]
    .bfill(axis=1)
    .iloc[:, 0]
    .fillna(0)
)'''

processed_all_americans = processed_all_americans.loc[:, ~processed_all_americans.columns.duplicated()]
combine_data = pd.read_csv("data/raw/combine/Combine_data_repaired_v2.csv")
combine_data["draft_year"] = combine_data["Drafted"].str.extract(r"(\d{4})")
combine_data["draft_round"] = combine_data["Drafted"].str.extract(r"/ (\d+)(?:st|nd|rd|th) /")
combine_data["draft_pick"] = combine_data["Drafted"].str.extract(r"/ (\d+)(?:st|nd|rd|th) pick /")

combine_data["name"] = combine_data["Player"].str.lower().str.strip()

#player_data["name"] = player_data["name_y"].str.lower().str.strip()
#player_data = player_data.drop(columns=["name_x", "name_y"])
combine_data = combine_data.drop(columns=["Player"])

processed_all_americans["name"] = (
    processed_all_americans["Name"]
    .str.lower()
    .str.strip()
)

processed_all_americans.columns = processed_all_americans.columns.str.strip()
processed_all_americans["name"] = processed_all_americans["Name"].str.lower().str.strip()
processed_all_americans["season"] = processed_all_americans["Year"]#.astype(int)
processed_all_americans["is_all_american"] = 1

processed_all_americans = processed_all_americans.drop(columns=[
    "Stats",    
    "ID",
    "Unnamed: 6"
], errors="ignore")

print(processed_all_americans.columns)

aa_features = (
    processed_all_americans[["name", "season", "is_all_american", "Unaminous"]]
    .drop_duplicates()
)

aa_features.to_csv("data/raw/draft/aa_features.csv")
unanimous_aa = {
    "Yes":1,
    "No":0
}
aa_features["unanimous"] = aa_features["Unaminous"].map(unanimous_aa)
#player_data.to_csv("data/processed/player_data.csv")

player_data = pd.merge(
    player_data,
    aa_features,
    on=["name", "season"],
    how="left"
)

player_data["is_all_american"] = player_data["is_all_american"].fillna(0).astype(int)
print(player_data.head)

draft_data = pd.read_csv("data/raw/draft/nfl_draft_pre_draft_only.csv")
draft_data = draft_data.drop(columns=["college_stats_label"])
draft_data = draft_data.drop(columns=["player_id"])
draft_data["name"] = draft_data["player"].str.lower().str.strip()
draft_data = draft_data.drop(columns=["player"])
draft_data["pro_team"] = draft_data["team"].str.lower().str.strip()
draft_data = draft_data.drop(columns=["team"])
draft_data["team"] = draft_data["college_university"].str.lower().str.strip()
draft_data = draft_data.drop(columns=["college_university"])
draft_data = draft_data.rename(columns={
    "round": "draft_round",
    "pick": "draft_pick"
})


print(draft_data.head())
draft_data.to_csv("data/processed/draft_data.csv")

combine_data.to_csv("data/processed/combine_data.csv")
processed_all_americans.to_csv("data/processed/all_americans.csv")
combine_data["team"] = combine_data["School"]
combine_data = combine_data.drop(columns=["School"])
combine_data["position"] = combine_data["Pos"]
combine_data = combine_data.drop(columns=["Pos"])



player_data = pd.merge(
    player_data,
    draft_data,
    on = ["name", "position","team"],
    how = "left"
)

player_data = pd.merge(
    player_data,
    combine_data,
    on = ["name","position","team"],
    how = "left" 
)
#player_data = player_data["position"].fillna("none")
player_data.to_csv("data/processed/player_data.csv")
