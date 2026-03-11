import pandas as pd
from pathlib import Path

rosters_19 = pd.read_json("data/raw/rosters/rosters_2019.json")
rosters_20 = pd.read_json("data/raw/rosters/rosters_2020.json")
rosters_21 = pd.read_json("data/raw/rosters/rosters_2021.json")
rosters_22 = pd.read_json("data/raw/rosters/rosters_2022.json")
rosters_23 = pd.read_json("data/raw/rosters/rosters_2023.json")
rosters_24 = pd.read_json("data/raw/rosters/rosters_2024.json")
rosters_25 = pd.read_json("data/raw/rosters/rosters_2025.json")

'''
print(rosters_19[['height','weight','team','position']].isnull().sum())
print(rosters_20[['height','weight','team','position']].isnull().sum())
print(rosters_21[['height','weight','team','position']].isnull().sum())
print(rosters_22[['height','weight','team','position']].isnull().sum())
print(rosters_23[['height','weight','team','position']].isnull().sum())
print(rosters_24[['height','weight','team','position']].isnull().sum())
print(rosters_25[['height','weight','team','position']].isnull().sum())
'''

rosters_19["name"] = rosters_19["firstName"] + " " + rosters_19["lastName"]
rosters_20["name"] = rosters_20["firstName"] + " " + rosters_20["lastName"]
rosters_21["name"] = rosters_21["firstName"] + " " + rosters_21["lastName"]
rosters_22["name"] = rosters_22["firstName"] + " " + rosters_22["lastName"]
rosters_23["name"] = rosters_23["firstName"] + " " + rosters_23["lastName"]
rosters_24["name"] = rosters_24["firstName"] + " " + rosters_24["lastName"]
rosters_25["name"] = rosters_25["firstName"] + " " + rosters_25["lastName"]

processed_rosters_19 = rosters_19[['name','height','weight','team','position', 'season']]
processed_rosters_20 = rosters_20[['name','height','weight','team','position', 'season']]
processed_rosters_21 = rosters_21[['name','height','weight','team','position', 'season']]
processed_rosters_22 = rosters_22[['name','height','weight','team','position', 'season']]
processed_rosters_23 = rosters_23[['name','height','weight','team','position', 'season']]
processed_rosters_24 = rosters_24[['name','height','weight','team','position', 'season']]
processed_rosters_25 = rosters_25[['name','height','weight','team','position', 'season']]

roster_years = [processed_rosters_19, processed_rosters_20, processed_rosters_21, processed_rosters_22, processed_rosters_23, processed_rosters_24, processed_rosters_25]
processed_roster = pd.concat(roster_years, ignore_index = True)
print(processed_roster)

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

print(stats_df.shape)
print(stats_df.dtypes)
print(stats_df.head())

all_stats = pd.concat(dfs, ignore_index=True) 
print(all_stats)  



player_data = pd.merge(
    processed_roster,
    all_stats,
    left_on=["name","team", "season"],
    right_on=["player","team", "season"],
    how="inner"
)

print(player_data.shape)
print(player_data.head())
print(player_data.isnull().sum())

print(player_data.columns)