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

processed_19 = rosters_19[['name', 'height','weight','team','position']]
processed_20 = rosters_20[['name','height','weight','team','position']]
processed_21 = rosters_21[['name','height','weight','team','position']]
processed_22 = rosters_22[['name','height','weight','team','position']]
processed_23 = rosters_23[['name','height','weight','team','position']]
processed_24 = rosters_24[['name','height','weight','team','position']]
processed_25 = rosters_25[['name','height','weight','team','position']]

