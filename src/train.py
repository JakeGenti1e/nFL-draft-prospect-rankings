import numpy as np
import pandas as pd
import xgboost as xgb 
from xgboost import XGBRegressor
from xgboost import plot_tree
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

draft_multiplier = 4
print("Loading data...")
df = pd.read_csv("player_data_2025_adjusted (2) - Copy.csv", low_memory=False)
df.columns = df.columns.str.replace("_x$", "", regex=True)
df = df.loc[:, ~df.columns.str.endswith("_y")]
df["team"] = df["team"].astype("category")
df.columns = df.columns.str.replace("_x_x$", "", regex=True)
df.columns = df.columns.str.replace("_y_y$", "", regex=True)
df["conference"] = df["conference"].astype("category")
df["position"] = df["position"].astype("category")
df["pro_team"] = df["pro_team"].astype("category")
df["unanimous"] = df["unanimous"].fillna(0)
df["is_all_american"] = df["is_all_american"].astype("Int64")
df["unanimous"] = df["unanimous"].astype("Int64")
df["unanimous"] = df["unanimous"]*2
df["Drafted"] = df["Drafted"].fillna(0)
df["height"] = df["height"]*0.1
df["weight"] = df["weight"]*0.1
df["career_stage"] = 0
df["rushing_YDS"] *= 1.5
df["rushing_TD"] *= 1.5
df["rushing_YPC"] *= 1.6
df["int_rate"] = df["passing_INT"] / df["passing_ATT"].replace(0, np.nan)
df["td_rate"] = df["passing_TD"] / df["passing_ATT"].replace(0, np.nan)
df["win_pct"] = (df["W"]/(df["W"]+df["L"]))
df["win_pct"] = df["win_pct"] + df["SRS"]
df["qb_score"] = (
    df["passing_YPA"] * 1.5 +
    df["passing_PCT"] * 0.30 +
    df["td_rate"] * 200 -
    df["int_rate"] * 300 +
    np.log1p(df["passing_ATT"]) * 1.5
)
df.loc[df["years_in_school"] <= 3, "career_stage"] = 4  
df.loc[df["years_in_school"] == 4, "career_stage"] = 1  
df.loc[df["years_in_school"] >= 5, "career_stage"] = 0  
df["passing_td_bonus"] = 0
df.loc[df["passing_TD"] >= 20, "passing_td_bonus"] = 2
df.loc[df["passing_TD"] >= 25, "passing_td_bonus"] = 3
df.loc[df["passing_TD"] >= 30, "passing_td_bonus"] = 4
df.loc[df["passing_TD"] >= 40, "passing_td_bonus"] = 6

df["first_season"] = df.groupby("playerId")["season"].transform("min")
num_cols = df.select_dtypes(include="number").columns
df["draft_score"] = 300-df["draft_pick_x"]
df = df.reset_index(drop = True)
print(df[["draft_pick_x", "draft_score"]].sample(20))
print(df["draft_score"].describe())
df["edge_pressure_score"] = (
    df["defensive_TFL"] * 1.5 +
    df["defensive_SACKS"] * 2.0
)

df["sack_share"] = df["defensive_SACKS"] / (df["defensive_TFL"] + 1)
df["db_score"] = (df["interceptions_INT"]*2)+(df["defensive_PD"])
df["ap_rank_score"] = 30-df["AP Rank"]
#print(df["team"].unique())
confrence_rankings = {
    "SEC": 2.5,
    "Big Ten": 2.5,
    "Notre Dame":2.4,
    "ACC": 2.3,
    "Pac-12": 2.3,
    "Big 12":2.3,
    "AAC":2.0,
    "MWC":1.8,
    "Pac-2":1.7,
    "Sun Belt":1.6,
    "MAC": 1.5,
    "C-USA": 1.4,
    "Independent": 1.3,
    "Service Academy": 1.0
}

df["confrence_weight"] = df["conference"].map(confrence_rankings)



print(sorted(df["conference"].dropna().astype(str).str.strip().unique()))
print("Creating splits...")
print(df["season"].dtype) 
print(df["season"].value_counts(dropna=False).sort_index())

player_score = {
    1:8,
    2:6,
    3:5,
    4:4,
    5:3,
    6:2,
    7:1
}

df["score"] = df["draft_round_x"].map(player_score)

df.loc[df["draft_pick_x"] <= 5, "score"] += 5
df.loc[df["draft_pick_x"] <= 10, "score"] += 3
df.loc[df["draft_pick_x"] <= 20, "score"] += 2



df["ap_rank_score"] = 30-df["AP Rank"]
qb_cols = [
    "career_stage",
    "name",
    "OSRS",
    "ap_rank_score",
    "win_pct",
    "Off",
    "height",
    "passing_PCT",
    "weight",
    "team",
    "season",
    "position",
    "confrence_weight",
    "is_all_american",
    "unanimous",
    "weighted_award_score",
    "passing_ATT",
    "passing_COMPLETIONS",
    "passing_INT",
    "passing_TD",
    "passing_YDS",
    "passing_YPA",
    "passing_td_bonus",
    "qb_score",
    "declared"
]

qb_df = df[df["position"] == "QB"][qb_cols + ["score"]]

qb_df.loc[qb_df["passing_ATT"] >= 300, "OSRS"]+=100
qb_df.loc[qb_df["passing_ATT"] >= 300, "win_pct"]+=15
qb_df.loc[qb_df["passing_ATT"] >= 300, "Off"]+=25

qb_df.loc[qb_df["passing_YDS"] >= 3000, "OSRS"]+=100
qb_df.loc[qb_df["passing_YDS"] >= 3000, "win_pct"]+=15
qb_df.loc[qb_df["passing_YDS"] >= 3000, "Off"]+=25

qb_df.loc[qb_df["passing_TD"] >= 25, "OSRS"]+=200
qb_df.loc[qb_df["passing_TD"] >= 25, "win_pct"]+=30
qb_df.loc[qb_df["passing_TD"] >= 25, "Off"]+=50

qb_df.loc[qb_df["passing_PCT"] >= 63, "OSRS"]+=100
qb_df.loc[qb_df["passing_PCT"] >= 63, "win_pct"]+=45
qb_df.loc[qb_df["passing_PCT"] >= 63, "Off"]+=35

qb_df.loc[qb_df["passing_PCT"] <= 63, "OSRS"]-=100
qb_df.loc[qb_df["passing_PCT"] <= 63, "win_pct"]-=45
qb_df.loc[qb_df["passing_PCT"] <= 63, "Off"]-=35

qb_df.loc[qb_df["passing_ATT"] <= 200, "qb_score"]-=300
qb_df.loc[qb_df["passing_ATT"] <= 300, "OSRS"]-=300
qb_df.loc[qb_df["passing_ATT"] <= 300, "win_pct"]-=45
qb_df.loc[qb_df["passing_ATT"] <= 300, "Off"]-=75

qb_df.loc[qb_df["passing_YDS"] <= 3000, "OSRS"]-=100
qb_df.loc[qb_df["passing_YDS"] <= 3000, "win_pct"]-=15
qb_df.loc[qb_df["passing_YDS"] <= 3000, "Off"]-=25

qb_df.loc[qb_df["passing_TD"] <= 25, "OSRS"]-=200
qb_df.loc[qb_df["passing_TD"] <= 25, "win_pct"]-=30
qb_df.loc[qb_df["passing_TD"] <= 25, "Off"]-=50
qb_df.loc[qb_df["passing_TD"] <= 20, "qb_score"]-=150

qb_df.loc[qb_df["is_all_american"] == 1, "OSRS"]+=250
qb_df.loc[qb_df["is_all_american"] == 1, "win_pct"]+=20



skill_cols = [
    "career_stage",
    "name",
    "height",
    "weight",
    "ap_rank_score",
    "win_pct",
    "OSRS",
    "Tot_Off",
    "Off",
    "confrence_weight",
    "season",
    "position",
    "is_all_american",
    "biletnikoff",
    "doak_walker",
    "john_mackey",
    "unanimous",
    "weighted_award_score",
    "receiving_LONG",
    "receiving_TD",
    "receiving_YDS",
    "receiving_YPR",
    "rushing_LONG",
    "rushing_TD",
    "rushing_YDS",
    "rushing_YPC",
    "heisman",
    "declared",
    "Vertical",
    "40yd",
    "3Cone",
    "Bench",
    "Broad Jump",
    "Shuttle"
]

skill_df = df[df["position"].isin(["RB","WR","TE"])][skill_cols + ["score"]]
skill_df.loc[skill_df["receiving_YDS"] >= 600, "OSRS"]+=100
skill_df.loc[skill_df["receiving_YDS"] >= 600, "Off"]+=25
skill_df.loc[skill_df["rushing_YDS"] >= 600, "OSRS"]+=100
skill_df.loc[skill_df["rushing_YDS"] >= 600, "Off"]+=25
skill_df.loc[skill_df["is_all_american"] == 1, "OSRS"]+=250
skill_df.loc[skill_df["receiving_YDS"] >= 600, "win_pct"]+=15
skill_df.loc[skill_df["rushing_YDS"] >= 600, "win_pct"]+=15
skill_df.loc[skill_df["40yd"]<=4.35, "OSRS"]+=500
skill_df.loc[skill_df["Vertical"]>=38, "OSRS"]+=475
skill_df.loc[skill_df["Broad Jump"]>=10.0, "OSRS"]+=450
skill_df.loc[skill_df["Shuttle"]<=4.05, "OSRS"]+=425
skill_df.loc[skill_df["3Cone"]<=6.70, "OSRS"]+=400


edge_cols = [
    "career_stage",
    "name",
    "position",
    "season",
    "DSRS",
    "ap_rank_score",
    "win_pct",
    "is_all_american",
    "unanimous",
    "ted_hendricks",      
    "vince_lombardi",
    "weighted_award_score",
    "confrence_weight",
    "defensive_SACKS",
    "defensive_TFL",
    "defensive_QB HUR",
    "defensive_TOT",
    "edge_pressure_score",
    "sack_share",
    "ronnie_lott",
    "defensive_SOLO",
    "declared",
    "Vertical",
    "40yd",
    "3Cone",
    "Bench",
    "Broad Jump",
    "Shuttle"
]

edge_df = df[df["position"].isin(["DE", "EDGE", "OLB","DL"])][edge_cols + ["score"]]
edge_df["defensive_SACKS"] = edge_df["defensive_SACKS"]*1.5
edge_df["defensive_TFL"] = edge_df["defensive_TFL"]*2
edge_df.loc[edge_df["edge_pressure_score"] >= 8, "DSRS"]+=100
edge_df.loc[edge_df["defensive_TOT"] >= 50, "DSRS"]+=500
edge_df.loc[edge_df["defensive_TOT"] <= 15, "DSRS"]-=1000
edge_df.loc[edge_df["is_all_american"] == 1, "DSRS"]+=250
edge_df.loc[edge_df["edge_pressure_score"] >= 8, "win_pct"]+=15
edge_df.loc[edge_df["defensive_TOT"] >= 50, "win_pct"]+=15
edge_df.loc[edge_df["weighted_award_score"] >= 1, "win_pct"]+=80
edge_df.loc[edge_df["weighted_award_score"] >= 1, "DSRS"]+=100
edge_df.loc[edge_df["is_all_american"] == 1, "DSRS"]+=120
edge_df.loc[edge_df["career_stage"]<=4, "DSRS"]+=100
edge_df.loc[edge_df["Vertical"] >= 38, "DSRS"]+=450
edge_df.loc[edge_df["40yd"] <= 4.90, "DSRS"] +=400
edge_df.loc[edge_df["3Cone"]<= 6.90, "DSRS"]+=475
edge_df.loc[edge_df["Broad Jump"] >= 10.0, "DSRS"]+=500
edge_df.loc[edge_df["Shuttle"] >= 4.20, "DSRS"]+=425
edge_df.loc[edge_df["Bench"] >= 25, "DSRS"]+=375

idl_cols = [
    "career_stage",
    "name",
    "season",
    "position",
    "DSRS",
    "Rush_Def",
    "Tot_Def",
    "is_all_american",
    "ap_rank_score",
    "weighted_award_score",
    "win_pct",
    "unanimous",
    "confrence_weight",
    "defensive_TFL",
    "defensive_TOT",
    "defensive_SOLO",
    "ronnie_lott",
    "declared",
    "Vertical",
    "40yd",
    "3Cone",
    "Bench",
    "Broad Jump",
    "Shuttle"
]

idl_df = df[df["position"].isin(["DT", "NT"])][idl_cols + ["score"]]
idl_df.loc[idl_df["defensive_TFL"] >= 7, "DSRS"]+=100
idl_df.loc[idl_df["defensive_TOT"] >= 40, "DSRS"]+=100
idl_df.loc[idl_df["career_stage"]<=4, "DSRS"]+=100
idl_df.loc[idl_df["is_all_american"] == 1, "DSRS"]+=250
idl_df.loc[idl_df["defensive_TFL"] >= 7, "win_pct"]+=15
idl_df.loc[idl_df["defensive_TOT"] >= 40, "win_pct"]+=15
idl_df.loc[idl_df["Vertical"] >= 34, "DSRS"]+=400
idl_df.loc[idl_df["40yd"] <= 4.90, "DSRS"] +=450
idl_df.loc[idl_df["3Cone"]<= 7.40, "DSRS"]+=370
idl_df.loc[idl_df["Broad Jump"] >= 9.6, "DSRS"]+=500
idl_df.loc[idl_df["Shuttle"] <= 4.40, "DSRS"]+=500
idl_df.loc[idl_df["Bench"] >= 30, "DSRS"]+=400

lb_cols = [
    "career_stage",
    "name",
    "season",
    "position",
    "DSRS",
    "Def",
    "ap_rank_score",
    "win_pct",
    "Tot_Def",
    "is_all_american",
    "unanimous",
    "dick_butkus",        
    "vince_lombardi",
    "weighted_award_score",
    "confrence_weight",
    "defensive_TOT",
    "defensive_TFL",
    "defensive_SOLO",
    "ronnie_lott",
    "declared",
    "Vertical",
    "40yd",
    "3Cone",
    "Bench",
    "Shuttle"
]

lb_df = df[df["position"].isin(["LB", "ILB"])][lb_cols + ["score"]]
lb_df.loc[lb_df["defensive_TFL"] >= 3, "DSRS"]+=100
lb_df.loc[lb_df["defensive_TOT"] >= 40, "DSRS"]+=100
lb_df.loc[lb_df["is_all_american"] == 1, "DSRS"]+=250
lb_df.loc[lb_df["defensive_TFL"] >= 7, "win_pct"]+=15
lb_df.loc[lb_df["defensive_TOT"] >= 7, "win_pct"]+=15
lb_df.loc[lb_df["career_stage"]<=4, "DSRS"]+=100
lb_df.loc[lb_df["weighted_award_score"] >= 1, "win_pct"]+=80
lb_df.loc[lb_df["weighted_award_score"] >= 1, "DSRS"]+=100
lb_df.loc[lb_df["defensive_TOT"] <= 20, "DSRS"]-=10000
lb_df.loc[lb_df["defensive_TOT"] <= 20, "DSRS"]-=10000
lb_df.loc[lb_df["Vertical"] >= 38, "DSRS"]+=400
lb_df.loc[lb_df["40yd"] <= 4.50, "DSRS"] +=450
lb_df.loc[lb_df["3Cone"]<= 6.90, "DSRS"]+=370
lb_df.loc[lb_df["Shuttle"] <= 4.25, "DSRS"]+=500
lb_df.loc[lb_df["Shuttle"] <= 4.25, "DSRS"]+=500
lb_df.loc[lb_df["Bench"] >= 25, "DSRS"]+400
db_cols = [
    "career_stage",
    "first_season",
    "name",
    "position",
    "season",
    "DSRS",
    "ap_rank_score",
    "win_pct",
    "Def",
    "Tot_Def",
    "Pass_Def",
    "is_all_american",
    "unanimous",
    "weighted_award_score",
    "confrence_weight",
    "interceptions_INT",
    "db_score",
    "Vertical",
    "40yd",
    "3Cone",
    "Shuttle",
    "interceptions_TD",
    "defensive_PD",
    "declared",
    "jim_thorpe",
    "defensive_TOT",
    "defensive_SOLO",
    "defensive_TFL",
    "ronnie_lott"
]

db_df = df[df["position"].isin(["CB", "DB","S"])][db_cols + ["score"]]
db_df.loc[db_df["defensive_PD"] >= 6, "DSRS"]+=300
db_df.loc[db_df["db_score"] >= 8, "DSRS"]+=300
db_df.loc[db_df["db_score"] >= 8, "win_pct"]+=30
db_df.loc[db_df["career_stage"]<=4, "DSRS"]+=100
db_df.loc[db_df["defensive_TOT"] <= 10, "DSRS"]-=20000
db_df.loc[db_df["defensive_TOT"] >= 40, "DSRS"]+=50
db_df.loc[db_df["defensive_TOT"] >= 80, "DSRS"]+=100
db_df.loc[db_df["defensive_TFL"] >= 3, "db_score"]+=50
db_df.loc[db_df["is_all_american"] == 1, "db_score"]+=200
db_df.loc[db_df["is_all_american"] == 1, "win_pct"]+=20
db_df.loc[db_df["is_all_american"] == 1, "win_pct"]+=20
db_df.loc[db_df["defensive_PD"] >= 6, "win_pct"]+=35
db_df.loc[db_df["defensive_PD"] == 0, "win_pct"]-=350
db_df.loc[db_df["db_score"] >= 10, "win_pct"]+=35
db_df.loc[db_df["weighted_award_score"] >= 1, "win_pct"]+=80
db_df.loc[db_df["weighted_award_score"] >= 1, "DSRS"]+=100
db_df.loc[db_df["Vertical"] >= 38, "DSRS"] +=500
db_df.loc[db_df["40yd"] <= 4.35, "DSRS"] +=500
db_df.loc[db_df["3Cone"]<= 6.70, "DSRS"]+=500
db_df.loc[db_df["Shuttle"] <= 4.05, "DSRS"]+=500




db_df["defensive_PD"] = db_df["defensive_PD"].fillna(0)*2
db_df = db_df.drop(columns=["first_season"])

qb_train = qb_df[qb_df["season"].isin([2019,2020,2021,2022])].copy()
qb_val = qb_df[qb_df["season"].isin([2023,2024])].copy()
qb_test = qb_df[(qb_df["season"] == 2025) & (qb_df["declared"] == 1)].copy()
qb_train = qb_train[(qb_train["score"].notna()) & (qb_train["position"].notna())]
qb_val = qb_val[qb_val["score"].notna()]


skill_train = skill_df[skill_df["season"].isin([2019,2020,2021,2022])].copy()
skill_val = skill_df[skill_df["season"].isin([2023,2024])].copy()
skill_test = skill_df[(skill_df["season"] == 2025) & (skill_df["declared"] == 1)].copy()
skill_train = skill_train[(skill_train["score"].notna()) & (skill_train["position"].notna())]
skill_val = skill_val[skill_val["score"].notna()]

edge_train = edge_df[edge_df["season"].isin([2019,2020,2021,2022])].copy()
edge_val = edge_df[edge_df["season"].isin([2023,2024])].copy()
edge_test = edge_df[(edge_df["season"] == 2025) & (edge_df["declared"] == 1)].copy()
edge_train = edge_train[(edge_train["score"].notna()) & (edge_train["position"].notna())]
edge_val = edge_val[edge_val["score"].notna()]

idl_train = idl_df[idl_df["season"].isin([2019,2020,2021,2022])].copy()
idl_val = idl_df[idl_df["season"].isin([2023,2024])].copy()
idl_test = idl_df[(idl_df["season"] == 2025) & (idl_df["declared"] == 1)].copy()
idl_train = idl_train[(idl_train["score"].notna()) & (idl_train["position"].notna())]
idl_val = idl_val[idl_val["score"].notna()]

lb_train = lb_df[lb_df["season"].isin([2019,2020,2021,2022])].copy()
lb_val = lb_df[lb_df["season"].isin([2023,2024])].copy()
lb_test = lb_df[(lb_df["season"] == 2025) & (lb_df["declared"] == 1)].copy()
lb_train = lb_train[(lb_train["score"].notna()) & (lb_train["position"].notna())]
lb_val = lb_val[lb_val["score"].notna()]

lb_train = lb_train.drop(columns=["season"])
lb_val = lb_val.drop(columns=["season"])
lb_test = lb_test.drop(columns=["season"])

db_train = db_df[db_df["season"].isin([2019,2020,2021,2022])].copy()
db_val = db_df[db_df["season"].isin([2023,2024])].copy()
db_test = db_df[(db_df["season"] == 2025) & (db_df["declared"] == 1)].copy()
db_train = db_train[(db_train["score"].notna()) & (db_train["position"].notna())]
db_val = db_val[db_val["score"].notna()]

db_train = db_train.drop(columns=["season"])
db_val = db_val.drop(columns=["season"])
db_test = db_test.drop(columns=["season"])

qb_X_train = qb_train.drop(columns=["score", "name"])
qb_y_train = qb_train["score"]

qb_X_val = qb_val.drop(columns=["score", "name"])
qb_y_val = qb_val["score"]

qb_X_test = qb_test.drop(columns=["score", "name"])


skill_X_train = skill_train.drop(columns=["score", "name"])
skill_y_train = skill_train["score"]

skill_X_val = skill_val.drop(columns=["score", "name"])
skill_y_val = skill_val["score"]

skill_X_test = skill_test.drop(columns=["score", "name"])

edge_X_train = edge_train.drop(columns=["score", "name"])
edge_y_train = edge_train["score"]

edge_X_val = edge_val.drop(columns=["score","name"])
edge_y_val = edge_val["score"]

edge_X_test = edge_test.drop(columns=["score", "name"])

idl_X_train = idl_train.drop(columns=["score", "name"])
idl_y_train = idl_train["score"]

idl_X_val = idl_val.drop(columns=["score","name"])
idl_y_val = idl_val["score"]

idl_X_test = idl_test.drop(columns=["score", "name"])

lb_X_train = lb_train.drop(columns=["score", "name"])
lb_y_train = lb_train["score"]

lb_X_val = lb_val.drop(columns=["score","name"])
lb_y_val = lb_val["score"]

lb_X_test = lb_test.drop(columns=["score", "name"])

db_X_train = db_train.drop(columns=["score", "name"])
db_y_train = db_train["score"]

db_X_val = db_val.drop(columns=["score","name"])
db_y_val = db_val["score"]

db_X_test = db_test.drop(columns=["score", "name"])


print(" qb train_df: ", qb_train.shape)
qb_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.70,
    subsample=0.70,
    reg_alpha =8,
    reg_lambda =8,
    min_child_weight = 8,
    max_depth = 2,
    learning_rate = 0.01,
    n_estimators = 90,
    random_state = 42
    )
print("test_df:", qb_test.shape)

qb_model.fit(qb_X_train, qb_y_train)



train_score = qb_model.score(qb_X_train, qb_y_train)
val_score = qb_model.score(qb_X_val, qb_y_val)
preds = qb_model.predict(qb_X_test)
skill_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.75,
    subsample = 0.75,
    reg_lambda = 8,
    min_child_weight = 8,
    max_depth = 2,
    learning_rate = 0.05,
    n_estimators = 100,
    reg_alpha= 2,
    random_state = 42
    )
skill_model.fit(skill_X_train, skill_y_train)
skill_preds = skill_model.predict(skill_X_test)



print(train_score)

#skill_model.fit(skill_X_train, skill_y_train)

print(val_score)

print(qb_test.columns)
results = qb_test.reset_index(drop=True).copy()
print(results.columns)

results["predicted_score"]= preds
rankings  = results.sort_values(["predicted_score","position"], ascending=[False, True])

print(rankings[["name", "predicted_score"]].head(32))

skill_results = skill_test.reset_index(drop=True).copy()
skill_results["predicted_score"]= skill_preds
skill_rankings  = skill_results.sort_values(["predicted_score","position"], ascending=[False, True])
print(skill_rankings[["name", "predicted_score"]].head(32))
edge_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.65,
    reg_lambda =6,
    min_child_weight = 2,
    max_depth = 2,
    learning_rate = 0.08,
    n_estimators = 115,
    reg_alpha= 2,
    random_state = 42
    )
edge_model.fit(edge_X_train, edge_y_train)
edge_preds = edge_model.predict(edge_X_test)

edge_results = edge_test.reset_index(drop=True).copy()
edge_results["predicted_score"]= edge_preds
edge_rankings  = edge_results.sort_values(["predicted_score","position"], ascending=[False, True])
print(edge_rankings[["name", "predicted_score"]].head(32))

idl_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.75,
    reg_lambda = 7,
    min_child_weight = 3,
    max_depth = 2,
    learning_rate = 0.08,
    n_estimators = 150,
    reg_alpha= 2,
    random_state = 42
    )
idl_model.fit(idl_X_train, idl_y_train)
idl_preds = idl_model.predict(idl_X_test)

idl_results = idl_test.reset_index(drop=True).copy()
idl_results["predicted_score"]= idl_preds
idl_rankings  = idl_results.sort_values(["predicted_score","position"], ascending=[False, True])
print(idl_rankings[["name", "predicted_score"]].head(32))

lb_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.75,
    reg_lambda = 8,
    min_child_weight = 4,
    max_depth = 3,
    learning_rate = 0.08,
    n_estimators = 150,
    reg_alpha= 2,
    random_state = 42
    )
lb_model.fit(lb_X_train, lb_y_train)
lb_preds = lb_model.predict(lb_X_test)

lb_results = lb_test.reset_index(drop=True).copy()
lb_results["predicted_score"]= lb_preds
lb_rankings  = lb_results.sort_values(["predicted_score","position"], ascending=[False, True])
print(lb_rankings[["name", "predicted_score"]].head(32))

db_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.75,
    reg_lambda = 6,
    min_child_weight = 2,
    max_depth = 2,
    learning_rate = 0.01,
    n_estimators = 70,
    reg_alpha= 2,
    random_state = 42
    )
db_model.fit(db_X_train, db_y_train)
db_preds = db_model.predict(db_X_test)

db_results = db_test.reset_index(drop=True).copy()
db_results["predicted_score"]= db_preds
db_rankings  = db_results.sort_values(["predicted_score","position"], ascending=[False, True])
print(db_rankings[["name", "predicted_score"]].head(32))

print("min:", preds.min())
print("max:", preds.max())
print("mean:", preds.mean())
print("std:", preds.std())


qb_feat_imp = pd.Series(qb_model.feature_importances_, index=qb_X_train.columns)
qb_feat_imp = qb_feat_imp.sort_values(ascending=False)

print(qb_feat_imp.head(15))

skill_feat_imp = pd.Series(skill_model.feature_importances_, index=skill_X_train.columns)
skill_feat_imp = skill_feat_imp.sort_values(ascending=False)

print(skill_feat_imp.head(15))

edge_feat_imp = pd.Series(edge_model.feature_importances_, index=edge_X_train.columns)
edge_feat_imp = edge_feat_imp.sort_values(ascending=False)

print(edge_feat_imp.head(15))

idl_feat_imp = pd.Series(idl_model.feature_importances_, index=idl_X_train.columns)
idl_feat_imp = idl_feat_imp.sort_values(ascending=False)

print(idl_feat_imp.head(15))

lb_feat_imp = pd.Series(lb_model.feature_importances_, index=lb_X_train.columns)
lb_feat_imp = lb_feat_imp.sort_values(ascending=False)

print(lb_feat_imp.head(15))

db_feat_imp = pd.Series(db_model.feature_importances_, index=db_X_train.columns)
db_feat_imp = db_feat_imp.sort_values(ascending=False)

print("db: ", db_feat_imp.head(15))

plt.figure(figsize=(100,50))
plot_tree(db_model, num_trees=0)
#plt.show()
