import numpy as np
import pandas as pd
import xgboost as xgb 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import graphviz
import matplotlib.pyplot as plt

draft_multiplier = 4
print("Loading data...")
df = pd.read_csv("data/processed/player_data_ii.csv", index_col = 0)
df["team"] = df["team"].astype("category")
df["conference"] = df["conference"].astype("category")
df["position"] = df["position"].astype("category")
df["pro_team"] = df["pro_team"].astype("category")
df["unanimous"] = df["unanimous"].fillna(0)
df["is_all_american"] = df["is_all_american"].astype("Int64")
df["unanimous"] = df["unanimous"].astype("Int64")
df["unanimous"] = df["unanimous"]*4
df["Drafted"] = df["Drafted"].fillna(0)
df["height"] = df["height"]*0.1
df["weight"] = df["weight"]*0.1
df["early_declare"] = (df["declared"] == 1) & (df["years_in_school"] <=3).astype(int)
df["first_season"] = df.groupby("playerId")["season"].transform("min")
df = df.drop(columns = ["draft_year_y", "draft_round_y", "draft_pick_y", "pro_team","puntReturns_AVG", "punting_LONG", "puntReturns_YDS", "kickReturns_LONG","kickReturns_YDS", "puntReturns_TD", "puntReturns_LONG", "punting_YPP", "puntReturns_NO", "kickReturns_AVG", "kickReturns_NO", "punting_NO", "punting_YDS", "kickReturns_TD", "punting_In 20", "Unaminous","College","age", "Ht","Wt"])
num_cols = df.select_dtypes(include="number").columns
df["draft_score"] = 300-df["draft_pick_x"]
df = df.reset_index(drop = True)
print(df[["draft_pick_x", "draft_score"]].sample(20))
print(df["draft_score"].describe())

#print(df["team"].unique())
confrence_rankings = {
    "SEC": 2.5,
    "Big Ten": 2.5,
    "ACC": 2.3,
    "Pac-12": 2.3,
    "Big 12":2.3,
    "Notre Dame":2.3,
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
    1:7,
    2:6,
    3:5,
    4:4,
    5:3,
    6:2,
    7:1
}

df["score"] = df["draft_round_x"].map(player_score)

qb_cols = [
    "early_declare",
    "first_season",
    "name",
    "height",
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
    "passing_PCT",
    "passing_TD",
    "passing_YDS",
    "passing_YPA",
    "declared"
]

qb_df = df[df["position"] == "QB"][qb_cols + ["score"]]


skill_cols = [
    "early_declare",
    "first_season",
    "name",
    "height",
    "weight",
    "confrence_weight",
    "team",
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
    "declared",
]

skill_df = df[df["position"].isin(["RB","WR","TE"])][skill_cols + ["score"]]

def_cols = [
    "early_declare",
    "first_season",
    "name",
    "height",
    "weight",
    "team",
    "season",
    "position",
    "is_all_american",
    "unanimous",
    "dick_butkus",
    "ted_hendricks",
    "vince_lombardi",
    "weighted_award_score",
    "confrence_weight",
    "interceptions_AVG",
    "interceptions_INT",
    "interceptions_TD",
    "interceptions_YDS",
    "defensive_PD",
    "defensive_QB HUR",
    "defensive_SACKS",
    "defensive_SOLO",
    "defensive_TD",
    "defensive_TFL",
    "defensive_TOT",
    "declared"
]

def_df = df[df["position"].isin(["DL","DE","NT","DT","LB", "CB", "S", "DB"])][def_cols + ["score"]]




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

def_train = def_df[def_df["season"].isin([2019,2020,2021,2022])].copy()
def_val = def_df[def_df["season"].isin([2023,2024])].copy()
def_test = def_df[(def_df["season"] == 2025) & (def_df["declared"] == 1)].copy()
def_train = def_train[(def_train["score"].notna()) & (def_train["position"].notna())]
def_val = def_val[def_val["score"].notna()]


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

def_X_train = def_train.drop(columns=["score", "name"])
def_y_train = def_train["score"]

def_X_val = def_val.drop(columns=["score","name"])
def_y_val = def_val["score"]

def_X_test = def_test.drop(columns=["score", "name"])

print(" qb train_df: ", qb_train.shape)
qb_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.75,
    subsample=0.75,
    reg_alpha =8,
    reg_lambda =8,
    min_child_weight = 8,
    max_depth = 2,
    learning_rate = 0.03,
    n_estimators = 100,
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

results["predicted_round"]= preds
rankings  = results.sort_values(["predicted_round","position"], ascending=[False, True])

print(rankings[["name", "predicted_round"]].head(32))

skill_results = skill_test.reset_index(drop=True).copy()
skill_results["predicted_round"]= skill_preds
skill_rankings  = skill_results.sort_values(["predicted_round","position"], ascending=[False, True])
print(skill_rankings[["name", "predicted_round"]].head(32))
def_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.75,
    reg_lambda = 7,
    min_child_weight = 5,
    max_depth = 6,
    learning_rate = 0.2,
    n_estimators = 400,
    reg_alpha= 3,
    random_state = 42
    )
def_model.fit(def_X_train, def_y_train)
def_preds = def_model.predict(def_X_test)

def_results = def_test.reset_index(drop=True).copy()
def_results["predicted_round"]= def_preds
def_rankings  = def_results.sort_values(["predicted_round","position"], ascending=[False, True])
print(def_rankings[["name", "predicted_round"]].head(32))


print("min:", preds.min())
print("max:", preds.max())
print("mean:", preds.mean())
print("std:", preds.std())

'''
# feature importances
qb_feat_imp = pd.Series(qb_model.feature_importances_, index=qb_train.columns)
qb_feat_imp = qb_feat_imp.sort_values(ascending=False)

# print top 20
print(qb_feat_imp.head(20))

# plot top 20
plt.figure(figsize=(10, 6))
qb_feat_imp.head(20).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 XGBoost Feature Importances")
plt.tight_layout()'''