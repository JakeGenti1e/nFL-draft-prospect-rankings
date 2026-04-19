import numpy as np
import pandas as pd
import xgboost as xgb 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import graphviz
import matplotlib.pyplot as plt

draft_multiplier = 4
print("Loading data...")
df = pd.read_csv("data/processed/player_data.csv", index_col = 0)
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

qb_cols = [
    "name",
    "height",
    "weight",
    "team",
    "season",
    "position",
    "weighted_award_score",
    "is_all_american",
    "unanimous",
    "passing_ATT",
    "passing_COMPLETIONS",
    "passing_INT",
    "passing_PCT",
    "passing_TD",
    "passing_YDS",
    "passing_YPA"
]

qb_df = df[df["position"] == "QB"][qb_cols + ["draft_pick_x"]]


skill_cols = [
    "name",
    "height",
    "weight",
    "team",
    "season",
    "position",
    "weighted_award_score",
    "is_all_american",
    "unanimous",
    "receiving_LONG",
    "receiving_REC",
    "receiving_TD",
    "receiving_YDS",
    "receiving_YPR",
    "rushing_CAR",
    "rushing_LONG",
    "rushing_TD",
    "rushing_YDS",
    "rushing_YPC"
]

skill_df = df[df["position"].isin(["RB","WR","TE"])][skill_cols + ["draft_pick_x"]]

def_cols = [
    "name",
    "height",
    "weight",
    "team",
    "season",
    "position",
    "weighted_award_score",
    "is_all_american",
    "unanimous",
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
    "defensive_TOT"
]

def_df = df[df["position"].isin(["DL","DE","NT","DT","LB", "CB", "S", "DB"])][def_cols + ["draft_pick_x"]]



qb_train = qb_df[qb_df["season"].isin([2019,2020,2021,2022])].copy()
qb_val = qb_df[qb_df["season"].isin([2023,2024])].copy()
qb_test = qb_df[qb_df["season"] == 2025].copy()
qb_train = qb_train[(qb_train["draft_pick_x"].notna()) & (qb_train["position"].notna())]
qb_val = qb_val[qb_val["draft_pick_x"].notna()]


skill_train = skill_df[skill_df["season"].isin([2019,2020,2021,2022])].copy()
skill_val = skill_df[skill_df["season"].isin([2023,2024])].copy()
skill_test = skill_df[skill_df["season"] == 2025].copy()
skill_train = skill_train[(skill_train["draft_pick_x"].notna()) & (skill_train["position"].notna())]
skill_val = skill_val[skill_val["draft_pick_x"].notna()]

def_train = def_df[def_df["season"].isin([2019,2020,2021,2022])].copy()
def_val = def_df[def_df["season"].isin([2023,2024])].copy()
def_test = def_df[def_df["season"] == 2025].copy()
def_train = def_train[(def_train["draft_pick_x"].notna()) & (def_train["position"].notna())]
def_val = def_val[def_val["draft_pick_x"].notna()]


qb_X_train = qb_train.drop(columns=["draft_pick_x", "name"])
qb_y_train = qb_train["draft_pick_x"]

qb_X_val = qb_val.drop(columns=["draft_pick_x", "name"])
qb_y_val = qb_val["draft_pick_x"]

qb_X_test = qb_test.drop(columns=["draft_pick_x", "name"])


skill_X_train = skill_train.drop(columns=["draft_pick_x", "name"])
skill_y_train = skill_train["draft_pick_x"]

skill_X_val = skill_val.drop(columns=["draft_pick_x", "name"])
skill_y_val = skill_val["draft_pick_x"]

skill_X_test = skill_test.drop(columns=["draft_pick_x", "name"])

def_X_train = def_train.drop(columns=["draft_pick_x", "name"])
def_y_train = def_train["draft_pick_x"]

def_X_val = def_val.drop(columns=["draft_pick_x", "name"])
def_y_val = def_val["draft_pick_x"]

def_X_test = def_test.drop(columns=["draft_pick_x", "name"])

print(" qb train_df: ", qb_train.shape)
qb_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.6,
    reg_lambda = 4,
    min_child_weight = 5,
    max_depth =4,
    learning_rate = 0.08,
    n_estimators = 500
    )
print("test_df:", qb_test.shape)

qb_model.fit(qb_X_train, qb_y_train)



train_score = qb_model.score(qb_X_train, qb_y_train)
val_score = qb_model.score(qb_X_val, qb_y_val)
preds = qb_model.predict(qb_X_test)
skill_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.8,
    reg_lambda = 5,
    min_child_weight = 5,
    max_depth =2,
    learning_rate = 0.03,
    n_estimators = 500
    )
skill_model.fit(skill_X_train, skill_y_train)
skill_preds = skill_model.predict(skill_X_test)



print(train_score)

#skill_model.fit(skill_X_train, skill_y_train)

print(val_score)

print(qb_test.columns)
results = qb_test.reset_index(drop=True).copy()
print(results.columns)

results["predicted_pick"]= preds
rankings  = results.sort_values(["predicted_pick","position"], ascending=[True, True])

print(rankings[["name", "predicted_pick"]].head(32))

skill_results = skill_test.reset_index(drop=True).copy()
skill_results["predicted_pick"]= skill_preds
skill_rankings  = skill_results.sort_values(["predicted_pick","position"], ascending=[True, True])
print(skill_rankings[["name", "predicted_pick"]].head(32))

def_model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.8,
    reg_lambda = 5,
    min_child_weight = 5,
    max_depth =2,
    learning_rate = 0.03,
    n_estimators = 500
    )
def_model.fit(def_X_train, def_y_train)
def_preds = def_model.predict(def_X_test)

def_results = def_test.reset_index(drop=True).copy()
def_results["predicted_pick"]= def_preds
def_rankings  = def_results.sort_values(["predicted_pick","position"], ascending=[True, True])
print(def_rankings[["name", "predicted_pick"]].head(32))


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