import numpy as np
import pandas as pd
import xgboost as xgb 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import graphviz
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
df["unanimous"] = df["unanimous"]*2
df["Drafted"] = df["Drafted"].fillna(0)
df["height"] = df["height"]*0.1
df["weight"] = df["weight"]*0.1
df = df.drop(columns = ["draft_year_y", "draft_round_y", "draft_pick_y", "pro_team","puntReturns_AVG", "punting_LONG", "puntReturns_YDS", "kickReturns_LONG","kickReturns_YDS", "puntReturns_TD", "puntReturns_LONG", "punting_YPP", "puntReturns_NO", "kickReturns_AVG", "kickReturns_NO", "punting_NO", "punting_YDS", "kickReturns_TD", "punting_In 20", "age", "Unaminous"])
num_cols = df.select_dtypes(include="number").columns
df["draft_pick_x"] = df["draft_pick_x"]
df["draft_score"] = 300-df["draft_pick_x"]
df = df.reset_index(drop = True)
print(df[["draft_pick_x", "draft_score"]].sample(20))
print(df["draft_score"].describe())
#print(df["team"].unique())
confrence_rankings = {
    "SEC": 1.5,
    "Big Ten": 1.4,
    "ACC": 1.4,
    "Pac-12": 1.4,
    "Notre Dame":1.4,
    "American Athletic":1.3,
    "Mountain West":1.2,
    "Mid-American": 1.1,
    "Conference USA": 1.1,
    "FBS Independents": 1.0,
    "Sun Belt": 1.0
}

df["confrence_weight"] = df["conference"].map(confrence_rankings)

print(sorted(df["conference"].dropna().astype(str).str.strip().unique()))
print("Creating splits...")
print(df["season"].dtype) 
print(df["season"].value_counts(dropna=False).sort_index())

train = df[df["season"] <=2022].copy()
val = df[(df["season"] >=2023) & (df["season"] <= 2024)].copy()
test = df[df["season"] == 2025].copy()

train = train[train["draft_pick_x"].notna()]
train = train[train["position"].notna()]
val = val[val["draft_pick_x"].notna()]

train.to_csv("data/raw/draft/train_data.csv")
test.to_csv("data/raw/draft/test_data.csv")

print("test_df:", test.shape)

print(train["draft_score"].describe())
print(train["draft_score"].isna().mean())
print(train["draft_score"].notna().mean())




X_train = train.drop(columns=["draft_score", "name", "playerId", "draft_pick_x", "draft_round_x", "Drafted", "draft_year_x"])
y_train = train["draft_score"]

X_val = val.drop(columns=["draft_score", "name", "playerId", "draft_pick_x", "draft_round_x", "Drafted", "draft_year_x"])
y_val = val["draft_score"]

X_test = test.drop(columns=["draft_score", "name", "playerId", "draft_pick_x", "draft_round_x", "Drafted", "draft_year_x"])


model = XGBRegressor(
    enable_categorical =True,
    colsample_bytree = 0.4,
    reg_lambda = 5,
    min_child_weight = 5,
    max_depth =4,
    learning_rate = 0.04,
    n_estimators = 400
    )
print("test_df:", test.shape)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
preds = model.predict(X_test)

print(train_score)
print(val_score)

print(test.columns)
results = test.reset_index(drop=True).copy()
print(results.columns)

results["predicted_pick"]= 300-preds
rankings  = results.sort_values(["position","predicted_pick"], ascending=[True,True])

print(rankings[["name", "predicted_pick", "team", "position"]].head(32))


print("min:", preds.min())
print("max:", preds.max())
print("mean:", preds.mean())
print("std:", preds.std())

import pandas as pd
import matplotlib.pyplot as plt

# feature importances
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
feat_imp = feat_imp.sort_values(ascending=False)

# print top 20
print(feat_imp.head(20))

# plot top 20
plt.figure(figsize=(10, 6))
feat_imp.head(20).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 XGBoost Feature Importances")
plt.tight_layout()
