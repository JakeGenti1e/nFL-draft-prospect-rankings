import numpy as np
import pandas as pd
from xgboost import XGBRegressor

print("Loading data...")
df = pd.read_csv("data/processed/features.csv")
df = df.drop(columns=["position_y"])
df["position_x"] = df["position_x"].astype("category")
df["team"] = df["team"].astype("category")
df["conference"] = df["conference"].astype("category")
df = df[df["draft_pick"].notna()]

print("Creating splits...")

train = df[df["season"] <=2022]
val = df[(df["season"] >=2023) & (df["season"] <= 2024)]
test = df[df["season"] == 2025]

X_train = train.drop(columns=["draft_pick", "name", "playerId"])
y_train = train["draft_pick"]

X_val = val.drop(columns=["draft_pick", "name", "playerId"])
y_val = val["draft_pick"]

X_test = test.drop(columns=["draft_pick", "name", "playerId"])
y_test = test["draft_pick"]


model = XGBRegressor(
    enable_categorical = True
)

model.fit(X_train, y_train)
model.score(X_val, y_val)


results = test.copy()
results["predicted_pick"] = model.predict(X_test)
rankings  = results.sort_values("season")
rankings = rankings.groupby("name")
rankings  = results.sort_values("predicted_pick")

print(rankings[["name", "predicted_pick"]].head(32))


