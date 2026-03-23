import pandas as pd

df = pd.read_csv("data/raw/combine/Combine data.csv", header=None)

df = df.iloc[:, 0].str.split(",", expand=True)
df = df.iloc[:, :14]

df.columns = [
    "player", "pos", "school", "college", "ht", "wt",
    "forty_yd", "vertical", "bench", "broad_jump",
    "three_cone", "shuttle", "drafted", "player_additional"
]

for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# drop fake header row
df = df.iloc[1:].reset_index(drop=True)

print(df.head())
print(df.shape)

df.to_csv("data/processed/combine_clean.csv", index=False)