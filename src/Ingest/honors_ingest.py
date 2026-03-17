import pandas as pd

df = pd.read_html("sportsref_download_files/sheet001.htm")[0]

print(df.head())