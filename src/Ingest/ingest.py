from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "draft.csv"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}\nPut draft.csv in the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    print("Loaded:", DATA_PATH.name)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))



if __name__ == "__main__":
    main()

def get_headers():
    if API_KEY is None:
        raise RuntimeError("CFBD_API_KEY not found in environment variables")

    return {
        "Authorization": f"Bearer {API_KEY}"
    }