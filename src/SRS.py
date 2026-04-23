import re
import pandas as pd
from pypdf import PdfReader

INPUT_FILE = "SRS Data.pdf"
OUTPUT_FILE = "srs_2019_2025_clean.csv"

# reverse chronological order
YEARS = [2025, 2024, 2023, 2022, 2021, 2020, 2019]

COLUMNS = [
    "Rk", "season", "School", "Conf", "AP Rank", "W", "L",
    "OSRS", "DSRS", "SRS", "Off", "Def",
    "Pass_Off", "Pass_Def", "Rush_Off", "Rush_Def", "Tot_Off", "Tot_Def"
]

def extract_chunks(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # split where a new row/header begins after wide PDF spacing
    chunks = re.split(r"\s{2,}(?=(?:\d+,|Rk,|,,,,))", text)
    return [c.strip() for c in chunks if c.strip()]

def parse_rows(chunks: list[str]) -> list[list[str]]:
    rows = []
    buf = ""

    for chunk in chunks:
        if (
            not chunk
            or chunk.startswith("Rk,")
            or chunk.startswith(",,,,")
            or chunk.startswith("Overall")
        ):
            continue

        buf = buf + chunk if buf else chunk

        # valid raw row has 17 columns => 16 commas
        if re.match(r"^\d+,", buf) and buf.count(",") >= 16:
            parts = [p.strip() for p in buf.split(",")]
            rows.append(parts[:17])
            buf = ""

    return rows

chunks = extract_chunks(INPUT_FILE)
raw_rows = parse_rows(chunks)

print("raw rows parsed:", len(raw_rows))
print("first 3 raw rows:")
for r in raw_rows[:3]:
    print(r)

# assign seasons by rank reset
seasoned_rows = []
year_idx = 0
prev_rk = None

for row in raw_rows:
    rk = int(row[0])

    # whenever rank resets to 1, move to next older season
    if prev_rk is not None and rk == 1 and prev_rk > 1:
        year_idx += 1

    if year_idx >= len(YEARS):
        break

    season = YEARS[year_idx]
    seasoned_rows.append([row[0], season] + row[1:])
    prev_rk = rk

df = pd.DataFrame(seasoned_rows, columns=COLUMNS)

# normalize whitespace a bit
for col in ["School", "Conf"]:
    df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

# optional type conversion
for col in ["Rk", "season", "AP Rank", "W", "L"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved clean file to {OUTPUT_FILE}")
print(df.head(10))
print(df["season"].value_counts().sort_index())