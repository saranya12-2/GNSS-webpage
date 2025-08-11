import pandas as pd
from datetime import datetime
import requests
from io import StringIO

# === Input URL ===
url = "https://omniweb.gsfc.nasa.gov/staging/omni2_b6LZj_SAs5.lst"

# === Download the file ===
response = requests.get(url)
response.raise_for_status()

# === Fixed-width file specs: [YEAR, DOY, HOUR, Dst-index] ===
colspecs = [(0, 4), (4, 8), (8, 11), (11, 17)]  # start & end positions
names = ["YEAR", "DOY", "HOUR", "DST"]

# === Read the fixed-width content ===
df = pd.read_fwf(StringIO(response.text), colspecs=colspecs, names=names)

# === Convert DOY to Date ===
df["Date"] = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str), format="%Y%j")

# === Reorder and save ===
df = df[["Date", "HOUR", "DST"]].rename(columns={"HOUR": "Hour", "DST": "Dst"})
df.to_csv("dst_2024_2025.csv", index=False)

print(f"✅ Saved {len(df)} records to dst_2024_2025.csv")
