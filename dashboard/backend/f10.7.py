import requests
import pandas as pd
from datetime import datetime, date

def download_and_average_f107(output_file="f107_2024_2025.csv"):
    url = "https://spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
    print("🔄 Downloading Canadian F10.7 data…")
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.splitlines()

    data = []

    for line in lines:
        if not line.strip() or line.startswith("#") or line.startswith("fluxdate"):
            continue
        try:
            parts = line.split()
            fluxdate = parts[0]  # e.g., 20250102
            obs_flux = float(parts[4])  # fluxobsflux
            dt = datetime.strptime(fluxdate, "%Y%m%d").date()

            if 2024 <= dt.year <= 2025 and dt <= date.today():
                data.append({"Date": dt, "F107": obs_flux})
        except Exception:
            continue

    # Convert to DataFrame and group by date
    df = pd.DataFrame(data)
    df_grouped = df.groupby("Date", as_index=False).mean()  # Average per day

    df_grouped.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df_grouped)} averaged daily F10.7 records to {output_file}")

if __name__ == "__main__":
    download_and_average_f107()
