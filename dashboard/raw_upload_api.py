from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import zipfile
import os
from datetime import datetime, timedelta
from io import TextIOWrapper, BytesIO
import json
raw_upload_api = FastAPI()

# === Satellite classifier map ===
def satellite_classifier():
    return {svid: "GPS" for svid in range(1, 38)}

SAT_MAP = satellite_classifier()

# === Convert GPS Week + Seconds to Calendar DateTime ===
def gps_to_utc_datetime(gps_week, gps_seconds):
    GPS_EPOCH = datetime(1980, 1, 6)
    return GPS_EPOCH + timedelta(weeks=int(gps_week), seconds=float(gps_seconds))

# === Save JSON Output ===
def save_json_output(data, out_path):
    def convert(o):
        if isinstance(o, (datetime, pd.Timestamp)):
            return o.strftime("%Y-%m-%d %H:%M:%S")
        raise TypeError(f"Type {type(o)} not serializable")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=convert)

# === Process DataFrame for GPS ===
def fast_process_df(df, source_type):
    try:
        df = df.copy()
        if source_type == "excel":
            if df.shape[1] < 17:
                return None, None, None, None
            df = df.iloc[:, [0, 1, 2, 16]]
            df.columns = ["GPS Week", "GPS Seconds", "Satellite ID", "TEC"]
        else:
            df.columns = ["GPS Week", "GPS Seconds", "Satellite ID", "TEC"]

        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=["Satellite ID", "GPS Week", "GPS Seconds", "TEC"], inplace=True)

        df["Satellite System"] = df["Satellite ID"].map(SAT_MAP)
        df = df[df["Satellite System"] == "GPS"]
        if df.empty:
            return None, None, None, None

        df["Calendar Time"] = df.apply(lambda row: gps_to_utc_datetime(row["GPS Week"], row["GPS Seconds"]), axis=1)

        df["Minute"] = df["Calendar Time"].dt.floor("min")
        min_df = df.dropna(subset=["TEC"])
        min_avg_df = min_df.groupby("Minute")["TEC"].mean().reset_index(name="TEC")
        min_avg_df["Minute"] = min_avg_df["Minute"].dt.strftime("%H:%M:%S")
        min_avg_df = min_avg_df.rename(columns={"Minute": "Time"})

        df["Hour"] = df["Calendar Time"].dt.floor("h")
        hour_df = df.dropna(subset=["TEC"])
        hour_avg_df = hour_df.groupby("Hour")["TEC"].mean().reset_index(name="TEC")
        hour_avg_df["Hour"] = hour_avg_df["Hour"].dt.strftime("%H:%M:%S")
        hour_avg_df = hour_avg_df.rename(columns={"Hour": "Time"})

        file_date = df.iloc[0]["Calendar Time"].strftime("%d%m%Y")

        return None, min_avg_df, hour_avg_df, file_date

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

@raw_upload_api.post("/process/zip")
async def process_zip_file(file: UploadFile = File(...), ground_station: str = Form(...)):
    content = await file.read()
    in_memory_zip = BytesIO(content)

    try:
        file_name = file.filename
        os.makedirs("uploaded", exist_ok=True)
        raw_path = os.path.join("uploaded", file_name)
        with open(raw_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(in_memory_zip, 'r') as zip_ref:
            ismr_files = [f for f in zip_ref.namelist() if f.endswith('.ismr')]
            print(f"Found {len(ismr_files)} ISMR files in zip: {ismr_files}")
            if not ismr_files:
                return JSONResponse({"error": "No .ismr files found in ZIP."})

            for name in ismr_files:
                print(f"Processing file: {name}")
                with zip_ref.open(name) as f:
                    df = pd.read_csv(TextIOWrapper(f, 'utf-8'), delimiter=',', header=None, usecols=[0, 1, 2, 16])
                    print(f"Shape of {name}: {df.shape}")
                    _, min_avg, hour_avg, file_date = fast_process_df(df, "zip")

                    if min_avg is None or hour_avg is None:
                        print(f"Skipping file {name} — data not suitable.")
                        continue

                    base_path = os.path.join("processed", ground_station)
                    min_dir = os.path.join(base_path, "min", file_date)
                    hour_dir = os.path.join(base_path, "hours", file_date)
                    os.makedirs(min_dir, exist_ok=True)
                    os.makedirs(hour_dir, exist_ok=True)

                    min_avg.to_excel(os.path.join(min_dir, "minute_avg.xlsx"), index=False)
                    hour_avg.to_excel(os.path.join(hour_dir, "hour_avg.xlsx"), index=False)

                    save_json_output(min_avg.to_dict(orient="records"), os.path.join(min_dir, "minute_avg.json"))
                    save_json_output(hour_avg.to_dict(orient="records"), os.path.join(hour_dir, "hour_avg.json"))

        return JSONResponse({"status": "Files processed and saved successfully."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Upload failed. {str(e)}"})


@raw_upload_api.post("/process/excel")
async def process_excel_file(file: UploadFile = File(...), ground_station: str = Form(...)):
    try:
        content = await file.read()

        # Save uploaded file for reference
        file_name = file.filename
        os.makedirs("uploaded", exist_ok=True)
        raw_path = os.path.join("uploaded", file_name)
        with open(raw_path, "wb") as f:
            f.write(content)

        with BytesIO(content) as in_memory_file:
            xls = pd.ExcelFile(in_memory_file)
            for sheet in xls.sheet_names:
                print(f"Processing sheet: {sheet}")
                df = pd.read_excel(xls, sheet_name=sheet)
                print(f"Shape of sheet: {df.shape}")
                
                _, min_avg, hour_avg, file_date = fast_process_df(df, "excel")

                if min_avg is None or hour_avg is None:
                    print(f"Skipping sheet '{sheet}' — no usable data.")
                    continue  # Skip to next sheet

                print(f"Saving outputs for {file_date}")

                # Prepare output directories
                base_path = os.path.join("processed", ground_station)
                min_dir = os.path.join(base_path, "min", file_date)
                hour_dir = os.path.join(base_path, "hours", file_date)
                os.makedirs(min_dir, exist_ok=True)
                os.makedirs(hour_dir, exist_ok=True)

                # Save as Excel
                min_avg.to_excel(os.path.join(min_dir, "minute_avg.xlsx"), index=False)
                hour_avg.to_excel(os.path.join(hour_dir, "hour_avg.xlsx"), index=False)

                # Save as JSON
                save_json_output(min_avg.to_dict(orient="records"), os.path.join(min_dir, "minute_avg.json"))
                save_json_output(hour_avg.to_dict(orient="records"), os.path.join(hour_dir, "hour_avg.json"))

        return JSONResponse({"status": "Files processed and saved successfully."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(raw_upload_api, host="127.0.0.1", port=8000)