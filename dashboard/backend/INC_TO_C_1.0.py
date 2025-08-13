import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError

# === Load Models ===
models = {
    "1": joblib.load(os.path.join("models", "best_rf_model.pkl")),
    "2": joblib.load(os.path.join("models", "gb_model.pkl")),
    "3": joblib.load(os.path.join("models", "xgb_model.pkl")),
    "4": joblib.load(os.path.join("models", "mlp_model.pkl")),
    "5": load_model(os.path.join("models", "lstm_model.h5"), compile=False),
    "6": load_model(os.path.join("models", "bilstm_model.h5"), compile=False),
}
models["5"].compile(optimizer='adam', loss=MeanSquaredError())
models["6"].compile(optimizer='adam', loss=MeanSquaredError())

# === Input/Output Paths ===
input_base = r"C:\\Users\\TIH24\\Desktop\\ML\\processed\\SASTRA\\hours"
output_base = r"output"

os.makedirs(output_base, exist_ok=True)
for i in range(1, 7):
    os.makedirs(os.path.join(output_base, str(i)), exist_ok=True)

# === Feature Engineering ===
def add_features(df, day_index):
    df["Day"] = day_index
    df["Hour"] = df["Time"].astype(int)
    df["SinTime"] = np.sin(2 * np.pi * df["Time"] / 24)
    df["CosTime"] = np.cos(2 * np.pi * df["Time"] / 24)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365)
    df["TEC_roll3"] = df["TEC"].rolling(3, min_periods=1).mean()
    df["TEC_prev"] = df["TEC"].shift(1).bfill()
    df["TEC_diff"] = df["TEC"].diff().fillna(0)
    df["TEC_diff2"] = df["TEC_diff"].diff().fillna(0)
    df["TEC_std5"] = df["TEC"].rolling(5, min_periods=1).std().fillna(0)
    return df

def reshape_for_lstm(X, time_steps=3):
    X_seq = np.array([X.shift(i).bfill().values for i in reversed(range(time_steps))])
    return np.transpose(X_seq, (1, 0, 2))

# === Main Day Processing ===
def process_day(day_folder, day_index):
    json_path = os.path.join(input_base, day_folder, "hour_avg.json")

    if not os.path.exists(json_path):
        print(f"Predicting missing day: {day_folder}")
        df = pd.DataFrame({'Time': np.arange(24), 'TEC': 100})  # Use dummy TEC values
        time_missing = True
    else:
        with open(json_path) as f:
            data = pd.DataFrame(json.load(f))

        all_hours = pd.DataFrame({'Time': np.arange(24)})

        def extract_hour(val):
            try:
                if isinstance(val, str) and ':' in val:
                    return int(val.split(":")[0])
                return int(val)
            except:
                return np.nan

        data["Time"] = data["Time"].apply(extract_hour)
        data = data.dropna(subset=["Time"])
        data["Time"] = data["Time"].astype(int)
        df = all_hours.merge(data, on='Time', how='left')
        df["TEC"] = pd.to_numeric(df["TEC"], errors="coerce")
        time_missing = df["TEC"].isna().sum() > 0
        df["TEC"] = df["TEC"].interpolate().bfill().ffill()

    df = add_features(df, day_index)
    features = ["Day", "Time", "Hour", "SinTime", "CosTime", "Day_sin", "Day_cos",
                "TEC_roll3", "TEC_prev", "TEC_diff", "TEC_diff2", "TEC_std5"]
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    if len(X) == 0:
        print(f"Skipping {day_folder}: insufficient data after preprocessing.")
        return

    df["Time"] = df["Time"].apply(lambda x: f"{int(x):02d}:00:00")

    if not time_missing:
        df = df.drop_duplicates(subset="Time")
        for mid in models.keys():
            out_path = os.path.join(output_base, mid, f"{day_folder}0.json")
            with open(out_path, 'w') as out_f:
                json.dump(df[["Time", "TEC"]].to_dict(orient="records"), out_f, indent=2)
        return

    X_lstm = reshape_for_lstm(X)

    for mid, model in models.items():
        try:
            if mid in ["5", "6"]:
                y_pred = model.predict(X_lstm).flatten()
            else:
                y_pred = model.predict(X)

            pred_df = pd.DataFrame({"Time": df["Time"].iloc[:len(y_pred)], "TEC": y_pred})
            pred_df = pred_df.drop_duplicates(subset="Time")
            out_path = os.path.join(output_base, mid, f"{day_folder}{mid}.json")
            with open(out_path, 'w') as out_f:
                json.dump(pred_df.to_dict(orient="records"), out_f, indent=2)
        except Exception as e:
            print(f"Error predicting with model {mid} on day {day_folder}: {e}")

# === Smart Missing Day Detection and Processing ===
def group_consecutive_dates(dates):
    if not dates:
        return []
    groups = []
    temp = [dates[0]]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            temp.append(dates[i])
        else:
            groups.append(temp)
            temp = [dates[i]]
    groups.append(temp)
    return groups

existing_folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]
existing_dates = []
for folder in existing_folders:
    try:
        dt = datetime.strptime(folder, "%d%m%Y")
        existing_dates.append(dt)
    except ValueError:
        continue

existing_dates = sorted(existing_dates)
if not existing_dates:
    raise ValueError("No valid date folders found in input_base.")

start_date = min(existing_dates)
end_date = max(existing_dates)
full_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
missing_dates = sorted(set(full_dates) - set(existing_dates))

predictable_missing = []
for group in group_consecutive_dates(missing_dates):
    if len(group) <= 5:
        predictable_missing.extend(group)
    else:
        print(f"Skipping missing block of {len(group)} days from {group[0].strftime('%d%m%Y')} to {group[-1].strftime('%d%m%Y')}")

final_days = sorted(existing_dates + predictable_missing)
final_folders = [d.strftime("%d%m%Y") for d in final_days]

# === Run All Days ===
for idx, folder_name in enumerate(final_folders, start=1):
    process_day(folder_name, idx)
