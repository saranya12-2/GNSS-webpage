# import os
# import json
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from sklearn.preprocessing import StandardScaler
# import joblib
# from tensorflow.keras.models import load_model
# from keras.losses import MeanSquaredError

# # === Load Models ===
# models = {
#     "1": joblib.load(os.path.join("models", "best_rf_model.pkl")),
#     "2": joblib.load(os.path.join("models", "gb_model.pkl")),
#     "3": joblib.load(os.path.join("models", "xgb_model.pkl")),
#     "4": joblib.load(os.path.join("models", "mlp_model.pkl")),
#     "5": load_model(os.path.join("models", "lstm_model.h5"), compile=False),
#     "6": load_model(os.path.join("models", "bilstm_model.h5"), compile=False),
# }
# models["5"].compile(optimizer='adam', loss=MeanSquaredError())
# models["6"].compile(optimizer='adam', loss=MeanSquaredError())

# # === Input/Output Paths ===
# input_base = r"C:\Users\Tharani\Desktop\GNSS data\GNSS data\REACT_NODE_VITE_TAILWIND\processed\SASTRA\hours"
# output_base = r"output1"

# os.makedirs(output_base, exist_ok=True)
# for i in range(1, 7):
#     os.makedirs(os.path.join(output_base, str(i)), exist_ok=True)

# # === Feature Engineering ===
# def add_features(df, day_index):
#     df["Day"] = day_index
#     df["Hour"] = pd.to_datetime(df["Time"]).dt.hour
#     df["SinTime"] = np.sin(2 * np.pi * df["Hour"] / 24)
#     df["CosTime"] = np.cos(2 * np.pi * df["Hour"] / 24)
#     df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365)
#     df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365)
#     df["TEC_roll3"] = df["TEC"].rolling(3, min_periods=1).mean()
#     df["TEC_prev"] = df["TEC"].shift(1).bfill()
#     df["TEC_diff"] = df["TEC"].diff().fillna(0)
#     df["TEC_diff2"] = df["TEC_diff"].diff().fillna(0)
#     df["TEC_std5"] = df["TEC"].rolling(5, min_periods=1).std().fillna(0)
#     return df

# def reshape_for_lstm(X, time_steps=3):
#     X_seq = np.array([X.shift(i).bfill().values for i in reversed(range(time_steps))])
#     return np.transpose(X_seq, (1, 0, 2))

# # === Main Day Processing ===
# def process_day(day_folder, day_index):
#     json_path = os.path.join(input_base, day_folder, "hour_avg.json")

#     if not os.path.exists(json_path):
#         print(f"Predicting missing day: {day_folder}")
#         df = pd.DataFrame({
#             'Time': [f"{i:02d}:00:00" for i in range(24)],
#             'TEC': 100,
#             'F107': 150,
#             'Dst': -20
#         })
#         time_missing = True
#     else:
#         with open(json_path) as f:
#             raw_data = json.load(f)
#         df = pd.DataFrame(raw_data)
#         df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
#         df = df.sort_values("Time")
#         df["TEC"] = pd.to_numeric(df["TEC"], errors="coerce")
#         df["F107"] = pd.to_numeric(df["F107"], errors="coerce")
#         df["Dst"] = pd.to_numeric(df["Dst"], errors="coerce")
#         df["TEC"] = df["TEC"].interpolate().bfill().ffill()
#         df["F107"] = df["F107"].interpolate().bfill().ffill()
#         df["Dst"] = df["Dst"].interpolate().bfill().ffill()
#         time_missing = df["TEC"].isna().sum() > 0

#     df = add_features(df, day_index)
#     features = ["Day", "Hour", "SinTime", "CosTime", "Day_sin", "Day_cos",
#                 "TEC_roll3", "TEC_prev", "TEC_diff", "TEC_diff2", "TEC_std5",
#                 "F107", "Dst"]
#     X = df[features].copy()
#     X.replace([np.inf, -np.inf], np.nan, inplace=True)
#     X.dropna(inplace=True)

#     if len(X) == 0:
#         print(f"Skipping {day_folder}: insufficient data after preprocessing.")
#         return

#     df["Time"] = df["Time"].apply(lambda x: f"{x}")

#     if not time_missing:
#         df = df.drop_duplicates(subset="Time")
#         for mid in models.keys():
#             out_path = os.path.join(output_base, mid, f"{day_folder}0.json")
#             with open(out_path, 'w') as out_f:
#                 json.dump(df[["Time", "TEC"]].to_dict(orient="records"), out_f, indent=2)
#         return

#     X_lstm = reshape_for_lstm(X)

#     for mid, model in models.items():
#         try:
#             if mid in ["5", "6"]:
#                 y_pred = model.predict(X_lstm).flatten()
#             else:
#                 y_pred = model.predict(X)

#             pred_df = pd.DataFrame({"Time": df["Time"].iloc[:len(y_pred)], "TEC": y_pred})
#             pred_df = pred_df.drop_duplicates(subset="Time")
#             out_path = os.path.join(output_base, mid, f"{day_folder}{mid}.json")
#             with open(out_path, 'w') as out_f:
#                 json.dump(pred_df.to_dict(orient="records"), out_f, indent=2)
#         except Exception as e:
#             print(f"Error predicting with model {mid} on day {day_folder}: {e}")

# # === Smart Missing Day Detection and Processing ===
# def group_consecutive_dates(dates):
#     if not dates:
#         return []
#     groups = []
#     temp = [dates[0]]
#     for i in range(1, len(dates)):
#         if (dates[i] - dates[i-1]).days == 1:
#             temp.append(dates[i])
#         else:
#             groups.append(temp)
#             temp = [dates[i]]
#     groups.append(temp)
#     return groups

# existing_folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]
# existing_dates = []
# for folder in existing_folders:
#     try:
#         dt = datetime.strptime(folder, "%d%m%Y")
#         existing_dates.append(dt)
#     except ValueError:
#         continue

# existing_dates = sorted(existing_dates)
# if not existing_dates:
#     raise ValueError("No valid date folders found in input_base.")

# start_date = min(existing_dates)
# end_date = max(existing_dates)
# full_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
# missing_dates = sorted(set(full_dates) - set(existing_dates))

# predictable_missing = []
# for group in group_consecutive_dates(missing_dates):
#     if len(group) <= 5:
#         predictable_missing.extend(group)
#     else:
#         print(f"Skipping missing block of {len(group)} days from {group[0].strftime('%d%m%Y')} to {group[-1].strftime('%d%m%Y')}")

# final_days = sorted(existing_dates + predictable_missing)
# final_folders = [d.strftime("%d%m%Y") for d in final_days]

# # === Run All Days ===
# for idx, folder_name in enumerate(final_folders, start=1):
#     process_day(folder_name, idx)
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from statsmodels.tsa.arima.model import ARIMA

# === Load Models ===
# Ensure these model files are in the specified 'models' directory
models = {
    "1": joblib.load(os.path.join("models", "best_rf_model.pkl")),
    "2": joblib.load(os.path.join("models", "gb_model.pkl")),
    "3": joblib.load(os.path.join("models", "xgb_model.pkl")),
    "4": joblib.load(os.path.join("models", "mlp_model.pkl")),
    "5": load_model(os.path.join("models", "lstm_model.h5"), compile=False),
    "6": load_model(os.path.join("models", "bilstm_model.h5"), compile=False),
    "7": joblib.load(os.path.join("models", "hybrid_residual_model.pkl"))
}

# Compile the Keras models
models["5"].compile(optimizer='adam', loss=MeanSquaredError())
models["6"].compile(optimizer='adam', loss=MeanSquaredError())

# === Input/Output Paths ===
input_base = r"C:\Users\Tharani\Desktop\GNSS data\GNSS data\REACT_NODE_VITE_TAILWIND\processed\SASTRA\hours"
output_base = r"output1"

os.makedirs(output_base, exist_ok=True)
for i in range(1, 8):
    os.makedirs(os.path.join(output_base, str(i)), exist_ok=True)
os.makedirs(os.path.join(output_base, "8_ARIMA"), exist_ok=True)

# === Feature Engineering Helpers ===
def add_features(df, day_index):
    """
    Adds time-based and statistical features to the DataFrame.
    """
    df = df.copy()
    df["Day"] = day_index
    df["Hour"] = pd.to_datetime(df["Time"]).dt.hour
    df["SinTime"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["CosTime"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365)
    df["TEC_roll3"] = df["TEC"].rolling(3, min_periods=1).mean()
    df["TEC_prev"] = df["TEC"].shift(1).bfill()
    df["TEC_diff"] = df["TEC"].diff().fillna(0)
    df["TEC_diff2"] = df["TEC_diff"].diff().fillna(0)
    df["TEC_std5"] = df["TEC"].rolling(5, min_periods=1).std().fillna(0)
    
    if "F107" not in df:
        df["F107"] = 150 
    if "Dst" not in df:
        df["Dst"] = -20
    
    return df

def reshape_for_lstm(X, time_steps=12):
    """
    Reshapes a DataFrame of features into a 3D array for LSTM models.
    """
    X_seq = np.array([X.shift(i).bfill().values for i in reversed(range(time_steps))])
    return np.transpose(X_seq, (1, 0, 2))

# === Main Day Processing Function ===
def process_day(day_folder, day_index):
    """
    Processes a single day's data, either predicting missing values or
    storing the existing data.
    """
    json_path = os.path.join(input_base, day_folder, "hour_avg.json")
    
    if not os.path.exists(json_path):
        print(f"Predicting missing day: {day_folder}")
        df = pd.DataFrame({
            'Time': [f"{i:02d}:00:00" for i in range(24)],
            'TEC': 100, 
            'F107': 150,
            'Dst': -20
        })
        is_missing = True
    else:
        with open(json_path) as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
        df = df.sort_values("Time").reset_index(drop=True)
        
        df["TEC"] = pd.to_numeric(df["TEC"], errors="coerce").interpolate().bfill().ffill()
        df["F107"] = pd.to_numeric(df["F107"], errors="coerce").interpolate().bfill().ffill()
        df["Dst"] = pd.to_numeric(df["Dst"], errors="coerce").interpolate().bfill().ffill()
        
        is_missing = df["TEC"].isna().sum() > 0 or df["TEC"].sum() == 0
    
    df = add_features(df, day_index)
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
    
    # === Save original data if it exists ===
    if not is_missing:
        df = df.drop_duplicates(subset="Time")
        # Save a copy of the original data in EVERY model folder
        for mid in list(models.keys()) + ["8_ARIMA"]:
            if mid == "8_ARIMA":
                out_path = os.path.join(output_base, mid, f"{day_folder}0.json")
            else:
                out_path = os.path.join(output_base, mid, f"{day_folder}0.json")
            with open(out_path, 'w') as out_f:
                json.dump(df[["Time", "TEC"]].to_dict(orient="records"), out_f, indent=2)
        return

    # --- Data preparation for models ---
    features = [
        "Day", "Hour", "SinTime", "CosTime", "Day_sin", "Day_cos",
        "TEC_roll3", "TEC_prev", "TEC_diff", "TEC_diff2", "TEC_std5",
        "F107", "Dst"
    ]
    X_full = df[features].copy()
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full.dropna(inplace=True)

    if len(X_full) == 0:
        print(f"Skipping {day_folder}: insufficient data after preprocessing.")
        return
        
    X_lstm = reshape_for_lstm(X_full, time_steps=12)
    X_flat = X_full.values

    # --- Make predictions and save files ---
    for mid, model in models.items():
        try:
            if mid in ["5", "6"]:
                y_pred = model.predict(X_lstm).flatten()
            elif mid == "7":
                lstm_pred = models["5"].predict(X_lstm).flatten()
                resid_pred = model.predict(X_flat)
                y_pred = lstm_pred + resid_pred[:, 0]
            else:
                y_pred = model.predict(X_flat)
                if y_pred.ndim > 1:
                    y_pred = y_pred[:, 0]
                y_pred = y_pred.flatten()

            pred_df = pd.DataFrame({"Time": df["Time"].iloc[:len(y_pred)], "TEC": y_pred})
            pred_df = pred_df.drop_duplicates(subset="Time")
            
            model_id_num = mid
            out_path = os.path.join(output_base, model_id_num, f"{day_folder}{model_id_num}.json")
            pred_df.to_json(out_path, orient="records", indent=2)

        except Exception as e:
            print(f"Error predicting with model {mid} on day {day_folder}: {e}")

    # 8: ARIMA model
    try:
        series = df["TEC"].values
        arima_model = ARIMA(series, order=(6, 1, 1)).fit()
        arima_pred = arima_model.forecast(steps=len(df))
        
        out_ar = pd.DataFrame({"Time": df["Time"], "TEC": arima_pred})
        out_ar.to_json(os.path.join(output_base, "8_ARIMA", f"{day_folder}8.json"),
                       orient="records", indent=2)
    except Exception as e:
        print(f"Error with ARIMA model on day {day_folder}: {e}")

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

final_days = sorted(list(set(existing_dates + predictable_missing)))
final_folders = [d.strftime("%d%m%Y") for d in final_days]

# === Run All Days ===
for idx, folder_name in enumerate(final_folders, start=1):
    process_day(folder_name, idx)