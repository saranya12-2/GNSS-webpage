import os
import pandas as pd
import numpy as np
import pywt
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === 1. Load and Process All CSVs ===
folder_path = r"C:\Users\TIH24\Desktop\GNSS data\REACT_NODE_VITE_TAILWIND\data_set"
all_data = []

for i, file_name in enumerate(sorted(os.listdir(folder_path))):
    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required_cols = {"Time", "TEC", "F107", "Hour", "Dst"}
    if not required_cols.issubset(df.columns):
        continue

    df["Time"] = pd.to_datetime(df["Time"])
    df["Day"] = i + 1

    gps_clean = df["TEC"].interpolate().bfill().ffill()
    coeffs = pywt.wavedec(gps_clean, 'db4', level=3)
    coeffs[1:] = [pywt.threshold(c, 0.1 * max(c), mode='soft') for c in coeffs[1:]]
    smooth_gps = pywt.waverec(coeffs, 'db4')[:len(df)]
    df["GPS_smooth"] = smooth_gps

    df["SinTime"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["CosTime"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365)
    df["TEC_roll3"] = df["GPS_smooth"].rolling(3, min_periods=1).mean()
    df["TEC_prev"] = df["GPS_smooth"].shift(1).fillna(method='bfill')
    df["TEC_diff"] = df["GPS_smooth"].diff().fillna(0)
    df["TEC_diff2"] = df["TEC_diff"].diff().fillna(0)
    df["TEC_std5"] = df["GPS_smooth"].rolling(5).std().fillna(0)

    df.dropna(inplace=True)
    all_data.append(df)

# === 2. Combine and Clean ===
df_all = pd.concat(all_data, ignore_index=True)
df_all = df_all[np.abs(df_all["GPS_smooth"] - df_all["GPS_smooth"].mean()) < 3 * df_all["GPS_smooth"].std()]

# === 3. Features and Target ===
features = ["Day", "Hour", "SinTime", "CosTime", "Day_sin", "Day_cos",
            "TEC_roll3", "TEC_prev", "TEC_diff", "TEC_diff2", "TEC_std5",
            "F107", "Dst"]
X = df_all[features]
y = df_all["GPS_smooth"]

# === 4. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Reshape for LSTM ===
def reshape_for_lstm(X, time_steps=3):
    X_reshaped = np.array([X.shift(i).fillna(method='bfill').values for i in reversed(range(time_steps))])
    return np.transpose(X_reshaped, (1, 0, 2))

X_train_lstm = reshape_for_lstm(pd.DataFrame(X_train), time_steps=3)
X_test_lstm = reshape_for_lstm(pd.DataFrame(X_test), time_steps=3)

# === 6. Train Models ===
rf = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(rf, {
    "n_estimators": [100, 300],
    "max_depth": [10, None],
    "min_samples_split": [2, 4],
    "max_features": ['sqrt']
}, n_iter=4, cv=3, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05)
gb.fit(X_train, y_train)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, objective='reg:squarederror')
xgb.fit(X_train, y_train)

mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True))
])
mlp.fit(X_train, y_train)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

lstm_model = build_lstm_model(X_train_lstm.shape[1:])
lstm_model.fit(X_train_lstm, y_train, validation_split=0.1, epochs=50,
               callbacks=[EarlyStopping(patience=5)], verbose=0)

def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

bilstm_model = build_bilstm_model(X_train_lstm.shape[1:])
bilstm_model.fit(X_train_lstm, y_train, validation_split=0.1, epochs=50,
                 callbacks=[EarlyStopping(patience=5)], verbose=0)

# === 7. Save Models ===
joblib.dump(best_rf, "./best_rf_model.pkl")
joblib.dump(gb, "./gb_model.pkl")
joblib.dump(xgb, "./xgb_model.pkl")
joblib.dump(mlp, "./mlp_model.pkl")
lstm_model.save("./lstm_model.h5")
bilstm_model.save("./bilstm_model.h5")

# === 8. Evaluation ===
def custom_accuracy(y_true, y_pred, tol):
    return np.mean(np.abs(y_true - y_pred) <= tol) * 100

models = {
    "Random Forest": best_rf,
    "Gradient Boosting": gb,
    "XGBoost": xgb,
    "MLP": mlp,
    "LSTM": lstm_model,
    "BiLSTM": bilstm_model
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_lstm).flatten() if name in ["LSTM", "BiLSTM"] else model.predict(X_test)
    results[name] = {
        "R²": round(r2_score(y_test, y_pred) * 100, 2),
        "MSE": round(mean_squared_error(y_test, y_pred), 2),
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "Acc ±2.5": round(custom_accuracy(y_test, y_pred, 2.5), 2),
        "Acc ±5.0": round(custom_accuracy(y_test, y_pred, 5.0), 2),
        "Acc ±7.5": round(custom_accuracy(y_test, y_pred, 7.5), 2)
    }

print(results)

# === 9. Plot Graphs ===
plt.figure(figsize=(18, 12))
for idx, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test_lstm).flatten() if name in ["LSTM", "BiLSTM"] else model.predict(X_test)
    plt.subplot(3, 2, idx)
    plt.plot(y_test.values[:200], label='Actual', linewidth=2)
    plt.plot(y_pred[:200], label='Predicted', linestyle='--')
    plt.title(f"{name} Prediction vs Actual")
    plt.xlabel("Sample Index")
    plt.ylabel("TEC Value")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 12))
for idx, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test_lstm).flatten() if name in ["LSTM", "BiLSTM"] else model.predict(X_test)
    plt.subplot(3, 2, idx)
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"{name} - Predicted vs Actual")
    plt.xlabel("Actual TEC")
    plt.ylabel("Predicted TEC")
    plt.grid(True)
plt.tight_layout()
plt.show()
