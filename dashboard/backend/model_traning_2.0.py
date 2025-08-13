# import os
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# import matplotlib.pyplot as plt

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# from statsmodels.tsa.arima.model import ARIMA
# import warnings
# warnings.filterwarnings("ignore")

# # === Load multiple CSVs from folder ===
# folder_path = r"C:\Users\Tharani\Desktop\GNSS data\GNSS data\REACT_NODE_VITE_TAILWIND\data_set"  # Update this path to your dataset folder
# all_dfs = []
# for file in sorted(os.listdir(folder_path)):
#     if file.endswith(".csv"):
#         df = pd.read_csv(os.path.join(folder_path, file))
#         df.columns = df.columns.str.strip()
#         if "Time" in df.columns and "TEC" in df.columns:
#             df = df.copy()
#             df['Time'] = pd.to_datetime(df['Time'])
#             df['GPS_smooth'] = df['TEC'].interpolate().bfill().ffill()
#             all_dfs.append(df)
# if not all_dfs:
#     raise ValueError("No valid CSV files with required columns found.")
# df = pd.concat(all_dfs, ignore_index=True)

# # === Add Hour and Day of Year features ===
# df['Hour'] = df['Time'].dt.hour
# df['DayOfYear'] = df['Time'].dt.dayofyear

# # Normalize only TEC
# scaler = MinMaxScaler()
# df['TEC_scaled'] = scaler.fit_transform(df[['GPS_smooth']])

# # === Add time-aware features ===
# df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
# df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
# df['Day_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
# df['Day_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# features = ['TEC_scaled', 'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos']
# data = df[features].values

# # === Sequence Preparation: 12-input -> 24-output ===
# def create_sequences_multi(data, time_steps=12, pred_steps=24):
#     X, y = [], []
#     for i in range(len(data) - time_steps - pred_steps):
#         X.append(data[i:i+time_steps])
#         y.append(data[i+time_steps:i+time_steps+pred_steps, 0])
#     return np.array(X), np.array(y)

# X_seq, y = create_sequences_multi(data, time_steps=12, pred_steps=24)
# X_flat = X_seq.reshape(X_seq.shape[0], -1)

# split_idx = int(0.8 * len(X_seq))
# X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
# y_train, y_test = y[:split_idx], y[split_idx:]
# X_train_flat, X_test_flat = X_flat[:split_idx], X_flat[split_idx:]

# # === Classical Models ===
# models = {
#     "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)),
#     "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03)),
#     "XGBoost": MultiOutputRegressor(XGBRegressor(n_estimators=300, learning_rate=0.03, objective='reg:squarederror')),
#     "MLP": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1500, early_stopping=True))
# }

# for name, model in models.items():
#     print(f"Training {name}...")
#     model.fit(X_train_flat, y_train)

# # === LSTM with Conv1D ===\
# lstm = Sequential([
#     Conv1D(32, kernel_size=3, activation='relu', input_shape=(12, X_seq.shape[2])),
#     MaxPooling1D(pool_size=2),
#     LSTM(64, return_sequences=True),
#     Dropout(0.3),
#     LSTM(32),
#     Dense(24)
# ])
# lstm.compile(optimizer='adam', loss='mse')
# callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5)]
# lstm.fit(X_train_seq, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=0)

# # === BiLSTM ===
# bilstm = Sequential([
#     Bidirectional(LSTM(64, return_sequences=True), input_shape=(12, X_seq.shape[2])),
#     Dropout(0.3),
#     LSTM(32),
#     Dropout(0.2),
#     Dense(24)
# ])
# bilstm.compile(optimizer='adam', loss='mse')
# bilstm.fit(X_train_seq, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=0)

# # === Hybrid: LSTM + XGBoost on Residuals ===
# # Get LSTM predictions on training set and compute residuals
# lstm_train_pred = lstm.predict(X_train_seq)
# residuals = y_train - lstm_train_pred
# # Train XGBoost to model residuals
# residual_model = MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, objective='reg:squarederror'))
# residual_model.fit(X_train_flat, residuals)

# # === Evaluate All Models ===
# def inverse_tec(pred):
#     # pred shape: (n_samples, pred_steps)
#     # reconstruct scaling: expand to original feature dimension placeholder
#     return scaler.inverse_transform(pred.reshape(-1,1)).reshape(pred.shape)

# all_models = models.copy()
# all_models.update({"LSTM": lstm, "BiLSTM": bilstm, "Hybrid": residual_model})

# results = {}
# for name, model in all_models.items():
#     if name == "LSTM":
#         y_pred = model.predict(X_test_seq)
#     elif name == "BiLSTM":
#         y_pred = model.predict(X_test_seq)
#     elif name == "Hybrid":
#         lstm_pred = lstm.predict(X_test_seq)
#         res_pred = model.predict(X_test_flat)
#         y_pred = lstm_pred + res_pred
#     else:
#         y_pred = model.predict(X_test_flat)

#     y_pred_inv = inverse_tec(y_pred)
#     y_test_inv = inverse_tec(y_test)

#     r2 = r2_score(y_test_inv[:, 0], y_pred_inv[:, 0])
#     mse = mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0])
#     mae = mean_absolute_error(y_test_inv[:, 0], y_pred_inv[:, 0])

#     results[name] = {"R²": round(r2, 3), "MSE": round(mse, 2), "MAE": round(mae, 2)}

#     plt.figure(figsize=(8, 4))
#     plt.plot(y_test_inv[0], label='Actual')
#     plt.plot(y_pred_inv[0], label='Predicted', linestyle='--')
#     plt.title(f"24-Hour Forecast - {name}")
#     plt.xlabel("Hour Ahead")
#     plt.ylabel("TEC")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # === ARIMA ===
# arima_series = df['GPS_smooth'].values
# train_arima, test_arima = arima_series[:-24], arima_series[-24:]

# arima_model = ARIMA(train_arima, order=(6,1,1)).fit()
# arima_pred = arima_model.forecast(steps=24)

# r2_arima = r2_score(test_arima, arima_pred)
# mse_arima = mean_squared_error(test_arima, arima_pred)
# mae_arima = mean_absolute_error(test_arima, arima_pred)
# results["ARIMA"] = {"R²": round(r2_arima, 3), "MSE": round(mse_arima, 2), "MAE": round(mae_arima, 2)}

# plt.figure(figsize=(8, 4))
# plt.plot(test_arima, label='Actual')
# plt.plot(arima_pred, label='ARIMA Predicted', linestyle='--')
# plt.title("24-Hour Forecast - ARIMA")
# plt.xlabel("Hour Ahead")
# plt.ylabel("TEC")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # === Print Results ===
# for name, metrics in results.items():
#     print(f"{name}: R²={metrics['R²']}, MSE={metrics['MSE']}, MAE={metrics['MAE']}")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.arima.model import ARIMA

# === 1. Load & Preprocess ===
folder = "./data_set"  # your folder
dfs = []
for f in sorted(os.listdir(folder)):
    if f.endswith(".csv"):
        tmp = pd.read_csv(os.path.join(folder, f))
        if {"Time","TEC"}.issubset(tmp.columns):
            tmp["Time"] = pd.to_datetime(tmp["Time"])
            tmp["GPS_smooth"] = tmp["TEC"].interpolate().bfill().ffill()
            dfs.append(tmp)
df = pd.concat(dfs, ignore_index=True)

# === 2. Feature Engineering ===
df["Hour"] = df["Time"].dt.hour
df["DayOfYear"] = df["Time"].dt.dayofyear
scaler = MinMaxScaler()
df["TEC_scaled"] = scaler.fit_transform(df[["GPS_smooth"]])
df["Hour_sin"] = np.sin(2*np.pi*df["Hour"]/24)
df["Hour_cos"] = np.cos(2*np.pi*df["Hour"]/24)
df["Day_sin"]  = np.sin(2*np.pi*df["DayOfYear"]/365)
df["Day_cos"]  = np.cos(2*np.pi*df["DayOfYear"]/365)

# === 3. Sequence Preparation (12→24) ===
features = ["TEC_scaled","Hour_sin","Hour_cos","Day_sin","Day_cos"]
mat = df[features].values

def make_seq(X, in_steps=12, out_steps=24):
    XX, yy = [], []
    for i in range(len(X)-in_steps-out_steps):
        XX.append(X[i:i+in_steps])
        yy.append(X[i+in_steps:i+in_steps+out_steps,0])
    return np.array(XX), np.array(yy)

X_seq, y = make_seq(mat)
X_flat = X_seq.reshape(len(X_seq), -1)

# Train-test split
split = int(0.8*len(X_seq))
X_tr_seq, X_te_seq = X_seq[:split], X_seq[split:]
y_tr, y_te         = y[:split], y[split:]
X_tr_flat, X_te_flat = X_flat[:split], X_flat[split:]

# === 4. Hyperparameter Tuning (Classical) ===
param_grids = {
    "RF":   {"estimator__n_estimators":[100,200,300], "estimator__max_depth":[None,10,20]},
    "GB":   {"estimator__n_estimators":[100,200],    "estimator__learning_rate":[0.01,0.1]},
    "XGB":  {"estimator__n_estimators":[100,200],    "estimator__learning_rate":[0.01,0.1], "estimator__colsample_bytree":[0.6,1.0]},
    "MLP":  {"estimator__hidden_layer_sizes":[(64,32),(128,64)], "estimator__alpha":[1e-4,1e-3]}
}
bases = {
    "RF": RandomForestRegressor(random_state=42),
    "GB": GradientBoostingRegressor(),
    "XGB": XGBRegressor(objective="reg:squarederror"),
    "MLP": MLPRegressor(max_iter=1000, early_stopping=True)
}
tuned = {}
for name, base in bases.items():
    mor = MultiOutputRegressor(base)
    rs = RandomizedSearchCV(
        mor, param_grids[name], n_iter=5, cv=3,
        scoring="neg_mean_squared_error", n_jobs=-1, verbose=1, random_state=42
    )
    rs.fit(X_tr_flat, y_tr)
    tuned[name] = rs.best_estimator_
    print(f"{name} tuned:", rs.best_params_)

# === 5. Hybrid LSTM + XGB Residual Tuning ===
# Train LSTM on training data
lstm = Sequential([
    Conv1D(32,3,activation="relu",input_shape=(12,5)),
    MaxPooling1D(2),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(24)
])
lstm.compile("adam","mse")
lstm.fit(X_tr_seq, y_tr, epochs=20, batch_size=32, validation_split=0.1, verbose=0,
         callbacks=[EarlyStopping(patience=5,restore_best_weights=True)])
residuals = y_tr - lstm.predict(X_tr_seq)

mor_res = MultiOutputRegressor(XGBRegressor(objective="reg:squarederror"))
rsr = RandomizedSearchCV(
    mor_res,
    {"estimator__n_estimators":[100,200], "estimator__learning_rate":[0.01,0.1]},
    n_iter=5,cv=3,scoring="neg_mean_squared_error",n_jobs=-1,verbose=1,random_state=42
)
rsr.fit(X_tr_flat, residuals)
tuned["Hybrid"] = rsr.best_estimator_
print("Hybrid tuned:", rsr.best_params_)
joblib.dump(tuned["Hybrid"], "hybrid_residual_model.pkl")
print("Hybrid residual model saved to hybrid_residual_model.pkl")
# === 6. ARIMA on last 24 hours ===
series = df["GPS_smooth"].values
train_ar, test_ar = series[:-24], series[-24:]
arima = ARIMA(train_ar, order=(5,1,0)).fit()
arima_pred = arima.forecast(24)

# === 7. Evaluate & Plot ===
def inv(y_pred):
    return scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

results = {}
# Classical + Hybrid
for name, m in tuned.items():
    if name=="Hybrid":
        y_p = lstm.predict(X_te_seq) + tuned["Hybrid"].predict(X_te_flat)
    else:
        y_p = m.predict(X_te_flat)
    y_t = y_te
    r2  = r2_score(y_t[:,0], y_p[:,0])
    mse = mean_squared_error(y_t[:,0], y_p[:,0])
    mae = mean_absolute_error(y_t[:,0], y_p[:,0])
    results[name] = (r2,mse,mae)
    plt.figure(figsize=(6,3))
    plt.plot(inv(y_t[0]), label="Actual")
    plt.plot(inv(y_p[0]), label="Predicted")
    plt.title(f"{name} 24h Forecast")
    plt.xlabel("Hour Ahead"); plt.ylabel("TEC")
    plt.legend(); plt.tight_layout(); plt.show()

# LSTM standalone
y_p_lstm = lstm.predict(X_te_seq)
r2 = r2_score(y_te[:,0], y_p_lstm[:,0])
mse= mean_squared_error(y_te[:,0], y_p_lstm[:,0])
mae= mean_absolute_error(y_te[:,0], y_p_lstm[:,0])
results["LSTM"] = (r2,mse,mae)
plt.figure(figsize=(6,3))
plt.plot(inv(y_te[0]), label="Actual")
plt.plot(inv(y_p_lstm[0]), label="LSTM")
plt.title("LSTM Standalone"); plt.legend(); plt.show()

# ARIMA
r2_ar = r2_score(test_ar, arima_pred)
mse_ar= mean_squared_error(test_ar, arima_pred)
mae_ar= mean_absolute_error(test_ar, arima_pred)
results["ARIMA"] = (r2_ar,mse_ar,mae_ar)
plt.figure(figsize=(6,3))
plt.plot(test_ar, label="Actual")
plt.plot(arima_pred, label="ARIMA")
plt.title("ARIMA 24h"); plt.legend(); plt.show()

# Print metrics
print("\nModel Performance (24h horizon):")
for k,(r2,mse,mae) in results.items():
    print(f"{k:12s} → R²={r2:.3f}, MSE={mse:.2f}, MAE={mae:.2f}")
