# import os
# import pandas as pd
# import numpy as np
# import pywt
# import joblib
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from scipy.stats import zscore
# import matplotlib.pyplot as plt

# # === 1. Load Data from Folder ===
# folder_path = r"C:\Users\Tharani\Desktop\ML\ML\data_set"
# all_data = []

# for i, file_name in enumerate(os.listdir(folder_path)):
#     if file_name.endswith('.csv'):
#         file_path = os.path.join(folder_path, file_name)
#         df = pd.read_csv(file_path)
#         df.columns = df.columns.str.strip()
#         df["Day"] = i + 1  # simple index-based day assignment

#         if "TEC" not in df.columns or "Time" not in df.columns:
#             continue

#         df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour + \
#                      pd.to_datetime(df["Time"], format="%H:%M:%S").dt.minute / 60

#         gps_clean = df["TEC"].interpolate().fillna(method="bfill").fillna(method="ffill")
#         coeffs = pywt.wavedec(gps_clean, 'db4', level=3)
#         coeffs[1:] = [pywt.threshold(c, 0.1 * max(c), mode='soft') for c in coeffs[1:]]
#         smooth_gps = pywt.waverec(coeffs, 'db4')[:len(df)]
#         df["GPS_smooth"] = smooth_gps

#         df["SinTime"] = np.sin(2 * np.pi * df["Time"] / 24)
#         df["CosTime"] = np.cos(2 * np.pi * df["Time"] / 24)
#         df["Hour"] = df["Time"].astype(int)
#         df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365)
#         df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365)
#         df["TEC_roll3"] = df["GPS_smooth"].rolling(window=3, min_periods=1).mean()
#         df["TEC_prev"] = df["GPS_smooth"].shift(1).fillna(method='bfill')
#         df["TEC_diff"] = df["GPS_smooth"].diff().fillna(0)
#         df["TEC_diff2"] = df["TEC_diff"].diff().fillna(0)
#         df["TEC_std5"] = df["GPS_smooth"].rolling(5).std().fillna(0)

#         df.dropna(inplace=True)
#         all_data.append(df)

# # === 2. Combine All Data ===
# df_all = pd.concat(all_data, ignore_index=True)
# df_all = df_all[np.abs(zscore(df_all["GPS_smooth"])) < 3]

# # === 3. Define Features and Labels ===
# features = ["Day", "Time", "Hour", "SinTime", "CosTime", "Day_sin", "Day_cos",
#             "TEC_roll3", "TEC_prev", "TEC_diff", "TEC_diff2", "TEC_std5"]
# X = df_all[features]
# y = df_all["GPS_smooth"]

# # === 4. Train-Test Split ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Reshape for LSTM ===
# def reshape_for_lstm(X, time_steps=3):
#     X_reshaped = np.array([X.shift(i).fillna(method='bfill').values for i in reversed(range(time_steps))])
#     return np.transpose(X_reshaped, (1, 0, 2))

# X_train_lstm = reshape_for_lstm(pd.DataFrame(X_train), time_steps=3)
# X_test_lstm = reshape_for_lstm(pd.DataFrame(X_test), time_steps=3)

# # === 5. Model Training ===
# rf_model = RandomForestRegressor(random_state=42)
# rf_params = {
#     "n_estimators": [100, 300, 500],
#     "max_depth": [10, 30, None],
#     "min_samples_split": [2, 4, 6],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ['sqrt', 'log2']
# }
# rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=5, cv=3, n_jobs=-1, random_state=42)
# rf_search.fit(X_train, y_train)
# best_rf = rf_search.best_estimator_

# gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
# gb_model.fit(X_train, y_train)

# xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, objective="reg:squarederror", random_state=42)
# xgb_model.fit(X_train, y_train)

# mlp_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000,
#                          activation="relu", early_stopping=True,
#                          learning_rate='adaptive', random_state=42))
# ])
# mlp_pipeline.fit(X_train, y_train)

# # === LSTM ===
# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(64, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

# lstm_model = build_lstm_model(X_train_lstm.shape[1:])
# lstm_model.fit(X_train_lstm, y_train, validation_split=0.1, epochs=50,
#                callbacks=[EarlyStopping(patience=5)], verbose=0)

# # === BiLSTM ===
# def build_bilstm_model(input_shape):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(64), input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

# bilstm_model = build_bilstm_model(X_train_lstm.shape[1:])
# bilstm_model.fit(X_train_lstm, y_train, validation_split=0.1, epochs=50,
#                  callbacks=[EarlyStopping(patience=5)], verbose=0)

# # === 6. Save Models ===
# joblib.dump(best_rf, "./best_rf_model.pkl")
# joblib.dump(gb_model, "./gb_model.pkl")
# joblib.dump(xgb_model, "./xgb_model.pkl")
# joblib.dump(mlp_pipeline, "./mlp_model.pkl")
# lstm_model.save("./lstm_model.h5")
# bilstm_model.save("./bilstm_model.h5")

# # === 7. Evaluation ===
# def custom_accuracy(y_true, y_pred, tol):
#     return np.mean(np.abs(y_true - y_pred) <= tol) * 100

# models = {
#     "Random Forest": best_rf,
#     "Gradient Boosting": gb_model,
#     "XGBoost": xgb_model,
#     "MLP": mlp_pipeline,
#     "LSTM": lstm_model,
#     "BiLSTM": bilstm_model
# }

# evaluation_results = {}
# for name, model in models.items():
#     if name in ["LSTM", "BiLSTM"]:
#         pred = model.predict(X_test_lstm).flatten()
#     else:
#         pred = model.predict(X_test)

#     r2 = r2_score(y_test, pred)
#     mse = mean_squared_error(y_test, pred)
#     mae = mean_absolute_error(y_test, pred)
#     acc_2_5 = custom_accuracy(y_test, pred, 2.5)
#     acc_5_0 = custom_accuracy(y_test, pred, 5.0)
#     acc_7_5 = custom_accuracy(y_test, pred, 7.5)

#     evaluation_results[name] = {
#         "R²": round(r2 * 100, 2),
#         "MSE": round(mse, 2),
#         "MAE": round(mae, 2),
#         "Acc ±2.5": round(acc_2_5, 2),
#         "Acc ±5.0": round(acc_5_0, 2),
#         "Acc ±7.5": round(acc_7_5, 2)
#     }

# print(evaluation_results)

# # === 8. Plot Predicted vs Actual ===
# plt.figure(figsize=(18, 12))

# for idx, (name, model) in enumerate(models.items(), 1):
#     if name in ["LSTM", "BiLSTM"]:
#         y_pred = model.predict(X_test_lstm).flatten()
#     else:
#         y_pred = model.predict(X_test)

#     plt.subplot(3, 2, idx)
#     plt.plot(y_test.values[:200], label='Actual', linewidth=2)
#     plt.plot(y_pred[:200], label='Predicted', linestyle='--')
#     plt.title(f"{name} Prediction vs Actual")
#     plt.xlabel("Sample Index")
#     plt.ylabel("TEC Value")
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# # === 8. Scatter Plot: Predicted vs Actual ===
# plt.figure(figsize=(18, 12))

# for idx, (name, model) in enumerate(models.items(), 1):
#     if name in ["LSTM", "BiLSTM"]:
#         y_pred = model.predict(X_test_lstm).flatten()
#     else:
#         y_pred = model.predict(X_test)

#     plt.subplot(3, 2, idx)
#     plt.scatter(y_test.values, y_pred, alpha=0.5, s=10, label='Predicted vs Actual')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
#     plt.title(f"{name} - Predicted vs Actual")
#     plt.xlabel("Actual TEC")
#     plt.ylabel("Predicted TEC")
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()


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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # CHANGED
from tensorflow.keras import regularizers  # CHANGED
from scipy.stats import zscore
import matplotlib.pyplot as plt

# === 1. Load & preprocess data ===
folder_path = r"C:\Users\Tharani\Desktop\ML\ML\data_set"
all_data = []

for i, file_name in enumerate(os.listdir(folder_path)):
    if not file_name.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(folder_path, file_name))
    df.columns = df.columns.str.strip()
    df["Day"] = i + 1
    if "TEC" not in df.columns or "Time" not in df.columns:
        continue

    # convert Time to fractional hours
    t = pd.to_datetime(df["Time"], format="%H:%M:%S")
    df["Time"] = t.dt.hour + t.dt.minute / 60

    # wavelet smoothing
    gps = df["TEC"].interpolate().fillna(method="bfill").fillna(method="ffill")
    coeffs = pywt.wavedec(gps, 'db4', level=3)
    coeffs[1:] = [
        pywt.threshold(c, 0.1 * max(c), mode='soft')
        for c in coeffs[1:]
    ]
    df["GPS_smooth"] = pywt.waverec(coeffs, 'db4')[:len(df)]

    # feature engineering
    df["SinTime"] = np.sin(2 * np.pi * df["Time"] / 24)
    df["CosTime"] = np.cos(2 * np.pi * df["Time"] / 24)
    df["Hour"] = df["Time"].astype(int)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 365)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 365)
    df["TEC_roll3"] = df["GPS_smooth"].rolling(3, min_periods=1).mean()
    df["TEC_prev"] = df["GPS_smooth"].shift(1).fillna(method='bfill')
    df["TEC_diff"] = df["GPS_smooth"].diff().fillna(0)
    df["TEC_diff2"] = df["TEC_diff"].diff().fillna(0)
    df["TEC_std5"] = df["GPS_smooth"].rolling(5).std().fillna(0)

    df.dropna(inplace=True)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all = df_all[np.abs(zscore(df_all["GPS_smooth"])) < 3]

# === 2. Features & labels ===
features = [
    "Day", "Time", "Hour", "SinTime", "CosTime",
    "Day_sin", "Day_cos", "TEC_roll3", "TEC_prev",
    "TEC_diff", "TEC_diff2", "TEC_std5"
]
X = df_all[features]
y = df_all["GPS_smooth"]

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Reshape for LSTM windows ===
def reshape_for_lstm(X, time_steps=3):
    arr = np.array([
        X.shift(i).fillna(method='bfill').values
        for i in reversed(range(time_steps))
    ])
    return np.transpose(arr, (1, 0, 2))

X_train_lstm = reshape_for_lstm(pd.DataFrame(X_train), time_steps=3)
X_test_lstm  = reshape_for_lstm(pd.DataFrame(X_test),  time_steps=3)

# === 5. Traditional ML models ===
# Random Forest with randomized search
rf_model = RandomForestRegressor(random_state=42)
rf_params = {
    "n_estimators": [100, 300, 500],
    "max_depth": [10, 30, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2']
}
rf_search = RandomizedSearchCV(
    rf_model, rf_params, n_iter=5, cv=3, n_jobs=-1, random_state=42
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, random_state=42
)
gb_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    objective="reg:squarederror", random_state=42
)
xgb_model.fit(X_train, y_train)

# MLP Pipeline
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        activation="relu",
        early_stopping=True,
        learning_rate='adaptive',
        random_state=42
    ))
])
mlp_pipeline.fit(X_train, y_train)

# === 6. Optimized LSTM model ===
def build_lstm_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(
        32,                        # CHANGED: fewer units
        return_sequences=True,     # CHANGED: for attention
        dropout=0.2,               # CHANGED
        recurrent_dropout=0.2,     # CHANGED
        kernel_regularizer=regularizers.l2(1e-4)  # CHANGED
    )(inp)
    x = BatchNormalization()(x)   # CHANGED
    attn = Attention()([x, x])    # CHANGED
    x = GlobalAveragePooling1D()(attn)  # CHANGED
    x = Dropout(0.3)(x)           # CHANGED
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

lstm_model = build_lstm_model(X_train_lstm.shape[1:])
callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True
    ),  # CHANGED: increased patience
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5
    )  # CHANGED
]
lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_split=0.1,
    epochs=100,      # CHANGED: more epochs
    batch_size=8,    # CHANGED: smaller batch size
    callbacks=callbacks,
    verbose=1
)

# === 7. Optimized BiLSTM model ===
def build_bilstm_model(input_shape):
    inp = Input(shape=input_shape)
    x = Bidirectional(
        LSTM(
            32, return_sequences=True,
            dropout=0.2, recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(inp)
    x = BatchNormalization()(x)
    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)
    x = Dropout(0.3)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

bilstm_model = build_bilstm_model(X_train_lstm.shape[1:])
bilstm_history = bilstm_model.fit(
    X_train_lstm, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# === 8. Save all models ===
joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(gb_model, "gb_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(mlp_pipeline, "mlp_model.pkl")
lstm_model.save("lstm_model.h5")
bilstm_model.save("bilstm_model.h5")

# === 9. Evaluation ===
def custom_accuracy(y_true, y_pred, tol):
    return np.mean(np.abs(y_true - y_pred) <= tol) * 100

models = {
    "Random Forest": best_rf,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
    "MLP": mlp_pipeline,
    "LSTM": lstm_model,
    "BiLSTM": bilstm_model
}

results = {}
for name, model in models.items():
    if name in ["LSTM", "BiLSTM"]:
        preds = model.predict(X_test_lstm).flatten()
    else:
        preds = model.predict(X_test)

    r2  = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    acc25 = custom_accuracy(y_test, preds, 2.5)
    acc50 = custom_accuracy(y_test, preds, 5.0)
    acc75 = custom_accuracy(y_test, preds, 7.5)

    results[name] = {
        "R²": round(r2 * 100, 2),
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "Acc ±2.5": round(acc25, 2),
        "Acc ±5.0": round(acc50, 2),
        "Acc ±7.5": round(acc75, 2)
    }

print(results)

# === 10. Plot Predictions vs Actuals ===
plt.figure(figsize=(18, 12))
for idx, (name, model) in enumerate(models.items(), 1):
    if name in ["LSTM", "BiLSTM"]:
        y_pred = model.predict(X_test_lstm).flatten()
    else:
        y_pred = model.predict(X_test)

    plt.subplot(3, 2, idx)
    plt.plot(y_test.values[:200], label='Actual', linewidth=2)
    plt.plot(y_pred[:200], label='Predicted', linestyle='--')
    plt.title(f"{name} Prediction vs Actual")
    plt.xlabel("Sample Index")
    plt.ylabel("TEC")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# === 11. Scatter Plot: Pred vs Actual ===
plt.figure(figsize=(18, 12))
for idx, (name, model) in enumerate(models.items(), 1):
    if name in ["LSTM", "BiLSTM"]:
        y_pred = model.predict(X_test_lstm).flatten()
    else:
        y_pred = model.predict(X_test)

    plt.subplot(3, 2, idx)
    plt.scatter(y_test.values, y_pred, alpha=0.5, s=10)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', linewidth=2
    )
    plt.title(f"{name} - Pred vs Actual")
    plt.xlabel("Actual TEC")
    plt.ylabel("Predicted TEC")
    plt.grid(True)

plt.tight_layout()
plt.show()
