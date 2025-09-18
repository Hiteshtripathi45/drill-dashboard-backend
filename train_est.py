import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed data (exclude Temp for input)
X_train = np.load('X_train.npy')[:, :, [1, 2, 3, 4, 5, 6]]  # RPM, Load, Vibration, Depth, temp_change, torque_est
X_test = np.load('X_test.npy')[:, :, [1, 2, 3, 4, 5, 6]]

# Load CSV to get Temp as target
df = pd.read_csv('drill-data.csv')

# FIX: parse Time correctly
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')

# Keep only valid rows
df = df.dropna(subset=['Time', 'Temp', 'RPM', 'Load', 'Vibration', 'Depth'])
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df = df.dropna(subset=['Temp'])

# Scale target (Temp)
scaler_y = MinMaxScaler()
df['Temp_scaled'] = scaler_y.fit_transform(df[['Temp']]).flatten()

# Align y with sequences (window_size=20)
window_size = 20
y_train_full = df['Temp_scaled'][window_size:window_size+len(X_train)]
y_test_full = df['Temp_scaled'][window_size+len(X_train):window_size+len(X_train)+len(X_test)]

# Flatten X for RF
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Safety check
print(f"X_train shape: {X_train_flat.shape}, y_train shape: {len(y_train_full)}")
if len(y_train_full) == 0:
    print("Error: No valid Temp data for training.")
    exit(1)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_flat, y_train_full)

# Evaluate (unscaled for real-world error)
predictions_scaled = rf.predict(X_test_flat)
y_test_unscaled = scaler_y.inverse_transform(y_test_full.values.reshape(-1, 1)).flatten()
predictions_unscaled = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
print(f'MSE on test (unscaled): {mse:.4f}')

# Save model and scaler
joblib.dump(rf, 'temp_est_model.joblib')
joblib.dump(scaler_y, 'temp_scaler.joblib')
print("Temp estimation model trained and saved.")
