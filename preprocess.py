import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load cleaned CSV
try:
    df = pd.read_csv('drill-data.csv')
    print(f"Loaded CSV with {len(df)} rows")
    print("First few rows:\n", df.head())
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Parse timestamp (fixed format issue)
# Option 1: robust parsing
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')

print(f"After parsing Time, {df['Time'].notna().sum()} rows remain (non-NaN Time)")

# Drop rows with NaN Time
df = df.dropna(subset=['Time'])
print(f"After dropping NaN Time, {len(df)} rows remain")

# Required columns
required_cols = ['Temp', 'RPM', 'Load', 'Vibration', 'Depth']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    exit(1)

# Convert to numeric (safety)
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in required columns
df = df.dropna(subset=required_cols)
print(f"After dropping NaN data, {len(df)} rows remain")

# Check minimum rows
if len(df) < 20:
    print(f"Error: Only {len(df)} rows after cleaning. Need at least 20 for sequences.")
    exit(1)

# Feature engineering
df['temp_change'] = df['Temp'].diff().fillna(0)
df['torque_est'] = (df['Load'] * 9.55) / df['RPM'].clip(lower=1)

# Simulated RUL
df['cycle'] = range(len(df))
df['RUL'] = 500 - (df['cycle'] % 500)

# Normalize features
scaler = MinMaxScaler()
features = ['Temp', 'RPM', 'Load', 'Vibration', 'Depth', 'temp_change', 'torque_est']

try:
    df[features] = scaler.fit_transform(df[features])
    print("Normalization successful")
except Exception as e:
    print(f"Normalization failed: {e}")
    print("DataFrame info:\n", df[features].info())
    exit(1)

# Create sequences for ML models
def create_sequences(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        seq = data[features].iloc[i:i+window_size].values
        X.append(seq)
        y.append(data['RUL'].iloc[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(df)
print(f"Created {len(X)} sequences")

if len(X) == 0:
    print("Error: No sequences created. Need at least 20 rows.")
    exit(1)

# Train-test split and save
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print("Preprocessing done. Files saved.")
