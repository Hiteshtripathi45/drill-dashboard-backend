import pandas as pd

# Load original CSV (change file name if needed)
df = pd.read_csv('drill-data (5).csv')

# Replace '—' with NaN
df.replace('—', pd.NA, inplace=True)

# Drop rows where all important data columns are NaN
df.dropna(subset=['Temp', 'RPM', 'Load', 'Vibration', 'Depth'], how='all', inplace=True)

# Convert Depth to numeric and fill missing with mean
df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
mean_depth = df['Depth'].mean()
df['Depth'].fillna(mean_depth, inplace=True)

# Ensure other columns are numeric
for col in ['Temp', 'RPM', 'Load', 'Vibration']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows still having NaN in required columns
df.dropna(subset=['Temp', 'RPM', 'Load', 'Vibration', 'Depth'], inplace=True)

# Save cleaned CSV
df.to_csv('drill-data.csv', index=False)
print(f"Cleaned CSV saved with {len(df)} rows. Mean Depth used: {mean_depth:.2f}")
