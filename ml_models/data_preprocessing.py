import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/dataset.txt",
    sep=';',
    na_values='?',
    low_memory=False
)

print("Initial shape:", df.shape)

# Combine Date + Time
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Drop original columns
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Convert all columns to float
cols = df.columns.drop('datetime')
df[cols] = df[cols].astype(float)

# Handle missing values
df = df.fillna(method='ffill')

# Set index
df.set_index('datetime', inplace=True)

print("Cleaned shape:", df.shape)
print(df.head())

# Resample data (hourly)
df_hourly = df.resample('H').mean()

# Save cleaned data
df_hourly.to_csv("data/hourly_data.csv")

print("Data preprocessing complete. Saved as hourly_data.csv")