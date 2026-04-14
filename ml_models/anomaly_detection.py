import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/hourly_data.csv", parse_dates=['datetime'], index_col='datetime')

# Use only one feature (start simple)
data = df[['Global_active_power']]

# Train model
model = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(data)

# Convert labels (-1 = anomaly, 1 = normal)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

print("Total anomalies detected:", df['anomaly'].sum())

# Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Global_active_power'], label='Power')
plt.scatter(df[df['anomaly']==1].index,
            df[df['anomaly']==1]['Global_active_power'],
            color='red', label='Anomaly')

plt.legend()
plt.title("Anomaly Detection in Power Consumption")
plt.show()