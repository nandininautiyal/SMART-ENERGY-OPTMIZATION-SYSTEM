import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/hourly_data.csv", parse_dates=['datetime'], index_col='datetime')

# Use one feature
data = df[['Global_active_power']]

# Rolling average (simple forecasting baseline)
data['prediction'] = data['Global_active_power'].rolling(window=24).mean()

# Drop NaN
data = data.dropna()

# Plot
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Global_active_power'], label='Actual')
plt.plot(data.index, data['prediction'], label='Predicted', color='red')

plt.legend()
plt.title("Power Consumption Forecast")
plt.show()