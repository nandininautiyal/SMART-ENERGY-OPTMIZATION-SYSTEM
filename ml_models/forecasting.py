"""
Power Consumption Forecasting
Paper: "Anomaly Detection for Power Consumption Data based on Isolated Forest"

Baseline forecasting using rolling window mean.
This serves as the comparison baseline for the anomaly detection model.
Future extensions: ARIMA, LSTM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_FILE = "data/hourly_data.csv"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "data/processed_spark_data.csv"

df = pd.read_csv(DATA_FILE, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df.dropna(subset=["Global_active_power"], inplace=True)

print(f"Loaded {len(df):,} records for forecasting")

# ── Rolling mean forecast (24h window) ────────────────────────────────────────
data = df[["Global_active_power"]].copy()
data["prediction_24h"] = data["Global_active_power"].rolling(window=24, min_periods=1).mean()
data["prediction_48h"] = data["Global_active_power"].rolling(window=48, min_periods=1).mean()
data["prediction_7d"]  = data["Global_active_power"].rolling(window=168, min_periods=1).mean()
data.dropna(inplace=True)

# ── Evaluation metrics ─────────────────────────────────────────────────────────
actual = data["Global_active_power"]
pred   = data["prediction_24h"]

rmse = np.sqrt(((actual - pred) ** 2).mean())
mae  = (actual - pred).abs().mean()
mape = ((actual - pred).abs() / (actual + 1e-9)).mean() * 100

print(f"\nForecasting Evaluation (24h rolling mean):")
print(f"  RMSE : {rmse:.4f} kW")
print(f"  MAE  : {mae:.4f} kW")
print(f"  MAPE : {mape:.2f}%")

# ── Plot ───────────────────────────────────────────────────────────────────────
tail = data.tail(500)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    ax.grid(color="#30363d", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# Panel 1: Actual vs 24h rolling
axes[0].plot(tail.index, tail["Global_active_power"], color="#00d4ff", linewidth=0.8,
             alpha=0.9, label="Actual Power")
axes[0].plot(tail.index, tail["prediction_24h"], color="#ffa502", linewidth=1.8,
             linestyle="--", label="24h Rolling Mean")
axes[0].plot(tail.index, tail["prediction_7d"],  color="#a29bfe", linewidth=1.2,
             linestyle=":", label="7-day Rolling Mean")
axes[0].set_title("Power Consumption Forecast — Rolling Window Baseline",
                  color="#e6edf3", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Power (kW)", color="#e6edf3")
axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

# Panel 2: Forecast error
error = tail["Global_active_power"] - tail["prediction_24h"]
axes[1].fill_between(tail.index, error, 0,
                     where=(error > 0), color="#ff4757", alpha=0.5, label="Over-forecast")
axes[1].fill_between(tail.index, error, 0,
                     where=(error < 0), color="#2ed573", alpha=0.5, label="Under-forecast")
axes[1].axhline(0, color="#30363d", linewidth=1)
axes[1].set_title(f"Forecast Error  (RMSE={rmse:.4f} kW, MAE={mae:.4f} kW, MAPE={mape:.2f}%)",
                  color="#e6edf3", fontsize=11)
axes[1].set_ylabel("Error (kW)", color="#e6edf3")
axes[1].set_xlabel("Time", color="#e6edf3")
axes[1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

plt.tight_layout()
plt.savefig("data/forecast_results.png", dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → data/forecast_results.png")
plt.show()
print("\nForecasting complete ✓")