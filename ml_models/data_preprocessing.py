"""
Data Preprocessing Pipeline
Paper: "Anomaly Detection for Power Consumption Data based on Isolated Forest"

This script cleans raw UCI Household Power Consumption dataset and
generates features aligned with Section III-A of the paper:
  - Preprocessing (negative/zero/outlier removal)
  - Mean-based features (7 weekday bucket averages)
  - Trend-based features (sliding window D/R indices)
  - Saves both hourly CSV and feature-engineered CSV
"""

import pandas as pd
import numpy as np
import os

# ─── LOAD RAW DATASET ─────────────────────────────────────────────────────────
DATA_PATH = "data/dataset.txt"
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at {DATA_PATH}")
    print("Download from: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption")
    exit(1)

print("Loading dataset…")
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values="?",
    low_memory=False,
)
print(f"Initial shape: {df.shape}")

# ─── PAPER §III-A-1: PREPROCESSING ───────────────────────────────────────────
# Combine Date + Time into datetime index
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df.drop(["Date", "Time"], axis=1, inplace=True)
df.set_index("datetime", inplace=True)

# Convert numeric columns
num_cols = df.columns.tolist()
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Forward-fill missing values (paper: remove NA/outlier, then clean)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)  # fallback for leading NAs

# Paper §III-A-1: Remove negative values, zero values, abnormal large values
df = df[df["Global_active_power"] > 0]
df = df[df["Global_active_power"] < 20]  # hard cap at 20 kW (> physical household max)

print(f"After cleaning: {df.shape}")

# ─── RESAMPLE TO HOURLY ───────────────────────────────────────────────────────
df_hourly = df.resample("H").mean()
df_hourly.dropna(inplace=True)
df_hourly.to_csv("data/hourly_data.csv")
print(f"Hourly data saved: {df_hourly.shape}")

# ─── PAPER §III-A-2: MEAN-BASED FEATURES ─────────────────────────────────────
# 7 weekday-bucket averages per user window (here: rolling per-day)
df_daily = df_hourly["Global_active_power"].resample("D").mean().dropna()
df_daily = df_daily.reset_index()
df_daily["weekday"] = df_daily["datetime"].dt.dayofweek  # 0=Mon, 6=Sun
df_daily["date"]    = df_daily["datetime"].dt.date

# Pivot to get 7-column weekday feature table
weekday_pivot = df_daily.pivot_table(
    values="Global_active_power",
    index=df_daily["datetime"].dt.isocalendar().week.values,
    columns="weekday",
    aggfunc="mean"
)
weekday_pivot.columns = [f"avg_day_{i}" for i in range(7)]
weekday_pivot.dropna(inplace=True)
weekday_pivot.to_csv("data/weekday_features.csv")
print(f"Weekday features saved: {weekday_pivot.shape}")

# ─── PAPER §III-A-3: TREND-BASED FEATURES ────────────────────────────────────
def compute_trend_features(series: pd.Series, w2_len: int = 7) -> pd.DataFrame:
    """
    For each sliding window position, compute D (downward) and R (rising) indices.
    D = avg1 / (avg3 + 0.001) / (std1 + std3 + 0.001)
    R = avg3 / (avg1 + 0.001) / (std1 + std3 + 0.001)
    """
    records = []
    vals    = series.values
    idx     = series.index

    for i in range(w2_len, len(vals) - w2_len):
        w1 = vals[i - w2_len: i]
        w3 = vals[i: i + w2_len]

        avg1, avg3 = w1.mean(), w3.mean()
        std1 = w1.std() + 1e-9
        std3 = w3.std() + 1e-9

        D = (avg1 / (avg3 + 0.001)) / (std1 + std3 + 0.001)
        R = (avg3 / (avg1 + 0.001)) / (std1 + std3 + 0.001)

        records.append({
            "datetime":              idx[i],
            "Global_active_power":   vals[i],
            "D_downward_trend":      round(D, 6),
            "R_rising_trend":        round(R, 6),
        })

    return pd.DataFrame(records).set_index("datetime")


print("Computing trend features (sliding window)…")
trend_df = compute_trend_features(df_hourly["Global_active_power"], w2_len=24)
trend_df.to_csv("data/trend_features.csv")
print(f"Trend features saved: {trend_df.shape}")

# ─── SAVE PROCESSED SPARK-READY CSV ──────────────────────────────────────────
df_hourly[["Global_active_power"]].to_csv("data/processed_spark_data.csv")
print("Spark-ready data saved: data/processed_spark_data.csv")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n=== Preprocessing Summary ===")
print(f"  Records (hourly):       {len(df_hourly):,}")
print(f"  GAP mean:               {df_hourly['Global_active_power'].mean():.4f} kW")
print(f"  GAP std:                {df_hourly['Global_active_power'].std():.4f} kW")
print(f"  Date range:             {df_hourly.index[0]} → {df_hourly.index[-1]}")
print(f"  Weekday feature rows:   {len(weekday_pivot)}")
print(f"  Trend feature rows:     {len(trend_df)}")
print("\nPreprocessing complete ✓")