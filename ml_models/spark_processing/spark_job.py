"""
spark_job.py — Pandas-based data processing (no PySpark required)
Replicates Spark SQL weekday aggregation from paper §III-A-2
Runs inside python:3.11-slim container
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_PATH    = "/app/data/dataset.txt"
CSV_PATH     = "/app/data/hourly_data.csv"
OUTPUT_PATH  = "/app/data/processed_spark_data.csv"
WEEKDAY_PATH = "/app/data/weekday_features.csv"
TREND_PATH   = "/app/data/trend_features.csv"


def load_data():
    if os.path.exists(CSV_PATH):
        print(f"[Job] Loading {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, low_memory=False)
        # Rename power column to standard name
        if "Global_active_power" in df.columns:
            df = df.rename(columns={"Global_active_power": "power_kw"})
        elif "power_kw" not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) == 0:
                raise ValueError("No numeric column found in hourly_data.csv")
            df = df.rename(columns={num_cols[0]: "power_kw"})
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                "2023-01-01", periods=len(df), freq="h"
            )
        return df

    if os.path.exists(DATA_PATH):
        print(f"[Job] Loading {DATA_PATH}")
        try:
            raw = pd.read_csv(
                DATA_PATH, header=None, sep=r'\s+',
                on_bad_lines='skip', engine='python',
                nrows=500000   # cap for memory safety
            )
        except Exception as e:
            print(f"[Job] CSV read error: {e} — trying semicolon separator")
            raw = pd.read_csv(
                DATA_PATH, header=0, sep=";",
                on_bad_lines='skip', engine='python',
                nrows=500000
            )

        # Find first column that is >50% numeric
        power_series = None
        for col in raw.columns:
            vals = pd.to_numeric(raw[col], errors='coerce')
            if vals.notna().sum() > len(raw) * 0.5:
                power_series = vals
                break

        if power_series is None:
            raise ValueError("No numeric column found in dataset.txt")

        df = pd.DataFrame({
            "power_kw":  power_series.values,
            "timestamp": pd.date_range(
                "2023-01-01", periods=len(power_series), freq="min"
            )
        })
        df = df.dropna(subset=["power_kw"])
        return df

    # Fallback: synthetic data
    print("[Job] No data file found — generating synthetic data")
    np.random.seed(42)
    n = 8760
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "power_kw":  np.abs(np.random.normal(5.0, 1.5, n))
    })


def process(df):
    # Paper §III-A-1: filter invalid readings
    df = df.copy()
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["power_kw", "timestamp"])
    df = df[(df["power_kw"] > 0) & (df["power_kw"] <= 20)]

    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["date"]    = df["timestamp"].dt.date.astype(str)

    # Paper §III-A-2: 7 weekday time-of-day buckets
    bins   = [0, 6, 9, 12, 14, 18, 21, 24]
    labels = ["night", "morning", "late_morning",
              "noon", "afternoon", "evening", "late_evening"]
    df["bucket"] = pd.cut(
        df["hour"], bins=bins, labels=labels,
        right=False, include_lowest=True
    )

    weekday_avg = (
        df.groupby(["weekday", "bucket"], observed=True)["power_kw"]
          .mean()
          .reset_index()
          .rename(columns={"power_kw": "avg_power_kw"})
    )

    # Paper §III-A-3: trend features D (deviation) and R (ratio)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rolling_mean"] = df["power_kw"].rolling(24, min_periods=1).mean()
    df["D"] = df["power_kw"] - df["rolling_mean"]
    df["R"] = df["power_kw"] / (df["rolling_mean"] + 1e-9)

    trend = df[["timestamp", "power_kw", "D", "R", "weekday", "hour"]].copy()

    print(f"[Job] Processed rows  : {len(df):,}")
    print(f"[Job] Weekday buckets : {len(weekday_avg)}")
    print(f"[Job] D range         : {df['D'].min():.3f} to {df['D'].max():.3f}")
    print(f"[Job] R range         : {df['R'].min():.3f} to {df['R'].max():.3f}")

    return df, weekday_avg, trend


def main():
    print("[Job] ====================================")
    print("[Job]  Smart Energy — Processing Job")
    print("[Job] ====================================")
    print(f"[Job] Python version : {sys.version}")
    print(f"[Job] pandas version : {pd.__version__}")
    print(f"[Job] Working dir    : {os.getcwd()}")
    print(f"[Job] Files in /app/data: {os.listdir('/app/data') if os.path.exists('/app/data') else 'NOT FOUND'}")

    df = load_data()
    print(f"[Job] Raw rows loaded : {len(df):,}")

    df, weekday_avg, trend = process(df)

    df.to_csv(OUTPUT_PATH, index=False)
    weekday_avg.to_csv(WEEKDAY_PATH, index=False)
    trend.to_csv(TREND_PATH, index=False)

    print(f"[Job] Saved -> {OUTPUT_PATH}")
    print(f"[Job] Saved -> {WEEKDAY_PATH}")
    print(f"[Job] Saved -> {TREND_PATH}")
    print("[Job] DONE ✓")
    sys.exit(0)   # explicit clean exit


if __name__ == "__main__":
    main()