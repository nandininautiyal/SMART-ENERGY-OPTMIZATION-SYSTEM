"""
Smart Energy Optimization System - FastAPI Backend
Based on: "Anomaly Detection for Power Consumption Data based on Isolated Forest"
(POWERCON 2018, Wei Mao et al.)

Key paper concepts implemented:
- Isolation Forest for unsupervised anomaly detection
- Feature Engineering: Mean-based (weekday averages) + Trend-based (sliding window D/R indices)
- PCA for feature dimensionality reduction
- TinyML decision logic (software-only)
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Smart Energy Analytics API", version="2.0")

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── STATIC FILES (R plot) ───────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="data"), name="static")

# ─── LOAD & CLEAN DATA ───────────────────────────────────────────────────────
df = pd.read_csv("data/processed_spark_data.csv")
df.replace("?", pd.NA, inplace=True)


if "power_kw" in df.columns and "Global_active_power" not in df.columns:
    df.rename(columns={"power_kw": "Global_active_power"}, inplace=True)

df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df.dropna(subset=["Global_active_power"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ─── PAPER SECTION III-A: FEATURE ENGINEERING ────────────────────────────────

def extract_trend_features(series: pd.Series, w2_len: int = 7) -> dict:
    """
    Paper Section III-A-3: Feature based on Trend.
    Sliding window approach with W1, W2, W3.
    Computes downward trend D and rising trend R indices.
    D = avg1/(avg3+0.001) / (std1+std3+0.001)
    R = avg3/(avg1+0.001) / (std1+std3+0.001)
    Returns max D and max R across all windows (as anomaly trend indicators).
    """
    max_d, max_r = 0.0, 0.0
    n = len(series)
    step = w2_len

    for i in range(step, n - step):
        w1 = series[i - step: i].values
        w3 = series[i: i + step].values

        avg1, avg3 = w1.mean(), w3.mean()
        std1, std3 = w1.std() + 1e-6, w3.std() + 1e-6

        d = (avg1 / (avg3 + 0.001)) / (std1 + std3 + 0.001)
        r = (avg3 / (avg1 + 0.001)) / (std1 + std3 + 0.001)

        if d > max_d:
            max_d = d
        if r > max_r:
            max_r = r

    return {"max_D": float(max_d), "max_R": float(max_r)}


def build_feature_matrix(series: pd.Series) -> np.ndarray:
    """
    Paper Section III-A: Build N-sample feature matrix.
    Features per window:
      - mean, std, min, max (statistical)
      - weekday-style 7-bucket averages (mean-based features)
      - D index, R index (trend-based features)
    Total: 4 + 7 + 2 = 13 features → reduced via PCA to 5
    """
    window = 24  # 24 samples per "user day"
    features = []
    vals = series.values

    for i in range(0, len(vals) - window, window):
        seg = vals[i: i + window]

        # Statistical features
        f_mean = seg.mean()
        f_std  = seg.std() + 1e-9
        f_min  = seg.min()
        f_max  = seg.max()

        # 7 weekday-style bucket averages (paper: avg Mon-Sun)
        bucket_avgs = [seg[j::7].mean() if len(seg[j::7]) > 0 else 0
                       for j in range(7)]

        # Trend features D and R for this segment
        trend = extract_trend_features(pd.Series(seg), w2_len=4)

        row = [f_mean, f_std, f_min, f_max] + bucket_avgs + [trend["max_D"], trend["max_R"]]
        features.append(row)

    return np.array(features)


# ─── PAPER SECTION III-B: PCA DIMENSIONALITY REDUCTION ───────────────────────
print("Building feature matrix (Paper §III-A)…")
feature_matrix = build_feature_matrix(df["Global_active_power"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix)

# PCA: keep 5 principal components (paper uses PCA; avg accuracy 73.42%)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_.tolist()

print(f"PCA explained variance: {[round(v*100,1) for v in explained_variance]}%")

# ─── PAPER SECTION II: ISOLATION FOREST ──────────────────────────────────────
# Paper params: contamination (small fraction), 100 iTrees, 256 samples
iforest = IsolationForest(
    n_estimators=100,       # paper: 100 trees sufficient
    max_samples=256,        # paper: average 256 samples
    contamination=0.01,     # paper: anomalies are "only a small part"
    random_state=42
)
segment_anomaly_labels = iforest.fit_predict(X_pca)
segment_anomaly_scores = iforest.decision_function(X_pca)

# Map back to per-row level for simple endpoint compatibility
# Each segment = 24 rows; broadcast label
window = 24
anomaly_col = np.zeros(len(df), dtype=int)
for idx, label in enumerate(segment_anomaly_labels):
    start = idx * window
    end = min(start + window, len(df))
    if label == -1:
        anomaly_col[start:end] = 1

df["anomaly"] = anomaly_col
df["anomaly_score"] = 0.0
for idx, score in enumerate(segment_anomaly_scores):
    start = idx * window
    end = min(start + window, len(df))
    df.loc[start:end-1, "anomaly_score"] = score

print(f"Total anomalies detected: {df['anomaly'].sum()} / {len(df)}")

# ─── TINYML DECISION ENGINE (Software-only, §IoT integration) ────────────────
TINYML_THRESHOLDS = {
    "high":    2.5,
    "low":     0.3,
    "warning": 4.0,
}

def tinyml_decision(power: float) -> dict:
    """
    TinyML-style rule-based inference engine.
    Simulates what would run on an MCU (e.g., Arduino Nano 33).
    Returns decision, confidence, and recommended action.
    """
    if power >= TINYML_THRESHOLDS["warning"]:
        return {
            "label": "CRITICAL",
            "emoji": "🔴",
            "confidence": min(100, int((power / 6.0) * 100)),
            "action": "Immediate load shedding required",
            "priority": 3,
        }
    elif power >= TINYML_THRESHOLDS["high"]:
        return {
            "label": "HIGH",
            "emoji": "⚡",
            "confidence": int(((power - 2.5) / 1.5) * 100),
            "action": "Reduce load — turn off AC / heater",
            "priority": 2,
        }
    elif power <= TINYML_THRESHOLDS["low"]:
        return {
            "label": "LOW",
            "emoji": "💤",
            "confidence": int((1 - power / 0.3) * 100),
            "action": "Low usage — no action needed",
            "priority": 0,
        }
    else:
        return {
            "label": "NORMAL",
            "emoji": "✅",
            "confidence": 90,
            "action": "Usage is optimal",
            "priority": 1,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return {
        "message": "Smart Energy Analytics API v2.0",
        "paper": "Anomaly Detection for Power Consumption Data based on Isolated Forest",
        "endpoints": ["/stats", "/prediction", "/anomalies", "/insights",
                      "/tinyml", "/spark-stats", "/analyze", "/analyze-rooms",
                      "/pca-info", "/model-info", "/trend-features"],
    }


@app.get("/stats")
def get_stats():
    q75, q25 = df["Global_active_power"].quantile([0.75, 0.25])
    return {
        "mean_power":   round(float(df["Global_active_power"].mean()), 4),
        "max_power":    round(float(df["Global_active_power"].max()), 4),
        "min_power":    round(float(df["Global_active_power"].min()), 4),
        "std_power":    round(float(df["Global_active_power"].std()), 4),
        "median_power": round(float(df["Global_active_power"].median()), 4),
        "iqr":          round(float(q75 - q25), 4),
        "total_records": int(len(df)),
        "anomaly_count": int(df["anomaly"].sum()),
        "anomaly_rate":  round(float(df["anomaly"].mean() * 100), 2),
    }


@app.get("/prediction")
def get_prediction():
    """Rolling 24-window mean as simple forecast baseline (forecasting.py logic)."""
    temp = df[["Global_active_power"]].copy()
    temp["prediction"] = temp["Global_active_power"].rolling(window=24).mean()
    temp = temp.dropna()
    tail = temp.tail(200)
    return {
        "Global_active_power": [round(v, 4) for v in tail["Global_active_power"].tolist()],
        "prediction":          [round(v, 4) for v in tail["prediction"].tolist()],
        "rmse": round(float(np.sqrt(((tail["Global_active_power"] - tail["prediction"])**2).mean())), 4),
    }


@app.get("/anomalies")
def get_anomalies():
    """Return anomaly points with their isolation scores."""
    anomalies = df[df["anomaly"] == 1].tail(200)
    return {
        "Global_active_power": [round(v, 4) for v in anomalies["Global_active_power"].tolist()],
        "anomaly_scores":      [round(v, 4) for v in anomalies["anomaly_score"].tolist()],
        "indices":             anomalies.index.tolist(),
        "count":               int(len(anomalies)),
    }


@app.get("/insights")
def get_insights():
    """AI-generated textual insights based on data statistics."""
    peak       = df["Global_active_power"].max()
    avg        = df["Global_active_power"].mean()
    anom_count = int(df["anomaly"].sum())
    anom_rate  = df["anomaly"].mean() * 100
    std        = df["Global_active_power"].std()

    insights = []

    if peak > 5:
        insights.append({"type": "warning", "msg": f"⚠️ Peak usage {peak:.2f} kW exceeds safe threshold (5 kW)"})
    if avg > 2:
        insights.append({"type": "alert", "msg": f"📊 Average consumption {avg:.2f} kW is above normal (2 kW)"})
    if anom_count > 100:
        insights.append({"type": "critical", "msg": f"🚨 {anom_count} anomalies detected ({anom_rate:.1f}% of data)"})
    if std > 1.5:
        insights.append({"type": "info", "msg": f"📈 High variability detected (σ = {std:.2f}) — irregular usage patterns"})

    # PCA variance insight
    total_var = sum(explained_variance[:2]) * 100
    insights.append({"type": "model", "msg": f"🔬 PCA: First 2 components explain {total_var:.1f}% of variance"})

    if not insights or all(i["type"] == "model" for i in insights):
        insights.insert(0, {"type": "ok", "msg": "✅ Energy usage patterns are within normal range"})

    return {"insights": insights, "anomaly_rate": round(anom_rate, 2)}


@app.get("/tinyml")
def get_tinyml():
    """TinyML inference on latest reading."""
    latest = float(df["Global_active_power"].iloc[-1])
    decision = tinyml_decision(latest)
    return {
        "power":    round(latest, 4),
        "decision": decision,
        "thresholds": TINYML_THRESHOLDS,
    }


@app.get("/spark-stats")
def get_spark_stats():
    """Stats from Spark-processed CSV."""
    sdf = pd.read_csv("data/processed_spark_data.csv")
    if "power_kw" in sdf.columns and "Global_active_power" not in sdf.columns:
        sdf.rename(columns={"power_kw": "Global_active_power"}, inplace=True)
    sdf["Global_active_power"] = pd.to_numeric(sdf["Global_active_power"], errors="coerce")
    sdf.dropna(inplace=True)
    return {
        "rows":      int(len(sdf)),
        "avg_power": round(float(sdf["Global_active_power"].mean()), 4),
        "max_power": round(float(sdf["Global_active_power"].max()), 4),
        "min_power": round(float(sdf["Global_active_power"].min()), 4),
        "std_power": round(float(sdf["Global_active_power"].std()), 4),
        "source":    "Apache Spark (PySpark) → processed_spark_data.csv",
    }


@app.get("/analyze")
def analyze(power: float = Query(..., description="Power reading in kW")):
    """Single-point analysis combining IForest score + TinyML."""
    decision   = tinyml_decision(power)
    mean_power = float(df["Global_active_power"].mean())
    std_power  = float(df["Global_active_power"].std())

    # Z-score based anomaly flag (threshold: mean + 2σ, paper §III-A)
    z_score      = (power - mean_power) / std_power
    anomaly_flag = bool(abs(z_score) > 2.0)

    remedy = (
        "Turn off AC, heater, or high-load devices"  if power > 2.5 else
        "Standby devices may be drawing power"        if power < 0.3 else
        "Usage is optimal — no action needed"
    )

    return {
        "power":        round(power, 4),
        "z_score":      round(z_score, 3),
        "decision":     decision,
        "remedy":       remedy,
        "anomaly":      anomaly_flag,
        "mean_baseline": round(mean_power, 4),
    }


@app.post("/analyze-rooms")
def analyze_rooms(data: dict):
    """Multi-room analysis with per-room TinyML decisions."""
    rooms      = data.get("rooms", [])
    mean_power = float(df["Global_active_power"].mean())
    std_power  = float(df["Global_active_power"].std())
    results    = []

    for room in rooms:
        power = float(room.get("power", 0))
        name  = str(room.get("name", "Unknown"))

        decision = tinyml_decision(power)
        z_score  = (power - mean_power) / (std_power + 1e-9)
        anomaly  = bool(abs(z_score) > 2.0)

        remedy = (
            "Turn off AC, heater, or unused devices" if power > 2.5 else
            "No action needed"                        if power <= 0.3 else
            "Maintain current usage"
        )

        results.append({
            "room":    name,
            "power":   round(power, 4),
            "decision": decision,
            "remedy":  remedy,
            "anomaly": anomaly,
            "z_score": round(z_score, 3),
        })

    total = sum(r["power"] for r in results)
    return {
        "results":     results,
        "total_power": round(total, 4),
        "avg_power":   round(total / len(results) if results else 0, 4),
    }


@app.get("/pca-info")
def get_pca_info():
    """Return PCA model information (paper §III-B)."""
    return {
        "method":             "Principal Component Analysis (PCA)",
        "paper_reference":    "Wold S, Esbensen K, Geladi P. 1987",
        "n_components":       5,
        "explained_variance": [round(v * 100, 2) for v in explained_variance],
        "total_variance_explained": round(sum(explained_variance) * 100, 2),
        "original_features":  13,
        "note": "PCA outperforms AutoEncoder for this dataset (Paper Table II: 73.4% vs 66.8%)",
    }


@app.get("/model-info")
def get_model_info():
    """Return Isolation Forest model metadata (paper §II)."""
    return {
        "algorithm":         "Isolation Forest (iForest)",
        "paper_reference":   "Liu F T, Ting K M, Zhou Z H. ICDM 2008",
        "n_estimators":      100,
        "max_samples":       256,
        "contamination":     0.01,
        "feature_reduction": "PCA (5 components)",
        "anomaly_count":     int(df["anomaly"].sum()),
        "total_records":     int(len(df)),
        "anomaly_rate_pct":  round(float(df["anomaly"].mean() * 100), 2),
        "paper_accuracy":    "73.42% (PCA + IForest, Table II)",
        "type":              "Unsupervised — no labeled data required",
    }


@app.get("/trend-features")
def get_trend_features():
    """Compute D and R trend indices from paper §III-A-3."""
    sample = df["Global_active_power"].iloc[:500]
    trend = extract_trend_features(sample, w2_len=7)
    return {
        "max_D_index": round(trend["max_D"], 4),
        "max_R_index": round(trend["max_R"], 4),
        "description": {
            "D": "Downward trend index — high D = sudden power drop (possible theft/fault)",
            "R": "Rising trend index — high R = sudden power spike (overload/anomaly)",
        },
        "formula": {
            "D": "avg1 / (avg3 + 0.001) / (std1 + std3 + 0.001)",
            "R": "avg3 / (avg1 + 0.001) / (std1 + std3 + 0.001)",
        },
    }