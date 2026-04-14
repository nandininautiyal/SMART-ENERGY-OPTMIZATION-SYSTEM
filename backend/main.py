from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd
from sklearn.ensemble import IsolationForest

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STATIC FILES (R plot) ----------------
app.mount("/static", StaticFiles(directory="data"), name="static")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/processed_spark_data.csv")

df.replace("?", pd.NA, inplace=True)
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], errors="coerce"
)
df.dropna(inplace=True)

# ---------------- ANOMALY MODEL ----------------
model = IsolationForest(contamination=0.01, random_state=42)
df["anomaly"] = model.fit_predict(df[["Global_active_power"]])
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

# ---------------- TINYML ----------------
def tinyml_decision(power):
    if power > 2:
        return "⚡ Reduce load"
    elif power < 0.5:
        return "💤 Low usage"
    else:
        return "✅ Normal usage"

# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {"message": "Energy Analytics API Running"}


@app.get("/stats")
def get_stats():
    return {
        "mean_power": float(df["Global_active_power"].mean()),
        "max_power": float(df["Global_active_power"].max()),
        "min_power": float(df["Global_active_power"].min()),
    }


@app.get("/prediction")
def get_prediction():
    temp = df[["Global_active_power"]].copy()
    temp["prediction"] = temp["Global_active_power"].rolling(window=24).mean()
    temp = temp.dropna()

    return {
        "Global_active_power": temp.tail(200)["Global_active_power"].tolist(),
        "prediction": temp.tail(200)["prediction"].tolist(),
    }


@app.get("/anomalies")
def get_anomalies():
    anomalies = df[df["anomaly"] == 1].tail(200)
    return {
        "Global_active_power": anomalies["Global_active_power"].tolist()
    }


@app.get("/insights")
def get_insights():
    peak = df["Global_active_power"].max()
    avg = df["Global_active_power"].mean()
    anomaly_count = len(df[df["anomaly"] == 1])

    insights = []

    if peak > 5:
        insights.append("⚠️ High peak usage detected")

    if avg > 2:
        insights.append("📊 Overall consumption is high")

    if anomaly_count > 100:
        insights.append("🚨 Frequent anomalies detected")

    if not insights:
        insights.append("✅ Energy usage is normal")

    return {"insights": insights}


@app.get("/tinyml")
def get_tinyml():
    latest = float(df["Global_active_power"].iloc[-1])

    return {
        "power": latest,
        "decision": tinyml_decision(latest),
    }


@app.get("/spark-stats")
def get_spark_stats():
    sdf = pd.read_csv("data/processed_spark_data.csv")

    return {
        "rows": int(len(sdf)),
        "avg_power": float(sdf["Global_active_power"].mean()),
        "max_power": float(sdf["Global_active_power"].max()),
        "min_power": float(sdf["Global_active_power"].min()),
    }


# ---------------- SINGLE INPUT ----------------
@app.get("/analyze")
def analyze(power: float = Query(...)):
    decision = tinyml_decision(power)
    anomaly_flag = bool(power > df["Global_active_power"].mean() + 2)

    remedy = (
        "Turn off unused appliances"
        if power > 2
        else "Usage is optimal"
    )

    return {
        "power": float(power),
        "decision": decision,
        "remedy": remedy,
        "anomaly": anomaly_flag,
    }


# ---------------- MULTI ROOM ANALYSIS ----------------
@app.post("/analyze-rooms")
def analyze_rooms(data: dict):
    rooms = data.get("rooms", [])
    results = []

    mean_power = float(df["Global_active_power"].mean())

    for room in rooms:
        power = float(room.get("power", 0))
        name = str(room.get("name", "Unknown"))

        if power > 2:
            decision = "⚡ Reduce load"
            remedy = "Turn off AC, heater, or unused devices"
        elif power < 0.5:
            decision = "💤 Low usage"
            remedy = "No action needed"
        else:
            decision = "✅ Normal usage"
            remedy = "Maintain usage"

        anomaly = bool(power > mean_power + 2)

        results.append({
            "room": name,
            "power": power,
            "decision": decision,
            "remedy": remedy,
            "anomaly": anomaly,
        })

    return {"results": results}