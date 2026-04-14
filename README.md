# Smart Energy Optimization System

An AI-powered full-stack application that analyzes energy consumption, detects anomalies, and provides optimization recommendations using Machine Learning, Spark, and R.

---

## Features

- Real-time energy analysis (room-wise input)
- Machine Learning-based anomaly detection (Isolation Forest)
- TinyML-inspired decision system for recommendations
- Apache Spark for large-scale data processing
- R-based statistical visualization (power distribution)
- Fully Dockerized (DevOps-ready)

---

## Tech Stack

- **Frontend:** React.js + Chart.js
- **Backend:** FastAPI (Python)
- **ML:** Scikit-learn (Isolation Forest)
- **Big Data:** Apache Spark
- **Analytics:** R
- **DevOps:** Docker + Docker Compose

---

## How It Works

1. User inputs power consumption (room-wise)
2. Backend analyzes:
   - Anomaly detection (ML)
   - Decision logic (TinyML)
3. System returns:
   - ⚠️ Alerts
   - 💡 Recommendations
4. Visualization:
   - Charts (React)
   - Statistical distribution (R)

---

## Sample Recommendations

- Reduce load → turn off heavy appliances
- Low usage → no action needed
- Anomaly detected → investigate unusual consumption

---

## Run with Docker

```bash
docker-compose -f docker/docker-compose.yml up --build


---

This project uses the **UCI Household Power Consumption Dataset**, a real-world dataset containing electricity usage measurements.

Due to GitHub size limitations, the full dataset is not included in this repository.

