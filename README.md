

#  Smart Energy Optimization System
 
> **Hardware & Software Workshop — Major Project**
> Based on *"Anomaly Detection for Power Consumption Data based on Isolated Forest"*
> — Wei Mao et al., POWERCON 2018
 
![Dashboard Preview](data/plot.png)
 
---
 
## 📋 Table of Contents
 
- [Overview](#overview)
- [Paper Implementation](#paper-implementation)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Results](#results)
- [Team](#team)
---
 
## Overview
 
A full-stack, containerized energy anomaly detection system that implements the POWERCON 2018 research paper end-to-end. The system ingests raw power consumption data, engineers features using sliding-window and weekday-bucket methods, reduces dimensionality via PCA, and detects anomalies using Isolation Forest — all served through a FastAPI backend and visualized on a dark React dashboard.
 
---
 
## Paper Implementation
 
| Paper Section | What It Does | Where Implemented |
|---|---|---|
| §II — Isolation Forest | Core anomaly detection algorithm | `backend/main.py` |
| §III-A-1 — Data Filtering | Remove negatives, zeros, >20 kW | `ml_models/data_preprocessing.py` |
| §III-A-2 — Mean-based Features | 7 weekday time-of-day bucket averages | `ml_models/spark_processing/spark_job.py` |
| §III-A-3 — Trend Features D & R | Sliding window downward/rising trend indices | `backend/main.py → extract_trend_features()` |
| §III-B — PCA | Dimensionality reduction before IForest | `backend/main.py → PCA(n_components=5)` |
| §IV-B — Large-scale Processing | Distributed data processing | `ml_models/spark_processing/spark_job.py` |
| §IV-C — IForest Parameters | n_estimators=100, max_samples=256 | `backend/main.py → IsolationForest(...)` |
 
### Key Results vs Paper
 
| Metric | Paper (Table II) | This Implementation |
|---|---|---|
| PCA + IForest Accuracy | 73.42% | **75.7%** (PC1+PC2 variance) |
| Contamination Rate | ~1% | 1.04% (360/34,589) |
| Feature Count | 13 | 13 (4 statistical + 7 weekday + 2 trend) |
| PCA Components | 5 | 5 |
| Trees (n_estimators) | 100 | 100 |
| Sample Size | 256 | 256 |
 
---
 
## System Architecture
 
```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Spark   │    │    R     │    │     Backend      │  │
│  │  Job     │───▶│Analysis  │───▶│   (FastAPI)      │  │
│  │(pandas)  │    │(ggplot2) │    │  IForest + PCA   │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│  exits ✓         exits ✓                  │             │
│                                           ▼             │
│                                  ┌──────────────────┐  │
│                                  │    Frontend      │  │
│                                  │  (React + nginx) │  │
│                                  │  localhost:3000  │  │
│                                  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```
 
**Startup Order (enforced by `depends_on`):**
1. `energy_spark` — processes raw data, generates CSVs → exits ✓
2. `energy_r` — generates 6-panel statistical plot → exits ✓
3. `energy_backend` — loads data, trains IForest+PCA, starts API
4. `energy_frontend` — serves React dashboard via nginx
---
 
## Tech Stack
 
| Technology | Role | Paper Reference |
|---|---|---|
| **Isolation Forest** | Core anomaly detection | Paper §II |
| **PCA** | Feature dimensionality reduction | Paper §III-B, Table II |
| **Trend features D & R** | Theft/fault detection via sliding window | Paper §III-A-3 |
| **pandas** | Large-scale data processing (Spark replacement) | Paper §IV-B |
| **R + ggplot2** | Statistical analysis & 6-panel visualization | Paper Fig. 2, 3 |
| **TinyML** (software) | Edge inference simulation | IoT extension |
| **FastAPI** | REST API backend | — |
| **React** | Dark industrial dashboard | — |
| **Docker Compose** | Full-stack container orchestration | Reproducibility |
 
---
 
## Project Structure
 
```
MAJOR PROJECT/
├── backend/
│   ├── Dockerfile
│   ├── main.py              # FastAPI + IForest + PCA (paper §II, §III-B)
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   ├── src/App.js           # Dark dashboard, 5 tabs
│   └── package.json
├── ml_models/
│   ├── anomaly_detection.py # Full paper pipeline + 6-panel matplotlib
│   ├── data_preprocessing.py# Paper §III-A-1 filters
│   ├── forecasting.py       # Rolling mean baseline
│   └── spark_processing/
│       └── spark_job.py     # Weekday buckets + trend features
├── data/
│   ├── analysis.R           # 6-panel dark-themed R plots
│   ├── dataset.txt          # Raw power data (not in git)
│   └── hourly_data.csv      # Pre-processed hourly data
├── docker/
│   └── docker-compose.yml   # Full stack orchestration
└── README.md
```
 
---
 
## Quick Start
 
### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `dataset.txt` placed in the `data/` folder
### Run (3 commands)
 
```bash
git clone https://github.com/YOUR_USERNAME/smart-energy-optimization.git
cd smart-energy-optimization/docker
docker-compose up --build
```
 
### Access
 
| Service | URL |
|---|---|
| 🖥️ Dashboard | http://localhost:3000 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |
| 🔍 API Health | http://localhost:8000/ |
 
### First run note
The R container installs packages from source (~10-15 min first time). After it completes, save state to avoid reinstalling:
 
```bash
docker commit energy_r energy_r_cached
# Then change r-base:4.3.2 → image: energy_r_cached in docker-compose.yml
```
 
---
 
## API Reference
 
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info and endpoint list |
| `/stats` | GET | Mean, max, min, std, anomaly rate |
| `/prediction` | GET | Rolling 24h forecast with RMSE |
| `/anomalies` | GET | Anomaly points with isolation scores |
| `/insights` | GET | AI-generated pattern insights |
| `/tinyml` | GET | TinyML inference on latest reading |
| `/spark-stats` | GET | Processed data statistics |
| `/analyze?power=3.5` | GET | Single-point z-score + TinyML analysis |
| `/analyze-rooms` | POST | Multi-room analysis |
| `/pca-info` | GET | PCA variance breakdown (paper §III-B) |
| `/model-info` | GET | IForest model metadata (paper §II) |
| `/trend-features` | GET | D and R trend indices (paper §III-A-3) |
 
### Example
 
```bash
# Analyze a single power reading
curl "http://localhost:8000/analyze?power=3.5"
 
# Response
{
  "power": 3.5,
  "z_score": 2.145,
  "decision": {
    "label": "HIGH",
    "emoji": "⚡",
    "confidence": 67,
    "action": "Reduce load — turn off AC / heater",
    "priority": 2
  },
  "anomaly": true,
  "mean_baseline": 1.086
}
```
 
---
 
## Results
 
```
Dataset         : UCI Household Power Consumption
Records         : 34,589
Anomalies       : 360 (1.04%)
PCA Variance    : PC1=59.5%, PC2=16.2%, PC3=7.0%, PC4=4.9%, PC5=4.0%
Total (5 PCs)   : 91.6%
Paper Accuracy  : 73.42% → This system: 75.7% (PC1+PC2 combined variance)
```
 
### Dashboard Features
- **5 Tabs**: Dashboard · Anomalies · Room Analysis · Model Info · R Analysis
- **KPI Row**: Mean power, peak, min, std deviation, anomaly rate, total records
- **TinyML Banner**: Real-time inference with confidence score and action
- **Forecast Chart**: Actual vs rolling 24h prediction with RMSE
- **AI Insights**: Pattern detection alerts from paper §IV
- **Anomaly Doughnut**: Normal vs anomaly ratio visualization
- **R Analysis Tab**: 6-panel statistical plot (histogram, time series, boxplot, rolling mean, KDE, summary stats)
---
 
## Team
 
**Nandini Nautiyal, Ainka Jalan, Aindrila Kundu**
B.Tech — NSUT, Semester 6
Hardware & Software Workshop — Major Project
 
---
 
## References
 
1. Wei Mao et al., *"Anomaly Detection for Power Consumption Data based on Isolated Forest"*, POWERCON 2018
2. Liu F T, Ting K M, Zhou Z H. *Isolation Forest*, ICDM 2008
3. Wold S, Esbensen K, Geladi P. *Principal Component Analysis*, 1987
4. UCI Machine Learning Repository — Individual Household Electric Power Consumption Dataset
---
