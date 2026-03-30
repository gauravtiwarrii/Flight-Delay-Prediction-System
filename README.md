# Flight Delay Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

**A production-grade, end-to-end Machine Learning system for predicting flight delays.**  
Integrates flight schedules, weather data, and airport traffic for real-time delay probability estimation.

[🌐 Live Demo](https://gauravtiwarrii.github.io/Flight-Delay-Prediction-System) · [📖 Docs](#-quick-start) · [🔌 API](#-api-reference)

</div>

---

## 📌 Overview

The **Flight Delay Prediction System** is a full ML pipeline that:
- Ingests and processes **100,000+ synthetic flight records** combining flight schedules, weather conditions, and airport congestion metrics
- Trains **3 machine learning models** (Logistic Regression, Random Forest, XGBoost) with 5-fold cross-validation
- Exposes predictions through a **Flask REST API** with 5 endpoints
- Provides an **interactive web dashboard** (Streamlit + standalone HTML GUI)
- Stores all data and predictions in **PostgreSQL** (with automatic SQLite fallback)

> 🏆 Best Model: **XGBoost** — F1: 0.583 | ROC-AUC: 0.710

---

## 🏗️ System Architecture

```
flight-delay/
│
├── src/
│   ├── pipeline/
│   │   ├── data_generator.py     # Synthetic dataset generation (100K records)
│   │   ├── data_cleaner.py       # IQR clipping, imputation, deduplication
│   │   ├── feature_engineer.py   # Feature creation, encoding, scaling
│   │   └── db_loader.py          # PostgreSQL / SQLite ingestion
│   │
│   ├── models/
│   │   ├── trainer.py            # Multi-model training with CV
│   │   ├── evaluator.py          # Plots: confusion matrix, ROC, feature importance
│   │   └── predictor.py          # Real-time inference wrapper
│   │
│   ├── api/
│   │   └── app.py                # Flask REST API (5 endpoints)
│   │
│   └── dashboard/
│       └── streamlit_app.py      # 4-tab Streamlit analytics dashboard
│
├── gui/
│   └── index.html                # Standalone web GUI (zero dependencies)
│
├── docs/                         # GitHub Pages (live demo)
├── models/                       # Saved model artifacts (.joblib, .json, .png)
├── data/                         # Raw & processed datasets + SQLite DB
├── config.py                     # Central configuration
├── run_pipeline.py               # End-to-end orchestrator
└── requirements.txt
```

---

## ✨ Features

### 🔄 Data Pipeline
| Step | Description |
|---|---|
| **Generation** | 100,000 synthetic records — flight schedule + weather + airport traffic |
| **Cleaning** | Median/mode imputation · IQR outlier clipping · duplicate removal |
| **Engineering** | 18 features: cyclical encoding, composite scores, historical aggregations |
| **Storage** | Bulk-insert to PostgreSQL (`flights_raw`, `flights_processed`, `model_predictions`, `model_metrics`) |

### 🛠️ Engineered Features
| Feature | Type | Description |
|---|---|---|
| `dep_hour_sin/cos` | Temporal | Cyclical encoding of departure hour |
| `dep_month_sin/cos` | Temporal | Cyclical encoding of departure month |
| `is_weekend` / `is_holiday` | Temporal | Indian public holiday calendar |
| `weather_severity_score` | Composite | Weighted: precipitation (35%) + wind (30%) + visibility (25%) + code (10%) |
| `congestion_level` | Bucketed | Low / Medium / High from continuous congestion index |
| `route_avg_delay` | Aggregated | Historical mean delay per origin→destination route |
| `carrier_avg_delay` | Aggregated | Historical mean delay per airline |

### 🤖 Model Performance (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6515 | 0.5371 | 0.5210 | 0.5814 | 0.6936 |
| Random Forest | 0.6520 | 0.5380 | 0.5190 | 0.5677 | 0.7093 |
| **XGBoost ⭐** | **0.6800** | **0.5863** | **0.5520** | **0.5830** | **0.7100** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- (Optional) PostgreSQL 14+

### 1. Clone the Repository
```bash
git clone https://github.com/gauravtiwarrii/Flight-Delay-Prediction-System.git
cd Flight-Delay-Prediction-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)
```bash
cp .env.example .env
# Edit .env if using PostgreSQL, otherwise SQLite is used automatically
```

### 4. Run the Full Pipeline
```bash
python run_pipeline.py
```
This executes all 6 steps:
```
Step 1: Data Generation    → 100,000 synthetic records
Step 2: Data Cleaning      → Imputation, outliers, duplicates
Step 3: Feature Engineering→ 18 ML-ready features
Step 4: Database Ingestion → 4 tables in SQLite/PostgreSQL
Step 5: Model Training     → LR + RandomForest + XGBoost (5-fold CV)
Step 6: Evaluation         → Confusion matrix, ROC curve, feature importance
```

**Re-run without regenerating data:**
```bash
python run_pipeline.py --skip-data --skip-train
```

### 5. Launch the Web GUI
Simply open in your browser — **no server required**:
```
gui/index.html
```
Or visit the [🌐 Live Demo](https://gauravtiwarrii.github.io/Flight-Delay-Prediction-System)

### 6. Launch the Flask API
```bash
python src/api/app.py
# → http://localhost:5000
```

### 7. Launch the Streamlit Dashboard
```bash
streamlit run src/dashboard/streamlit_app.py
# → http://localhost:8501
```

---

## 🔌 API Reference

Base URL: `http://localhost:5000`

### `GET /health`
Health check.
```json
{ "status": "ok", "service": "Flight Delay Prediction API", "version": "1.0.0" }
```

### `POST /predict`
Single flight prediction.

**Request:**
```json
{
  "airline":          "IndiGo",
  "origin":           "DEL",
  "destination":      "BOM",
  "scheduled_dep":    "2024-07-15T18:30:00",
  "distance_km":      1150,
  "temperature_c":    34.0,
  "wind_speed_kmh":   45.0,
  "visibility_km":    4.0,
  "precipitation_mm": 22.0,
  "weather_code":     4,
  "congestion_index": 0.80
}
```

**Response:**
```json
{
  "status":              "ok",
  "is_delayed":          1,
  "delay_prob":          0.7307,
  "delay_prob_pct":      73.07,
  "risk_level":          "🔴 High Risk",
  "estimated_delay_min": 65.8,
  "model_name":          "XGBoost"
}
```

### `POST /batch_predict`
Upload a CSV file for batch predictions.
```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "file=@flights.csv"
```

### `GET /model_info`
Returns model metadata and cross-validated performance metrics.

### `GET /stats`
Returns aggregate delay statistics from the database.

---

## 🗄️ Database Schema

```sql
-- Raw flight records
flights_raw (flight_id, airline, origin, destination,
             scheduled_dep, distance_km, temperature_c,
             wind_speed_kmh, visibility_km, precipitation_mm,
             congestion_index, delay_minutes, is_delayed)

-- Feature-engineered records for model training
flights_processed (flight_id, dep_hour, dep_day_of_week,
                   weather_severity_score, congestion_level,
                   route_avg_delay, is_delayed, ...)

-- Real-time predictions log
model_predictions (flight_id, airline, origin, destination,
                   predicted_delay, delay_prob, model_version, timestamp)

-- Model training metrics
model_metrics (model_name, accuracy, precision_s, recall,
               f1_score, roc_auc, is_best, run_date)
```

---

## 🌐 GUI Preview

The standalone GUI (`gui/index.html`) works directly in your browser with **zero dependencies**:

| Tab | What you see |
|---|---|
| 🔮 **Live Prediction** | Flight form + animated probability gauge + risk factor breakdown |
| 📊 **Analytics** | Delay by airline · by hour · monthly trend · by weather |
| 📈 **Model Performance** | 3-model comparison · feature importance · CV metrics table |

When the Flask API is running, predictions use the real trained model. Otherwise, the GUI enters **Demo Mode** with a client-side prediction engine.

---

## 📁 Generated Artifacts

After running `python run_pipeline.py`:

```
data/
├── raw/flights_raw.csv           # 100K raw records
├── processed/flights_processed.csv  # Feature-engineered
└── flight_delay.db               # SQLite DB (37 MB, 4 tables)

models/
├── best_model.joblib             # Trained XGBoost model
├── scaler.joblib                 # StandardScaler
├── encoders.joblib               # LabelEncoders
├── metrics.json                  # CV metrics for all models
├── confusion_matrix.png          # Evaluation plot
├── roc_curve.png                 # ROC-AUC curve
└── feature_importance.png        # Top 20 feature importances
```

---

## ⚙️ Configuration

Edit `config.py` or create a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `USE_SQLITE` | `true` | Use SQLite instead of PostgreSQL |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_DB` | `flight_delay` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASS` | `password` | Database password |
| `N_SYNTHETIC_ROWS` | `100000` | Rows to generate |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-Learn, XGBoost |
| API | Flask, Flask-CORS |
| Dashboard | Streamlit, Plotly |
| Web GUI | HTML5, CSS3, Vanilla JS, Chart.js |
| Database | PostgreSQL (psycopg2), SQLite |
| Serialization | Joblib |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## 👨‍💻 Author

**Gaurav Tiwari**  
[![GitHub](https://img.shields.io/badge/GitHub-gauravtiwarrii-181717?style=flat&logo=github)](https://github.com/gauravtiwarrii)

---

<div align="center">
<sub>Built with ❤️ · March 2025</sub>
</div>
