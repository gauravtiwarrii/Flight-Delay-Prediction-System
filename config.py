"""
config.py — Central configuration for Flight Delay Prediction System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if missing
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Database Configuration ────────────────────────────────────────────────────
# PostgreSQL (primary). Set these in a .env file or environment variables.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB   = os.getenv("POSTGRES_DB",   "flight_delay")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.getenv("POSTGRES_PASS", "password")

POSTGRES_URI = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# SQLite fallback — used when Postgres is unavailable
SQLITE_PATH = DATA_DIR / "flight_delay.db"
USE_SQLITE_FALLBACK = os.getenv("USE_SQLITE", "true").lower() == "true"

# ─── Dataset Configuration ─────────────────────────────────────────────────────
RAW_FLIGHTS_CSV    = RAW_DIR / "flights_raw.csv"
PROCESSED_CSV      = PROCESSED_DIR / "flights_processed.csv"
N_SYNTHETIC_ROWS   = 100_000          # synthetic dataset size
RANDOM_SEED        = 42

# ─── Model Configuration ───────────────────────────────────────────────────────
TARGET_BINARY      = "is_delayed"     # 0/1 classification target
TARGET_REGRESSION  = "delay_minutes"  # regression target (optional)

FEATURE_COLS = [
    "dep_hour", "dep_day_of_week", "dep_month", "is_weekend", "is_holiday",
    "distance_km", "airline_encoded", "origin_encoded", "dest_encoded",
    "temperature_c", "wind_speed_kmh", "visibility_km", "precipitation_mm",
    "weather_severity_score", "congestion_index", "congestion_level",
    "route_avg_delay", "carrier_avg_delay",
]

TRAIN_TEST_SPLIT   = 0.20             # 80/20 split
CV_FOLDS           = 5

# Model artifact paths
BEST_MODEL_PATH    = MODELS_DIR / "best_model.joblib"
SCALER_PATH        = MODELS_DIR / "scaler.joblib"
ENCODER_PATH       = MODELS_DIR / "encoders.joblib"
METRICS_PATH       = MODELS_DIR / "metrics.json"
FEATURE_IMP_PLOT   = MODELS_DIR / "feature_importance.png"
CONFUSION_PLOT     = MODELS_DIR / "confusion_matrix.png"
ROC_PLOT           = MODELS_DIR / "roc_curve.png"

# ─── Flask API Configuration ───────────────────────────────────────────────────
API_HOST  = "0.0.0.0"
API_PORT  = 5000
API_DEBUG = False

# ─── Streamlit Configuration ───────────────────────────────────────────────────
STREAMLIT_TITLE = "✈️ Flight Delay Prediction System"

# ─── Domain Constants ──────────────────────────────────────────────────────────
AIRLINES = [
    "IndiGo", "Air India", "SpiceJet", "Vistara",
    "GoFirst", "AirAsia India", "Alliance Air", "Star Air"
]

AIRPORTS = [
    "DEL", "BOM", "BLR", "MAA", "CCU", "HYD",
    "COK", "PNQ", "AMD", "GOI", "JAI", "LKO"
]

WEATHER_CODES = {
    0: "Clear", 1: "Partly Cloudy", 2: "Overcast",
    3: "Drizzle", 4: "Rain", 5: "Thunderstorm",
    6: "Fog", 7: "Haze"
}
