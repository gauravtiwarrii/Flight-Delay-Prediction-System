"""
predictor.py — Real-time inference wrapper.
Used by Flask API and Streamlit dashboard.
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    BEST_MODEL_PATH, SCALER_PATH, ENCODER_PATH,
    METRICS_PATH, FEATURE_COLS, WEATHER_CODES
)


class FlightDelayPredictor:
    """
    Loads the trained model, scaler, and encoders.
    Provides a predict() method suitable for API and dashboard use.

    Input (dict):
        airline, origin, destination, scheduled_dep (ISO str),
        distance_km, temperature_c, wind_speed_kmh, visibility_km,
        precipitation_mm, weather_code, congestion_index

    Output (dict):
        is_delayed (0|1), delay_prob (float), risk_level (str),
        estimated_delay_min (float), model_name (str)
    """

    def __init__(self):
        self.model_    = joblib.load(BEST_MODEL_PATH)
        self.scaler_   = joblib.load(SCALER_PATH)
        self.encoders_ = joblib.load(ENCODER_PATH)

        with open(METRICS_PATH) as f:
            self.metrics_ = json.load(f)

        self.model_name_    = self.metrics_.get("_meta", {}).get("best_model", "Unknown")
        self.feature_names_ = self.metrics_.get("_meta", {}).get("features", FEATURE_COLS)

    # ── Feature Assembly ──────────────────────────────────────────────────────
    def _build_features(self, inp: dict) -> np.ndarray:
        dt = pd.to_datetime(inp.get("scheduled_dep", datetime.now().isoformat()))

        dep_hour       = dt.hour
        dep_dow        = dt.dayofweek
        dep_month      = dt.month
        is_weekend     = int(dep_dow >= 5)
        is_holiday     = 0    # simplified for inference
        distance_km    = float(inp.get("distance_km", 800))

        # Label encode categoricals
        def safe_encode(col, val):
            le = self.encoders_.get(col)
            if le is None:
                return 0
            val_str = str(val).upper()
            known   = set(le.classes_)
            return int(le.transform([val_str])[0]) if val_str in known else 0

        airline_enc  = safe_encode("airline",     inp.get("airline",     "IndiGo"))
        origin_enc   = safe_encode("origin",      inp.get("origin",      "DEL"))
        dest_enc     = safe_encode("destination", inp.get("destination", "BOM"))

        temperature_c     = float(inp.get("temperature_c",    28.0))
        wind_speed_kmh    = float(inp.get("wind_speed_kmh",   15.0))
        visibility_km     = float(inp.get("visibility_km",    10.0))
        precipitation_mm  = float(inp.get("precipitation_mm",  0.0))
        weather_code      = int(inp.get("weather_code",         0))
        congestion_index  = float(inp.get("congestion_index", 0.40))

        # Weather severity
        weather_severity  = (
            (precipitation_mm / 80)    * 0.35 +
            (wind_speed_kmh   / 120)   * 0.30 +
            (1 - visibility_km / 15)   * 0.25 +
            (weather_code / 7)         * 0.10
        )

        # Congestion level
        congestion_level = (
            0 if congestion_index <= 0.33 else
            1 if congestion_index <= 0.66 else 2
        )

        # Use 0 for aggregations at inference time (no historical context)
        route_avg_delay   = float(inp.get("route_avg_delay",   0.0))
        carrier_avg_delay = float(inp.get("carrier_avg_delay", 0.0))

        raw_vector = {
            "dep_hour":                dep_hour,
            "dep_day_of_week":         dep_dow,
            "dep_month":               dep_month,
            "is_weekend":              is_weekend,
            "is_holiday":              is_holiday,
            "distance_km":             distance_km,
            "airline_encoded":         airline_enc,
            "origin_encoded":          origin_enc,
            "dest_encoded":            dest_enc,
            "temperature_c":           temperature_c,
            "wind_speed_kmh":          wind_speed_kmh,
            "visibility_km":           visibility_km,
            "precipitation_mm":        precipitation_mm,
            "weather_severity_score":  weather_severity,
            "congestion_index":        congestion_index,
            "congestion_level":        congestion_level,
            "route_avg_delay":         route_avg_delay,
            "carrier_avg_delay":       carrier_avg_delay,
        }

        # Order exactly as FEATURE_COLS / feature_names_
        feature_vec = np.array(
            [raw_vector.get(f, 0.0) for f in self.feature_names_],
            dtype=float
        ).reshape(1, -1)

        # Scale
        feature_vec = self.scaler_.transform(feature_vec)
        return feature_vec

    # ── Public Predict ────────────────────────────────────────────────────────
    def predict(self, inp: dict) -> dict:
        """
        Args:
            inp (dict): flight input parameters
        Returns:
            dict with prediction results
        """
        X = self._build_features(inp)

        pred  = int(self.model_.predict(X)[0])
        prob  = float(self.model_.predict_proba(X)[0][1])

        # Estimate delay minutes from probability (rough heuristic)
        est_delay = round(prob * 90, 1) if pred == 1 else round(prob * 10, 1)

        risk = (
            "🔴 High Risk"   if prob >= 0.70 else
            "🟠 Medium Risk" if prob >= 0.40 else
            "🟢 Low Risk"
        )

        return {
            "is_delayed":          pred,
            "delay_prob":          round(prob, 4),
            "delay_prob_pct":      round(prob * 100, 2),
            "risk_level":          risk,
            "estimated_delay_min": est_delay,
            "model_name":          self.model_name_,
            "input_summary": {
                "airline":     inp.get("airline"),
                "route":       f"{inp.get('origin')} → {inp.get('destination')}",
                "dep_time":    inp.get("scheduled_dep"),
                "weather":     WEATHER_CODES.get(int(inp.get("weather_code", 0)), "Clear"),
                "congestion":  f"{float(inp.get('congestion_index', 0.4)):.0%}",
            }
        }

    def model_info(self) -> dict:
        meta = self.metrics_.get("_meta", {})
        models_perf = {
            k: v for k, v in self.metrics_.items()
            if not k.startswith("_")
        }
        return {
            "best_model":    meta.get("best_model"),
            "n_samples":     meta.get("n_samples"),
            "delay_rate":    meta.get("delay_rate"),
            "feature_count": len(meta.get("features", [])),
            "features":      meta.get("features", []),
            "performance":   models_perf,
        }


# ── Singleton accessor ────────────────────────────────────────────────────────
_predictor_instance = None

def get_predictor() -> FlightDelayPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FlightDelayPredictor()
    return _predictor_instance


if __name__ == "__main__":
    p = get_predictor()
    result = p.predict({
        "airline":        "IndiGo",
        "origin":         "DEL",
        "destination":    "BOM",
        "scheduled_dep":  "2024-07-15T18:30:00",
        "distance_km":    1150,
        "temperature_c":  34.0,
        "wind_speed_kmh": 45.0,
        "visibility_km":  4.0,
        "precipitation_mm": 22.0,
        "weather_code":   4,
        "congestion_index": 0.80,
    })
    import json
    print(json.dumps(result, indent=2))
