"""
app.py — Flask REST API for Flight Delay Prediction System
Endpoints:
  GET  /health          → health check
  POST /predict         → single prediction
  POST /batch_predict   → CSV batch prediction
  GET  /model_info      → model metadata + performance
  GET  /stats           → aggregate delay statistics from DB
"""

import sys
import json
import io
import traceback
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import API_HOST, API_PORT, API_DEBUG
from src.models.predictor import get_predictor
from src.pipeline.db_loader import DBLoader

app   = Flask(__name__)
CORS(app)

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_db: DBLoader = None

def get_db() -> DBLoader:
    global _db
    if _db is None:
        _db = DBLoader()
        _db.create_tables()
    return _db


# ─── Helpers ──────────────────────────────────────────────────────────────────
def error_response(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code

def ok_response(data: dict):
    return jsonify({"status": "ok", **data}), 200


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return ok_response({
        "service": "Flight Delay Prediction API",
        "version": "1.0.0",
        "model_loaded": True,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction endpoint.

    Request JSON:
        {
            "airline":         "IndiGo",
            "origin":          "DEL",
            "destination":     "BOM",
            "scheduled_dep":   "2024-07-15T18:30:00",
            "distance_km":     1150,
            "temperature_c":   34.0,
            "wind_speed_kmh":  45.0,
            "visibility_km":   4.0,
            "precipitation_mm": 22.0,
            "weather_code":    4,
            "congestion_index": 0.80
        }

    Response JSON:
        {
            "status": "ok",
            "is_delayed": 1,
            "delay_prob": 0.8432,
            "delay_prob_pct": 84.32,
            "risk_level": "🔴 High Risk",
            "estimated_delay_min": 75.9,
            "model_name": "XGBoost",
            ...
        }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return error_response("Request body must be valid JSON.")

        predictor = get_predictor()
        result    = predictor.predict(data)

        # Optionally persist prediction
        try:
            get_db().save_prediction(
                flight_id     = data.get("flight_id", "MANUAL"),
                airline       = data.get("airline",   "Unknown"),
                origin        = data.get("origin",    "???"),
                dest          = data.get("destination","???"),
                predicted     = result["is_delayed"],
                prob          = result["delay_prob"],
                model_version = result["model_name"],
            )
        except Exception:
            pass   # non-fatal

        return ok_response(result)

    except FileNotFoundError:
        return error_response(
            "Model artifacts not found. Please run the training pipeline first.", 503)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch prediction from uploaded CSV.
    CSV must contain same columns as /predict body.
    Returns predictions as JSON array.
    """
    try:
        if "file" not in request.files:
            return error_response("No file uploaded. Use key 'file' with a CSV.")

        f  = request.files["file"]
        df = pd.read_csv(f)

        if df.empty:
            return error_response("CSV is empty.")

        predictor = get_predictor()
        results   = []
        for _, row in df.iterrows():
            try:
                r = predictor.predict(row.to_dict())
                results.append({**row.to_dict(), **r})
            except Exception as e:
                results.append({"error": str(e), **row.to_dict()})

        return ok_response({
            "total":    len(results),
            "delayed":  sum(1 for r in results if r.get("is_delayed") == 1),
            "results":  results,
        })

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.route("/model_info", methods=["GET"])
def model_info():
    """Returns model metadata, feature list, and cross-validated performance."""
    try:
        predictor = get_predictor()
        info      = predictor.model_info()
        return ok_response(info)
    except FileNotFoundError:
        return error_response("Model not trained yet.", 503)
    except Exception as e:
        return error_response(str(e), 500)


@app.route("/stats", methods=["GET"])
def stats():
    """Returns aggregate delay statistics from the database."""
    try:
        db    = get_db()
        data  = db.query_delay_stats()
        return ok_response({"statistics": data})
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[API] Starting Flask server on {API_HOST}:{API_PORT}...")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
