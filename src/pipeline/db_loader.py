"""
db_loader.py — Loads processed flight data into PostgreSQL or SQLite fallback.
"""

import sys
import json
import pandas as pd
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    POSTGRES_URI, SQLITE_PATH, USE_SQLITE_FALLBACK,
    PROCESSED_CSV, PROCESSED_DIR
)

# ─── Table DDL ─────────────────────────────────────────────────────────────────
DDL_FLIGHTS_RAW = """
CREATE TABLE IF NOT EXISTS flights_raw (
    flight_id          TEXT PRIMARY KEY,
    airline            TEXT,
    flight_number      TEXT,
    origin             TEXT,
    destination        TEXT,
    scheduled_dep      TIMESTAMP,
    distance_km        REAL,
    sched_duration_min REAL,
    temperature_c      REAL,
    wind_speed_kmh     REAL,
    visibility_km      REAL,
    precipitation_mm   REAL,
    humidity_pct       REAL,
    weather_code       INTEGER,
    weather_label      TEXT,
    congestion_index   REAL,
    runway_occupancy   REAL,
    gate_delay_min     REAL,
    taxiing_time_min   REAL,
    delay_minutes      REAL,
    is_delayed         INTEGER
);
"""

DDL_FLIGHTS_PROCESSED = """
CREATE TABLE IF NOT EXISTS flights_processed (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_id             TEXT,
    dep_hour              REAL,
    dep_day_of_week       REAL,
    dep_month             REAL,
    is_weekend            REAL,
    is_holiday            REAL,
    distance_km           REAL,
    airline_encoded       REAL,
    origin_encoded        REAL,
    dest_encoded          REAL,
    temperature_c         REAL,
    wind_speed_kmh        REAL,
    visibility_km         REAL,
    precipitation_mm      REAL,
    weather_severity_score REAL,
    congestion_index      REAL,
    congestion_level      REAL,
    route_avg_delay       REAL,
    carrier_avg_delay     REAL,
    delay_minutes         REAL,
    is_delayed            INTEGER
);
"""

DDL_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    flight_id       TEXT,
    airline         TEXT,
    origin          TEXT,
    destination     TEXT,
    predicted_delay INTEGER,
    delay_prob      REAL,
    model_version   TEXT
);
"""

DDL_MODEL_METRICS = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name  TEXT,
    accuracy    REAL,
    precision_s REAL,
    recall      REAL,
    f1_score    REAL,
    roc_auc     REAL,
    is_best     INTEGER
);
"""


class DBLoader:
    """
    Handles all database operations.
    Auto-detects whether to use PostgreSQL or SQLite.
    """

    def __init__(self):
        self.use_sqlite = USE_SQLITE_FALLBACK
        self.conn       = None
        self._connect()

    def _connect(self):
        if not self.use_sqlite:
            try:
                import psycopg2
                self.conn = psycopg2.connect(POSTGRES_URI)
                print("[DBLoader] Connected to PostgreSQL.")
                return
            except Exception as e:
                print(f"[DBLoader] PostgreSQL unavailable ({e}). Falling back to SQLite.")
                self.use_sqlite = True

        self.conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        print(f"[DBLoader] Connected to SQLite → {SQLITE_PATH}")

    def _execute(self, sql: str):
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def create_tables(self):
        """Create all tables if they don't exist."""
        for ddl in [DDL_FLIGHTS_RAW, DDL_FLIGHTS_PROCESSED,
                    DDL_PREDICTIONS, DDL_MODEL_METRICS]:
            self._execute(ddl)
        print("[DBLoader] All tables created / verified.")

    # ── Bulk load raw CSV into flights_raw ──────────────────────────────────
    def load_raw(self, df: pd.DataFrame):
        raw_cols = [
            "flight_id", "airline", "flight_number", "origin", "destination",
            "scheduled_dep", "distance_km", "sched_duration_min",
            "temperature_c", "wind_speed_kmh", "visibility_km",
            "precipitation_mm", "humidity_pct", "weather_code", "weather_label",
            "congestion_index", "runway_occupancy", "gate_delay_min",
            "taxiing_time_min", "delay_minutes", "is_delayed"
        ]
        subset = df[[c for c in raw_cols if c in df.columns]].copy()
        subset["scheduled_dep"] = subset["scheduled_dep"].astype(str)

        if self.use_sqlite:
            subset.to_sql("flights_raw", self.conn, if_exists="replace",
                          index=False, chunksize=5000)
        else:
            # PostgreSQL: use executemany for transactional safety
            cur = self.conn.cursor()
            cols   = subset.columns.tolist()
            ph     = ",".join(["%s"] * len(cols))
            sql    = f"INSERT INTO flights_raw ({','.join(cols)}) VALUES ({ph}) ON CONFLICT DO NOTHING"
            data   = [tuple(r) for r in subset.itertuples(index=False)]
            cur.executemany(sql, data)
            self.conn.commit()

        print(f"[DBLoader] Loaded {len(subset):,} rows → flights_raw")

    # ── Load processed features ──────────────────────────────────────────────
    def load_processed(self, df: pd.DataFrame):
        proc_cols = [
            "flight_id", "dep_hour", "dep_day_of_week", "dep_month",
            "is_weekend", "is_holiday", "distance_km", "airline_encoded",
            "origin_encoded", "dest_encoded", "temperature_c", "wind_speed_kmh",
            "visibility_km", "precipitation_mm", "weather_severity_score",
            "congestion_index", "congestion_level", "route_avg_delay",
            "carrier_avg_delay", "delay_minutes", "is_delayed"
        ]
        subset = df[[c for c in proc_cols if c in df.columns]].copy()
        if self.use_sqlite:
            subset.to_sql("flights_processed", self.conn, if_exists="replace",
                          index=False, chunksize=5000)
        print(f"[DBLoader] Loaded {len(subset):,} rows → flights_processed")

    # ── Save prediction ──────────────────────────────────────────────────────
    def save_prediction(self, flight_id: str, airline: str, origin: str,
                        dest: str, predicted: int, prob: float,
                        model_version: str = "v1"):
        sql = """
        INSERT INTO model_predictions
            (flight_id, airline, origin, destination, predicted_delay, delay_prob, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """ if self.use_sqlite else """
        INSERT INTO model_predictions
            (flight_id, airline, origin, destination, predicted_delay, delay_prob, model_version)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """
        cur = self.conn.cursor()
        cur.execute(sql, (flight_id, airline, origin, dest,
                          int(predicted), float(prob), model_version))
        self.conn.commit()

    # ── Save model metrics ───────────────────────────────────────────────────
    def save_metrics(self, metrics: dict):
        for name, m in metrics.items():
            sql = """
            INSERT INTO model_metrics
                (model_name, accuracy, precision_s, recall, f1_score, roc_auc, is_best)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """ if self.use_sqlite else """
            INSERT INTO model_metrics
                (model_name, accuracy, precision_s, recall, f1_score, roc_auc, is_best)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """
            cur = self.conn.cursor()
            cur.execute(sql, (
                name,
                m.get("accuracy",  0),
                m.get("precision", 0),
                m.get("recall",    0),
                m.get("f1",        0),
                m.get("roc_auc",   0),
                int(m.get("is_best", False))
            ))
            self.conn.commit()
        print("[DBLoader] Model metrics saved.")

    # ── Query helpers ────────────────────────────────────────────────────────
    def query_delay_stats(self) -> dict:
        """Aggregate delay statistics for the API and dashboard."""
        cur = self.conn.cursor()
        queries = {
            "total_flights":   "SELECT COUNT(*) FROM flights_raw",
            "total_delayed":   "SELECT COUNT(*) FROM flights_raw WHERE is_delayed=1",
            "avg_delay_min":   "SELECT AVG(delay_minutes) FROM flights_raw WHERE is_delayed=1",
            "by_airline": """
                SELECT airline, COUNT(*) as total,
                       SUM(is_delayed) as delayed,
                       ROUND(AVG(delay_minutes),2) as avg_delay
                FROM flights_raw GROUP BY airline ORDER BY delayed DESC
            """,
            "by_hour": """
                SELECT CAST(strftime('%H', scheduled_dep) AS INTEGER) as hour,
                       COUNT(*) as total, SUM(is_delayed) as delayed
                FROM flights_raw GROUP BY hour ORDER BY hour
            """ if self.use_sqlite else """
                SELECT EXTRACT(HOUR FROM scheduled_dep)::INT as hour,
                       COUNT(*) as total, SUM(is_delayed) as delayed
                FROM flights_raw GROUP BY hour ORDER BY hour
            """,
        }

        stats = {}
        for key, sql in queries.items():
            try:
                cur.execute(sql)
                if key in ["total_flights", "total_delayed", "avg_delay_min"]:
                    stats[key] = cur.fetchone()[0]
                else:
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    stats[key] = [dict(zip(cols, r)) for r in rows]
            except Exception as e:
                stats[key] = f"Error: {e}"

        stats["delay_rate"] = round(
            (stats.get("total_delayed", 0) or 0) /
            max(stats.get("total_flights", 1), 1) * 100, 2
        )
        return stats

    def close(self):
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    import pandas as pd
    db = DBLoader()
    db.create_tables()
    df = pd.read_csv(PROCESSED_CSV)
    db.load_processed(df)
    stats = db.query_delay_stats()
    print(json.dumps({k: v for k, v in stats.items()
                      if not isinstance(v, list)}, indent=2))
    db.close()
