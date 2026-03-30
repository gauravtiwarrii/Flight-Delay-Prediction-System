"""
feature_engineer.py — Feature engineering and transformation pipeline
Creates temporal, aggregated, encoded, and composite features.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FEATURE_COLS, TARGET_BINARY, PROCESSED_CSV,
    SCALER_PATH, ENCODER_PATH, RANDOM_SEED
)

INDIAN_HOLIDAYS_2023_2024 = [
    "2023-01-26", "2023-03-08", "2023-04-07", "2023-04-14",
    "2023-08-15", "2023-10-02", "2023-10-24", "2023-11-14",
    "2023-12-25", "2024-01-26", "2024-03-25", "2024-04-14",
    "2024-08-15", "2024-10-02", "2024-10-31", "2024-12-25",
]
HOLIDAY_SET = set(pd.to_datetime(INDIAN_HOLIDAYS_2023_2024).date)


class FeatureEngineer:
    """
    Transforms cleaned flight data into ML-ready features.
    Handles encoding, scaling, and feature creation consistently
    across train and inference time.
    """

    def __init__(self):
        self.label_encoders_: dict = {}
        self.scaler_: StandardScaler = StandardScaler()
        self.fitted_: bool = False

    # ── 1. Temporal Features ──────────────────────────────────────────────────
    def _add_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df["scheduled_dep"])
        df["dep_hour"]       = dt.dt.hour
        df["dep_day_of_week"]= dt.dt.dayofweek    # Monday=0, Sunday=6
        df["dep_month"]      = dt.dt.month
        df["dep_quarter"]    = dt.dt.quarter
        df["is_weekend"]     = (df["dep_day_of_week"] >= 5).astype(int)
        df["is_holiday"]     = dt.dt.date.map(lambda d: int(d in HOLIDAY_SET))

        # Sine/Cosine encoding for cyclical features
        df["dep_hour_sin"]   = np.sin(2 * np.pi * df["dep_hour"]        / 24)
        df["dep_hour_cos"]   = np.cos(2 * np.pi * df["dep_hour"]        / 24)
        df["dep_month_sin"]  = np.sin(2 * np.pi * df["dep_month"]       / 12)
        df["dep_month_cos"]  = np.cos(2 * np.pi * df["dep_month"]       / 12)
        df["dep_dow_sin"]    = np.sin(2 * np.pi * df["dep_day_of_week"] / 7)
        df["dep_dow_cos"]    = np.cos(2 * np.pi * df["dep_day_of_week"] / 7)
        return df

    # ── 2. Weather Severity Score ─────────────────────────────────────────────
    def _add_weather_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalise each component to [0, 1] and weight
        precip_norm = (df["precipitation_mm"].clip(0, 80) / 80) * 0.35
        wind_norm   = (df["wind_speed_kmh"].clip(0, 120)  / 120)* 0.30
        vis_norm    = (1 - df["visibility_km"].clip(0, 15) / 15) * 0.25   # inverted
        wc_norm     = (df["weather_code"] / 7) * 0.10

        df["weather_severity_score"] = (
            precip_norm + wind_norm + vis_norm + wc_norm
        ).round(4)
        return df

    # ── 3. Congestion Level (Bucketed) ────────────────────────────────────────
    def _add_congestion_level(self, df: pd.DataFrame) -> pd.DataFrame:
        df["congestion_level"] = pd.cut(
            df["congestion_index"],
            bins=[0, 0.33, 0.66, 1.01],
            labels=[0, 1, 2],     # low, medium, high
            include_lowest=True
        ).astype(int)
        return df

    # ── 4. Historical Aggregations (route & carrier) ──────────────────────────
    def _add_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute historical mean delays per route and carrier."""
        df["route"] = df["origin"] + "_" + df["destination"]

        if "delay_minutes" in df.columns:
            route_avg  = df.groupby("route")["delay_minutes"].mean().rename("route_avg_delay")
            carrier_avg= df.groupby("airline")["delay_minutes"].mean().rename("carrier_avg_delay")
            df = df.join(route_avg, on="route")
            df = df.join(carrier_avg, on="airline")
        else:
            df["route_avg_delay"]   = 0.0
            df["carrier_avg_delay"] = 0.0

        return df

    # ── 5. Label Encoding ─────────────────────────────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_map = {
            "airline":     "airline_encoded",
            "origin":      "origin_encoded",
            "destination": "dest_encoded",
        }
        for raw_col, enc_col in cat_map.items():
            if raw_col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[enc_col] = le.fit_transform(df[raw_col].astype(str))
                self.label_encoders_[raw_col] = le
            else:
                le = self.label_encoders_[raw_col]
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[enc_col] = df[raw_col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
        return df

    # ── 6. Scale Numeric Features ─────────────────────────────────────────────
    def _scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        scale_cols = [c for c in FEATURE_COLS if c in df.columns]
        if fit:
            df[scale_cols] = self.scaler_.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler_.transform(df[scale_cols])
        return df

    # ── Public API ─────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[FeatureEngineer] Running fit_transform...")
        df = df.copy()
        df = self._add_temporal(df)
        df = self._add_weather_severity(df)
        df = self._add_congestion_level(df)
        df = self._add_aggregations(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_features(df, fit=True)
        self.fitted_ = True
        print(f"[FeatureEngineer] Done. Shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "Call fit_transform first."
        df = df.copy()
        df = self._add_temporal(df)
        df = self._add_weather_severity(df)
        df = self._add_congestion_level(df)
        df = self._add_aggregations(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_features(df, fit=False)
        return df

    def save(self):
        joblib.dump(self.scaler_,        SCALER_PATH)
        joblib.dump(self.label_encoders_, ENCODER_PATH)
        print(f"[FeatureEngineer] Saved scaler → {SCALER_PATH}")
        print(f"[FeatureEngineer] Saved encoders → {ENCODER_PATH}")

    @classmethod
    def load(cls):
        fe = cls()
        fe.scaler_         = joblib.load(SCALER_PATH)
        fe.label_encoders_ = joblib.load(ENCODER_PATH)
        fe.fitted_         = True
        return fe


def engineer_features(df: pd.DataFrame, save_artifacts: bool = True) -> tuple:
    """Convenience wrapper: returns (processed_df, feature_engineer)."""
    fe = FeatureEngineer()
    processed = fe.fit_transform(df)
    if save_artifacts:
        processed.to_csv(PROCESSED_CSV, index=False)
        fe.save()
        print(f"[FeatureEngineer] Processed dataset → {PROCESSED_CSV}")
    return processed, fe


if __name__ == "__main__":
    from data_cleaner import load_and_clean
    df_clean = load_and_clean()
    df_feat, fe = engineer_features(df_clean)
    print(df_feat[["dep_hour", "weather_severity_score", "congestion_level",
                   "route_avg_delay", "is_delayed"]].head())
