"""
data_cleaner.py — Data cleaning pipeline
Handles missing values, duplicates, outliers, and type coercion.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_FLIGHTS_CSV, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for flight delay dataset.
    All transformations are logged for auditability.
    """

    def __init__(self, verbose: bool = True):
        self.verbose   = verbose
        self.report_   = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[DataCleaner] {msg}")

    # ── 1. Remove Duplicates ──────────────────────────────────────────────────
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["flight_id"], keep="first")
        dropped = before - len(df)
        self.report_["duplicates_removed"] = dropped
        self._log(f"Duplicates removed: {dropped:,}")
        return df

    # ── 2. Handle Missing Values ──────────────────────────────────────────────
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        null_before = df.isnull().sum().sum()

        # Numeric → median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Categorical → mode imputation
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

        null_after = df.isnull().sum().sum()
        self.report_["nulls_imputed"] = int(null_before - null_after)
        self._log(f"Nulls imputed: {int(null_before - null_after):,}")
        return df

    # ── 3. Clip Outliers via IQR ──────────────────────────────────────────────
    def _clip_outliers(self, df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.DataFrame:
        total_clipped = 0
        for col in cols:
            if col not in df.columns:
                continue
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            lo  = q1 - factor * iqr
            hi  = q3 + factor * iqr
            clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            total_clipped += clipped
        self.report_["outliers_clipped"] = int(total_clipped)
        self._log(f"Outliers clipped (IQR×{factor}): {total_clipped:,}")
        return df

    # ── 4. Type Coercion & Validation ─────────────────────────────────────────
    def _enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure scheduled_dep is datetime
        if "scheduled_dep" in df.columns:
            df["scheduled_dep"] = pd.to_datetime(df["scheduled_dep"], errors="coerce")

        # Categorical string columns
        str_cols = ["airline", "origin", "destination", "flight_number",
                    "weather_label", "flight_id"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        # Boolean / binary
        if "is_delayed" in df.columns:
            df["is_delayed"] = df["is_delayed"].astype(int)

        self._log("Type coercion complete.")
        return df

    # ── 5. Drop Invalid Rows ───────────────────────────────────────────────────
    def _drop_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        # Drop rows with invalid datetime
        if "scheduled_dep" in df.columns:
            df = df.dropna(subset=["scheduled_dep"])
        # Drop rows where origin == destination
        if "origin" in df.columns and "destination" in df.columns:
            df = df[df["origin"] != df["destination"]]
        # Drop rows with negative delay_minutes
        if "delay_minutes" in df.columns:
            df = df[df["delay_minutes"] >= 0]
        dropped = before - len(df)
        self.report_["invalid_rows_dropped"] = dropped
        self._log(f"Invalid rows dropped: {dropped:,}")
        return df

    # ── Main Fit-Transform ─────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log(f"Starting clean. Input shape: {df.shape}")
        df = df.copy()
        df = self._remove_duplicates(df)
        df = self._impute_missing(df)
        df = self._enforce_types(df)
        df = self._clip_outliers(df, cols=[
            "wind_speed_kmh", "precipitation_mm", "delay_minutes",
            "gate_delay_min", "congestion_index"
        ])
        df = self._drop_invalid(df)
        df = df.reset_index(drop=True)
        self.report_["final_shape"] = df.shape
        self._log(f"Cleaning complete. Output shape: {df.shape}")
        self._log(f"Report: {self.report_}")
        return df


def load_and_clean(path=RAW_FLIGHTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    cleaner = DataCleaner(verbose=True)
    return cleaner.fit_transform(df)


if __name__ == "__main__":
    df = load_and_clean()
    print(df.dtypes)
    print(df.describe())
