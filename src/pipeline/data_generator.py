"""
data_generator.py — Generates synthetic flight, weather, and airport traffic data
Produces ~100,000 realistic records for training and analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    AIRLINES, AIRPORTS, WEATHER_CODES,
    N_SYNTHETIC_ROWS, RANDOM_SEED, RAW_FLIGHTS_CSV
)

rng = np.random.default_rng(RANDOM_SEED)


def _generate_flight_data(n: int) -> pd.DataFrame:
    """Core flight schedule and operational features."""
    # Date range: last 2 years
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2024, 12, 31)
    total_secs = int((end_date - start_date).total_seconds())
    random_secs = rng.integers(0, total_secs, size=n)
    scheduled_dep = [
        start_date + timedelta(seconds=int(s)) for s in random_secs
    ]

    airlines   = rng.choice(AIRLINES, size=n)
    origins    = rng.choice(AIRPORTS, size=n)
    # Destination must differ from origin
    dests = np.array([
        rng.choice([a for a in AIRPORTS if a != orig])
        for orig in origins
    ])

    # Flight number
    flight_nums = [
        f"{al[:2].upper()}{rng.integers(100, 9999)}"
        for al in airlines
    ]

    # Distance (km) — rough Indian domestic range
    distances = rng.integers(200, 3200, size=n).astype(float)

    # Scheduled flight duration (mins) ~ distance / 8 ± noise
    sched_duration = (distances / 8 + rng.normal(0, 15, n)).clip(40, 400)

    df = pd.DataFrame({
        "flight_id":       [f"FLT{i:07d}" for i in range(n)],
        "airline":          airlines,
        "flight_number":    flight_nums,
        "origin":           origins,
        "destination":      dests,
        "scheduled_dep":    scheduled_dep,
        "distance_km":      distances,
        "sched_duration_min": sched_duration.round(1),
    })
    return df


def _generate_weather_data(n: int, scheduled_dep: list) -> pd.DataFrame:
    """Weather conditions at departure time."""
    months = np.array([d.month for d in scheduled_dep])

    # India: monsoon June–Sep (month 6–9)
    is_monsoon = np.isin(months, [6, 7, 8, 9])
    base_precip = np.where(is_monsoon, 5.0, 0.2)

    temperature_c   = rng.normal(28, 8, n).clip(-5, 48).round(1)
    wind_speed_kmh  = rng.exponential(18, n).clip(0, 120).round(1)
    visibility_km   = rng.uniform(0.5, 15.0, n).round(2)
    precipitation_mm = (rng.exponential(base_precip, n)).clip(0, 80).round(2)
    humidity_pct    = rng.uniform(20, 100, n).round(1)

    # Weather code — biased towards clear/partly cloudy
    weather_probs   = [0.30, 0.25, 0.15, 0.08, 0.08, 0.06, 0.05, 0.03]
    weather_code    = rng.choice(list(WEATHER_CODES.keys()), size=n, p=weather_probs)
    weather_label   = [WEATHER_CODES[c] for c in weather_code]

    return pd.DataFrame({
        "temperature_c":    temperature_c,
        "wind_speed_kmh":   wind_speed_kmh,
        "visibility_km":    visibility_km,
        "precipitation_mm": precipitation_mm,
        "humidity_pct":     humidity_pct,
        "weather_code":     weather_code,
        "weather_label":    weather_label,
    })


def _generate_traffic_data(n: int, scheduled_dep: list) -> pd.DataFrame:
    """Airport congestion and operational traffic."""
    hours = np.array([d.hour for d in scheduled_dep])

    # Peak hours: 6–9, 17–21
    is_morning_peak  = (hours >= 6)  & (hours <= 9)
    is_evening_peak  = (hours >= 17) & (hours <= 21)
    congestion_base  = np.where(is_morning_peak | is_evening_peak, 0.70, 0.35)

    congestion_index = (congestion_base + rng.normal(0, 0.12, n)).clip(0.0, 1.0).round(3)
    runway_occupancy = rng.uniform(0.1, 0.95, n).round(3)
    gate_delay_min   = rng.exponential(4, n).clip(0, 45).round(1)
    taxiing_time_min = rng.normal(15, 5, n).clip(3, 40).round(1)

    return pd.DataFrame({
        "congestion_index":   congestion_index,
        "runway_occupancy":   runway_occupancy,
        "gate_delay_min":     gate_delay_min,
        "taxiing_time_min":   taxiing_time_min,
    })


def _generate_delay_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic delay_minutes and is_delayed labels
    using domain-informed logic across weather, traffic, and time features.
    """
    n = len(df)

    # Base delay probability per feature signals
    bad_weather    = ((df["precipitation_mm"] > 10) |
                      (df["wind_speed_kmh"]   > 60) |
                      (df["visibility_km"]    <  2) |
                      (df["weather_code"]    >=  4)).astype(float) * 0.28
    high_traffic   = (df["congestion_index"] > 0.65).astype(float) * 0.25
    gate_delay     = (df["gate_delay_min"]   > 10).astype(float)  * 0.12
    long_route     = (df["distance_km"]      > 1800).astype(float)* 0.05

    hours = np.array([d.hour for d in df["scheduled_dep"]])
    peak_hour = (((hours >= 6) & (hours <= 9)) |
                 ((hours >= 17) & (hours <= 21))).astype(float) * 0.10

    delay_prob = (0.15 + bad_weather + high_traffic + gate_delay +
                  long_route + peak_hour).clip(0.05, 0.95)

    is_delayed_arr     = rng.binomial(1, delay_prob, n)
    delay_base         = rng.exponential(35, n)       # mean ~35 min
    delay_minutes_arr  = np.where(
        is_delayed_arr == 1,
        (delay_base + rng.normal(0, 10, n)).clip(15, 360),
        rng.uniform(0, 14, n)                         # on-time: <15 min variance
    ).round(1)

    df["delay_minutes"] = delay_minutes_arr
    df["is_delayed"]    = is_delayed_arr
    return df


def generate_dataset(n: int = N_SYNTHETIC_ROWS, save: bool = True) -> pd.DataFrame:
    """
    Full synthetic dataset generation pipeline.
    Returns a combined DataFrame and optionally saves to CSV.
    """
    print(f"[DataGenerator] Generating {n:,} synthetic flight records...")

    flight_df  = _generate_flight_data(n)
    weather_df = _generate_weather_data(n, flight_df["scheduled_dep"].tolist())
    traffic_df = _generate_traffic_data(n, flight_df["scheduled_dep"].tolist())

    df = pd.concat([flight_df, weather_df, traffic_df], axis=1)
    df = _generate_delay_targets(df)

    # Inject controlled nulls (simulates real-world data quality issues)
    null_cols  = ["visibility_km", "precipitation_mm", "gate_delay_min", "humidity_pct"]
    null_frac  = 0.03
    for col in null_cols:
        mask = rng.random(n) < null_frac
        df.loc[mask, col] = np.nan

    # Inject duplicates (~0.5%)
    dup_indices = rng.choice(n, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

    if save:
        df.to_csv(RAW_FLIGHTS_CSV, index=False)
        print(f"[DataGenerator] Saved raw data → {RAW_FLIGHTS_CSV}")
        print(f"[DataGenerator] Shape: {df.shape} | Delay rate: {df['is_delayed'].mean():.2%}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
