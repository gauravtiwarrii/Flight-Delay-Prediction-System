"""
run_pipeline.py — Master orchestrator for Flight Delay Prediction System
Runs all pipeline steps in sequence.

Usage:
    python run_pipeline.py                # full pipeline
    python run_pipeline.py --skip-data   # skip data generation (reuse existing)
    python run_pipeline.py --skip-train  # skip model training
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_FLIGHTS_CSV, PROCESSED_CSV, BEST_MODEL_PATH,
    METRICS_PATH, SQLITE_PATH
)


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")


def run_pipeline(skip_data=False, skip_train=False):
    start = time.time()

    # ── Step 1: Data Generation ────────────────────────────────────────────
    step("1/6  Data Generation")
    if skip_data and RAW_FLIGHTS_CSV.exists():
        print(f"[SKIP] Using existing raw data: {RAW_FLIGHTS_CSV}")
    else:
        from src.pipeline.data_generator import generate_dataset
        generate_dataset()

    # ── Step 2: Data Cleaning ──────────────────────────────────────────────
    step("2/6  Data Cleaning")
    from src.pipeline.data_cleaner import DataCleaner
    import pandas as pd
    df_raw    = pd.read_csv(RAW_FLIGHTS_CSV)
    cleaner   = DataCleaner(verbose=True)
    df_clean  = cleaner.fit_transform(df_raw)
    print(f"[Pipeline] Clean shape: {df_clean.shape}")

    # ── Step 3: Feature Engineering ────────────────────────────────────────
    step("3/6  Feature Engineering")
    from src.pipeline.feature_engineer import engineer_features
    df_feat, fe = engineer_features(df_clean, save_artifacts=True)
    print(f"[Pipeline] Feature shape: {df_feat.shape}")

    # ── Step 4: Load into Database ─────────────────────────────────────────
    step("4/6  Database Ingestion")
    from src.pipeline.db_loader import DBLoader
    db = DBLoader()
    db.create_tables()
    db.load_raw(df_clean)
    db.load_processed(df_feat)

    # ── Step 5: Model Training ─────────────────────────────────────────────
    step("5/6  Model Training")
    if skip_train and BEST_MODEL_PATH.exists():
        print(f"[SKIP] Using existing model: {BEST_MODEL_PATH}")
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        from src.models.trainer import train_all
        metrics = train_all(csv_path=PROCESSED_CSV)

    # Save model metrics to DB
    db.save_metrics({
        k: v for k, v in metrics.items() if not k.startswith("_")
    })

    # ── Step 6: Evaluation & Plots ─────────────────────────────────────────
    step("6/6  Model Evaluation")
    from src.models.evaluator import full_evaluation
    full_evaluation()

    db.close()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    best = metrics.get("_meta", {}).get("best_model", "Unknown")
    best_f1 = metrics.get(best, {}).get("f1", 0)
    print(f"\n  Best model : {best}  (F1={best_f1:.4f})")
    print(f"  Database   : {SQLITE_PATH}")
    print(f"  Model      : {BEST_MODEL_PATH}")
    print(f"\n  To start the API:       python src/api/app.py")
    print(f"  To start the dashboard: streamlit run src/dashboard/streamlit_app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flight Delay Prediction — Pipeline Orchestrator"
    )
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip data generation if raw CSV already exists")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training if model already saved")
    args = parser.parse_args()

    try:
        run_pipeline(skip_data=args.skip_data, skip_train=args.skip_train)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
