"""
trainer.py — Trains Logistic Regression, Random Forest, and XGBoost models.
Selects best by F1 score and saves artifacts.
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_CSV, FEATURE_COLS, TARGET_BINARY,
    BEST_MODEL_PATH, METRICS_PATH, CV_FOLDS,
    TRAIN_TEST_SPLIT, RANDOM_SEED
)


MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=500, C=1.0, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_SEED
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=10, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=2,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
    ),
}


def load_features(csv_path=PROCESSED_CSV) -> tuple:
    df = pd.read_csv(csv_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[TARGET_BINARY].values.astype(int)
    return X, y, available


def evaluate_model(model, X, y) -> dict:
    """Cross-validated evaluation on the full dataset."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                             return_train_score=False, n_jobs=-1)
    return {
        "accuracy":  float(results["test_accuracy"].mean()),
        "precision": float(results["test_precision"].mean()),
        "recall":    float(results["test_recall"].mean()),
        "f1":        float(results["test_f1"].mean()),
        "roc_auc":   float(results["test_roc_auc"].mean()),
    }


def train_all(csv_path=PROCESSED_CSV) -> dict:
    """Train all models, select best, save to disk."""
    print("[Trainer] Loading processed features...")
    X, y, feature_names = load_features(csv_path)
    print(f"[Trainer] Features: {len(feature_names)}  |  Samples: {len(y):,}  |  "
          f"Delay rate: {y.mean():.2%}")

    all_metrics: dict = {}

    for name, model in MODELS.items():
        print(f"\n[Trainer] Training {name}...")
        metrics = evaluate_model(model, X, y)
        all_metrics[name] = metrics
        print(
            f"  Accuracy : {metrics['accuracy']:.4f}  |  "
            f"Precision: {metrics['precision']:.4f}  |  "
            f"Recall   : {metrics['recall']:.4f}  |  "
            f"F1       : {metrics['f1']:.4f}  |  "
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

    # Select best by F1
    best_name  = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_model = MODELS[best_name]
    print(f"\n[Trainer] ✓ Best model: {best_name} (F1={all_metrics[best_name]['f1']:.4f})")

    # Final fit on full data
    best_model.fit(X, y)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"[Trainer] Saved model → {BEST_MODEL_PATH}")

    # Mark best in metrics
    for k in all_metrics:
        all_metrics[k]["is_best"] = (k == best_name)
    all_metrics["_meta"] = {
        "best_model": best_name,
        "features":   feature_names,
        "n_samples":  int(len(y)),
        "delay_rate": float(y.mean()),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[Trainer] Metrics saved → {METRICS_PATH}")

    return all_metrics


if __name__ == "__main__":
    metrics = train_all()
